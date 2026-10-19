// src/kernels/cuda/allreduce.cu - the cross-GPU sum, see include/strata/kernels/allreduce.hpp.
//
// A port of `third_party/llama.cpp/ggml/src/ggml-cuda/allreduce.cu:62-202` (the "internal" path: two GPUs
// in one process, no NCCL, no NVLink).  The three differences from it - the device-side token, ping-pong
// staging and the monotonic arrival compare - are argued in the header; this file implements them.
#include "strata/kernels/allreduce.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace strata::kernels {
namespace {

/// One arrival slot per block, 64 B apart: two cards polling one cache line is the false-sharing stall the
/// upstream comment calls out (allreduce.cu:69-72).  The stride is a compile-time constant because the KERNEL
/// and the host both index these slots - the host from `release()` and from the stall report - and a kernel
/// indexing them 4 bytes apart while the host writes them 64 bytes apart would leave every block but the
/// first one waiting on a slot nobody writes.
constexpr int kArrivalInts = 16;                       // 64 B / sizeof(uint32_t)
constexpr size_t kSlotBytes = kArrivalInts * sizeof(uint32_t);

/// The give-up bound, and WHY IT HAS TO BE THIS SHORT.
///
/// A waiter spins with `__nanosleep`, which is a HINT: its real duration varies by an order of magnitude
/// across drivers.  MEASURED on the RTX 3080 pair this was written on, the loop advances at roughly 1 us per
/// turn against the instruction's documented ~100 ns.  So the first value here (50M) was ~50 SECONDS per
/// call and the second (5M) still ~5 s - which is not a safety net, it is an outage: while a kernel pair
/// spun, Windows logged repeated `nvlddmkm` event 153 (the driver's own hang report), the desktop froze,
/// Task Manager hung, and every GPU application on the machine - the engine, and the harness this was being
/// developed in - had to be restarted.  A window's legitimate wait is under a millisecond, so the bound only
/// has to cover a scheduling hiccup: 20k turns is ~20 ms even at the pessimistic microsecond per turn.
///
/// THE RULE THIS ENCODES: never hold a GPU for seconds waiting on another one.  If the peer is gone, give up
/// and REPORT it (`timed_out_rank`), so the engine can refuse the window and restart the stage - do not spin.
/// RAISED FROM 20k, AND THE REASON IS MEASURED.  The bound above assumes "a window's legitimate wait is under a
/// millisecond", which holds for two ranks inside one window - but the two ranks of a TENSOR-PARALLEL window do
/// not enter the next window together: rank 0 also runs the sampler and the state commit between windows, so its
/// next reduce is issued later than rank 1's.  Measured on the RTX 3080 pair with the per-call instrumentation:
/// both ranks called reduce 49 times, in the same order, and rank 1 then gave up waiting at call 49 (the first
/// reduce of the second window) while rank 0's call 49 arrived afterwards.  The counts prove the protocol is
/// intact; only the bound was too tight.  200k turns is ~200 ms even at the pessimistic microsecond per turn.
///
/// RAISED FROM 200k TO 4M, measured on the same pair.  The give-up was firing while the two ranks HAD met: the
/// `TPDIAG stall` dump prints, from the host, `signalled=384 saw_peer=384` on BOTH ranks at the moment of the
/// give-up - and `tp_host_runtime.hpp`'s own note reads "both non-zero on both sides with a give-up means the
/// visibility of the write, not the order".  A false give-up is not harmless: the caller aborts the window
/// (`late >= 0`), and a run that reached it produced `HTTP 400` with the layer-0 residual split
/// (rank0 14.5292413 / rank1 12.3435555), while every run whose layer-0 residual agreed also produced the
/// single card's token.  Under WDDM one side's launch can be delayed by SECONDS, so the bound has to sit above
/// that jitter; a genuinely stuck window is still caught by the engine's own watchdog, which is what printed
/// `verify: an earlier window never finished on the GPU (#267); restart the engine` in this session's runs.
// MEASURED AND REVERTED: raising this x10 (to 40000000) for the pair did NOT unblock the probe runs (still a
// 420 s timeout) and did not improve the 8-sample spread (3 distinct against 2).  So the give-up bound is not
// what stalls a probed pair, and the engine's value is left as it was.
constexpr uint64_t kGiveUpSpins = 4000000ull;
    // give-up path, which used to deadlock (asymmetric) and still read unvalidated peer staging.  With the
    // symmetric give-up in place a real abandonment releases both ranks, so the cheap way to tell a genuine
    // hang from jitter is to raise the bound and see whether the stalls disappear.u;
constexpr uint32_t kTimedOut = 0xDEADBEEFu;

__device__ __forceinline__ void ar_signal(uint32_t * p, uint32_t token) { *(volatile uint32_t *) p = token; }
__device__ __forceinline__ uint32_t ar_read(const uint32_t * p) { return *(const volatile uint32_t *) p; }

/// bf16 as this tree stores it: the high 16 bits of an f32 (`include/strata/kernels/gr.hpp:10`).  The round
/// is to nearest even, what `ggml_cuda_cast` does, and it must round rather than truncate: the two ranks
/// compare their sums bit for bit, and a truncation biases both halves the same way.  A carry out of the
/// low half cannot reach the sign or turn an infinity into a NaN (0x7F80 +/- stays 0x7F80).
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/// The two wire paths differ only in staging and reading back, so the kernel body is shared through these.
template <typename Wire> struct WireOps;
template <> struct WireOps<float> {
    using type = float;
    static constexpr int elems = 4;                                       // 16 B per store
    static __device__ __forceinline__ type from(float v) { return v; }
    static __device__ __forceinline__ float to(type v) { return v; }
};
template <> struct WireOps<uint16_t> {
    using type = uint16_t;
    static constexpr int elems = 8;                                       // 16 B per store
    static __device__ __forceinline__ type from(float v) { return f32_to_bf16(v); }
    static __device__ __forceinline__ float to(type v) { return bf16_to_f32(v); }
};

/// One reduce.  Every block stages its own stripe of `x` into this rank's half of the staging region,
/// signals its arrival token, waits for the peer's, then adds the peer's half back into `x`, in place.
/// `x` is read for staging and read again for the add, and written once; nothing touches it in between, so
/// in place is safe.  Both operands round through the wire type before the add, which is what makes the two
/// ranks agree bit for bit instead of merely agreeing closely.
template <typename Wire>
__global__ void ar_reduce_kernel(float * x, typename WireOps<Wire>::type * mine0,
                                 typename WireOps<Wire>::type * mine1,
                                 const typename WireOps<Wire>::type * peer0,
                                 const typename WireOps<Wire>::type * peer1,
                                 const uint32_t * peer_timeout, int count,
                                 uint32_t * my_arrival, const uint32_t * peer_arrival,
                                 uint32_t * token, uint32_t * timed_out, uint32_t * progress) {
    using Ops = WireOps<Wire>;
    using T = typename Ops::type;
    constexpr int ELEMS = Ops::elems;

    const int tid = threadIdx.x, bid = blockIdx.x;
    const int gtid = bid * blockDim.x + tid, gnt = gridDim.x * blockDim.x;
    const int nvec = count / ELEMS, tail = nvec * ELEMS;
    if (bid == 0 && tid == 0) *progress = 1u;      // kArProgressEntered

    // the staging half is chosen by the HOST (`calls_[r] & 1`, one value per rank).  A device-derived parity was
    // tried and reverted: the token is per BLOCK, so different blocks of one rank could pick different halves.
    typename WireOps<Wire>::type * const mine_h = mine0;
    const typename WireOps<Wire>::type * const peer_h = peer0;

    union { T w[ELEMS]; uint4 v; } u;
    for (int i = gtid; i < nvec; i += gnt) {
        const int off = i * ELEMS;
        #pragma unroll
        for (int k = 0; k < ELEMS; ++k) u.w[k] = Ops::from(x[off + k]);
        *(uint4 *) (mine_h + off) = u.v;
    }
    if (bid == 0 && tid < count - tail) mine_h[tail + tid] = Ops::from(x[tail + tid]);

    __threadfence_system();
    __syncthreads();

    if (bid == 0 && tid == 0) *(volatile uint32_t *) (progress + 1) = 1u;   // kArProgressStaged

    if (tid == 0) {
        // ONE BUMP PER EXECUTION, PER BLOCK.  The blocks of one rank must all derive the SAME token or the
        // peer's per-block arrival ints would be compared against different numbers, and blocks cannot share
        // a register - so each block owns a counter.  A device atomic and not a host value: a captured graph
        // would otherwise replay the same token forever (see the header).
        const uint32_t tok = atomicAdd(&token[bid], 1u) + 1u;
        ar_signal(&my_arrival[bid * kArrivalInts], tok);
        if (bid == 0) *(volatile uint32_t *) (progress + 2) = tok;           // kArProgressSignalled
        __threadfence_system();       // our staged bytes are visible system-wide before the token
        uint32_t spins = 0;
        // MEASURED, NOT ASSUMED.  Changing this to `!= tok` (an exact call match) did NOT fix the divergence:
        // eight identical requests then gave ['-','The','The','The','The','用户','The','The'] - the steady state
        // was still the wrong token and still varied, so the change is reverted rather than kept on a theory.
        // The `>= tok` form is what the two-rank test in `tp_allreduce_test` passes with, and the remaining
        // divergence is being chased with the same 8-request measurement.
        while (ar_read(&peer_arrival[bid * kArrivalInts]) < tok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(100);
#elif defined(__HIP_PLATFORM_AMD__)
            __builtin_amdgcn_s_sleep(4);      // allreduce.cu:167
#endif
            // SYMMETRIC GIVE-UP: the peer may already have abandoned this call.  Without this check the two
            // ranks wait for each other until each hits its own bound (the engine reports it as a stall at
            // whichever layer was in flight), and then this rank reads staging the peer never wrote.
            if (peer_timeout != nullptr && ar_read(peer_timeout) == kTimedOut) {
                *timed_out = kTimedOut;
                break;
            }
            if (++spins >= kGiveUpSpins) { *timed_out = kTimedOut; break; }
        }
        if (bid == 0) *(volatile uint32_t *) (progress + 3) = ar_read(&peer_arrival[0]);
    }
    __syncthreads();
    __threadfence_system();           // acquire the peer's staged bytes

    union { T w[ELEMS]; uint4 v; } p;
    for (int i = gtid; i < nvec; i += gnt) {
        const int off = i * ELEMS;
        p.v = *(const uint4 *) (peer_h + off);
        #pragma unroll
        for (int k = 0; k < ELEMS; ++k) x[off + k] = Ops::to(Ops::from(x[off + k])) + Ops::to(p.w[k]);
    }
    if (bid == 0 && tid < count - tail)
        x[tail + tid] = Ops::to(Ops::from(x[tail + tid])) + Ops::to(peer_h[tail + tid]);
}

inline size_t ar_region_bytes(int64_t max_floats, ArWire w) {
    const size_t raw = (size_t) max_floats * (w == ArWire::F32 ? sizeof(float) : sizeof(uint16_t));
    return (raw + 15u) & ~(size_t) 15u;     // the staging stores are 16 B
}

/// Where each piece sits inside one rank's mapped-pinned allocation.  One place computes it, so the host and
/// device pointer arithmetic cannot drift apart - a mismatch there is a kernel writing over another region's
/// bytes, which decodes to plausible values and produces plausible output.
struct ArLayout {
    size_t region = 0;
    size_t arrival = 0;
    size_t timeout = 0;
    size_t progress = 0;
    size_t total = 0;
};
inline ArLayout ar_layout(size_t region_bytes, int blocks) {
    ArLayout L;
    L.arrival = 2 * region_bytes;
    L.timeout = L.arrival + 2 * (size_t) blocks * kSlotBytes;
    L.progress = L.timeout + kSlotBytes;
    L.total = L.progress + kSlotBytes;
    return L;
}

}  // namespace

ArPair::~ArPair() { shutdown(); }

bool ArPair::init(const int * ordinals, int64_t max_floats, ArWire wire, int blocks, std::string & err) {
    if (ready_) { err = "allreduce: the pair is already set up"; return false; }
    if (ordinals == nullptr) { err = "allreduce: no devices given"; return false; }
    if (max_floats < 1) {
        err = "allreduce: max_floats must be positive (got " + std::to_string(max_floats) + ")";
        return false;
    }
    if (blocks < 1 || blocks > 64) {
        err = "allreduce: blocks must be 1..64 (got " + std::to_string(blocks) + ")";
        return false;
    }
    // TWO RANKS MUST BE ON DIFFERENT CARDS.  An earlier edit here allowed a same-device pair as an A/B, and the
    // engine's own collective test refuses it (`tests/core/tp_allreduce_test.cu:109-110` checks both the refusal
    // and the message) - so that allowance left `tp_allreduce_test` red, which the tensor-parallel work must not
    // do: the tests are part of what "usable" means.  A same-device A/B can be run as two processes instead.
    if (ordinals[0] == ordinals[1]) {
        // MEASURED AND REVERTED (round 1 of the new plan): the same-device A/B was tried as "remove the cross-card
        // expert path and see whether the divergence goes".  It does not run: two ranks on one card have to execute
        // concurrently, and whoever holds the device first leaves the other's kernel unable to start, so the pair
        // spins at the first collective until the 420 s harness timeout.  That is exactly why this refusal exists
        // and why the engine's own test demands it, so the allowance is gone again.
        err = "allreduce: both ranks on device " + std::to_string(ordinals[0]) +
              " - this is not a tensor-parallel pair";
        return false;
    }
    dev_[0] = ordinals[0];
    dev_[1] = ordinals[1];
    wire_ = wire;
    blocks_ = blocks;
    max_floats_ = max_floats;

    const size_t region_bytes = ar_region_bytes(max_floats, wire);
    const ArLayout L = ar_layout(region_bytes, blocks);

    for (int r = 0; r < kArRanks; ++r) {
        if (cudaSetDevice(dev_[r]) != cudaSuccess) {
            err = "allreduce: cudaSetDevice(" + std::to_string(dev_[r]) + ") failed";
            shutdown();
            return false;
        }
        // Mapped + Portable, the allocation style of the layer split's hand-off
        // (src/program/generate.cpp:7829): one allocation both cards can see, so a rank's kernel writes bytes
        // the other reads with no IPC handle and no per-buffer registration.
        void * h = nullptr;
        const cudaError_t e = cudaHostAlloc(&h, L.total, cudaHostAllocMapped | cudaHostAllocPortable);
        if (e != cudaSuccess) {
            err = "allreduce: rank " + std::to_string(r) + ": " + std::to_string(L.total >> 20) +
                  " MiB of mapped pinned memory: " + cudaGetErrorString(e);
            shutdown();
            return false;
        }
        host_[r] = h;
        std::memset(h, 0, L.total);

        void * dv = nullptr;
        if (cudaHostGetDevicePointer(&dv, h, 0) != cudaSuccess) {
            err = "allreduce: rank " + std::to_string(r) + ": the mapped pinned memory has no device pointer";
            shutdown();
            return false;
        }
        uint8_t * hb = (uint8_t *) h, * db = (uint8_t *) dv;
        view_[r] = ArRankView{};
        view_[r].dev = dev_[r];
        // The flush event: one per rank, reused by every `flush` call.  Timing disabled, so it is a marker and
        // not a clock read (see the header for why a trailing submission is mandatory under WDDM).
        cudaEvent_t fev = nullptr;
        if (cudaEventCreateWithFlags(&fev, cudaEventDisableTiming) != cudaSuccess) {
            err = "allreduce: rank " + std::to_string(r) + ": the flush event could not be created";
            shutdown();
            return false;
        }
        flush_ev_[r] = (void *) fev;
        for (int p = 0; p < 2; ++p) {
            view_[r].mine[p] = db + L.region + (size_t) p * region_bytes;
            view_[r].arrival[p] = (uint32_t *) (db + L.arrival + (size_t) p * (size_t) blocks * kSlotBytes);
            h_arrival_[r][p] = (uint32_t *) (hb + L.arrival + (size_t) p * (size_t) blocks * kSlotBytes);
        }
        view_[r].timeout = (uint32_t *) (db + L.timeout);
        h_timeout_[r] = (uint32_t *) (hb + L.timeout);
        view_[r].progress = (uint32_t *) (db + L.progress);
        h_progress_[r] = (uint32_t *) (hb + L.progress);
        // the peer's halves are filled in once both ranks have allocated; the token counters are the one
        // piece that is real device memory, because the kernel bumps them rather than the host.
        if (cudaMalloc((void **) &view_[r].token, (size_t) blocks * sizeof(uint32_t)) != cudaSuccess ||
            cudaMemset(view_[r].token, 0, (size_t) blocks * sizeof(uint32_t)) != cudaSuccess) {
            err = "allreduce: rank " + std::to_string(r) + ": the token counters could not be allocated";
            shutdown();
            return false;
        }
        pinned_bytes_ += L.total;
    }

    // Cross-wire the two views.  Rank r writes its own region 0 and reads the peer's region 1, and the other
    // way round for rank 1, so calls never share a buffer.
    for (int r = 0; r < kArRanks; ++r) {
        const int o = 1 - r;
        if (cudaSetDevice(dev_[r]) != cudaSuccess) {
            err = "allreduce: cudaSetDevice(" + std::to_string(dev_[r]) + ") failed";
            shutdown();
            return false;
        }
        void * dv = nullptr;
        if (cudaHostGetDevicePointer(&dv, host_[o], 0) != cudaSuccess) {
            err = "allreduce: rank " + std::to_string(r) + " cannot see rank " + std::to_string(o) +
                  "'s staged memory; the two ranks must be in one process on one host";
            shutdown();
            return false;
        }
        uint8_t * db = (uint8_t *) dv;
        for (int p = 0; p < 2; ++p) {
            view_[r].peer[p] = db + L.region + (size_t) p * region_bytes;
            // What rank r waits on is the OTHER rank's arrival area, and it has to be THIS device's view of
            // those bytes - not the pointer rank o's own kernels use.  Under UVA the two happen to be the same
            // address, and taking it from this rank's own mapping is what makes that an accident that cannot
            // bite.
            view_[r].peer_arrival[p] = (const uint32_t *) (db + L.arrival + (size_t) p * (size_t) blocks * kSlotBytes);
        }
        // The peer's GIVE-UP word, seen from THIS rank's device - the same reasoning as `peer_arrival` above.
        view_[r].peer_timeout = (const uint32_t *) (db + L.timeout);
    }

    int a2b = 0, b2a = 0;
    (void) cudaDeviceCanAccessPeer(&a2b, dev_[0], dev_[1]);
    (void) cudaDeviceCanAccessPeer(&b2a, dev_[1], dev_[0]);
    peer_mode_ = (a2b && b2a) ? 1 : 0;

    calls_[0] = calls_[1] = 0;
    ready_ = true;
    return true;
}

void ArPair::shutdown() {
    for (int r = 0; r < kArRanks; ++r) {
        if (flush_ev_[r] != nullptr) {
            (void) cudaSetDevice(dev_[r] < 0 ? 0 : dev_[r]);
            (void) cudaEventDestroy((cudaEvent_t) flush_ev_[r]);
            flush_ev_[r] = nullptr;
        }
        if (view_[r].token != nullptr) {
            (void) cudaSetDevice(dev_[r] < 0 ? 0 : dev_[r]);
            (void) cudaFree(view_[r].token);
        }
        if (host_[r] != nullptr) (void) cudaFreeHost(host_[r]);
        host_[r] = nullptr;
        view_[r] = ArRankView{};
        h_arrival_[r][0] = h_arrival_[r][1] = nullptr;
        h_timeout_[r] = nullptr;
        h_progress_[r] = nullptr;
    }
    pinned_bytes_ = 0;
    peer_mode_ = -1;
    ready_ = false;
}

namespace {
const ArPair* g_ar_debug = nullptr;
}

const void* ArPair::host_staging(int rank, int half) const {
    if (rank < 0 || rank >= kArRanks || host_[rank] == nullptr) return nullptr;
    const size_t rb = ar_region_bytes(max_floats_, wire_);
    return (const uint8_t*) host_[rank] + (size_t) (half ? 1 : 0) * rb;
}

void ar_debug_set_pair(ArPair* p) { g_ar_debug = p; }

const void* ar_debug_staging(int rank, int half) {
    return g_ar_debug == nullptr ? nullptr : g_ar_debug->host_staging(rank, half);
}

void ArPair::launch(int rank, float * x, int count, void * stream) {
    const int r = rank <= 0 ? 0 : (rank >= kArRanks ? kArRanks - 1 : rank);
    ++calls_[r];
    const ArRankView & v = view_[r];
    // The host call count selects the staging parity: both ranks must use the same parity for the same
    // logical call, and alternating parities keep call k+1 from overwriting the staging call k is reading.
    const int parity = (int) (calls_[r] & 1ull);
    if (wire_ == ArWire::F32) {
        ar_reduce_kernel<float><<<blocks_, kArThreads, 0, (cudaStream_t) stream>>>(
            x, (float *) v.mine[parity], (float *) v.mine[1 - parity], (const float *) v.peer[parity],
            (const float *) v.peer[1 - parity], v.peer_timeout, count, v.arrival[parity],
            v.peer_arrival[parity], v.token, v.timeout, v.progress);
    } else {
        ar_reduce_kernel<uint16_t><<<blocks_, kArThreads, 0, (cudaStream_t) stream>>>(
            x, (uint16_t *) v.mine[parity], (uint16_t *) v.mine[1 - parity], (const uint16_t *) v.peer[parity],
            (const uint16_t *) v.peer[1 - parity], v.peer_timeout, count, v.arrival[parity],
            v.peer_arrival[parity], v.token, v.timeout, v.progress);
    }
}

bool ArPair::reduce_host(int rank, float * x, int count, void * stream, std::string & err) {
    if (!ready_) { err = "allreduce: reduce before a successful init"; return false; }
    if (rank < 0 || rank >= kArRanks) {
        err = "allreduce: rank " + std::to_string(rank) + " is outside 0.." + std::to_string(kArRanks - 1);
        return false;
    }
    if (x == nullptr) { err = "allreduce: null buffer"; return false; }
    if (count < 1 || count > max_floats_) {
        err = "allreduce: count " + std::to_string(count) + " is outside 1.." + std::to_string(max_floats_) +
              "; the pair is sized for a verify window, and a prompt chunk goes through the copy-engine route";
        return false;
    }
    // A launch goes to the CURRENT device, not to the stream's, so a caller that has not set its device
    // queues the kernel on the wrong card and the peer spins until it gives up.  `launch` leaves that to the
    // caller - inside a capture the engine has already set it - and this checked entry point cannot.
    if (cudaSetDevice(dev_[rank]) != cudaSuccess) {
        err = "allreduce: cudaSetDevice(" + std::to_string(dev_[rank]) + ") failed";
        return false;
    }
    launch(rank, x, count, stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = "allreduce: rank " + std::to_string(rank) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

void ArPair::flush(int rank, void * stream) {
    // A RANK ON THE DEFAULT STREAM STILL NEEDS THIS, AND RETURNING EARLY HERE IS WHAT MADE THE MEETING
    // INTERMITTENT.  Under WDDM a launch sits in the command buffer until a later submission arrives on that
    // stream.  `TpRank` falls back to the DEFAULT stream when `cudaStreamCreateWithFlags` fails, and this
    // function then returned without submitting anything - so that rank's reduce kernel never began, its peer
    // waited, and the window died with `rank N gave up waiting for its peer`.  Measured on two cards: the same
    // binary and config completed a request, then failed the next run.  `cudaEventRecord(ev, nullptr)` is a
    // submission on the LEGACY DEFAULT stream, which is exactly the stream such a rank launched on, so passing
    // the null through is the correct flush rather than skipping it.
    // TPDIAG: STRATA_TP_OLD_FLUSH=1 restores the pre-round-19 early return, which is how the two arms of the
    // test that asks whether that change is what stopped the collective summing correctly are run.
    if (const char* of = std::getenv("STRATA_TP_OLD_FLUSH"); of != nullptr && of[0] == (char) 49) {
        if (stream == nullptr) return;
    }
    if (!ready_) return;
    const int r = rank <= 0 ? 0 : (rank >= kArRanks ? kArRanks - 1 : rank);
    if (cudaSetDevice(dev_[r]) != cudaSuccess) return;
    if (flush_ev_[r] == nullptr) return;
    (void) cudaEventRecord((cudaEvent_t) flush_ev_[r], (cudaStream_t) stream);
}

void ArPair::release() {
    if (!ready_) return;
    // Past any token, in both parities of both ranks: `peer_arrival < token` is then false for every waiter,
    // so a rank whose peer is gone drains instead of spinning.  The values are NOT the sum afterwards, so the
    // caller must refuse the window - the contract of Verifier::release_gpu_waits.
    for (int r = 0; r < kArRanks; ++r)
        for (int p = 0; p < 2; ++p)
            for (int b = 0; b < blocks_; ++b)
                *(volatile uint32_t *) (h_arrival_[r][p] + (size_t) b * (kSlotBytes / sizeof(uint32_t))) =
                    0xFFFFFFFFu;
}

void ArPair::clear_timeout() {
    if (!ready_) return;
    for (int r = 0; r < kArRanks; ++r)
        if (h_timeout_[r] != nullptr) *(volatile uint32_t *) h_timeout_[r] = 0u;
}

int ArPair::timed_out_rank() const {
    if (!ready_) return -1;
    for (int r = 0; r < kArRanks; ++r)
        if (*(volatile const uint32_t *) h_timeout_[r] == kTimedOut) return r;
    return -1;
}

bool ArPair::token_counts(uint32_t * out) const {
    if (!ready_ || out == nullptr) return false;
    for (int r = 0; r < kArRanks; ++r) {
        if (cudaSetDevice(dev_[r]) != cudaSuccess) return false;
        if (cudaMemcpy(&out[r], view_[r].token, sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
    }
    return true;
}

uint32_t ArPair::peek_progress(int rank, int which) const {
    if (!ready_ || which < 0 || which > kArProgressSawPeer) return 0;
    return *(volatile const uint32_t *) (h_progress_[rank <= 0 ? 0 : kArRanks - 1] + which);
}

uint32_t ArPair::peek_arrival(int rank, bool peer, int parity, int block) const {
    if (!ready_) return 0;
    const int r = !peer ? (rank <= 0 ? 0 : kArRanks - 1) : (rank <= 0 ? kArRanks - 1 : 0);
    const int p = parity & 1;
    const int b = block < 0 ? 0 : (block >= blocks_ ? blocks_ - 1 : block);
    return *(volatile const uint32_t *) (h_arrival_[r][p] + (size_t) b * kArrivalInts);
}

}  // namespace strata::kernels
