// include/strata/kernels/allreduce.hpp - the cross-GPU sum tensor parallelism needs.
//
// The protocol is llama.cpp's (`third_party/llama.cpp/ggml/src/ggml-cuda/allreduce.cu`): the two ranks
// stage their halves through MAPPED PINNED HOST MEMORY, meet on an arrival token, and each operand passes
// through a narrow WIRE TYPE before the add.  The wire rounding is not an optimisation detail: it is what
// makes the two cards' results BIT-IDENTICAL, because both sides truncate to the same type before adding.
// Both cards agreeing is a requirement here, not a nicety - every later layer reads the reduced value, and
// the router must pick the same expert ids on both ranks or the two cards compute different models.
//
// THREE DELIBERATE DIFFERENCES FROM llama.cpp.  Each is a bug in the captured-graph world, not a taste:
//
//   1. THE TOKEN COMES FROM THE DEVICE.  llama.cpp's caller holds the token and passes it in.  Strata's
//      verify window is ONE CAPTURED CUDA GRAPH replayed hundreds of times (`core/verify.hpp`), and a token
//      baked into a graph is the same on every replay: the peer's arrival int would still hold it from the
//      previous replay, so the wait would fall through BEFORE the bytes were staged.  Here each block owns a
//      device counter and bumps it with one atomicAdd per execution.  A counter per BLOCK and not one per
//      rank, because the blocks of one rank must all derive the same token and blocks cannot share a
//      register, and because one slot per block is what the arrival protocol already uses.
//
//   2. PING-PONG STAGING.  With a single staging region the peer's call k+1 overwrites the bytes this rank
//      is still reading for call k - the writing side runs ahead by design, since its spin holds it until the
//      peer ARRIVES, not until the peer has finished reading.  Two regions selected by call parity fix it: a
//      rank can be one call ahead and calls k and k+1 never share bytes.  It cannot be two ahead, because
//      entering call k+2 requires the peer's arrival at k+1.
//
//   3. MONOTONIC ARRIVAL (`>= token`, not `== token`).  Equality cannot be released from the host, which
//      does not know the device-side token.  With a monotonic compare, writing UINT32_MAX into the arrival
//      slots lets a stalled waiter fall through, which is what the watchdog and the give-up paths need
//      (`core/verify.hpp`'s `release_gpu_waits`).  Ping-pong keeps the compare safe: within one parity the
//      tokens increase by two per call, so a stale arrival int can never satisfy the next wait.
//
// PARITY IS THE HOST'S, THE TOKEN IS THE DEVICE'S, AND THEY ARE ALLOWED TO DISAGREE.  The region a call uses
// must be known when the call is recorded, so `launch` counts its own calls per rank and uses `count & 1`.
// The kernel's token counts EXECUTIONS instead.  A captured graph is recorded once but replayed many times,
// so after the first replay the two numbers differ - and that is fine: what the protocol needs is (a) both
// ranks use the same parity for the same logical call, which they do because both record the same call
// sequence, and (b) the tokens written into one parity's slot strictly increase, which they do because the
// call sites of one parity are two apart and a replay adds the whole graph's call count.
//
// ONE PAIR SERVES ONE USE: either a captured window graph, or host-driven calls.  Interleaving them (capture
// a graph, then drive host calls, then replay) can write a parity slot out of order and the monotonic
// compare would then let a wait through early.  The engine does not do this: a stage captures its window
// once and from then on only replays.
//
// WHAT THIS DOES NOT DO (v1): the peer-access path.  llama.cpp probes for peer access and, when it is there,
// keeps the same protocol but stages through the peer's device memory instead of host memory.  The cards this
// was built for report no P2P at all (`cudaDeviceCanAccessPeer` is 0 between the two RTX 3080s), so a peer
// path would be untestable code on the only rig available - and this tree does not ship claims it has not
// measured.  `peer_mode()` reports what the probe found; the staged path below runs either way.
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace strata::kernels {

/// What the two ranks round through before adding.  BF16 is the default: the model's own activations are
/// bf16 in the contract the kernels follow (`docs/activation-contract.md`), and a verify window is
/// kVerifyMaxT * n_embd = 8 * 2560 floats, so the two extra bytes per element that F32 stages buy nothing
/// the model can see.  F16 is not offered: the tree has no portable `__half` usage (bf16 is passed as
/// `uint16_t`, `include/strata/kernels/gr.hpp:10`), and nothing needs a third rounding.
enum class ArWire : int {
    F32 = 0,      ///< wire == the accumulator; the honest baseline for a parity test
    Bf16 = 1,
};

/// Rank count.  v1 is exactly two: `n_head_kv` is 2 (`core/geometry.hpp`), so a third rank would have no KV
/// head to own.
inline constexpr int kArRanks = 2;
/// Blocks per reduce.  Each block stripes a disjoint slice of the payload and has its own arrival slot, so
/// more blocks pump more PCIe stores in parallel (allreduce.cu:74-77).
inline constexpr int kArBlocks = 8;
inline constexpr int kArThreads = 256;

/// One rank's device-side view of the pair's memory.  These are the pointers a captured graph records, so
/// they must outlive every graph that captured a reduce.
struct ArRankView {
    int dev = -1;
    void* mine[2] = {};                     ///< this rank WRITES here, indexed by call parity
    const void* peer[2] = {};               ///< it reads the peer's half here
    uint32_t* arrival[2] = {};              ///< its own arrival tokens: [parity][block]
    const uint32_t* peer_arrival[2] = {};   ///< the peer's, which it waits on
    uint32_t* token = nullptr;              ///< [block] device counters, one bump per execution
    uint32_t* timeout = nullptr;
    /// The PEER's timeout word.  A wait that only polls `peer_arrival` cannot see a peer that has already
    /// given up, so the two ranks deadlock until each hits its own bound - and the give-up path then reads a
    /// staging area the peer never wrote.  Passed so either rank's give-up releases both.
    const uint32_t* peer_timeout = nullptr;            ///< [1] set to kTimedOut when a wait gave up (mapped: host-readable)
    /// [4] mapped progress words, host-readable with no sync: entered / staged / signalled / saw-peer.
    /// A wait that never completes has four possible causes - the kernel never started, it faulted in the
    /// staging, it signalled but the peer cannot see it, or the peer never signalled - and without these the
    /// only way to tell them apart is to guess.  The engine's stall report wants them too.
    uint32_t* progress = nullptr;
};

/// The indices into `ArRankView::progress`.
enum : int {
    kArProgressEntered = 0,     ///< block 0 began executing
    kArProgressStaged = 1,      ///< its stripe is in the staging region and fenced
    kArProgressSignalled = 2,   ///< it wrote its arrival token (the low 32 bits of the token)
    kArProgressSawPeer = 3,     ///< its wait finished: it read a peer token >= its own
};

/// The two cards a tensor-parallel group spans, and the memory their collectives stage through.
///
/// ONE INSTANCE PER PROCESS, shared by both ranks.  The engine runs its ranks as two host threads in one
/// process (a rank cannot share a stream, a doorbell or a CPU expert pool with its peer), and each rank's
/// kernels have to address both ranks' staging areas anyway, which is exactly what one mapped allocation per
/// rank gives.
class ArPair {
public:
    ArPair() = default;
    ~ArPair();
    ArPair(const ArPair&) = delete;
    ArPair& operator=(const ArPair&) = delete;

    /// `ordinals[0..kArRanks)` are the ranks in order (the engine's stage order: the card that drives the
    /// display and carries the head is rank 0).  `max_floats` is the largest count one reduce will carry -
    /// a verify window, so `kVerifyMaxT * n_embd`.  A prompt chunk is deliberately NOT served by this
    /// kernel: at thousands of rows the staged bytes dominate and llama.cpp switches to a copy-engine route
    /// (`allreduce.cu:30-37`), which is the prefill milestone's job.
    ///
    /// Allocates one mapped-pinned region per rank (two staging halves, the arrival slots and the timeout
    /// flag through it) plus the device token counters, then probes peer access.
    bool init(const int* ordinals, int64_t max_floats, ArWire wire, int blocks, std::string& err);
    void shutdown();
    bool ready() const { return ready_; }

    ArWire wire() const { return wire_; }
    int blocks() const { return blocks_; }
    int64_t max_floats() const { return max_floats_; }
    /// 1 when both directions reported peer access, 0 when the staged path is what runs, -1 before `init`
    /// probed it.
    int peer_mode() const { return peer_mode_; }

    /// Publish a pair for the host-side staging readback (diagnostics only; never read by a kernel path).
    /// Declared here as free functions so a caller does not need the private member.

    friend void ar_debug_set_pair(ArPair* p);
    friend const void* ar_debug_staging(int rank, int half);

    /// HOST view of a rank's staging half.  The region is mapped-pinned (`cudaHostAllocMapped`), so the host can
    /// read exactly the bytes a rank's kernel staged - the one link in the reduce that has never been observed
    /// directly, and the reason a wrong sum can otherwise only be guessed at.  Defined in `allreduce.cu`: it needs
    /// `ar_region_bytes`, which lives there.
    const void* host_staging(int rank, int half) const;
    /// Mapped pinned bytes the pair holds, for the startup line and the memory guard.
    uint64_t pinned_bytes() const { return pinned_bytes_; }
    const ArRankView& rank(int r) const { return view_[r <= 0 ? 0 : (r >= kArRanks ? kArRanks - 1 : r)]; }

    /// Sum `x` (count floats, ON RANK rank's DEVICE) with the peer's half, in place, on `stream`, which must
    /// be a stream of that rank's device.  Callable inside a graph capture: no host work and no
    /// synchronisation.  The caller MUST record the other rank's call too, with the same `count`, in the same
    /// order - the kernels wait for each other, and a rank that skips one leaves the other spinning.
    void launch(int rank, float* x, int count, void* stream);
    /// `launch` with the argument checks and the launch error.  For host-driven callers (the parity test, the
    /// first bring-up of a stage) and NOT for the captured token path.
    bool reduce_host(int rank, float* x, int count, void* stream, std::string& err);

    /// FORCE THIS RANK'S STREAM TO BE SUBMITTED.  Not an optimisation and not optional: MEASURED on the RTX
    /// 3080 pair under WDDM, a launch stays in a command buffer until a LATER submission arrives on that
    /// stream.  One launch alone did not start for 400 ms; with two ranks launching, only the FIRST rank's
    /// kernel started, because the second rank's launch was what flushed it, and the second rank's own work
    /// stayed pending.  A window whose kernels never start looks EXACTLY like a peer that never arrives, so
    /// without this the engine blames the wrong card.
    ///
    /// So after the last `launch` of a window, EVERY rank's stream needs one trailing submission.  This is
    /// that submission: an event record with timing disabled.  It costs host microseconds, runs no kernel, and
    /// cannot change the stream's results.
    void flush(int rank, void* stream);

    /// Raise both ranks' arrival slots past any token, so a waiter whose peer is gone drains instead of
    /// spinning forever.  Only for the give-up paths: afterwards the reduced values are NOT the sum, so the
    /// caller must refuse the window - the contract of `Verifier::release_gpu_waits`.
    void release();
    /// Which rank's kernel gave up waiting (a peer that never arrived), or -1.  Reads mapped memory, so it
    /// needs no sync.
    int timed_out_rank() const;
    /// CLEAR the give-up flag on both ranks.  The kernel only ever SETS it and the host only ever READS it (the
    /// mapped word is host-visible and host-writable), so a host-side clear is safe - and it is REQUIRED: the
    /// host checks the flag right after a launch, when THIS call's kernel has not run yet, so it reads the
    /// PREVIOUS call's value.  Left uncleared, one give-up - for instance the degenerate window the verifier runs
    /// while it is being set up - made every later reduce report a give-up, while the progress words showed both
    /// ranks meeting every block (measured on two cards: entered=1 staged=1 signalled=384 saw_peer=384 on BOTH).
    void clear_timeout();
    /// The token rank r's block-0 counter currently holds, for the startup line ("the pair has run N
    /// reduces").  Copies from the device, so it synchronises; pass a `kArRanks`-element array.
    bool token_counts(uint32_t* out) const;
    /// How many reduces `launch` has recorded for rank r.  This is the host's call count and what picks the
    /// staging parity; it is deliberately not the kernels' token, which counts executions (see the header).
    uint64_t calls(int rank) const { return calls_[rank <= 0 ? 0 : (rank >= kArRanks ? kArRanks - 1 : rank)]; }
    /// Host read of one mapped arrival slot, with NO synchronisation: a snapshot of a pair that may be
    /// running, not a barrier.  `peer` selects the slot this rank waits on rather than the one it signals.
    /// For the stall report - "rank 1 signalled 7 and is waiting for a 7 that rank 0 never wrote" is the
    /// difference between a protocol bug and a driver that does not make the write visible.
    uint32_t peek_arrival(int rank, bool peer, int parity, int block) const;
    /// Host read of one progress word (`kArProgressEntered` and friends).  Also no synchronisation.
    uint32_t peek_progress(int rank, int which) const;

private:
    bool ready_ = false;
    int dev_[kArRanks] = {-1, -1};

/// Diagnostics: publish a pair, then read a rank's staging half from the HOST (the region is mapped-pinned).
    ArWire wire_ = ArWire::Bf16;
    int blocks_ = kArBlocks;
    int64_t max_floats_ = 0;
    int peer_mode_ = -1;
    uint64_t pinned_bytes_ = 0;
    uint64_t calls_[kArRanks] = {0, 0};    ///< host-side call count per rank: the staging parity
    void* host_[kArRanks] = {nullptr, nullptr};
    /// One event per rank, created at init and reused by every `flush`.  Held as `void*` so this header stays
    /// free of CUDA types - the same reason every stream parameter in this tree is a `void*`.
    void* flush_ev_[kArRanks] = {nullptr, nullptr};
    ArRankView view_[kArRanks];
    // The host's view of the same bytes the kernels signal through: `release` writes the arrival slots and
    // `timed_out_rank` reads the timeout flag, both from the host and both without a sync.  Mapped pinned
    // memory has one address for the host and one per device, and the device pointer cannot be inverted, so
    // the pair has to keep both.
    uint32_t* h_arrival_[kArRanks][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    uint32_t* h_timeout_[kArRanks] = {nullptr, nullptr};
    uint32_t* h_progress_[kArRanks] = {nullptr, nullptr};
};

/// Diagnostics: publish a pair, then read a rank's staging half from the HOST (mapped-pinned region).
void ar_debug_set_pair(ArPair* p);
const void* ar_debug_staging(int rank, int half);

}  // namespace strata::kernels
