// tests/core/tp_allreduce_test.cu - the cross-GPU sum tensor parallelism is built on (milestone M1's gate).
//
// What this has to catch, in the order the failures would bite:
//
//   1. THE TWO CARDS DISAGREE.  The whole design leans on the reduced value being identical on both ranks:
//      the router reads it, and two ranks that pick different expert ids are two different models, silently.
//      So every case checks the two cards' bytes against EACH OTHER and against a host reference.
//   2. A REPLAYED GRAPH SKIPS THE WAIT.  A verify window is one captured graph replayed hundreds of times,
//      so a token passed in by the host would be the same on every replay and the peer's arrival int would
//      already hold it.  The capture case below replays five times and checks the sum each time; it fails
//      loudly on the naive protocol.
//   3. A DEAD PEER HANGS THE BOX.  The kernel must give up and say so, and `release()` must be able to let a
//      waiter through from the host.
//
// Two visible GPUs are required, and the test SKIPS (77) without them - it is about the collective, not about
// this machine.  Set STRATA_TP_TEST_DEVICES=0,1 to name the pair; by default the two fastest visible cards are
// used, which is what the engine's stage order does.
//
// STRATA_TP_TEST_GIVEUP=1 adds the case where a rank waits with no peer at all and has to time out (~5 s).
#include "strata/kernels/allreduce.hpp"

#include "strata/core/geometry.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::kernels;

static int failed = 0;
#define CHECK(x)                                                              \
    do {                                                                      \
        if (!(x)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);  \
            ++failed;                                                         \
        }                                                                     \
    } while (0)

// ---- the host's copy of the wire round, so the reference is not the kernel's own code path ----------
static uint16_t host_f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static float host_bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}
static float host_wire(float v, ArWire w) {
    return w == ArWire::Bf16 ? host_bf16_to_f32(host_f32_to_bf16(v)) : v;
}
static float host_sum(float a, float b, ArWire w) { return host_wire(a, w) + host_wire(b, w); }

static bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct Buf {
    int dev = 0;
    float* d = nullptr;
    std::vector<float> host;
};

static bool upload(Buf& b, const std::vector<float>& src) {
    if (cudaSetDevice(b.dev) != cudaSuccess) return false;
    if (cudaMemcpy(b.d, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    return true;
}
static bool download(Buf& b) {
    if (cudaSetDevice(b.dev) != cudaSuccess) return false;
    b.host.assign(b.host.size(), 0.0f);
    return cudaMemcpy(b.host.data(), b.d, b.host.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

int main() {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 2) {
        std::printf("tp_allreduce_test: skipped, %d visible device(s)\n", nd);
        return 77;
    }
    int devs[2] = {0, 1};
    if (const char* env = std::getenv("STRATA_TP_TEST_DEVICES")) {
        int a = -1, b = -1;
        if (std::sscanf(env, "%d,%d", &a, &b) == 2 && a >= 0 && b >= 0 && a < nd && b < nd) {
            devs[0] = a;
            devs[1] = b;
        }
    }

    const int64_t n_embd = strata::core::kCompiledGeometry.n_embd;
    const int64_t max_floats = (int64_t) kVerifyMaxT * n_embd;
    std::printf("tp_allreduce_test: devices %d + %d, n_embd %lld, max_floats %lld\n", devs[0], devs[1],
                (long long) n_embd, (long long) max_floats);

    // ---- argument checks that must refuse ------------------------------------------------------
    {
        ArPair p;
        std::string err;
        const int same[2] = {devs[0], devs[0]};
        CHECK(!p.init(same, max_floats, ArWire::Bf16, kArBlocks, err));
        CHECK(err.find("not a tensor-parallel pair") != std::string::npos);
        CHECK(!p.init(devs, 0, ArWire::Bf16, kArBlocks, err));
        CHECK(!p.init(devs, max_floats, ArWire::Bf16, 0, err));
    }

    ArPair pair;
    std::string err;
    if (!pair.init(devs, max_floats, ArWire::Bf16, kArBlocks, err)) {
        std::fprintf(stderr, "FAIL: init: %s\n", err.c_str());
        return 1;
    }
    std::printf("tp_allreduce_test: wire bf16, blocks %d, pinned %llu B, peer mode %d\n", pair.blocks(),
                (unsigned long long) pair.pinned_bytes(), pair.peer_mode());

    Buf ra{devs[0]}, rb{devs[1]};
    const size_t cap = (size_t) max_floats;
    if (cudaSetDevice(ra.dev) != cudaSuccess || cudaMalloc((void**) &ra.d, cap * sizeof(float)) != cudaSuccess ||
        cudaSetDevice(rb.dev) != cudaSuccess || cudaMalloc((void**) &rb.d, cap * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: the two buffers could not be allocated\n");
        return 1;
    }
    ra.host.assign(cap, 0.0f);
    rb.host.assign(cap, 0.0f);

    cudaStream_t sa = nullptr, sb = nullptr;
    CHECK(cudaSetDevice(ra.dev) == cudaSuccess && cudaStreamCreateWithFlags(&sa, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaSetDevice(rb.dev) == cudaSuccess && cudaStreamCreateWithFlags(&sb, cudaStreamNonBlocking) == cudaSuccess);

    // One host-driven reduce: fill both ranks, launch both, read both back, compare against the reference.
    auto one_reduce = [&](int count, ArWire wire, const char* what) {
        std::vector<float> a((size_t) count), b((size_t) count);
        for (int i = 0; i < count; ++i) {
            a[(size_t) i] = (float) ((i % 97) - 48) * 0.25f;
            b[(size_t) i] = (float) ((i % 31) - 15) * -0.5f;
        }
        if (!upload(ra, a) || !upload(rb, b)) { std::fprintf(stderr, "FAIL %s: upload\n", what); ++failed; return; }
        std::string e;
        // The current device has to match the stream's device at every launch, or the kernel is queued on the
        // wrong card and the peer spins until it gives up.  reduce_host sets it too; the test does it here as
        // well so the failure mode cannot be mistaken for a protocol bug.
        CHECK(cudaSetDevice(ra.dev) == cudaSuccess);
        if (!pair.reduce_host(0, ra.d, count, sa, e)) { std::fprintf(stderr, "FAIL %s: %s\n", what, e.c_str()); ++failed; return; }
        CHECK(cudaSetDevice(rb.dev) == cudaSuccess);
        if (!pair.reduce_host(1, rb.d, count, sb, e)) { std::fprintf(stderr, "FAIL %s: %s\n", what, e.c_str()); ++failed; return; }
        // FLUSH EACH RANK'S STREAM, and the test is worthless without it: under WDDM the last launch on a
        // stream is not submitted until a later submission arrives on it, so without these two records rank
        // 1's kernel never starts, rank 0 waits for a peer that is not late but absent, and the sums only
        // come out right because a `cudaStreamSynchronize` later flushed the same work anyway.  A wait that
        // nobody verifies is a wait that can be skipped - which is why the progress words are checked below.
        (void) cudaSetDevice(ra.dev);
        pair.flush(0, sa);
        (void) cudaSetDevice(rb.dev);
        pair.flush(1, sb);
        CHECK(cudaSetDevice(ra.dev) == cudaSuccess && cudaStreamSynchronize(sa) == cudaSuccess);
        CHECK(cudaSetDevice(rb.dev) == cudaSuccess && cudaStreamSynchronize(sb) == cudaSuccess);
        if (!download(ra) || !download(rb)) { std::fprintf(stderr, "FAIL %s: download\n", what); ++failed; return; }
        // The handshake has to have HAPPENED, not merely to have produced the right arithmetic: every block
        // must have staged, signalled, and read a peer token at least as large as its own.
        for (int r = 0; r < kArRanks; ++r) {
            if (pair.peek_progress(r, kArProgressEntered) == 0 || pair.peek_progress(r, kArProgressStaged) == 0 ||
                pair.peek_progress(r, kArProgressSignalled) == 0 || pair.peek_progress(r, kArProgressSawPeer) == 0) {
                ++failed;
                std::fprintf(stderr, "FAIL %s: rank %d never completed the handshake (entered %u, staged %u, "
                                     "signalled %u, saw %u)\n", what, r, pair.peek_progress(r, kArProgressEntered),
                             pair.peek_progress(r, kArProgressStaged), pair.peek_progress(r, kArProgressSignalled),
                             pair.peek_progress(r, kArProgressSawPeer));
            }
        }
        CHECK(pair.timed_out_rank() == -1);
        CHECK(bit_equal(ra.host, rb.host));
        int wrong = 0, first = -1;
        for (int i = 0; i < count; ++i) {
            const float want = host_sum(a[(size_t) i], b[(size_t) i], wire);
            if (std::memcmp(&want, &ra.host[(size_t) i], sizeof(float)) != 0) {
                if (first < 0) first = i;
                ++wrong;
            }
        }
        if (wrong || !bit_equal(ra.host, rb.host)) {
            ++failed;
            std::fprintf(stderr, "FAIL %s: %d of %d elements differ from the reference (first at %d)\n", what, wrong,
                         count, first);
        }
    };

    // ---- the sizes a verify window carries ------------------------------------------------------
    for (int t : {1, 2, 3, 6, kVerifyMaxT}) {
        char what[64];
        std::snprintf(what, sizeof what, "bf16 T=%d", t);
        one_reduce((int) ((int64_t) t * n_embd), ArWire::Bf16, what);
    }

    // ---- the same buffer reduced fifty times: parity alternation and the monotonic token ----------
    // Each call's result must be exactly the pair's values summed once - a stale arrival int or a region
    // reused a call too early shows up here as an accumulation or a wrong value.
    for (int i = 0; i < 50; ++i) {
        char what[64];
        std::snprintf(what, sizeof what, "repeat %d", i);
        one_reduce((int) (6 * n_embd), ArWire::Bf16, what);
    }

    // ---- the F32 wire ---------------------------------------------------------------------------
    one_reduce((int) (6 * n_embd), ArWire::F32, "f32 T=6");
    one_reduce((int) n_embd, ArWire::F32, "f32 T=1");

    // ---- out-of-range counts are refused, not truncated ------------------------------------------
    {
        std::string e;
        CHECK(!pair.reduce_host(0, ra.d, (int) max_floats + 1, sa, e));
        CHECK(e.find("outside") != std::string::npos);
        CHECK(!pair.reduce_host(2, ra.d, 2560, sa, e));
        CHECK(!pair.reduce_host(0, nullptr, 2560, sa, e));
    }

    // ---- A CAPTURED GRAPH, REPLAYED.  This is the case a host-side token cannot survive -------------
    {
        const int count = (int) (6 * n_embd);
        std::vector<float> a((size_t) count), b((size_t) count);
        for (int i = 0; i < count; ++i) {
            a[(size_t) i] = (float) (i % 13) * 0.5f;
            b[(size_t) i] = (float) (i % 7) * -0.25f;
        }
        CHECK(upload(ra, a) && upload(rb, b));

        cudaGraph_t ga = nullptr, gb = nullptr;
        cudaGraphExec_t xa = nullptr, xb = nullptr;
        CHECK(cudaSetDevice(ra.dev) == cudaSuccess);
        CHECK(cudaStreamBeginCapture(sa, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
        pair.launch(0, ra.d, count, sa);
        CHECK(cudaStreamEndCapture(sa, &ga) == cudaSuccess);
        CHECK(cudaGraphInstantiate(&xa, ga, 0) == cudaSuccess);
        CHECK(cudaSetDevice(rb.dev) == cudaSuccess);
        CHECK(cudaStreamBeginCapture(sb, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
        pair.launch(1, rb.d, count, sb);
        CHECK(cudaStreamEndCapture(sb, &gb) == cudaSuccess);
        CHECK(cudaGraphInstantiate(&xb, gb, 0) == cudaSuccess);

        for (int replay = 0; replay < 5; ++replay) {
            // Put the ORIGINAL values back before every replay, so each replay has the same answer.  This is
            // also what makes a skipped wait visible: with the naive host-side token the second replay's wait
            // falls through instantly, and whether that shows up as a wrong value or as a torn read, the two
            // cards' bytes stop agreeing and the check below fails.
            CHECK(upload(ra, a) && upload(rb, b));
            CHECK(cudaSetDevice(ra.dev) == cudaSuccess);
            CHECK(cudaGraphLaunch(xa, sa) == cudaSuccess);
            CHECK(cudaSetDevice(rb.dev) == cudaSuccess);
            CHECK(cudaGraphLaunch(xb, sb) == cudaSuccess);
            // A graph launch submits the whole graph, but it is still the LAST launch on each stream here, so
            // each needs its trailing submission before the two graphs can meet (see `ArPair::flush`).
            (void) cudaSetDevice(ra.dev);
            pair.flush(0, sa);
            (void) cudaSetDevice(rb.dev);
            pair.flush(1, sb);
            CHECK(cudaSetDevice(ra.dev) == cudaSuccess && cudaStreamSynchronize(sa) == cudaSuccess);
            CHECK(cudaSetDevice(rb.dev) == cudaSuccess && cudaStreamSynchronize(sb) == cudaSuccess);
            const bool got_a = download(ra), got_b = download(rb);
            CHECK(got_a && got_b);
            if (!got_a || !got_b) break;
            int wrong = 0, first = -1;
            for (int i = 0; i < count; ++i) {
                const float want = host_sum(a[(size_t) i], b[(size_t) i], ArWire::Bf16);
                if (std::memcmp(&want, &ra.host[(size_t) i], sizeof(float)) != 0) {
                    if (first < 0) first = i;
                    ++wrong;
                }
            }
            if (wrong || !bit_equal(ra.host, rb.host)) {
                ++failed;
                std::fprintf(stderr, "FAIL capture replay %d: %d of %d wrong (first at %d), cards agree %d\n", replay,
                             wrong, count, first, (int) bit_equal(ra.host, rb.host));
            }
        }
        uint32_t counts[kArRanks] = {0, 0};
        CHECK(pair.token_counts(counts));
        std::printf("tp_allreduce_test: after the capture case the token counters are %u / %u\n", counts[0], counts[1]);

        CHECK(cudaGraphExecDestroy(xa) == cudaSuccess);
        CHECK(cudaGraphDestroy(ga) == cudaSuccess);
        CHECK(cudaGraphExecDestroy(xb) == cudaSuccess);
        CHECK(cudaGraphDestroy(gb) == cudaSuccess);
    }

    // ---- a rank waiting with no peer: release() lets it through, and the give-up path reports -------------
    {
        // Only rank 0 is launched, so its kernel waits for an arrival that never comes.  Release it from the
        // host; the pair's parity is left skewed afterwards, which is why this is the last case.
        const int count = (int) n_embd;
        std::vector<float> a((size_t) count, 1.0f);
        CHECK(upload(ra, a));
        // The pair's ranks are now out of step by design; a fresh pair keeps the rest of the test honest.
        ArPair solo;
        std::string e;
        if (solo.init(devs, max_floats, ArWire::Bf16, kArBlocks, e)) {
            CHECK(cudaSetDevice(ra.dev) == cudaSuccess);
            solo.launch(0, ra.d, count, sa);
            solo.release();
            CHECK(cudaSetDevice(ra.dev) == cudaSuccess && cudaStreamSynchronize(sa) == cudaSuccess);
            CHECK(solo.timed_out_rank() == -1);   // it fell through on the release, it did not give up
        } else {
            std::fprintf(stderr, "FAIL: a second pair could not be set up: %s\n", e.c_str());
            ++failed;
        }
    }

    if (std::getenv("STRATA_TP_TEST_GIVEUP") != nullptr) {
        // No release at all: the kernel's own bound has to fire and be reported.
        ArPair solo;
        std::string e;
        if (solo.init(devs, max_floats, ArWire::Bf16, kArBlocks, e)) {
            std::vector<float> a((size_t) n_embd, 1.0f);
            CHECK(upload(ra, a));
            const int before = solo.timed_out_rank();
            CHECK(before == -1);
            CHECK(cudaSetDevice(ra.dev) == cudaSuccess);
            solo.launch(0, ra.d, n_embd, sa);
            CHECK(cudaStreamSynchronize(sa) == cudaSuccess);   // returns on its own bound, ~5 s
            CHECK(solo.timed_out_rank() == 0);
        }
    }

    CHECK(cudaSetDevice(ra.dev) == cudaSuccess && cudaFree(ra.d) == cudaSuccess);
    CHECK(cudaSetDevice(rb.dev) == cudaSuccess && cudaFree(rb.d) == cudaSuccess);
    CHECK(cudaSetDevice(ra.dev) == cudaSuccess && cudaStreamDestroy(sa) == cudaSuccess);
    CHECK(cudaSetDevice(rb.dev) == cudaSuccess && cudaStreamDestroy(sb) == cudaSuccess);
    pair.shutdown();

    if (failed == 0) std::printf("tp_allreduce_test: ok\n");
    return failed ? 1 : 0;
}
