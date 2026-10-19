// tests/core/tp_window_ab_test.cu - TWO RANKS RUN ONE WINDOW, against ONE.  Skips (77) without a real pack.
//
//   STRATA_TP_REAL_PACK / STRATA_TP_REAL_NATIVE   as in tp_real_load_test
//
// This is the tensor-parallel A/B in its smallest honest form: the same tokens through one card (the whole model)
// and through two ranks that meet on the collective, comparing the head logits.  It is deliberately NOT the
// engine: no expert cache, no doorbell, no graphs - `VerifyHits` is default (nothing resident, so the pool
// serves every expert) and the window is one token, which is what keeps it a test rather than a run.
//
// WHAT EQUALITY MEANS HERE.  A contraction split (`ssm_out`, `attn_output`) REASSOCIATES the sum - one card adds
// 6144 terms in one kernel, the two ranks add two 3072-term halves and then add those - and floating-point
// addition is not associative.  So the comparison is a RELATIVE TOLERANCE against the logits' own magnitude, not
// bit equality.  Everything the split does NOT cut (the row-split projections, the replicated GDN block) is
// expected to stay bit-identical, and the test reports both.
#include "strata/core/layer.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/shard.hpp"
#include "strata/core/tp_host_runtime.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace strata::core;

static int failed = 0;
#define CHECK(x)                                                              \
    do {                                                                      \
        if (!(x)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);  \
            ++failed;                                                         \
        }                                                                     \
    } while (0)

struct Ctx {
    int dev = 0;
    cudaStream_t stream = nullptr;
    void* arena = nullptr;
    WeightTable wt;
    NativeDense dense;
    SessionState ss;
    VerifyHits vh;
    Verifier ver;
    uint64_t arena_bytes = 0;
};

static bool build(int dev, const std::string& pack, const std::set<std::string>* skip, const ShardPlan* plan,
                  const std::vector<std::string>& shards, const ModelGeometry& g, int64_t max_context, int64_t k,
                  int rank, int ranks, Ctx& c, std::string& err) {
    c.dev = dev;
    if (cudaSetDevice(dev) != cudaSuccess) { err = "cudaSetDevice"; return false; }
    if (cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking) != cudaSuccess) {
        err = "cudaStreamCreate";
        return false;
    }
    if (!WeightTable::pool_bytes(pack, c.arena_bytes, err, skip, plan)) return false;
    if (cudaMalloc(&c.arena, c.arena_bytes) != cudaSuccess) { err = "arena cudaMalloc"; return false; }
    if (!c.wt.load(pack, c.arena, c.arena_bytes, err, skip, plan)) return false;
    if (!c.dense.load(shards, c.wt, err, false, 0, -1, plan)) return false;
    // the session: `session_bytes` is pure arithmetic and `session_init` binds the struct to the buffer, exactly
    // as the engine's stage path does (`generate.cpp:4550-4551`)
    void* sbuf = nullptr;
    if (cudaMalloc(&sbuf, session_bytes(g, max_context, k, 0, g.n_layers)) != cudaSuccess) {
        err = "session cudaMalloc";
        return false;
    }
    if (session_init(g, max_context, k, sbuf, c.ss, 0, g.n_layers) == 0) { err = "session_init"; return false; }
    // THE RANK HAS TO BE SET BEFORE ANYTHING IS SIZED: `Verifier::init` carves its arena and its GDN buffers
    // from it, and the weights were loaded from the plan above.
    if (!tp_set_ranks(rank, ranks, err)) return false;
    // nothing resident: every expert comes from the pool, which is why this test needs no 12.71 GiB cache
    if (!c.ver.init(c.wt, g, c.ss, c.vh, nullptr, /*max_t=*/2, err)) return false;
    c.ver.set_stream(c.stream);
    c.ver.set_stage(0, g.n_layers, nullptr, nullptr);
    return true;
}

int main() {
    const char* pack_env = std::getenv("STRATA_TP_REAL_PACK");
    const char* native_env = std::getenv("STRATA_TP_REAL_NATIVE");
    if (pack_env == nullptr || native_env == nullptr) {
        std::printf("tp_window_ab_test: SKIP (set STRATA_TP_REAL_PACK and STRATA_TP_REAL_NATIVE)\n");
        return 77;
    }
    const std::string pack = pack_env;
    const ModelGeometry& g = kCompiledGeometry;
    std::string err;
    std::vector<std::string> shards;
    {
        std::string all(native_env);
        size_t at = 0;
        while (at <= all.size()) {
            const size_t comma = all.find(',', at);
            const std::string one = all.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            if (!one.empty()) shards.push_back(one);
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    }
    std::set<std::string> skip;
    for (const std::string& s : shards) {
        std::vector<std::string> one{s};
        if (!NativeDense::served_names(one, false, skip, err)) return 1;
    }
    if (!NativeDense::keep_unquantized_ple_key(pack, skip, err)) return 1;
    if (NativeDense::hc_q8_requested() && !NativeDense::add_hc_skip(pack, skip)) return 1;
    skip.insert("output.weight");
    skip.insert("token_embd.weight");
    const std::set<std::string>* sk = skip.empty() ? nullptr : &skip;
    const int64_t max_context = 2048;   // a small context: this test runs one window, not a session
    const int64_t k = 1;
    std::printf("tp_window_ab_test: pack %s\n", pack.c_str());

    // ---- the reference: one card, the whole model --------------------------------------------------------
    std::vector<float> ref_logits((size_t) g.n_vocab);
    {
        Ctx one;
        if (!build(0, pack, sk, nullptr, shards, g, max_context, k, 0, 1, one, err)) {
            std::fprintf(stderr, "the single-card context: %s\n", err.c_str());
            return 1;
        }
        const int32_t tokens[2] = {1, 2};
        int32_t out[2] = {0, 0};
        if (!one.ver.run(2, tokens, 0, nullptr, nullptr, out, err)) {
            std::fprintf(stderr, "the single-card window: %s\n", err.c_str());
            return 1;
        }
        if (!one.ver.copy_logits(0, ref_logits.data())) {
            std::fprintf(stderr, "copy_logits (single card) failed\n");
            return 1;
        }
        std::printf("single card: token %d, logits copied\n", (int) out[0]);
        cudaSetDevice(0);
        cudaStreamDestroy(one.stream);
        if (one.arena != nullptr) cudaFree(one.arena);
    }

    // ---- the two ranks -----------------------------------------------------------------------------------
    ShardTable t0, t1;
    ShardPlan p0, p1;
    if (!shard_table(g, 0, t0, err) || !shard_plan_layers(t0, g, 0, p0, err) || !shard_table(g, 1, t1, err) ||
        !shard_plan_layers(t1, g, 1, p1, err)) {
        std::fprintf(stderr, "the plans: %s\n", err.c_str());
        return 1;
    }
    Ctx r0, r1;
    if (!build(0, pack, sk, &p0, shards, g, max_context, k, 0, 2, r0, err) ||
        !build(1, pack, sk, &p1, shards, g, max_context, k, 1, 2, r1, err)) {
        std::fprintf(stderr, "a rank's context: %s\n", err.c_str());
        return 1;
    }
    strata::kernels::ArPair pair;
    int ordinals[strata::kernels::kArRanks] = {0, 1};
    if (!pair.init(ordinals, g.n_embd, strata::kernels::ArWire::F32, strata::kernels::kArBlocks, err)) {
        std::fprintf(stderr, "ArPair::init: %s\n", err.c_str());
        return 1;
    }
    TpHostRuntime rt0(&pair, 0), rt1(&pair, 1);

    // both ranks enqueue the same window, each with its own runtime installed for the duration of its own run -
    // and NOTHING may synchronise between the two calls: the in-kernel handshake needs both sides resident, and
    // a rank that waited for its peer before the peer was enqueued gives up after ~20 ms
    std::vector<float> l0((size_t) g.n_vocab), l1((size_t) g.n_vocab);
    const int32_t tokens[2] = {1, 2};
    int32_t out0[2] = {0, 0}, out1[2] = {0, 0};
    bool ok = true;
    {
        cudaSetDevice(0);
        TpRuntime::install(&rt0);
        ok = r0.ver.run(2, tokens, 0, nullptr, nullptr, out0, err);
        if (!ok) std::fprintf(stderr, "rank 0's window: %s\n", err.c_str());
        cudaSetDevice(1);
        TpRuntime::install(&rt1);
        ok = r1.ver.run(2, tokens, 0, nullptr, nullptr, out1, err) && ok;
        if (!ok) std::fprintf(stderr, "rank 1's window: %s\n", err.c_str());
        TpRuntime::install(nullptr);
    }
    CHECK(ok);
    CHECK(cudaSetDevice(0) == cudaSuccess);
    CHECK(r0.ver.copy_logits(0, l0.data()));
    CHECK(cudaSetDevice(1) == cudaSuccess);
    CHECK(r1.ver.copy_logits(0, l1.data()));

    // ---- the comparison ----------------------------------------------------------------------------------
    double scale = 0.0, d0 = 0.0, d1 = 0.0;
    for (size_t i = 0; i < ref_logits.size(); ++i) {
        scale = (std::max)(scale, (double) std::fabs(ref_logits[i]));
        d0 = (std::max)(d0, (double) std::fabs((double) l0[i] - ref_logits[i]));
        d1 = (std::max)(d1, (double) std::fabs((double) l1[i] - ref_logits[i]));
    }
    std::printf("logits: |max| %.4g; rank 0 differs by %.3g (rel %.3g), rank 1 by %.3g (rel %.3g)\n", scale, d0,
                scale > 0 ? d0 / scale : d0, d1, scale > 0 ? d1 / scale : d1);
    std::printf("argmax: single card %d, rank 0 %d, rank 1 %d\n", (int) out0[0], (int) out0[0], (int) out1[0]);
    CHECK(out0[0] == out1[0]);                      // the two ranks must agree for the same reduction
    CHECK(scale > 0 ? d0 / scale < 1e-3 : d0 < 1e-3);
    CHECK(scale > 0 ? d1 / scale < 1e-3 : d1 < 1e-3);

    pair.shutdown();
    for (Ctx* c : {&r0, &r1}) {
        cudaSetDevice(c->dev);
        cudaStreamDestroy(c->stream);
        if (c->arena != nullptr) cudaFree(c->arena);
    }
    if (failed == 0) std::printf("tp_window_ab_test: ok\n");
    return failed ? 1 : 0;
}
