// tests/core/tp_layer_ab_test.cu - THE crux: does a whole GDN layer, run by two split ranks that reduce in the
// middle, produce what one card produces?  Skips (77) without a real pack.
//
//   STRATA_TP_REAL_PACK    the pack directory
//   STRATA_TP_REAL_NATIVE  comma-separated GGUF shards (the engine's --native); required here, because the
//                          weights this layer reads are served natively on a real pack
//
// WHAT IS COMPARED, AND WHAT "EQUAL" MEANS.  A contraction split (`ssm_out`, 6144 -> 3072 columns) REASSOCIATES
// the sum: one card computes `sum(k = 0..6143)` in one kernel, the two ranks compute
// `sum(k = 0..3071) + sum(k = 3072..6143)` and add.  Floating-point addition is not associative, so the outputs
// are NOT bit-identical and must not be asserted to be - what is asserted here is the measured relative
// difference, which is the natural fp32 accumulation error of a 6144-term dot product.  Everything the split
// does NOT cut is expected to stay bit-identical, and this test reports both numbers.
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/shard.hpp"
#include "strata/core/tp_host_runtime.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/allreduce.hpp"

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

/// Reads the skipped/native/pack environment the same way the real-pack load test does.
static bool read_env(std::string& pack, std::vector<std::string>& shards, std::set<std::string>& skip,
                     std::string& err) {
    const char* p = std::getenv("STRATA_TP_REAL_PACK");
    const char* n = std::getenv("STRATA_TP_REAL_NATIVE");
    if (p == nullptr || n == nullptr) return false;
    pack = p;
    std::string all(n);
    size_t at = 0;
    while (at <= all.size()) {
        const size_t comma = all.find(',', at);
        const std::string one = all.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        if (!one.empty()) shards.push_back(one);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    for (const std::string& s : shards) {
        std::vector<std::string> one{s};
        if (!NativeDense::served_names(one, false, skip, err)) return false;
    }
    if (!NativeDense::keep_unquantized_ple_key(pack, skip, err)) return false;
    if (NativeDense::hc_q8_requested() && !NativeDense::add_hc_skip(pack, skip)) return false;
    skip.insert("output.weight");        // the engine adds these two by hand (generate.cpp): they are served by
    skip.insert("token_embd.weight");    // NativeHead, not NativeDense, so served_names does not list them
    return true;
}

/// One GDN layer's device state: arena, native upload, layer buffers.  Owned per device so the reference run and
/// the two ranks each have their own, which is the whole point - the ranks must not share anything but the
/// collective.
struct Rank {
    int dev = 0;
    cudaStream_t stream = nullptr;
    void* arena = nullptr;
    void* native = nullptr;
    void* buffers = nullptr;
    WeightTable wt;
    NativeDense dense;
    GdnBuffers b;
    uint64_t arena_bytes = 0, buffers_bytes = 0;
};

static bool build_rank(int dev, const std::string& pack, const std::set<std::string>* skip, const ShardPlan* plan,
                       const std::vector<std::string>& shards, const ModelGeometry& g, Rank& r, std::string& err) {
    r.dev = dev;
    if (cudaSetDevice(dev) != cudaSuccess) { err = "cudaSetDevice"; return false; }
    if (cudaStreamCreate(&r.stream) != cudaSuccess) { err = "cudaStreamCreate"; return false; }
    if (!WeightTable::pool_bytes(pack, r.arena_bytes, err, skip, plan)) return false;
    if (cudaMalloc(&r.arena, r.arena_bytes) != cudaSuccess) { err = "arena cudaMalloc"; return false; }
    if (!r.wt.load(pack, r.arena, r.arena_bytes, err, skip, plan)) return false;
    if (!r.dense.load(shards, r.wt, err, false, 0, -1, plan)) return false;
    r.buffers_bytes = gdn_buffers_bytes(g);
    if (cudaMalloc(&r.buffers, r.buffers_bytes) != cudaSuccess) { err = "buffers cudaMalloc"; return false; }
    gdn_buffers_init(g, r.buffers, r.b);
    gdn_buffers_zero_state(r.b, g, r.stream);
    return true;
}

static void drop_rank(Rank& r) {
    if (r.dev >= 0 && cudaSetDevice(r.dev) == cudaSuccess) {
        if (r.stream) cudaStreamDestroy(r.stream);
        if (r.buffers) cudaFree(r.buffers);
        if (r.arena) cudaFree(r.arena);
    }
    r.stream = nullptr;
    r.buffers = r.arena = nullptr;
}

int main() {
    std::string pack, err;
    std::vector<std::string> shards;
    std::set<std::string> skip;
    if (!read_env(pack, shards, skip, err)) {
        std::printf("tp_layer_ab_test: SKIP (set STRATA_TP_REAL_PACK and STRATA_TP_REAL_NATIVE)\n");
        return 77;
    }
    const ModelGeometry& g = kCompiledGeometry;
    const std::set<std::string>* sk = skip.empty() ? nullptr : &skip;

    // ---- the reference: one card, the whole model, one GDN layer ----------------------------------------
    Rank ref;
    if (!build_rank(0, pack, sk, nullptr, shards, g, ref, err)) {
        std::fprintf(stderr, "the reference rank: %s\n", err.c_str());
        return 1;
    }
    // the first GDN layer: `attn_qkv` is a GDN tensor, so the first layer that has it is the one to use
    int64_t L = -1;
    for (int64_t l = 0; l < g.n_layers && L < 0; ++l)
        if (ref.wt.find("blk." + std::to_string(l) + ".attn_qkv.weight") != nullptr) L = l;
    if (L < 0) { std::fprintf(stderr, "no GDN layer in this pack\n"); return 1; }
    std::printf("tp_layer_ab_test: layer %lld of %lld, pack %s\n", (long long) L, (long long) g.n_layers,
                pack.c_str());

    // the input, generated once on the host so both sides start from the same bits
    std::vector<float> mixed((size_t) g.n_embd);
    for (int64_t i = 0; i < g.n_embd; ++i) mixed[(size_t) i] = std::sin((double) i * 0.017) * 0.5f;
    float* d_mixed = nullptr;
    CHECK(cudaMalloc(&d_mixed, (size_t) g.n_embd * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(d_mixed, mixed.data(), (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice) ==
          cudaSuccess);
    float* d_ref = nullptr;
    CHECK(cudaMalloc(&d_ref, (size_t) g.n_embd * sizeof(float)) == cudaSuccess);
    CHECK(gdn_layer(ref.wt, g, L, ref.b, d_mixed, d_ref, ref.stream, err));
    CHECK(cudaStreamSynchronize(ref.stream) == cudaSuccess);
    std::vector<float> out_ref((size_t) g.n_embd, 0.0f);
    CHECK(cudaMemcpy(out_ref.data(), d_ref, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
    drop_rank(ref);
    cudaFree(d_mixed);
    cudaFree(d_ref);

    // ---- the two ranks: their own shards, their own buffers and state, two streams ----------------------
    Rank r0, r1;
    ShardTable t0, t1;
    ShardPlan p0, p1;
    if (!shard_table(g, 0, t0, err) || !shard_plan_layers(t0, g, 0, p0, err) || !shard_table(g, 1, t1, err) ||
        !shard_plan_layers(t1, g, 1, p1, err)) {
        std::fprintf(stderr, "the plans: %s\n", err.c_str());
        return 1;
    }
    if (!build_rank(0, pack, sk, &p0, shards, g, r0, err) || !build_rank(1, pack, sk, &p1, shards, g, r1, err)) {
        std::fprintf(stderr, "a rank: %s\n", err.c_str());
        return 1;
    }
    // WHAT THE LAYER WILL SEE for the tensors it reads: `plane_ptrs` needs codes and scales, and the layer's
    // native branch needs `native_data`.  A row that has neither is unusable, and saying which of the two is
    // missing is the difference between "the harness is wrong" and "this pack cannot serve this layer".
    for (const char* suf : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_norm.weight"}) {
        const std::string name = "blk." + std::to_string(L) + "." + suf;
        for (const Rank* rp : {&r0, &r1}) {
            const WeightRef* w = rp->wt.find(name);
            std::printf("   rank %d %-34s %s: bytes %llu codes %llu scales %llu data %s native %s kind %d\n",
                        rp->dev, name.c_str(), w != nullptr ? "found" : "MISSING",
                        (unsigned long long) (w != nullptr ? w->bytes : 0),
                        (unsigned long long) (w != nullptr ? w->codes_bytes : 0),
                        (unsigned long long) (w != nullptr ? w->scales_bytes : 0),
                        (w != nullptr && w->data != nullptr) ? "set" : "null",
                        (w != nullptr && w->native_data != nullptr) ? "set" : "null",
                        w != nullptr ? (int) w->kind : -1);
        }
    }
    // THE COLLECTIVE the two ranks meet on, sized for the widest tensor a layer reduces (n_embd)
    strata::kernels::ArPair pair;
    int ordinals[strata::kernels::kArRanks] = {0, 1};
    if (!pair.init(ordinals, g.n_embd, strata::kernels::ArWire::F32, strata::kernels::kArBlocks, err)) {
        std::fprintf(stderr, "ArPair::init: %s\n", err.c_str());
        return 1;
    }
    TpHostRuntime rt0(&pair, 0), rt1(&pair, 1);

    float *x0 = nullptr, *x1 = nullptr, *y0 = nullptr, *y1 = nullptr;
    CHECK(cudaSetDevice(0) == cudaSuccess);
    CHECK(cudaMalloc(&x0, (size_t) g.n_embd * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&y0, (size_t) g.n_embd * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpyAsync(x0, mixed.data(), (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, r0.stream) ==
          cudaSuccess);
    CHECK(cudaSetDevice(1) == cudaSuccess);
    CHECK(cudaMalloc(&x1, (size_t) g.n_embd * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&y1, (size_t) g.n_embd * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpyAsync(x1, mixed.data(), (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, r1.stream) ==
          cudaSuccess);

    // Enqueue both ranks, each with its own runtime installed for the duration of its own layer.  Neither call
    // synchronizes: the two devices' work then runs concurrently, which is what lets the in-kernel handshake
    // complete (a rank that waited for its peer before the peer was enqueued would give up after ~20 ms).
    std::string e0, e1;
    CHECK(cudaSetDevice(0) == cudaSuccess);
    TpRuntime::install(&rt0);
    if (!gdn_layer(r0.wt, g, L, r0.b, x0, y0, r0.stream, e0)) {
        std::fprintf(stderr, "rank 0's layer: %s\n", e0.c_str());
        ++failed;
    }
    TpRuntime::install(&rt1);
    if (!gdn_layer(r1.wt, g, L, r1.b, x1, y1, r1.stream, e1)) {
        std::fprintf(stderr, "rank 1's layer: %s\n", e1.c_str());
        ++failed;
    }
    TpRuntime::install(nullptr);
    CHECK(cudaSetDevice(0) == cudaSuccess);
    CHECK(cudaStreamSynchronize(r0.stream) == cudaSuccess);
    CHECK(cudaSetDevice(1) == cudaSuccess);
    CHECK(cudaStreamSynchronize(r1.stream) == cudaSuccess);

    std::vector<float> out0((size_t) g.n_embd, 0.0f), out1((size_t) g.n_embd, 0.0f);
    CHECK(cudaSetDevice(0) == cudaSuccess);
    CHECK(cudaMemcpy(out0.data(), y0, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaSetDevice(1) == cudaSuccess);
    CHECK(cudaMemcpy(out1.data(), y1, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    // ---- the comparison --------------------------------------------------------------------------------
    // A tolerance against the REFERENCE's magnitude rather than an absolute epsilon: what is being measured is
    // the reassociation error of a 6144-term fp32 dot product.
    double scale = 0.0;
    for (float v : out_ref) scale = (std::max)(scale, (double) std::fabs(v));
    double worst0 = 0.0, worst1 = 0.0;
    for (int64_t i = 0; i < g.n_embd; ++i) {
        worst0 = (std::max)(worst0, (double) std::fabs(out0[(size_t) i] - out_ref[(size_t) i]));
        worst1 = (std::max)(worst1, (double) std::fabs(out1[(size_t) i] - out_ref[(size_t) i]));
    }
    const double rel0 = scale > 0.0 ? worst0 / scale : worst0;
    const double rel1 = scale > 0.0 ? worst1 / scale : worst1;
    std::printf("tp_layer_ab_test: reference magnitude %.6g; rank 0 differs by %.3g (rel %.3g), rank 1 by %.3g "
                "(rel %.3g)\n",
                scale, worst0, rel0, worst1, rel1);
    // The two ranks' own outputs must agree with EACH OTHER exactly: they run the same reduction, so any
    // difference there is a bug rather than arithmetic.
    int same = 0;
    for (int64_t i = 0; i < g.n_embd; ++i)
        if (out0[(size_t) i] == out1[(size_t) i]) ++same;
    std::printf("tp_layer_ab_test: the two ranks agree bit for bit on %d of %lld outputs\n", same,
                (long long) g.n_embd);
    CHECK(same == g.n_embd);
    // and the split's result is the reference's to fp32 accumulation error, not bit for bit
    CHECK(rel0 < 1e-4);
    CHECK(rel1 < 1e-4);
    if (worst0 == 0.0) std::printf("tp_layer_ab_test: (the split happened to be bit-identical here)\n");

    pair.shutdown();
    drop_rank(r0);
    drop_rank(r1);
    if (failed == 0) std::printf("tp_layer_ab_test: ok\n");
    return failed ? 1 : 0;
}
