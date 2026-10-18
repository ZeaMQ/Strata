// tests/core/tp_real_load_test.cu - does a rank's split plan load a REAL pack?  Skips without one.
//
// The synthetic test (`tp_shard_load_test`) proves the byte arithmetic on a fixture, and the plan probe proves
// the table covers every row of an index.  Neither can show whether a REAL pack - its quantisations, its plane
// layouts, all 48 layers - loads under a rank's plan and still passes the per-layer shape checks.  That needs a
// real pack, so this test asks for one by environment variable and SKIPS (77) when there is none:
//
//   STRATA_TP_REAL_PACK    the pack directory (e.g. ...\Strata-data\packs\sc117-iq3_s)
//   STRATA_TP_REAL_NATIVE  optional comma-separated GGUF shards the native path serves (the engine's --native)
//
// Seconds and a few GB of VRAM: no model is run, and the pack's dense arena (1.4 GB for the pack this was
// written against) is read from disk, not generated.
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/shard.hpp"
#include "strata/core/weights.hpp"

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

int main() {
    const char* pack_env = std::getenv("STRATA_TP_REAL_PACK");
    if (pack_env == nullptr || pack_env[0] == '\0') {
        std::printf("tp_real_load_test: SKIP (set STRATA_TP_REAL_PACK to a pack directory)\n");
        return 77;   // the project's skip code
    }
    const std::string pack = pack_env;
    const ModelGeometry& g = kCompiledGeometry;
    std::string err;

    // The engine's own setup: this is why a pack can refuse to load on its own (`output.weight` lives only in
    // the GGUF for the pack this was written against), and the same skip set has to be given here.
    std::set<std::string> skip;
    if (const char* native_env = std::getenv("STRATA_TP_REAL_NATIVE")) {
        std::string all(native_env);
        size_t at = 0;
        while (at <= all.size()) {
            const size_t comma = all.find(',', at);
            const std::string one = all.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            if (!one.empty()) {
                std::vector<std::string> shards{one};
                if (!NativeDense::served_names(shards, false, skip, err)) {
                    std::fprintf(stderr, "served_names %s: %s\n", one.c_str(), err.c_str());
                    return 1;
                }
            }
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    }
    if (!NativeDense::keep_unquantized_ple_key(pack, skip, err)) {
        std::fprintf(stderr, "keep_unquantized_ple_key: %s\n", err.c_str());
        return 1;
    }
    if (NativeDense::hc_q8_requested() && !NativeDense::add_hc_skip(pack, skip)) {
        std::fprintf(stderr, "add_hc_skip failed\n");
        return 1;
    }
    // THE TWO THE ENGINE ADDS BY HAND (`generate.cpp`: `skip.insert("output.weight")` when a native head is
    // configured - and `--native-head-gguf` follows `--native` - and `"token_embd.weight"` for a native pack).
    // Neither is in `served_names`: they are served by `NativeHead`, not by NativeDense, and without them the
    // pack refuses to load with "this pack holds the tensor only in its GGUF form".
    if (std::getenv("STRATA_TP_REAL_NATIVE") != nullptr) {
        skip.insert("output.weight");
        skip.insert("token_embd.weight");
    }
    const std::set<std::string>* sk = skip.empty() ? nullptr : &skip;
    std::printf("tp_real_load_test: %s, %zu tensors served natively\n", pack.c_str(), skip.size());

    uint64_t full_bytes = 0;
    if (!WeightTable::pool_bytes(pack, full_bytes, err, sk, nullptr)) {
        std::fprintf(stderr, "pool_bytes: %s\n", err.c_str());
        return 1;
    }
    void* full_dev = nullptr;
    if (cudaMalloc(&full_dev, full_bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc %llu failed\n", (unsigned long long) full_bytes);
        return 1;
    }
    WeightTable full;
    if (!full.load(pack, full_dev, full_bytes, err, sk, nullptr)) {
        std::fprintf(stderr, "the unsharded load: %s\n", err.c_str());
        return 1;
    }

    uint64_t shard_bytes[kShardRanks] = {0, 0};
    uint64_t native_bytes[kShardRanks] = {0, 0};
    for (int rank = 0; rank < kShardRanks; ++rank) {
        ShardTable table;
        ShardPlan plan;
        if (!shard_table(g, rank, table, err) || !shard_plan_layers(table, g, rank, plan, err)) {
            std::fprintf(stderr, "rank %d: the plan: %s\n", rank, err.c_str());
            return 1;
        }
        uint64_t bytes = 0;
        if (!WeightTable::pool_bytes(pack, bytes, err, sk, &plan)) {
            std::fprintf(stderr, "rank %d: pool_bytes: %s\n", rank, err.c_str());
            return 1;
        }
        void* dev = nullptr;
        if (cudaMalloc(&dev, bytes) != cudaSuccess) {
            std::fprintf(stderr, "rank %d: cudaMalloc %llu failed\n", rank, (unsigned long long) bytes);
            return 1;
        }
        WeightTable wt;
        if (!wt.load(pack, dev, bytes, err, sk, &plan)) {
            std::fprintf(stderr, "rank %d: THE SHARDED LOAD REFUSED THE REAL PACK: %s\n", rank, err.c_str());
            return 1;
        }
        shard_bytes[rank] = bytes;
        std::string why;
        const bool ok = check_all(wt, g, why, &table);
        if (!ok) {
            std::fprintf(stderr, "rank %d: check_all REFUSED THE REAL PACK: %s\n", rank, why.c_str());
            ++failed;
        }
        // Find a layer that HAS the two tensors: `attn_q` lives only on the 12 QSA layers, so hard-coding
        // `blk.1` (which this probe did first) prints -1 and checks nothing at all.
        int64_t q_layer = -1, o_layer = -1;
        for (int64_t l = 0; l < g.n_layers && (q_layer < 0 || o_layer < 0); ++l) {
            const std::string n = std::to_string(l);
            if (q_layer < 0 && wt.find("blk." + n + ".attn_q.weight") != nullptr) q_layer = l;
            if (o_layer < 0 && wt.find("blk." + n + ".attn_output.weight") != nullptr) o_layer = l;
        }
        const WeightRef* q = q_layer >= 0 ? wt.find("blk." + std::to_string(q_layer) + ".attn_q.weight") : nullptr;
        const WeightRef* o = o_layer >= 0 ? wt.find("blk." + std::to_string(o_layer) + ".attn_output.weight") : nullptr;
        std::printf("rank %d: %llu MiB of the full %llu MiB, check_all %s; blk.%lld.attn_q ne1=%lld (whole "
                    "12288), blk.%lld.attn_output ne0=%lld (whole 6144)\n",
                    rank, (unsigned long long) (bytes >> 20), (unsigned long long) (full_bytes >> 20),
                    ok ? "ok" : "FAILED", (long long) q_layer,
                    (long long) (q != nullptr ? q->ne1 : -1), (long long) o_layer,
                    (long long) (o != nullptr ? o->ne0 : -1));
        CHECK(q != nullptr && q->ne1 == 12288 / kShardRanks);
        CHECK(o != nullptr && o->ne0 == 6144 / kShardRanks);
        cudaFree(dev);

        // ---- AND THE PART THAT ACTUALLY DECIDES THE WINDOW TIME: the NATIVE upload ------------------------
        // The canonical arena above barely shrinks (1440 of 1466 MiB on this pack) because the big dense
        // projections are not in it at all - they are served from the GGUF, which is why the engine is given
        // --native.  This is where a rank's rows are cut, and it is the first time that path sees a real GGUF.
        {
            const char* native_env = std::getenv("STRATA_TP_REAL_NATIVE");
            if (native_env == nullptr) {
                std::printf("   (no STRATA_TP_REAL_NATIVE: the native path is not exercised)\n");
                continue;
            }
            std::vector<std::string> shards;
            std::string all(native_env);
            size_t at = 0;
            while (at <= all.size()) {
                const size_t comma = all.find(',', at);
                const std::string one = all.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
                if (!one.empty()) shards.push_back(one);
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
            uint64_t want = 0;
            if (!NativeDense::weight_bytes_for(shards, wt, false, 0, -1, want, err, &plan)) {
                std::fprintf(stderr, "rank %d: weight_bytes_for(native, sharded): %s\n", rank, err.c_str());
                return 1;
            }
            NativeDense dense;
            if (!dense.load(shards, wt, err, false, 0, -1, &plan)) {
                std::fprintf(stderr, "rank %d: THE NATIVE SHARDED UPLOAD REFUSED THE REAL GGUF: %s\n", rank,
                             err.c_str());
                return 1;
            }
            // THE PRE-SIZED NUMBER IS WHAT THE DEVICE LOSES, not what the matrices hold: `weight_bytes_for`
            // rounds each matrix up to the granule `cudaMalloc` backs it with, exactly as `allocated_bytes()`
            // does.  Comparing it against `weight_bytes()` (the raw total) looks like a 12% disagreement and is
            // nothing of the sort - the two accessors answer different questions, and the budget one is this.
            CHECK(dense.allocated_bytes() == want);
            CHECK(dense.weight_bytes() <= dense.allocated_bytes());
            native_bytes[rank] = want;
            std::printf("rank %d: native %llu MiB over %zu tensors (pre-sized %llu MiB)\n", rank,
                        (unsigned long long) (dense.weight_bytes() >> 20), dense.tensor_count(),
                        (unsigned long long) (want >> 20));
        }
    }
    // the whole native side, for comparison
    if (std::getenv("STRATA_TP_REAL_NATIVE") != nullptr) {
        std::vector<std::string> shards;
        std::string all(std::getenv("STRATA_TP_REAL_NATIVE"));
        size_t at = 0;
        while (at <= all.size()) {
            const size_t comma = all.find(',', at);
            const std::string one = all.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            if (!one.empty()) shards.push_back(one);
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
        uint64_t whole = 0;
        if (!NativeDense::weight_bytes_for(shards, full, false, 0, -1, whole, err, nullptr)) {
            std::fprintf(stderr, "weight_bytes_for(native, whole): %s\n", err.c_str());
            return 1;
        }
        std::printf("native: %llu MiB whole, %llu + %llu MiB for the two ranks\n",
                    (unsigned long long) (whole >> 20), (unsigned long long) (native_bytes[0] >> 20),
                    (unsigned long long) (native_bytes[1] >> 20));
        // THIS is the number that decides the window time, and it must really halve: each rank holds half of
        // the cut matrices and all of the replicated ones, so it is under the whole - and the two together are
        // over it, by exactly the replicated part.
        CHECK(native_bytes[0] < whole);
        CHECK(native_bytes[1] < whole);
        CHECK(native_bytes[0] + native_bytes[1] > whole);
    }
    cudaFree(full_dev);
    CHECK(shard_bytes[0] < full_bytes);
    CHECK(shard_bytes[1] < full_bytes);

    // The two shards hold the replicated tensors twice, so together they exceed the whole - what has to hold is
    // that neither rank alone needs the whole model, and that the cut tensors really are cut.
    std::printf("tp_real_load_test: shards %llu + %llu MiB against the full %llu MiB\n",
                (unsigned long long) (shard_bytes[0] >> 20), (unsigned long long) (shard_bytes[1] >> 20),
                (unsigned long long) (full_bytes >> 20));
    if (failed == 0) std::printf("tp_real_load_test: ok\n");
    return failed ? 1 : 0;
}
