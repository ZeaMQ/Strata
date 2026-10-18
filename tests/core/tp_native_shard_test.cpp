// tests/core/tp_native_shard_test.cpp - the native (GGUF-layout) shard runs, as pure arithmetic.
//
// No GPU, no GGUF, no model: `shard_runs` takes plain numbers, so the boundaries and every refusal can be
// checked directly.  A Q8_0-shaped matrix is used throughout - 32 elements per block, 34 bytes per block - so
// the numbers are the real ones and a row is `ne0 / 32 * 34` bytes.
#include <cstdio>
#include <string>
#include <vector>

#include "strata/core/native_shard.hpp"

using namespace strata::core;

static int failed = 0;
#define CHECK(x)                                                             \
    do {                                                                     \
        if (!(x)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ++failed;                                                        \
        }                                                                    \
    } while (0)

static constexpr int kB = 32;    // Q8_0: elements per block
static constexpr int kBB = 34;   // Q8_0: bytes per block

static TensorShard output_shard(int64_t begin, int64_t end) {
    TensorShard ts;
    ts.axis = ShardAxis::Output;
    ts.ranges = 1;
    ts.range[0] = {begin, end};
    return ts;
}
static TensorShard input_shard(int64_t begin, int64_t end) {
    TensorShard ts;
    ts.axis = ShardAxis::Input;
    ts.ranges = 1;
    ts.range[0] = {begin, end};
    return ts;
}

int main() {
    std::vector<ShardRun> runs;
    std::string err;

    // ---- a rank that keeps the whole matrix: one run, and it is the payload -------------------------
    {
        const uint64_t full = 8 * (32 / kB) * kBB;      // 8 rows of 32 elements
        CHECK(shard_runs(32, 8, full, kB, kBB, nullptr, runs, err));
        CHECK(runs.size() == 1 && runs[0].src == 0 && runs[0].len == full);
        TensorShard rep;                                // replicated is the same thing
        CHECK(shard_runs(32, 8, full, kB, kBB, &rep, runs, err));
        CHECK(runs.size() == 1 && runs[0].len == full);
    }

    // ---- an OUTPUT split: whole rows, one run per range --------------------------------------------
    {
        const uint64_t full = 8 * kBB;                  // 8 rows of 32 elements
        CHECK(shard_runs(32, 8, full, kB, kBB, nullptr, runs, err));
        // rank 1 takes rows [4, 8): one run, starting at row 4
        TensorShard ts = output_shard(4, 8);
        CHECK(shard_runs(32, 8, full, kB, kBB, &ts, runs, err));
        CHECK(runs.size() == 1);
        CHECK(runs[0].src == 4 * kBB && runs[0].len == 4 * kBB);
        CHECK(shard_runs_bytes(runs) == full / 2);
        // the GDN q|k|v shape: three ranges, and their bytes are the rows they name
        TensorShard three;
        three.axis = ShardAxis::Output;
        three.ranges = 3;
        three.range[0] = {0, 2};
        three.range[1] = {4, 6};
        three.range[2] = {6, 8};
        CHECK(shard_runs(32, 8, full, kB, kBB, &three, runs, err));
        CHECK(runs.size() == 3);
        CHECK(runs[0].src == 0 && runs[0].len == 2 * kBB);
        CHECK(runs[1].src == 4 * kBB && runs[1].len == 2 * kBB);
        CHECK(runs[2].src == 6 * kBB && runs[2].len == 2 * kBB);
        // the destination is the runs packed in order, so the second run's destination follows the first
        CHECK(shard_runs_bytes(runs) == 6 * kBB);
    }

    // ---- an INPUT split: the same block range inside EVERY row --------------------------------------
    {
        // 64 columns x 4 rows, so a row is 2 blocks = 68 bytes
        const uint64_t full = 4 * 2 * kBB;
        TensorShard ts = input_shard(0, 32);            // the low half of the contraction axis
        CHECK(shard_runs(64, 4, full, kB, kBB, &ts, runs, err));
        CHECK(runs.size() == 4);                        // one run per row
        for (int row = 0; row < 4; ++row) {
            CHECK(runs[(size_t) row].src == (uint64_t) row * 2 * kBB);
            CHECK(runs[(size_t) row].len == kBB);       // one of the row's two blocks
        }
        CHECK(shard_runs_bytes(runs) == full / 2);
        // and the other half starts one block into every row
        TensorShard hi = input_shard(32, 64);
        CHECK(shard_runs(64, 4, full, kB, kBB, &hi, runs, err));
        CHECK(runs.size() == 4);
        for (int row = 0; row < 4; ++row) {
            CHECK(runs[(size_t) row].src == (uint64_t) row * 2 * kBB + kBB);
            CHECK(runs[(size_t) row].len == kBB);
        }
        // a K-quant's super-block is 256 elements, and 128 is NOT a whole number of them: the format's own
        // block size is what decides, which is why this takes block_elements rather than assuming 128.
        TensorShard kq = input_shard(0, 128);
        CHECK(!shard_runs(512, 2, 2 * 2 * 34, 256, 144, &kq, runs, err));
        CHECK(err.find("256-element blocks") != std::string::npos);
    }

    // ---- every refusal -----------------------------------------------------------------------------
    {
        // a payload that is not a whole number of rows
        CHECK(!shard_runs(32, 8, 8 * kBB - 1, kB, kBB, nullptr, runs, err));
        CHECK(err.find("whole number of 8 rows") != std::string::npos);
        // no extent
        CHECK(!shard_runs(0, 8, 272, kB, kBB, nullptr, runs, err));
        CHECK(!shard_runs(32, 0, 272, kB, kBB, nullptr, runs, err));
        // an output range past the last row, and an empty one
        TensorShard past = output_shard(6, 9);
        CHECK(!shard_runs(32, 8, 8 * kBB, kB, kBB, &past, runs, err));
        CHECK(err.find("outside the matrix's 8 rows") != std::string::npos);
        TensorShard empty = output_shard(4, 4);
        CHECK(!shard_runs(32, 8, 8 * kBB, kB, kBB, &empty, runs, err));
        // a contraction range that is not a whole number of blocks, and one past the last column
        TensorShard mis = input_shard(0, 16);
        CHECK(!shard_runs(64, 4, 4 * 2 * kBB, kB, kBB, &mis, runs, err));
        CHECK(err.find("32-element blocks") != std::string::npos);
        TensorShard wide = input_shard(32, 96);
        CHECK(!shard_runs(64, 4, 4 * 2 * kBB, kB, kBB, &wide, runs, err));
        // a contraction split with no block geometry to check against
        TensorShard ok = input_shard(0, 32);
        CHECK(!shard_runs(64, 4, 4 * 2 * kBB, 0, 0, &ok, runs, err));
        CHECK(err.find("block geometry") != std::string::npos);
        // a split with no ranges at all
        TensorShard none;
        none.axis = ShardAxis::Output;
        none.ranges = 0;
        CHECK(!shard_runs(32, 8, 8 * kBB, kB, kBB, &none, runs, err));
    }

    // ---- the property the whole design rests on: the two ranks' runs TILE the payload exactly ---------
    {
        const uint64_t full = 8 * (64 / kB) * kBB;
        for (int rank = 0; rank < 2; ++rank) {
            TensorShard ts = output_shard(rank * 4, (rank + 1) * 4);
            CHECK(shard_runs(64, 8, full, kB, kBB, &ts, runs, err));
            CHECK(shard_runs_bytes(runs) == full / 2);
            for (const ShardRun& r : runs) CHECK(r.src + r.len <= full);
        }
        for (int rank = 0; rank < 2; ++rank) {
            TensorShard ts = input_shard(rank * 32, (rank + 1) * 32);
            CHECK(shard_runs(64, 8, full, kB, kBB, &ts, runs, err));
            CHECK(shard_runs_bytes(runs) == full / 2);
            for (const ShardRun& r : runs) CHECK(r.src + r.len <= full);
        }
    }

    if (failed == 0) std::printf("tp_native_shard_test: ok\n");
    return failed ? 1 : 0;
}
