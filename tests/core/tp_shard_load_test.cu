// tests/core/tp_shard_load_test.cu - the sharded weight load, proved BYTE FOR BYTE.
//
// This is the only evidence that matters for the loader change: a shard must be exactly the bytes of the rows
// it claims, plane by plane.  A sharded load that is off by a row still produces a tensor of the right SHAPE
// with plausible values in it, which is the failure this project has paid for most often.
//
// The pack is synthetic and written here, so every number is known: a three-plane quantized tensor, a
// bf16-promoted one, a plain f32 one, and one whose scale plane is fp16 and therefore WIDENED on the way in
// (the case where the source plane and the destination plane are different sizes, which is where a row-range
// shard can go wrong quietly).  Needs a GPU: the loader allocates pinned staging and copies to a device arena.
#include "strata/core/weights.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;

static int failed = 0;
#define CHECK(x)                                                             \
    do {                                                                     \
        if (!(x)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ++failed;                                                        \
        }                                                                    \
    } while (0)
// ---- the synthetic pack ---------------------------------------------------------------------------
// Fields, in the order tools/pack_index.py writes them:
//   name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook
//   has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind
struct T {
    const char* name;
    int kind;
    int64_t ne0, ne1;
    int code_bits, group, has_offset, scales_fp16;
    uint64_t src_off, src_bytes, dst_off, dst_bytes;
    uint64_t codes, scales, offset;
};

static void write_pack(const std::string& dir, uint64_t pool, const std::vector<T>& ts,
                       const std::vector<uint8_t>& blob) {
    // The directory has to exist: `fopen` returns null for a missing path, and `fprintf` on a null FILE* is a
    // crash rather than an error, which is exactly how this test first failed.
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::FILE* f = std::fopen((dir + "/index.txt").c_str(), "wb");
    if (f == nullptr) { std::fprintf(stderr, "FAIL: cannot write %s/index.txt\n", dir.c_str()); std::exit(1); }
    std::fprintf(f, "# align 256 pool %llu tensors %d\n", (unsigned long long) pool, (int) ts.size());
    for (const T& t : ts)
        std::fprintf(f,
                     "%s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d\n", t.name, 0,
                     t.kind, (unsigned long long) t.src_off, (unsigned long long) t.src_bytes,
                     (unsigned long long) t.dst_off, (unsigned long long) t.dst_bytes, (long long) t.ne0,
                     (long long) t.ne1, t.code_bits, 0, t.group, 0, t.has_offset,
                     (unsigned long long) t.codes, (unsigned long long) t.scales,
                     (unsigned long long) t.offset, t.scales_fp16, 0);
    std::fclose(f);
    f = std::fopen((dir + "/dense.bin").c_str(), "wb");
    if (f == nullptr) { std::fprintf(stderr, "FAIL: cannot write %s/dense.bin\n", dir.c_str()); std::exit(1); }
    std::fwrite(blob.data(), 1, blob.size(), f);
    std::fclose(f);
}

/// Deterministic bytes with a per-row signature, so a shard that took the wrong rows is obvious: every byte
/// carries its own row index in the high nibble.
static std::vector<uint8_t> make_blob(uint64_t bytes, int rows) {
    std::vector<uint8_t> b((size_t) bytes);
    for (uint64_t i = 0; i < bytes; ++i) {
        const int row = rows > 0 ? (int) ((i * rows) / (bytes ? bytes : 1)) : 0;
        b[(size_t) i] = (uint8_t) (((row & 0x0F) << 4) | (int) (i & 0x0F));
    }
    return b;
}

/// Load into a fresh device arena and copy it back.  The DEVICE pointer comes back as well as the host copy,
/// because the only way to know where a tensor sits inside its arena is `ref->data - arena`: the WeightRef
/// records a DEVICE pointer, and subtracting a HOST pointer from it (the first version of this test did
/// exactly that) yields a number that looks like an offset and is not one.
static bool device_load(const std::string& dir, const ShardPlan* plan, void*& dev, uint64_t& need,
                        WeightTable& wt, std::vector<uint8_t>& bytes_out, std::string& err) {
    dev = nullptr;
    if (!WeightTable::pool_bytes(dir, need, err, nullptr, plan)) return false;
    if (cudaMalloc(&dev, need ? need : 256) != cudaSuccess) { err = "cudaMalloc"; dev = nullptr; return false; }
    if (!wt.load(dir, dev, need, err, nullptr, plan)) {
        cudaFree(dev);
        dev = nullptr;
        return false;
    }
    bytes_out.assign((size_t) need, 0);
    if (cudaMemcpy(bytes_out.data(), dev, (size_t) need, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "cudaMemcpy back";
        cudaFree(dev);
        dev = nullptr;
        return false;
    }
    return true;
}

/// A tensor's offset inside its arena, from two device pointers.
static size_t offset_in(const WeightRef* r, const void* arena_dev) {
    return (size_t) ((const uint8_t*) r->data - (const uint8_t*) arena_dev);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string dir = "D:\\AI\\temp\\strata-tp\\pack-fixture";
    std::string err;

    // ---- the tensors, and their numbers -----------------------------------------------------------
    // A: three planes.  8 rows x 8 blocks of 32, codes 16 B/block, scales 4 B, offsets 4 B.  Its scale plane
    // is already f32 in the pack, so source and destination are the same size (1536 B).
    T A{"quant3", 0, 256, 8, 4, 32, 1, 0, 0, 1536, 0, 1536, 1024, 256, 256};
    // B: bf16 promoted from an f32 container: 4 B in, 2 B out.
    T B{"promoted_bf16", 1, 256, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // C: plain f32, the control: nothing is transformed.
    T C{"plain_f32", 2, 64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // D: fp16 SCALES that widen: 544 B in, 576 B out.  The case where the two planes differ in size.
    T D{"fp16_scales", 0, 128, 4, 8, 32, 0, 1, 0, 0, 0, 0, 512, 64, 0};
    // E: a ONE-DIMENSIONAL f32 vector split on the CONTRACTION axis - `ssm_a` / `ssm_dt.bias` in the real model
    // (48 v-head scalars).  It has no row structure at all, so a shard of it is simply that half of the bytes,
    // and it is the case the plain-run branch of the contraction split exists for.
    T E{"plain_vec", 2, 48, 1, 0, 0, 0, 0, 0, 192, 0, 192, 0, 0, 0};
    // F: a QUANTIZED tensor cut on the CONTRACTION axis - the case `emit` does per-row block arithmetic for.
    // Same geometry as A (256 wide, 8 rows, 32 elements per unit, 4 bits per code) and no offset plane, so the
    // expected bytes can be worked out by hand: 8 units a row, 4 of them kept, 8 rows.
    T F{"quant_input", 0, 256, 8, 4, 32, 0, 0, 0, 1280, 0, 1280, 1024, 256, 0};

    // The byte counts that depend on the shape, BEFORE the offsets are laid out: B and C are promoted/plain
    // element counts, and D's scale plane is fp16 in the pack and f32 in the arena, so its two sizes differ.
    B.src_bytes = (uint64_t) B.ne0 * B.ne1 * 4;
    B.dst_bytes = (uint64_t) B.ne0 * B.ne1 * 2;
    C.src_bytes = (uint64_t) C.ne0 * C.ne1 * 4;
    C.dst_bytes = C.src_bytes;
    D.scales = (uint64_t) D.ne1 * (D.ne0 / D.group) * 4;                 // ENGINE form: f32 scales
    D.src_bytes = D.codes + (uint64_t) D.ne1 * (D.ne0 / D.group) * 2;    // PACK form: fp16 scales
    D.dst_bytes = D.codes + D.scales;

    uint64_t at = 0;
    for (T* t : {&A, &B, &C, &D, &E, &F}) {
        t->src_off = at;
        at += t->src_bytes;
    }
    const uint64_t blob_bytes = at;
    at = 0;
    for (T* t : {&A, &B, &C, &D, &E, &F}) {
        t->dst_off = at;
        at = (at + t->dst_bytes + 255) / 256 * 256;
    }
    const uint64_t pool = at;
    const std::vector<uint8_t> blob = make_blob(blob_bytes, 16);
    write_pack(dir, pool, {A, B, C, D, E, F}, blob);
    std::printf("tp_shard_load_test: fixture %llu B of source, %llu B of arena\n",
                (unsigned long long) blob_bytes, (unsigned long long) pool);

    // ---- the full load, as the reference -----------------------------------------------------------
    WeightTable full;
    uint64_t full_bytes = 0;
    void* full_dev = nullptr;
    std::vector<uint8_t> full_arena;
    if (!device_load(dir, nullptr, full_dev, full_bytes, full, full_arena, err)) {
        std::fprintf(stderr, "FAIL: the unsharded load: %s\n", err.c_str());
        return 1;
    }
    CHECK(full_bytes == pool);

    // ---- THE DEFAULT PATH, MEASURED: a plan that replicates everything must be byte-identical ---------
    // This is the strongest form of "opt-in, the default path is untouched" that can be checked without two
    // engine builds: the SAME pack, loaded with a plan in hand whose every entry says "this rank holds it
    // whole", has to produce the same arena bytes as no plan at all - not the same size, the same bytes.
    {
        ShardPlan everything;
        for (const T& t : {A, B, C, D, E, F}) {
            TensorShard rep;
            rep.axis = ShardAxis::Replicated;
            everything[t.name] = rep;
        }
        WeightTable wt;
        uint64_t bytes = 0;
        void* dev = nullptr;
        std::vector<uint8_t> arena;
        if (!device_load(dir, &everything, dev, bytes, wt, arena, err)) {
            std::fprintf(stderr, "FAIL: the all-replicated load: %s\n", err.c_str());
            return 1;
        }
        CHECK(bytes == full_bytes);
        CHECK(arena.size() == full_arena.size());
        if (arena != full_arena) {
            ++failed;
            size_t i = 0;
            while (i < arena.size() && i < full_arena.size() && arena[i] == full_arena[i]) ++i;
            std::fprintf(stderr, "FAIL: an all-replicated plan changed the arena at byte %llu\n",
                         (unsigned long long) i);
        }
        // ... and each tensor's shape and segmentation are unchanged too
        for (const T& t : {A, B, C, D, E, F}) {
            const WeightRef* a = full.find(t.name);
            const WeightRef* b = wt.find(t.name);
            CHECK(a != nullptr && b != nullptr);
            if (a == nullptr || b == nullptr) continue;
            CHECK(a->ne0 == b->ne0 && a->ne1 == b->ne1 && a->bytes == b->bytes);
            CHECK(a->data != nullptr && b->data != nullptr);
        }
        if (dev != nullptr) cudaFree(dev);
    }

    // ---- rank 0 and rank 1: every tensor split on its OUTPUT axis ----------------------------------
    ShardPlan p[2];
    for (int rank = 0; rank < 2; ++rank) {
        for (const T& t : {A, B, C, D, E, F}) {
            TensorShard ts;
            ts.axis = ShardAxis::Output;
            ts.ranges = 1;
            ts.range[0] = {rank * (t.ne1 / 2), (rank + 1) * (t.ne1 / 2)};
            p[rank][t.name] = ts;
        }
        // and one tensor this rank must not hold at all
        TensorShard drop;
        drop.axis = ShardAxis::Rank0;
        drop.keep = (rank == 0);
        p[rank]["plain_f32"] = drop;
        // E is cut on the CONTRACTION axis: 48 scalars -> 24, and the shard is that half of the bytes
        TensorShard vec;
        vec.axis = ShardAxis::Input;
        vec.ranges = 1;
        vec.range[0] = {rank * 24, (rank + 1) * 24};
        p[rank]["plain_vec"] = vec;
        // F is quantized AND cut on the contraction axis: half of each row's 8 units, so 4 units of codes and
        // 4 units of scales out of every one of the 8 rows.
        TensorShard qin;
        qin.axis = ShardAxis::Input;
        qin.ranges = 1;
        qin.range[0] = {rank * 128, (rank + 1) * 128};
        p[rank]["quant_input"] = qin;
    }

    for (int rank = 0; rank < 2; ++rank) {
        WeightTable wt;
        uint64_t bytes = 0;
        void* rank_dev = nullptr;
        std::vector<uint8_t> arena;
        if (!device_load(dir, &p[rank], rank_dev, bytes, wt, arena, err)) {
            std::fprintf(stderr, "FAIL: the sharded load of rank %d: %s\n", rank, err.c_str());
            return 1;
        }
        std::printf("tp_shard_load_test: rank %d arena %llu B of the full %llu B\n", rank,
                    (unsigned long long) bytes, (unsigned long long) full_bytes);
        CHECK(bytes < full_bytes);

        for (const T& t : {A, B, C, D, E, F}) {
            const WeightRef* f = full.find(t.name);
            const WeightRef* s = wt.find(t.name);
            if (f == nullptr || s == nullptr) { CHECK(false); continue; }

            if (std::strcmp(t.name, "quant_input") == 0) {
                // A QUANTIZED CONTRACTION SPLIT, checked plane by plane and row by row.
                //
                // The full tensor's destination is plane-major: 8 rows x 8 units x 16 B of codes = 1024 B, then
                // 8 rows x 8 units x 4 B of scales = 256 B.  This rank keeps 4 of each row's 8 units, and the
                // four units it keeps are CONSECUTIVE, so both the shard's run and the full tensor's matching
                // bytes are a contiguous 64 B (codes) / 16 B (scales) run - one memcmp a plane a row.
                CHECK(s->ne0 == 128 && s->ne1 == 8);
                const size_t f_at = offset_in(f, full_dev);
                const size_t s_at = offset_in(s, rank_dev);
                const size_t w = (size_t) rank * 4;                  // the first unit this rank keeps
                bool good = f_at + 1280 <= full_arena.size() && s_at + 640 <= arena.size();
                for (size_t row = 0; good && row < 8; ++row) {
                    // codes: the shard's 4 units of this row against the full tensor's units w..w+4
                    good = std::memcmp(full_arena.data() + f_at + row * 128 + w * 16,
                                       arena.data() + s_at + row * 64, 64) == 0;
                    // scales: the same units out of the scale plane, which starts 1024 B in
                    good = good && std::memcmp(full_arena.data() + f_at + 1024 + row * 32 + w * 4,
                                               arena.data() + s_at + 512 + row * 16, 16) == 0;
                }
                if (!good) {
                    ++failed;
                    std::fprintf(stderr, "FAIL quant_input rank %d: the contraction split is not units %d..%d "
                                         "of every row\n", rank, (int) w, (int) (w + 4));
                }
                continue;
            }

            if (std::strcmp(t.name, "plain_vec") == 0) {
                // THE CONTRACTION SPLIT of a plain vector: there is no row structure, so the shard is exactly
                // its half of the source bytes - and that is what it must equal, byte for byte, against the
                // full load's corresponding offset.
                CHECK(s->ne0 == 24 && s->ne1 == 1 && s->elements == 24);
                CHECK(s->bytes == 96);
                const size_t f_at = offset_in(f, full_dev) + (size_t) rank * 96;
                const size_t s_at = offset_in(s, rank_dev);
                if (f_at + 96 > full_arena.size() || s_at + 96 > arena.size()) {
                    ++failed;
                    std::fprintf(stderr, "FAIL plain_vec rank %d: out of arena\n", rank);
                } else if (std::memcmp(full_arena.data() + f_at, arena.data() + s_at, 96) != 0) {
                    ++failed;
                    std::fprintf(stderr, "FAIL plain_vec rank %d: the shard is not the full vector's "
                                         "elements %d..%d\n", rank, rank * 24, (rank + 1) * 24);
                }
                continue;
            }

            if (std::strcmp(t.name, "plain_f32") == 0) {
                // the Rank0 tensor: rank 0 has it whole, rank 1 has no bytes and is not resident
                if (rank == 0) {
                    CHECK(s->resident && s->bytes == f->bytes && s->ne1 == f->ne1);
                } else {
                    CHECK(!s->resident && s->data == nullptr && s->bytes == 0);
                }
                continue;
            }

            // the shape: the output axis is halved, the contraction axis is untouched
            CHECK(s->ne0 == f->ne0);
            CHECK(s->ne1 == f->ne1 / 2);
            CHECK(s->elements == s->ne0 * s->ne1);
            CHECK(s->bytes == f->bytes / 2);
            CHECK(s->codes_bytes == f->codes_bytes / 2);
            CHECK(s->scales_bytes == f->scales_bytes / 2);
            CHECK(s->offset_bytes == f->offset_bytes / 2);

            // ---- THE CHECK: every plane of the shard is byte-for-byte the rows it claims -----------
            const size_t f_base = offset_in(f, full_dev);
            const size_t s_base = offset_in(s, rank_dev);
            const uint64_t rows_full = (uint64_t) f->ne1;
            const uint64_t rows_kept = (uint64_t) s->ne1;
            const uint64_t row0 = (uint64_t) rank * rows_kept;
            for (int plane = 0; plane < 3; ++plane) {
                const uint64_t fb = plane == 0 ? f->codes_bytes : (plane == 1 ? f->scales_bytes : f->offset_bytes);
                const uint64_t sb = plane == 0 ? s->codes_bytes : (plane == 1 ? s->scales_bytes : s->offset_bytes);
                if (fb == 0) { CHECK(sb == 0); continue; }
                const uint64_t foff = plane == 0 ? 0 : (plane == 1 ? f->codes_bytes : f->codes_bytes + f->scales_bytes);
                const uint64_t soff = plane == 0 ? 0 : (plane == 1 ? s->codes_bytes : s->codes_bytes + s->scales_bytes);
                const uint64_t per_row = fb / rows_full;
                CHECK(per_row * rows_full == fb);
                CHECK(sb == per_row * rows_kept);
                // where the plane starts inside each arena, from the DEVICE pointers of both arenas
                const size_t f_at = f_base + foff + (size_t) (row0 * per_row);
                const size_t s_at = s_base + soff;
                if (f_at + sb > full_arena.size() || s_at + sb > arena.size()) {
                    ++failed;
                    std::fprintf(stderr, "FAIL %s rank %d plane %d: out of arena\n", t.name, rank, plane);
                    continue;
                }
                bool same = std::memcmp(full_arena.data() + f_at, arena.data() + s_at, (size_t) sb) == 0;
                if (!same) {
                    ++failed;
                    std::fprintf(stderr, "FAIL %s rank %d plane %d: the shard's %llu B are not the full "
                                         "tensor's rows %llu..%llu\n",
                                 t.name, rank, plane, (unsigned long long) sb, (unsigned long long) row0,
                                 (unsigned long long) (row0 + rows_kept));
                }
            }
        }
    }

    // ---- the refusals -----------------------------------------------------------------------------
    {
        ShardPlan bad;
        TensorShard ts;
        ts.axis = ShardAxis::Input;
        ts.ranges = 1;
        // A QUANTIZED contraction split IS loaded now (one run per row per plane), so the refusal this case
        // checks is the one that keeps it honest: 48 columns of a 256-wide tensor whose index says 32 elements
        // per unit is NOT a whole number of units, and loading it would take one and a half blocks out of every
        // row and report a shape that looks right.  128 - half of 256 - would be loaded and checked elsewhere.
        ts.range[0] = {0, 48};
        bad["quant3"] = ts;
        // Sizing it is fine - the size arithmetic is the same on both axes - and the refusal comes from the
        // segment builder, which is the only place that knows the units, and it names the tensor.
        WeightTable wt;
        uint64_t need = 0;
        CHECK(WeightTable::pool_bytes(dir, need, err, nullptr, &bad));
        void* arena = nullptr;
        CHECK(cudaMalloc(&arena, need) == cudaSuccess);
        CHECK(!wt.load(dir, arena, need, err, nullptr, &bad));
        CHECK(err.find("quant3") != std::string::npos);
        CHECK(err.find("not a whole number of 32-element units") != std::string::npos);
        (void) cudaFree(arena);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) { CHECK(false); }
    if (failed == 0) std::printf("tp_shard_load_test: ok\n");
    return failed ? 1 : 0;
}
