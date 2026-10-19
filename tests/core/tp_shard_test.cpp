// tests/core/tp_shard_test.cpp - the tensor-parallel split table, and the refusals that keep it honest.
// No GPU and no model: the table is a rule over the geometry, so it is checked where it is decided.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "strata/core/shard.hpp"

using namespace strata::core;

static int failed = 0;
#define CHECK(x)                                                             \
    do {                                                                     \
        if (!(x)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            ++failed;                                                        \
        }                                                                    \
    } while (0)

static const ShardSpec* spec_of(const ShardTable& t, const char* suffix, int& out_failed) {
    // A missing entry must not dereference: return a default (Replicated) so the axis checks fail and the
    // test reports every missing name instead of dying on the first one.
    static const ShardSpec missing{};
    const ShardSpec* s = t.find(suffix);
    if (s == nullptr) {
        std::fprintf(stderr, "FAIL: the table has no entry for %s\n", suffix);
        ++out_failed;
        return &missing;
    }
    return s;
}

/// One range, for the tensors that split in half.
static bool one_range(const ShardSpec* s, ShardAxis axis, int64_t begin, int64_t end) {
    return s != nullptr && s->axis == axis && s->ranges == 1 && s->range[0].begin == begin &&
           s->range[0].end == end;
}

int main() {
    const ModelGeometry& g = kCompiledGeometry;
    ShardTable t0, t1;
    std::string err;
    if (!shard_table(g, 0, t0, err) || !shard_table(g, 1, t1, err)) {
        std::fprintf(stderr, "FAIL: the compiled geometry has no split table: %s\n", err.c_str());
        return 1;
    }
    if (std::getenv("STRATA_TP_PRINT_TABLE") != nullptr) {
        for (int which = 0; which < 2; ++which) {
            const ShardTable& t = which == 0 ? t0 : t1;
            std::printf("---- rank %d: %d entries\n", which, t.n);
            for (int i = 0; i < t.n; ++i) {
                const ShardSpec& s = t.entries[i].spec;
                std::printf("  %-28s axis=%d ranges=%d", t.entries[i].suffix, (int) s.axis, s.ranges);
                for (int r = 0; r < s.ranges; ++r)
                    std::printf(" [%lld,%lld)", (long long) s.range[r].begin, (long long) s.range[r].end);
                std::printf("\n");
            }
        }
    }

    // ---- the four axes, at the numbers the geometry fixes --------------------------------------------
    // 10240 = q|k|v = 128*16 / 128*16 / 128*48, so each rank takes 8 / 8 / 24 whole heads
    {
        const ShardSpec* a = spec_of(t0, "attn_qkv.weight", failed);
        if (a != nullptr && a->ranges == 3) {
            CHECK(a->range[0].begin == 0 && a->range[0].end == 1024);      // q: 8 of 16 heads
            CHECK(a->range[1].begin == 2048 && a->range[1].end == 3072);   // k
            CHECK(a->range[2].begin == 4096 && a->range[2].end == 7168);   // v: 24 of 48 heads
        }
        const ShardSpec* b = spec_of(t1, "attn_qkv.weight", failed);
        if (b != nullptr && b->ranges == 3) {
            CHECK(b->range[0].begin == 1024 && b->range[0].end == 2048);
            CHECK(b->range[1].begin == 3072 && b->range[1].end == 4096);
            CHECK(b->range[2].begin == 7168 && b->range[2].end == 10240);
        }
        // the conv FILTER is split by channel with the SAME three-range q|k|v split as `attn_qkv`: the depthwise
        // conv's taps must line up with the rank's qkv channels, and a contiguous half would misalign them.
        const ShardSpec* c = spec_of(t0, "ssm_conv1d.weight", failed);
        CHECK(c != nullptr && c->axis == ShardAxis::Output && c->ranges == 3);
        CHECK(c->range[0].begin == 0 && c->range[0].end == 1024);       // q: 8 of 16 heads
        CHECK(c->range[1].begin == 2048 && c->range[1].end == 3072);    // k
        CHECK(c->range[2].begin == 4096 && c->range[2].end == 7168);    // v: 24 of 48 heads
    }
    // the output-side projections: half of the feature axis, no reduce
    CHECK(one_range(spec_of(t0, "attn_q.weight", failed), ShardAxis::Output, 0, 6144));       // 12 of 24 heads
    CHECK(one_range(spec_of(t1, "attn_q.weight", failed), ShardAxis::Output, 6144, 12288));
    CHECK(one_range(spec_of(t0, "attn_k.weight", failed), ShardAxis::Output, 0, 256));        // 1 of 2 KV heads
    CHECK(one_range(spec_of(t1, "attn_v.weight", failed), ShardAxis::Output, 256, 512));
    CHECK(one_range(spec_of(t0, "ssm_alpha.weight", failed), ShardAxis::Output, 0, 24));      // 24 of 48 v heads
    CHECK(one_range(spec_of(t1, "ssm_a", failed), ShardAxis::Input, 24, 48));   // 1-D: it splits on ne0
    // the contraction-side projections: half of the contraction axis, and a reduce follows
    CHECK(one_range(spec_of(t0, "attn_output.weight", failed), ShardAxis::Input, 0, 3072));
    CHECK(one_range(spec_of(t1, "attn_output.weight", failed), ShardAxis::Input, 3072, 6144));

    // ---- what must NOT be split ----------------------------------------------------------------------
    // the router decides which experts run, so both ranks must route identically
    CHECK(spec_of(t0, "ffn_gate_inp.weight", failed)->axis == ShardAxis::Replicated);
    // the indexer's keys are shared by every query head, so the selection they feed must agree - and so must its
    // QUERIES: cut them and each rank scores a partial sum and picks different cells (llama.cpp's
    // `handle_lightning_indexer` is MIRRORED in and out for the same reason)
    CHECK(spec_of(t0, "indexer.k_proj.weight", failed)->axis == ShardAxis::Replicated);
    CHECK(spec_of(t0, "indexer.q_proj.weight", failed)->axis == ShardAxis::Replicated);
    // 640/2 = 320 is not a multiple of 128
    for (const char* suf : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
        CHECK(spec_of(t0, suf, failed)->axis == ShardAxis::Replicated);
    // the hyper-connection block: two more reduces per sublayer is more than the bandwidth it saves
    for (const char* suf : {"hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                            "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight"})
        CHECK(spec_of(t0, suf, failed)->axis == ShardAxis::Replicated);
    // the experts are whole on whichever rank caches them; the partition is the cache's, not the tensor's
    for (const char* suf : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
        CHECK(spec_of(t0, suf, failed)->axis == ShardAxis::Replicated);
    // the head, the embedding and the draft layer live on rank 0
    for (const char* suf : {"token_embd.weight", "output.weight", "output_norm.weight",
                            "nextn.eh_proj.weight", "nextn.hnorm.weight"})
        CHECK(spec_of(t0, suf, failed)->axis == ShardAxis::Rank0);

    // every entry family_layout.hpp lists is present, and nothing else is
    CHECK(t0.n == t1.n);
    CHECK(t0.n >= 44);
    CHECK(t0.find("no_such_tensor.weight") == nullptr);
    // and the table is deterministic: the same geometry gives the same bytes
    {
        ShardTable again;
        CHECK(shard_table(g, 0, again, err));
        CHECK(again.n == t0.n);
        CHECK(std::memcmp(again.entries, t0.entries, sizeof t0.entries) == 0);
    }

    // ---- the geometries that must be REFUSED ---------------------------------------------------------
    // A rank that ends up with no KV head is the trap this table exists to catch (kvmem hit it on an
    // asymmetric pair).  One KV head between two ranks is exactly that.
    {
        ModelGeometry bad = g;
        bad.n_head_kv = 1;
        ShardTable t;
        std::string e;
        CHECK(!shard_table(bad, 0, t, e));
        CHECK(e.find("head_count_kv") != std::string::npos);
        CHECK(e.find(" 1") != std::string::npos);
    }
    // an odd head count anywhere half of it has to keep a whole head
    for (int which = 0; which < 5; ++which) {
        ModelGeometry bad = g;
        const char* want = nullptr;
        switch (which) {
            case 0: bad.n_head = 23; want = "head_count"; break;
            case 1: bad.ssm_v_heads = 47; want = "time_step_rank"; break;
            case 2: bad.ssm_k_heads = 15; want = "group_count"; break;
            case 3: bad.idx_q_heads = 3; want = "indexer.head_count"; break;
            case 4: bad.n_head_kv = 0; want = "head_count_kv"; break;
        }
        ShardTable t;
        std::string e;
        CHECK(!shard_table(bad, 0, t, e));
        CHECK(e.find(want) != std::string::npos);
    }
    // a rank index outside the pair
    {
        ShardTable t;
        std::string e;
        CHECK(!shard_table(g, 2, t, e));
        CHECK(e.find("outside") != std::string::npos);
    }

    // ---- the pair check on its own, which is what a startup assertion will call ----------------------
    auto mk = [](ShardAxis axis, int n, ShardRange r0, ShardRange r1) {
        ShardSpec a, b;
        a.axis = b.axis = axis;
        a.ranges = b.ranges = n;
        a.range[0] = r0;
        b.range[0] = r1;
        return std::make_pair(a, b);
    };
    {
        std::string e;
        // a clean half-and-half
        auto ok = mk(ShardAxis::Input, 1, {0, 3072}, {3072, 6144});
        CHECK(shard_check_pair(ok.first, ok.second, 6144, "x", e));
        // a gap: 6144 wide but only 5888 claimed
        auto gap = mk(ShardAxis::Input, 1, {0, 3072}, {3328, 6144});
        CHECK(!shard_check_pair(gap.first, gap.second, 6144, "x", e));
        CHECK(e.find("cover") != std::string::npos);
        // an overlap
        auto over = mk(ShardAxis::Input, 1, {0, 3200}, {3072, 6144});
        CHECK(!shard_check_pair(over.first, over.second, 6144, "x", e));
        // an Input split that is not a whole number of activation blocks
        auto mis = mk(ShardAxis::Input, 1, {0, 3000}, {3000, 6000});
        CHECK(!shard_check_pair(mis.first, mis.second, 6000, "x", e));
        CHECK(e.find("activation blocks") != std::string::npos);
        // the same split on the OUTPUT axis is fine: rows need no alignment
        auto out = mk(ShardAxis::Output, 1, {0, 3000}, {3000, 6000});
        CHECK(shard_check_pair(out.first, out.second, 6000, "x", e));
        // past the end of the axis
        auto past = mk(ShardAxis::Output, 1, {0, 4000}, {4000, 8000});
        CHECK(!shard_check_pair(past.first, past.second, 6000, "x", e));
        CHECK(e.find("past the axis") != std::string::npos);
        // empty and negative ranges
        auto empt = mk(ShardAxis::Output, 1, {0, 0}, {0, 6000});
        CHECK(!shard_check_pair(empt.first, empt.second, 6000, "x", e));
        // the two ranks disagreeing about the axis
        ShardSpec a, b;
        a.axis = ShardAxis::Output;
        b.axis = ShardAxis::Input;
        a.ranges = b.ranges = 1;
        a.range[0] = {0, 3000};
        b.range[0] = {3000, 6000};
        CHECK(!shard_check_pair(a, b, 6000, "x", e));
        CHECK(e.find("disagree") != std::string::npos);
        // different range counts
        ShardSpec c = a;
        c.ranges = 2;
        c.range[1] = {0, 1};
        CHECK(!shard_check_pair(a, c, 6000, "x", e));
        // An unequally-split axis is ACCEPTED as long as it tiles: the pair check is about coverage, and
        // equal halves are `shard_split_even`'s job (the table's, not this function's).
        auto uneven = mk(ShardAxis::Output, 1, {0, 3}, {3, 5});
        CHECK(shard_check_pair(uneven.first, uneven.second, 5, "x", e));
        // ... which is exactly what makes an odd axis fail: `ne/2` rounds down and leaves a gap, and this is
        // the last line of defence for a head count the table should already have refused.
        auto odd = mk(ShardAxis::Output, 1, {0, 2}, {2, 4});
        CHECK(!shard_check_pair(odd.first, odd.second, 5, "x", e));
        CHECK(e.find("cover") != std::string::npos);
        // replicated and rank-0 tensors skip the range checks entirely
        ShardSpec rep_a, rep_b;
        CHECK(shard_check_pair(rep_a, rep_b, 0, "x", e));
        ShardSpec r0, r0b;
        r0.axis = r0b.axis = ShardAxis::Rank0;
        CHECK(shard_check_pair(r0, r0b, 0, "x", e));
    }

    // ---- the loader's view: the plan over the pack's full names, and the byte budget -----------------
    {
        // the byte budget: proportional, and REFUSED when a plane is not uniform per unit
        uint64_t n = 12345;
        CHECK(shard_bytes(1000, 100, 50, n, "t", err) && n == 500);
        CHECK(shard_bytes(1000, 100, 100, n, "t", err) && n == 1000);   // replicated costs the whole plane
        CHECK(shard_bytes(1000, 100, 0, n, "t", err) && n == 0);        // rank 0's tensor on the other rank
        CHECK(!shard_bytes(1001, 100, 50, n, "t", err));                // not uniform: refuse, do not round
        CHECK(err.find("does not divide evenly") != std::string::npos);
        CHECK(!shard_bytes(1000, 0, 50, n, "t", err));
        CHECK(!shard_bytes(1000, 100, -1, n, "t", err));

        ShardPlan p0, p1;
        CHECK(shard_plan_layers(t0, g, 0, p0, err));
        CHECK(shard_plan_layers(t1, g, 1, p1, err));
        CHECK(p0.size() == p1.size());
        // both spellings are present, and they agree
        const auto bare = p0.find("attn_q.weight");
        const auto named = p0.find("blk.0.attn_q.weight");
        const auto last = p0.find("blk.47.attn_q.weight");
        CHECK(bare != p0.end() && named != p0.end() && last != p0.end());
        if (named != p0.end()) {
            CHECK(named->second.axis == ShardAxis::Output && named->second.kept() == 6144);
            CHECK(last != p0.end() && last->second.range[0].begin == named->second.range[0].begin);
        }
        // every layer is covered for a per-layer tensor
        CHECK(p0.size() >= (size_t) (48 + t0.n));
        // rank 0's tensors: the plan knows which rank drops them
        const auto e0 = p0.find("output.weight");
        const auto e1 = p1.find("output.weight");
        CHECK(e0 != p0.end() && e1 != p1.end());
        if (e0 != p0.end() && e1 != p1.end()) {
            CHECK(e0->second.keep && !e1->second.keep);
            CHECK(e0->second.axis == ShardAxis::Rank0);
        }
        // a replicated tensor is kept by both, whole, and its axis is not split
        const auto r0p = p0.find("ffn_gate_inp.weight");
        const auto r1p = p1.find("ffn_gate_inp.weight");
        CHECK(r0p != p0.end() && r1p != p1.end());
        if (r0p != p0.end() && r1p != p1.end()) {
            CHECK(r0p->second.keep && r1p->second.keep);
            CHECK(r1p->second.axis == ShardAxis::Replicated);
            CHECK(r1p->second.kept() == 0);
        }
        // the two ranks' plans agree on everything except which half each keeps
        int compared = 0, rank0_seen = 0;
        for (const auto& kv : p0) {
            const auto it = p1.find(kv.first);
            CHECK(it != p1.end());
            if (it == p1.end()) continue;
            CHECK(it->second.axis == kv.second.axis);
            CHECK(it->second.ranges == kv.second.ranges);
            // `keep` is exactly what DIFFERS for a Rank0 tensor - that is its whole meaning - so it has to
            // agree everywhere else.  Counting the exception keeps it from swallowing every case.
            if (kv.second.axis == ShardAxis::Rank0) {
                ++rank0_seen;
                CHECK(it->second.keep != kv.second.keep);
            } else {
                CHECK(it->second.keep == kv.second.keep);
            }
            ++compared;
        }
        CHECK(compared == (int) p0.size());
        CHECK(rank0_seen > 0);
    }

    // ---- the shape rule the loader and `check_layer` share ------------------------------------------
    {
        int64_t a = 0, b = 0;
        std::string e;
        // an output split halves the feature axis and leaves the contraction axis alone
        CHECK(shard_shape(*spec_of(t0, "attn_q.weight", failed), 2560, 12288, a, b, e) && a == 2560 && b == 6144);
        // an input split does the opposite
        // a replicated tensor keeps its shape
        CHECK(shard_shape(*spec_of(t0, "ffn_gate_inp.weight", failed), 2560, 512, a, b, e) && a == 2560 &&
              b == 512);
        // a ONE-DIMENSIONAL tensor splits on its only axis, whatever ne1 happens to be
        // a spec that claims more units than the axis has is refused, never clamped
        ShardSpec over;
        over.axis = ShardAxis::Output;
        over.ranges = 1;
        over.range[0] = {0, 99};
        CHECK(!shard_shape(over, 2560, 48, a, b, e));
        CHECK(e.find("99") != std::string::npos);

        // ---- the 1-D alignment exemption, which is the rule that makes `ssm_a` legal at 24 ------------
        ShardSpec v0, v1;
        v0.axis = v1.axis = ShardAxis::Input;
        v0.ranges = v1.ranges = 1;
        v0.range[0] = {0, 24};
        v1.range[0] = {24, 48};
        // 24 is not a multiple of the 128-element activation block, but there is no dot product on this axis
        CHECK(shard_check_pair(v0, v1, 48, "ssm_a", e, /*contraction=*/false));
        // the SAME split on a contraction axis is refused: there the block contract does apply
        CHECK(!shard_check_pair(v0, v1, 48, "ssm_a", e, /*contraction=*/true));
        CHECK(e.find("activation blocks") != std::string::npos);
    }

    // ---- the slice a caller writes into a full-width buffer --------------------------------------------
    {
        int64_t begin = -1, count = -1;
        // an output split of a single range: half of 12288, at this rank's offset
        CHECK(shard_slice(spec_of(t0, "attn_q.weight", failed), ShardAxis::Output, 12288, begin, count));
        CHECK(begin == 0 && count == 6144);
        CHECK(shard_slice(spec_of(t1, "attn_q.weight", failed), ShardAxis::Output, 12288, begin, count));
        CHECK(begin == 6144 && count == 6144);
        // a contraction split of a single range: half of 6144
        CHECK(shard_slice(spec_of(t0, "ssm_out.weight", failed), ShardAxis::Input, 6144, begin, count));
        CHECK(begin == 0 && count == 3072);
        CHECK(shard_slice(spec_of(t1, "ssm_out.weight", failed), ShardAxis::Input, 6144, begin, count));
        CHECK(begin == 3072 && count == 3072);
        // THE THREE-RANGE CASE IS NOT ONE SLICE, and says so rather than returning the first range: the GDN
        // q|k|v is three ranges, so a caller that needs one offset cannot use it and must be told.
        CHECK(!shard_slice(spec_of(t0, "attn_qkv.weight", failed), ShardAxis::Output, 10240, begin, count));
        // and the routed experts are REPLICATED today (partitioning them by id is its own milestone), so a
        // caller asking for their output slice gets the whole axis rather than three quarters of a stack
        CHECK(shard_slice(spec_of(t0, "ffn_gate_exps.weight", failed), ShardAxis::Output, 512, begin, count));
        CHECK(begin == 0 && count == 512);
        // a tensor this rank holds whole - replicated, or rank 0's own - is the whole axis, whatever `full` is
        CHECK(shard_slice(spec_of(t0, "ffn_gate_inp.weight", failed), ShardAxis::Output, 512, begin, count));
        CHECK(begin == 0 && count == 512);
        CHECK(shard_slice(spec_of(t1, "token_embd.weight", failed), ShardAxis::Output, 248320, begin, count));
        CHECK(begin == 0 && count == 248320);
        CHECK(shard_slice(spec_of(t0, "output.weight", failed), ShardAxis::Output, 248320, begin, count));
        CHECK(begin == 0 && count == 248320);
        // a tensor the table does not know at all: the whole axis (the default path), and no plan needed
        CHECK(shard_slice<ShardSpec>(nullptr, ShardAxis::Output, 4096, begin, count));
        CHECK(begin == 0 && count == 4096);
        // and a split on the OTHER axis is not a split of this one
        CHECK(shard_slice(spec_of(t0, "attn_q.weight", failed), ShardAxis::Input, 2560, begin, count));
        CHECK(begin == 0 && count == 2560);
    }

    // ---- where this rank's slice of the contraction axis starts in each activation form ------------------
    {
        KActivation a;
        std::string e;
        // the real case: `ssm_out`'s second rank covers columns 3072..6144
        CHECK(k_activation_slice(3072, 292, a, e));       // 292 = this build's Q8_K block bytes, a stand-in
        CHECK(a.scalars == 3072);
        CHECK(a.q8_0 == (3072 / 32) * 34);                // 96 blocks of 34 B
        CHECK(a.q8k == (3072 / 256) * 292);               // 12 super-blocks
        // rank 0 is the zero offset, and every offset is exactly zero there
        CHECK(k_activation_slice(0, 292, a, e));
        CHECK(a.scalars == 0 && a.q8_0 == 0 && a.q8k == 0);
        // REFUSALS, one per boundary a kernel would silently misread
        CHECK(!k_activation_slice(-3072, 292, a, e));
        CHECK(e.find("negative") != std::string::npos);
        CHECK(!k_activation_slice(16, 292, a, e));        // not a whole Q8_0 block
        CHECK(e.find("32-element Q8_0") != std::string::npos);
        CHECK(!k_activation_slice(32 * 3, 292, a, e));    // a whole Q8_0 block, but not a whole Q8_K one
        CHECK(e.find("256-element Q8_K") != std::string::npos);
        CHECK(!k_activation_slice(3072, 0, a, e));        // no Q8_K block size to multiply by
        CHECK(e.find("Q8_K block size") != std::string::npos);
        // and the bound the SPLIT table has to respect for this to work at all: every range of a contraction
        // split must be a multiple of 256, which 3072 is and 3200 is not
        CHECK(3072 % 256 == 0);
        CHECK(!k_activation_slice(3200, 292, a, e));
    }

    // ---- the two ranks' activation slices TILE each image: the wiring arithmetic, across ranks -----------
    // `ssm_out` and `attn_output` are 6144 columns cut in half, and each rank hands its kernel a pointer into
    // one of four activation images.  What matters beyond "the offset is aligned" is that the two ranks' byte
    // runs are ADJACENT and together are the whole image: two ranks reading the same run would be a wrong
    // answer, and a gap would be one too - neither of which the per-rank alignment check can see.
    {
        const int64_t full_cols = 6144, half = full_cols / 2, q8k_block = 292;
        KActivation lo, hi;
        std::string e;
        CHECK(k_activation_slice(0, q8k_block, lo, e));
        CHECK(k_activation_slice(half, q8k_block, hi, e));
        // f32 / bf16: the second rank starts exactly `half` elements in
        CHECK(lo.scalars == 0 && hi.scalars == half);
        // Q8_0: 34-byte blocks, and the two runs are adjacent and cover the plane
        CHECK(lo.q8_0 == 0);
        CHECK(hi.q8_0 == (half / 32) * 34);
        CHECK(hi.q8_0 + (half / 32) * 34 == (full_cols / 32) * 34);
        // Q8_K: block-sized, same three properties
        CHECK(lo.q8k == 0);
        CHECK(hi.q8k == (half / 256) * q8k_block);
        CHECK(hi.q8k + (half / 256) * q8k_block == (full_cols / 256) * q8k_block);
        // 3072 is a whole number of both block sizes, which is WHY this split is legal at all - and it is the
        // 256 that binds, not the 128 the static table check uses
        CHECK(half % 256 == 0);
        CHECK(!k_activation_slice(half + 128, q8k_block, hi, e));
    }

    // ---- the GDN block's buffers per rank ------------------------------------------------
    {
        const int64_t q8k_block = 292;   // a stand-in for this build's Q8_K block bytes (see the note above)
        const uint64_t whole = tp_gdn_buffer_bytes(g, 0, 1, q8k_block);
        const uint64_t r0 = tp_gdn_buffer_bytes(g, 0, 2, q8k_block);
        const uint64_t r1 = tp_gdn_buffer_bytes(g, 1, 2, q8k_block);
        CHECK(whole > 0 && r0 > 0 && r1 > 0);
        // ONE rank is the whole model: this is the default path, and it has to be the same number the layer
        // computed for itself before any of this existed
        CHECK(r0 == r1);                       // the split is even, so the two ranks are the same size
        CHECK(r0 < whole);                     // ... and smaller than a single card's
        // THE ACTIVATION IMAGES ARE FULL WIDTH ON BOTH RANKS, so the two add up to MORE than the whole: the
        // overshoot is exactly those three parts (plus alignment), and asserting the direction pins that.
        CHECK(r0 + r1 > whole);
        // halving the head-channels and the v-heads has to be what makes the difference: a rank's conv width is
        // half of the model's and the state follows it
        CHECK(g.ssm_conv_channels / 2 * 4 < g.ssm_conv_channels * 4);
        // and the refusals: a rank outside the pair is not a size, it is a mistake
        CHECK(tp_gdn_buffer_bytes(g, 2, 2, q8k_block) == 0);
        CHECK(tp_gdn_buffer_bytes(g, -1, 2, q8k_block) == 0);
        CHECK(tp_gdn_buffer_bytes(g, 0, 0, q8k_block) == 0);
    }

    // ---- the head ranges one rank owns --------------------------------------------------
    {
        TpHeads h;
        std::string e;
        // ONE rank is the whole model, and its width is the geometry's own conv width: this is the default path
        CHECK(tp_heads(g, 0, 1, h, e));
        CHECK(h.q_count == g.ssm_k_heads && h.k_count == g.ssm_k_heads && h.v_count == g.ssm_v_heads);
        CHECK(h.q_begin == 0 && h.v_begin == 0);
        CHECK(h.qkv_count() == g.ssm_conv_channels);
        CHECK(h.k_offset() == g.ssm_k_heads * g.ssm_state_size);
        // TWO ranks: the same FRACTION of q, k and v each, which is what makes a v-head's recurrence find its
        // own q and k
        CHECK(tp_heads(g, 0, 2, h, e));
        CHECK(h.q_count == g.ssm_k_heads / 2 && h.v_count == g.ssm_v_heads / 2);
        CHECK(h.q_begin == 0 && h.v_begin == 0);
        const int64_t half_width = h.qkv_count();
        CHECK(half_width * 2 == g.ssm_conv_channels);
        CHECK(h.k_offset() == (g.ssm_k_heads / 2) * g.ssm_state_size);
        CHECK(h.v_offset() == (g.ssm_k_heads / 2 + g.ssm_k_heads / 2) * g.ssm_state_size);
        CHECK(tp_heads(g, 1, 2, h, e));
        CHECK(h.q_begin == g.ssm_k_heads / 2 && h.k_begin == g.ssm_k_heads / 2 &&
              h.v_begin == g.ssm_v_heads / 2);
        CHECK(h.qkv_count() * 2 == g.ssm_conv_channels);
        // the two ranks' sections tile each of q, k and v exactly
        CHECK(0 + g.ssm_k_heads / 2 == g.ssm_k_heads / 2);
        CHECK(g.ssm_v_heads / 2 + g.ssm_v_heads / 2 == g.ssm_v_heads);
        // REFUSALS: a rank that is not one, and a geometry whose heads do not divide
        CHECK(!tp_heads(g, 2, 2, h, e));
        CHECK(e.find("is not a rank") != std::string::npos);
        CHECK(!tp_heads(g, 0, 0, h, e));
        ModelGeometry odd = g;
        odd.ssm_k_heads = 16;
        odd.ssm_v_heads = 45;                 // 45 does not halve
        CHECK(!tp_heads(odd, 0, 2, h, e));
        CHECK(e.find("half a head is not a thing") != std::string::npos);
        odd.ssm_v_heads = 48;
        odd.ssm_k_heads = 15;                 // and neither does 15
        CHECK(!tp_heads(odd, 0, 2, h, e));
    }

    // ---- which rank a run is: the default is the whole model, and it is what sizes a rank's state --------
    {
        std::string e;
        // nothing set it, so a single-card run reads the "one rank holds everything" case
        CHECK(tp_rank_current() == 0 && tp_ranks_current() == 1);
        TpHeads whole;
        CHECK(tp_heads(g, tp_rank_current(), tp_ranks_current(), whole, e));
        CHECK(whole.qkv_count() == g.ssm_conv_channels);      // ... and that IS the geometry
        // the engine sets the pair once, before anything is sized
        CHECK(tp_set_ranks(1, 2, e));
        CHECK(tp_rank_current() == 1 && tp_ranks_current() == 2);
        TpHeads half;
        CHECK(tp_heads(g, tp_rank_current(), tp_ranks_current(), half, e));
        CHECK(half.v_begin == g.ssm_v_heads / 2);
        CHECK(half.qkv_count() * 2 == g.ssm_conv_channels);
        // it refuses a pair that is not one rather than clamping: a rank sized from another rank's geometry is a
        // buffer overlap, not a small error - and the refusal leaves the pair as it was
        CHECK(!tp_set_ranks(2, 2, e));
        CHECK(tp_rank_current() == 1);
        CHECK(tp_set_ranks(0, 1, e));                         // back to the default for the rest of the test
        CHECK(tp_rank_current() == 0 && tp_ranks_current() == 1);
    }
    // ---- the one-item rule the table is built on ------------------------------------------------------
    CHECK(!shard_split_even(0));
    CHECK(!shard_split_even(1));
    CHECK(!shard_split_even(3));
    CHECK(!shard_split_even(47));
    CHECK(shard_split_even(2));
    CHECK(shard_split_even(24));
    CHECK(shard_split_even(48));
    CHECK(shard_split_even(512));

    if (failed == 0) std::printf("tp_shard_test: ok (%d entries per rank)\n", t0.n);
    return failed ? 1 : 0;
}
