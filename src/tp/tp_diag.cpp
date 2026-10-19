// src/tp/tp_diag.cpp - TPDIAG's device log, moved VERBATIM out of src/core/verify.cpp (modularization: the
// infrastructure lives here so a rebase onto an upstream release conflicts only at the probe seams that stay in
// verify.cpp; when the question is answered, deleting this file, the header, and those seams is the whole
// removal).  The dead helpers the block carried (tp_wprobe_reserve + its scratch, tpdiag_ple_mutex,
// tpdiag_hc_record, g_tpdiag_budget, g_tpdiag_armed - zero call sites at move time) were dropped, not moved.
#include "strata/tp/tp_diag.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/core/shard.hpp"        // tp_rank_current
#include "strata/kernels/ngram.hpp"     // NG_HC_DIM

namespace strata::tp {

bool tpdiag_layer_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_TP_DIAG_LAYERS"); return v != nullptr && v[0] != '0'; }();
    return on;
}

/// TPDIAG's device log: the residual as it leaves each layer, written INSIDE the capture by a device-to-device
/// copy - the one copy kind a capture allows, where a synchronise fails the whole capture (measured).  It is read
/// back only after `cudaStreamEndCapture`, so both ranks can be compared layer by layer.
/// ONE LOG PER RANK: the two ranks run in two threads and both write here, so a single shared buffer
/// reports whichever rank wrote last, which reads as an intermittent divergence that is not there.
float* g_tpdiag_log[2] = {nullptr, nullptr};
size_t g_tpdiag_cap[2] = {0, 0};

void tpdiag_log_reserve(size_t dn) {
    if (!tpdiag_layer_on() || dn == 0) return;
    const size_t need = dn * (size_t) 560;
    // BOTH indices: rank 1's `init` runs on the thread that drives rank 0 (`TpRank::build`), so whatever rank
    // its own thread reports later must already have a log of its own.
    for (int r = 0; r < 2; ++r) {
        if (g_tpdiag_log[r] != nullptr && g_tpdiag_cap[r] >= need) continue;
        if (g_tpdiag_log[r] != nullptr) { (void) cudaFree(g_tpdiag_log[r]); g_tpdiag_log[r] = nullptr; g_tpdiag_cap[r] = 0; }
        if (cudaMalloc(&g_tpdiag_log[r], need * sizeof(float)) == cudaSuccess) g_tpdiag_cap[r] = need;
        else { g_tpdiag_log[r] = nullptr; g_tpdiag_cap[r] = 0; }
    }
    std::fprintf(stderr, "TPDIAG reserve rank=%d need=%zu MiB log0=%d log1=%d\n",
                 strata::core::tp_rank_current(), (need * sizeof(float)) >> 20,
                 g_tpdiag_log[0] != nullptr ? 1 : 0, g_tpdiag_log[1] != nullptr ? 1 : 0);
}

/// TPDIAG: is the FULL probe set on?  Every record becomes a node in the captured window, and the full set
/// (about 430 per window) HANGS the GPU stage - measured on two cards: with them on, 587 s and the engine's own
/// `verify: an earlier window never finished on the GPU (#267)`; with them off, 83 s and a clean run.  So by
/// default only the slots the current question needs are recorded (the PLE inputs, 556/557) and everything else
/// needs STRATA_TP_DIAG_FULL=1.
bool tpdiag_full_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_TP_DIAG_FULL"); return v != nullptr && v[0] == (char) 49; }();
    return on;
}

void tpdiag_pre_record(const float* buf, int64_t l, int grp, int T, int64_t HC, int64_t N, void* cs) {
    if (!tpdiag_full_on()) return;
    if (!tpdiag_layer_on() || l < 0 || l >= 48) return;
    const int rk = strata::core::tp_rank_current() < 2 ? strata::core::tp_rank_current() : 0;
    if (g_tpdiag_log[rk] == nullptr) return;
    const size_t dn = (size_t) T * (size_t) HC * (size_t) N;
    const size_t slot = 96 + (size_t) l * 2 + (size_t) (grp < 2 ? grp : 0);
    if (dn != 0 && dn * (slot + 1) <= g_tpdiag_cap[rk])
        (void) cudaMemcpyAsync(g_tpdiag_log[rk] + dn * slot, buf, dn * sizeof(float), cudaMemcpyDeviceToDevice,
                               (cudaStream_t) cs);
}

/// The residual ENTERING the layer loop (slot 192): if this already differs between the ranks, the divergence is
/// upstream of every layer.
/// A reduce's operand, straight before and straight after the collective (slots 200..203).  The two ranks'
/// PARTIALS must differ (each holds its own half of the contraction) while their SUMS must agree, so a
/// disagreement here separates "the collective delivered the wrong thing" from "this rank's own half is wrong".
void tpdiag_reduce_record(const float* buf, int slot, int64_t n, int64_t stride, void* cs) {
    if (!tpdiag_layer_on() || n <= 0 || stride <= 0) return;
    // 200..247 and 260..307 are the pre/post slots of the two reduced contractions (T*N floats each, ~10 KB):
    // they are cheap and they answer the open question - whether the all-reduce hands BOTH ranks the same sum -
    // so they stay on with only STRATA_TP_DIAG_LAYERS=1 (a full-probe run often dies of the collective instead).
    // 200..307 are back behind the full-probe gate: they were verified 48/48 already and cost 14 blocking
    // reads a window, which is what pushed a run past its timeout.
    if (slot != 474 && slot != 475 && slot != 476 && slot != 192 && slot != 458) return;   // minimal: projection I/O + GDN input
    const int rk = strata::core::tp_rank_current() < 2 ? strata::core::tp_rank_current() : 0;
    if (g_tpdiag_log[rk] == nullptr) return;
    const size_t off = (size_t) slot * (size_t) stride;
    if (off + (size_t) n <= g_tpdiag_cap[rk])
        (void) cudaMemcpyAsync(g_tpdiag_log[rk] + off, buf, (size_t) n * sizeof(float), cudaMemcpyDeviceToDevice,
                               (cudaStream_t) cs);
}

void tpdiag_input_record(const float* buf, int T, int64_t HC, int64_t N, void* cs) {
    if (!tpdiag_layer_on()) return;
    const int rk = strata::core::tp_rank_current() < 2 ? strata::core::tp_rank_current() : 0;
    if (g_tpdiag_log[rk] == nullptr) return;
    const size_t dn = (size_t) T * (size_t) HC * (size_t) N;
    const size_t slot = 192;
    if (dn != 0 && dn * (slot + 1) <= g_tpdiag_cap[rk])
        (void) cudaMemcpyAsync(g_tpdiag_log[rk] + dn * slot, buf, dn * sizeof(float), cudaMemcpyDeviceToDevice,
                               (cudaStream_t) cs);
}

void tpdiag_log_dump(int T, int64_t HC, int64_t N, const char* tag, int64_t win) {
    if (!tpdiag_layer_on()) return;
    const int rk = strata::core::tp_rank_current() < 2 ? strata::core::tp_rank_current() : 0;
    if (g_tpdiag_log[rk] == nullptr) return;
    const size_t dn = (size_t) T * (size_t) HC * (size_t) N;
    if (dn == 0) return;
    std::vector<float> h(dn);
    // MINIMAL DUMP (fast): the layer-0 bracket.  Every slot here is recorded with just STRATA_TP_DIAG_LAYERS=1
    // (the 400..495 range), and the FIRST one whose line differs between the two ranks is where the divergence
    // enters the window.  All reads are T*N floats except 448/449/462/463 (T*HC*N), 461/469/470 (T*HC),
    // and 467/468 (the weights, 61440).
    auto rd = [&](int slot, int64_t n, const char* nm) {
        const size_t off = (size_t) slot * dn;
        if (off + (size_t) n > g_tpdiag_cap[rk]) return;
        if (cudaMemcpy(h.data(), g_tpdiag_log[rk] + off, (size_t) n * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return;
        double s = 0.0, s2 = 0.0;
        for (int64_t i = 0; i < n; ++i) { const double v = h[(size_t) i]; s += v; s2 += v * v; }
        if (s == 0.0 && s2 == 0.0) return;   // an unrecorded slot
        std::fprintf(stderr, "TPDIAG %ss%d_%s win=%lld rank=%d sum=%.9g sumsq=%.9g v0=%.9g\n", tag, slot, nm,
                     (long long) win, strata::core::tp_rank_current(), s, s2, (double) h[0]);
    };
    const int64_t TN = (int64_t) T * N, THC = (int64_t) T * HC, THCN = (int64_t) dn;
    rd(192, THCN, "input");
    rd(476, (int64_t) 3072, "yin");
    rd(458, TN, "mix1");
    rd(448, THCN, "R_wb");
    rd(457, TN, "mix_pre1");
    rd(458, TN, "mix_rd1");
    rd(449, THCN, "R_rd1");
    rd(473, TN, "bo_rd1");
    rd(475, (int64_t) N, "redpre");
    rd(474, TN, "redout");
    rd(465, TN, "mix_pre2");
    rd(466, TN, "mix_rd2");
    rd(467, 61440, "wd1");
    rd(468, 61440, "wi1");
    rd(469, THC, "lo");
    rd(470, THC, "rs");
    rd(471, THC, "inj");
    rd(472, TN, "bo_pre2");
    rd(460, TN, "bo_moe");
    rd(461, THC, "inj2");
    rd(462, THCN, "R_wb_last");
    rd(463, THCN, "R_upd");
    rd(455, TN, "mix_router");
    // The original had `return;` here followed by ~157 lines of full-probe-set dumping that was already
    // unreachable in verify.cpp (dead since the minimal dump answered the question).  Dropped, not moved:
    // `git log -p` on this file's predecessor (src/core/verify.cpp, pre-modularization) has it verbatim.
}

}  // namespace strata::tp
