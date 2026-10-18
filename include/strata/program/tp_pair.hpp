// include/strata/program/tp_pair.hpp - ONE TENSOR-PARALLEL RANK'S STATE, and how it is built.
//
// The engine drives two ranks that each hold a slice of every split weight and run the SAME window on the same
// tokens.  A rank is therefore: its own device, its own shard of the weights (the canonical arena and the native
// upload), its own session, a nominal expert tier, and its own `Verifier` on its own stream.  The collective is
// shared and owned here too, because the two ranks meet on it.
//
// WHAT THIS DELIBERATELY DOES NOT OWN: the expert POOL.  A `Verifier`'s misses are served by a pool driver the
// engine already builds (`drive_pool_multi`/`drive`, passed to `Verifier::run`), and the rank's `run` call site
// hands it its own - so this header stays out of the doorbell's business and the engine keeps serving experts
// exactly as it does today.
//
// WHY THE RANK IS SET BEFORE `init`: `Verifier::init` carves its arena and its GDN buffers from
// `tp_ranks_current()`, and the weights above were loaded from the rank's plan.  A rank that sized itself from
// another rank's geometry would overlap buffers, so `tp_set_ranks` comes first and is checked.
#pragma once

#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/shard.hpp"
#include "strata/core/tp_host_runtime.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::program {

/// Switches to a device for the DURATION OF A SCOPE and restores the previous one, exactly as the engine's own
/// `OnDevice` guard does.  Leaving a rank's device current is not a small thing: everything the engine allocates
/// afterwards - its pinned arena, its logits - would land on the wrong card.
struct OnTpDevice {
    int prev = -1;
    explicit OnTpDevice(int d) {
        (void) cudaGetDevice(&prev);
        (void) cudaGetLastError();   // a stale error is not this call's (this engine probes failing allocations)
        (void) cudaSetDevice(d);
    }
    ~OnTpDevice() {
        if (prev >= 0) (void) cudaSetDevice(prev);
    }
    OnTpDevice(const OnTpDevice&) = delete;
    OnTpDevice& operator=(const OnTpDevice&) = delete;
};

/// One rank of a tensor-parallel pair.
struct TpRank {
    int dev = -1;
    int rank = 0;
    int ranks = 2;
    cudaStream_t stream = nullptr;
    void* arena = nullptr;
    void* session_buf = nullptr;
    strata::core::ExpertCache xcache;  ///< rank 0's expert cache, replicated here (an ExpertCache so the prompt
                                       ///< path's lend/refill and the verify hit path both work on this card)
    int32_t* residency = nullptr;  ///< device [n_layers * n_expert], every entry -1 (nothing resident)
    uint64_t arena_bytes = 0;
    strata::core::WeightTable wt;
    strata::core::NativeDense dense;
    strata::core::SessionState ss;
    strata::core::VerifyHits vh;
    strata::core::Verifier ver;

    /// OPENS THIS RANK'S DEVICE AND STREAM - and it has to happen EARLY, before this engine allocates the
    /// single-card arenas it needs for its own context.  That is the order the layer split creates its stages in
    /// (`generate.cpp`: every stage's streams are made before the expert cache is filled), and a stream asked for
    /// later on a card the process has already been busy with comes back `out of memory` even though the card is
    /// empty.  Everything else (`load`) can wait.
    bool open(int device, int which, int how_many, std::string& err) {
        dev = device;
        rank = which;
        ranks = how_many;
        int prev_open = -1;
        (void) cudaGetDevice(&prev_open);
        // FIRST, and it has to be before this card's context exists: the flags the engine's own multi-GPU path
        // gives a second card (`RemoteExperts::preflight`).  `cudaDeviceMapHost` is what lets this rank see the
        // mapped pinned staging the collective runs through, and without it a context made by a bare
        // `cudaSetDevice` leaves WDDM accounting so confused that `cudaMalloc` on the FIRST card fails outright
        // (measured: the expert cache's 1.27 GiB, then even 0.16 GiB, `out of memory` with 16 GiB free).
#if !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)
        if (const char* spin = std::getenv("STRATA_REMOTE_SPIN"); !(spin && spin[0] == '0'))
            (void) cudaInitDevice(dev, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
        (void) cudaGetLastError();
#endif
        (void) cudaGetLastError();
        const cudaError_t sd = cudaSetDevice(dev);
        int now = -1;
        (void) cudaGetDevice(&now);
        if (sd != cudaSuccess && now != dev) {
            int n_dev = 0;
            cudaGetDeviceCount(&n_dev);
            const cudaError_t se = cudaGetLastError();
            std::string names;
            for (int d = 0; d < n_dev; ++d) {
                cudaDeviceProp pr{};
                if (cudaGetDeviceProperties(&pr, d) == cudaSuccess) names += " " + std::to_string(d) + ":" + pr.name;
            }
            err = std::string("tensor parallel: cudaSetDevice(") + std::to_string(dev) + ") failed: " +
                  cudaGetErrorString(se) + "; this process sees " + std::to_string(n_dev) + " device(s):" + names;
            return false;
        }
        // A bare switch to this rank's device and a stream on it.  Nothing else: `cudaMalloc`/`cudaFree` here
        // (a "warm-up") was measured to make the engine's expert-cache sizing read almost no free VRAM on the
        // OTHER card, which sizes the cache to zero slots - the layer split's own stage loop, which the pricing
        // coexists with, does no such allocation before its streams.
        //
        // The stream is BEST EFFORT.  This rank has to be opened after the engine has allocated its expert cache
        // (opened earlier, WDDM fails that cache's `cudaMalloc` on a card with 16 GiB free), and by then
        // `cudaStreamCreateWithFlags` on the second card comes back `out of memory` - the same call succeeds
        // before those allocations, so it is the allocation, not the card.  The default stream needs no
        // allocation and is a legitimate home for this rank: it owns the card, so there is nothing to overlap.
        if (stream == nullptr) {
            (void) cudaGetLastError();
            const cudaError_t se = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (se != cudaSuccess) {
                stream = nullptr;   // the default stream; the rank runs on it instead
                (void) cudaGetLastError();
                int here = -1;
                (void) cudaGetDevice(&here);
                std::fprintf(stderr, "strata tp: rank %d on device %d: its own stream was refused (%s, current %d), "
                                     "running on the default stream\n", rank, dev, cudaGetErrorString(se), here);
            }
        }
        {
            cudaDeviceProp pr{};
            if (cudaGetDeviceProperties(&pr, dev) == cudaSuccess)
                std::fprintf(stderr, "strata tp: rank %d opened device %d (%s) (stream %s)\n", rank, dev, pr.name,
                             stream != nullptr ? "up" : "not yet");
            else
                std::fprintf(stderr, "strata tp: rank %d opened device %d (stream %s)\n", rank, dev,
                             stream != nullptr ? "up" : "not yet");
        }
        (void) cudaSetDevice(prev_open);   // see `OnTpDevice`: this rank's device must not stay current
        return true;
    }

    /// Loads this rank's shards and builds its session, tier and verifier.  `skip` is the engine's own skip set
    /// (the tensors the native path serves) - the same one the single-card path uses.  `src_cache`, when given,
    /// is rank 0's expert cache: it is REPLICATED on this rank's card so the replicated MoE is computed
    /// identically on both ranks (the nominal tier made this rank the only all-miss one, and the MoE outputs
    /// diverged from an identical input).
    bool build(const std::string& pack, const std::vector<std::string>& native_shards,
               const std::set<std::string>* skip, const strata::core::ModelGeometry& g, int64_t max_context,
               int64_t k, int device, int which, int how_many, const strata::core::NativeHead* head, int max_t,
               std::string& err, const strata::core::VerifyHits* src_cache = nullptr,
               int64_t src_cache_bytes = 0, const std::vector<int64_t>* slot_bytes = nullptr) {
        dev = device;
        rank = which;
        ranks = how_many;
        // The card the cache is replicated FROM: the caller's device before this rank's is opened (rank 0's).
        int src_dev = -1;
        (void) cudaGetDevice(&src_dev);
        if (!open(dev, rank, ranks, err)) return false;
        const OnTpDevice on_rank(dev);   // every allocation below belongs to THIS rank's card, and the engine's
                                         // current device is restored when this function returns
        // A STALE ERROR IS NOT THIS CALL'S.  `cudaSetDevice` reports the context's last error state, and this
        // engine probes allocations that are meant to fail (its memory guard), so the error it returns can be an
        // out-of-memory from something else entirely.  The device it actually switched to is the authority.
        (void) cudaGetLastError();
        const cudaError_t sd = cudaSetDevice(dev);
        int now = -1;
        (void) cudaGetDevice(&now);
        if (sd != cudaSuccess && now != dev) {
            int n_dev = 0;
            cudaGetDeviceCount(&n_dev);
            const cudaError_t se = cudaGetLastError();
            std::string names;
            for (int d = 0; d < n_dev; ++d) {
                cudaDeviceProp pr{};
                if (cudaGetDeviceProperties(&pr, d) == cudaSuccess) {
                    names += " " + std::to_string(d) + ":" + pr.name;
                }
            }
            err = std::string("tensor parallel: cudaSetDevice(") + std::to_string(dev) + ") failed: " +
                  cudaGetErrorString(se) + "; this process sees " + std::to_string(n_dev) + " device(s):" + names;
            return false;
        }
        (void) cudaGetLastError();   // see the note above: this engine leaves failures latched on purpose
        // BEST EFFORT, for the reason `open` gives: by the time a rank is built the engine has already allocated
        // its expert cache, and `cudaStreamCreateWithFlags` on the second card then comes back `out of memory`.
        // The default stream needs no allocation, and this rank owns its card, so it runs there instead.
        if (stream == nullptr) {
            const cudaError_t se = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            if (se != cudaSuccess) {
                stream = nullptr;
                (void) cudaGetLastError();
                std::fprintf(stderr, "strata tp: rank %d on device %d: its own stream was refused (%s), running on "
                                     "the default stream\n", rank, dev, cudaGetErrorString(se));
            }
        }
        // this rank's plan decides what it holds; `plan` outlives the load call below
        strata::core::ShardTable table;
        strata::core::ShardPlan plan;
        if (!strata::core::shard_table(g, rank, table, err) ||
            !strata::core::shard_plan_layers(table, g, rank, plan, err)) {
            return false;
        }
        if (!strata::core::WeightTable::pool_bytes(pack, arena_bytes, err, skip, &plan)) return false;
        (void) cudaGetLastError();
        if (cudaMalloc(&arena, arena_bytes) != cudaSuccess) {
            err = "tensor parallel: the rank's arena failed";
            return false;
        }
        if (!wt.load(pack, arena, arena_bytes, err, skip, &plan)) return false;
        // the native upload takes only this rank's rows of each matrix it serves
        if (!dense.load(native_shards, wt, err, false, 0, -1, &plan)) return false;
        // the session, exactly as the layer split builds one (`generate.cpp:4550-4551`)
        (void) cudaGetLastError();
        if (cudaMalloc(&session_buf, strata::core::session_bytes(g, max_context, k, 0, g.n_layers)) != cudaSuccess) {
            err = "tensor parallel: the rank's session failed";
            return false;
        }
        if (strata::core::session_init(g, max_context, k, session_buf, ss, 0, g.n_layers) == 0) {
            err = "tensor parallel: session_init failed";
            return false;
        }
        if (src_cache != nullptr && src_cache->n_slots > 0 && src_cache->cache_base != nullptr &&
            src_cache->d_res != nullptr && src_cache->blob > 0 && std::getenv("STRATA_TP_NO_CACHE") == nullptr) {
            // REPLICATE RANK 0'S EXPERT CACHE ON THIS CARD.  The MoE is replicated, not sharded, so both ranks
            // must compute it identically: the same residents through the same GPU hit kernel and the same
            // misses through the shared pool.  A nominal tier made this rank the only all-miss one - its misses
            // ran on the CPU pool while rank 0's residents ran on the GPU, and the route tail skip then dropped
            // experts on this rank that rank 0 computed.  The measured divergence was exactly that: the MoE
            // output differed from an identical input (slot 460: rank 0 sum 14.0142878, rank 1 sum 12.7674137).
            // The bytes and the residency table are copied so this rank's hit path is rank 0's twin.  The copy
            // is cross-device (`cudaMemcpyDefault` under UVA stages it, these cards have no peer access).
            vh.n_slots = src_cache->n_slots;
            vh.blob = src_cache->blob;
            // The arena's real span, not n_slots * blob: the slots are SIZED (`ExpertCache::open_sized`), so
            // `blob` (the largest expert) overestimates most of them and n_slots * blob reads past the arena.
            const size_t cache_bytes = src_cache_bytes > 0 ? (size_t) src_cache_bytes
                                                           : (size_t) vh.n_slots * (size_t) vh.blob;
            (void) cudaGetLastError();
            // This rank's cache is a real ExpertCache (rank 0's twin), so the prompt path can lend its tail slots
            // and the refill can put the experts back - a bare allocation would have neither.
            if (slot_bytes == nullptr || slot_bytes->empty()) {
                err = "tensor parallel: the expert cache replica needs the source cache's slot sizes";
                return false;
            }
            if (!xcache.open_sized(*slot_bytes, g.n_layers, g.n_expert, err)) return false;
            // The copy is staged through the host: `cudaMemcpyPeer` between these cards reports "invalid
            // argument" (they have no peer access), so the bytes go rank 0's card -> host -> this card.
            {
                // Chunked and pinned: one 1.75 GiB pageable D2H comes back "invalid argument" on this driver,
                // so the bytes move rank 0's card -> pinned host -> this card in 64 MiB pieces.
                const uint8_t* sp = src_cache->cache_base;
                const size_t chunk = 64u << 20;
                uint8_t* stage = nullptr;
                cudaError_t e = cudaHostAlloc((void**) &stage, chunk, cudaHostAllocDefault);
                if (e != cudaSuccess) {
                    err = std::string("tensor parallel: the replica staging buffer: ") + cudaGetErrorString(e);
                    return false;
                }
                bool cok = true;
                uint8_t* dcache = (uint8_t*) xcache.device_slot(0);
                for (size_t off = 0; off < cache_bytes && cok; off += chunk) {
                    const size_t nb = (off + chunk <= cache_bytes) ? chunk : cache_bytes - off;
                    if ((e = cudaSetDevice(src_dev)) != cudaSuccess ||
                        (e = cudaMemcpy(stage, sp + off, nb, cudaMemcpyDeviceToHost)) != cudaSuccess ||
                        (e = cudaSetDevice(dev)) != cudaSuccess ||
                        (e = cudaMemcpy(dcache + off, stage, nb, cudaMemcpyHostToDevice)) != cudaSuccess) {
                        err = std::string("tensor parallel: the cache replica copy: ") + cudaGetErrorString(e) +
                              " at " + std::to_string(off >> 20) + " MiB";
                        cok = false;
                    }
                }
                (void) cudaSetDevice(dev);
                (void) cudaFreeHost(stage);
                if (!cok) return false;
            }
            vh.cache_base = (const uint8_t*) xcache.device_slot(0);
            vh.slot_off = xcache.slot_offsets();
            const size_t nres = (size_t) (g.n_layers * g.n_expert);
            (void) cudaGetLastError();
            {
                std::vector<int32_t> stage(nres);
                const cudaError_t rerr = [&]() -> cudaError_t {
                    if (cudaMalloc(&residency, nres * sizeof(int32_t)) != cudaSuccess) return cudaGetLastError();
                    if (cudaSetDevice(src_dev) != cudaSuccess) return cudaGetLastError();
                    const cudaError_t e1 = cudaMemcpy(stage.data(), src_cache->d_res, nres * sizeof(int32_t),
                                                      cudaMemcpyDeviceToHost);
                    if (e1 != cudaSuccess) return e1;
                    if (cudaSetDevice(dev) != cudaSuccess) return cudaGetLastError();
                    return cudaMemcpy(residency, stage.data(), nres * sizeof(int32_t), cudaMemcpyHostToDevice);
                }();
                if (rerr != cudaSuccess) {
                    err = std::string("tensor parallel: the rank's residency table failed: ") +
                          cudaGetErrorString(rerr);
                    return false;
                }
            }
            vh.d_res = residency;
            vh.h_res = src_cache->h_res;
            vh.slot_off = src_cache->slot_off;
            std::fprintf(stderr, "strata tp: rank %d: the expert cache is replicated (%lld slots, %llu MiB)\n", rank,
                         (long long) vh.n_slots, (unsigned long long) (cache_bytes >> 20));
        } else {
        // A NOMINAL EXPERT TIER.  `Verifier::init` refuses a tier whose three fields are empty (`verify.cpp:587`
        // checks `d_res`, `cache_base` and `blob > 0` - the CONTENT is not its business), and every entry here is
        // -1, so no expert is resident and the pool serves all of them.  That is what makes two ranks fit on one
        // card's worth of VRAM, and it is honest: this engine's expert tier belongs to the VRAM budget, not to
        // the tensor split.
        vh.n_slots = 1;
        vh.blob = 4 * 1024 * 1024;   // only ever used to stride a cache nothing is ever placed in
        (void) cudaGetLastError();
        {
            std::vector<int64_t> one_slot(1, (int64_t) vh.blob);
            if (!xcache.open_sized(one_slot, g.n_layers, g.n_expert, err)) {
                err = "tensor parallel: the rank's expert slot failed: " + err;
                return false;
            }
        }
        vh.cache_base = (const uint8_t*) xcache.device_slot(0);
        vh.slot_off = xcache.slot_offsets();
        const size_t n = (size_t) (g.n_layers * g.n_expert);
        std::vector<int32_t> none(n, -1);
        (void) cudaGetLastError();
        if (cudaMalloc(&residency, n * sizeof(int32_t)) != cudaSuccess ||
            cudaMemcpy(residency, none.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "tensor parallel: the rank's residency table failed";
            return false;
        }
        vh.d_res = residency;
        vh.h_res = nullptr;   // the host table is a fast path for the doorbell, which this run does not use
        }
        // THE RANK BEFORE ANYTHING IS SIZED, and then PUT BACK.  The pair is per THREAD (`shard.hpp`), and this
        // call runs on the thread that drives rank 0: leaving rank 1 set here would hand rank 0's own capture and
        // windows rank 1's head ranges.  Rank 1's thread sets it again for its own window (`generate.cpp`).
        const int save_rank = strata::core::tp_rank_current();
        const int save_ranks = strata::core::tp_ranks_current();
        if (!strata::core::tp_set_ranks(rank, ranks, err)) return false;
        const bool inited = ver.init(wt, g, ss, vh, head, max_t, err);
        std::string init_err = err;
        ver.set_stream(stream);
        ver.set_stage(0, g.n_layers, nullptr, nullptr);   // every layer, no hand-off anywhere
        std::string restore_err;
        (void) strata::core::tp_set_ranks(save_rank, save_ranks, restore_err);
        if (!inited) {
            err = "tensor parallel: rank " + std::to_string(rank) + "'s verifier: " + init_err;
            return false;
        }
        return true;
    }

    void free() {
        if (dev < 0) return;
        cudaSetDevice(dev);
        if (stream != nullptr) cudaStreamDestroy(stream);
        if (residency != nullptr) cudaFree(residency);
        xcache.close();
        if (session_buf != nullptr) cudaFree(session_buf);
        if (arena != nullptr) cudaFree(arena);
        stream = nullptr;
        residency = nullptr;
        session_buf = nullptr;
        arena = nullptr;
    }
};

}  // namespace strata::program
