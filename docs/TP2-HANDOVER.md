# Tensor parallel (TP=2) for Strata — work in progress, **not ready to merge**

**Based on:** Strata **0.1.44.1**, commit **`296b59d2`** (2026-10-14), CUDA source build, Windows.
Everything below was measured on that tree; a different base commit may not apply cleanly.

**Provenance:** produced with AI assistance in one long session that ran **over 400 assistant turns and
4 000 tool steps** (the session's own counters), **not human-reviewed**. The fast path described in 4 stayed
unsolved through thirteen separate fix attempts; the fourteenth — writing down every allocator, writer and
reader of the three KV pools in one table — found it. Treat it as a bug report with a patch
attached, not as a submission.

**Status: half-finished, but usable.** A dual-GPU tensor-parallel path exists and is *correct* for every
prompt length I measured. Its fast prompt path was broken when this document was first published and is
**fixed as of 2026-10-19** (commit `7f55fd3`, section 0): prefill now runs at 1 187–1 519 tok/s, about half
the layer split's, so the experiment is usable but still does not beat the split. Sections 4, 4a and 5 are
kept as the record of the search. Please do not ship this as-is.

If a thread for tensor parallelism already exists, merge this report into it instead of opening a new one.

---

## 0. RESOLVED (2026-10-19): the fast path's non-finite logits — root cause and fix

Commit `7f55fd3` ("fix(TP): the batched prompt path's K/V append wrote the model's page stride into the
rank's pools"). Everything in sections 4, 4a and 5 below is the record of the search; the search ended here.

**Root cause, in one paragraph.** The batched prompt path's only inconsistent component was its K/V writer.
The resident slots, the pinned host copy and the staging pool are all allocated with `qsa_shapes_rank`
(one K/V head per page at TP=2), and every reader — the attention kernels (`(page * n_kv_heads + kvh) *
page_size`), `kv_stream`'s block runs, the gather and the decode kernels — addresses that stride.
`kv_append_kernel` was still writing with the model's 2 heads per page and the rank's GLOBAL head offset:
every row landed at twice the offset the readers used, and for the rank's upper pages past the pool's end.
The attention then read never-written rows; an INT8 row that was never written has scale 0 and the dequant
divides by it — 0/0 — so the first QSA layer's attention went non-finite and #879 fired at the end of the
prompt. The verify-window path never calls this kernel, which is why it was always correct.

**Why thirteen attempts missed it.** Each changed one side of a pair and left the other. Writing `page*1`
while `head_base` was still global moved rank 1's rows into rank 0's next page — that is the measured
"100 % non-finite, max |x| 0" reversal in section 5, and it wrongly cleared the writer of suspicion. The fix
fell out of the ledger, not of another probe.

**Measured after the fix** (same machine, same config, `STRATA_TP_FASTPROMPT=1`, warm expert cache): prompts
of 2 803 / 5 533 / 11 176 tokens all read in full, no non-finite anywhere in the engine log; 1 187 and
1 519 tok/s steady state (the first large prompt pays an ~86 s graph capture); decode ~90 tok/s; `12*7`
answers `84`, and the summarize/reasoning answers are coherent and correct end to end. The fast path is now
the path I would use; the verify windows stay the default and the safe fallback.

**Correction the same day — equal-length comparison.** The "~1.5 K vs 2.1–2.9 K" gap quoted above compared
different prompt lengths. Running the SAME documents through both paths on this machine (warm expert cache,
`max_tokens=1`, client-side wall clock, 2026-10-19): ~11.7k tokens TP 1 510–1 520 vs PP 1 519–1 535;
24.8k TP 1 907 vs PP 1 904; 123.6k TP **2 994 / 3 027** vs PP **2 987 / 3 046** (two runs each); 380k TP
**2 993** vs PP **2 941** (after a warm ladder, see 0b). The fixed fast path MATCHES the layer split at
every length measured, and the author's 2900+ is the long-context end of the same rising curve (both paths:
~1 s fixed per-request cost amortizes and the expert cache warms across chunks) — a live observation of
3 248 tok/s at 380k+ on the split agrees. `nvidia-smi dmon -s t` during the 123.6k run: 4–7 GB/s mean per
card (Rx+Tx), ~22 GB/s peak bursts — PCIe is not saturated. The shared ceiling is the host-side expert
pipeline (grouping + streaming), which is why the two schemes converge; the stage-2 "halve the expert work"
analysis in 4a explains why TP cannot BEAT the split, not why it was slower — it is not. Decode: see 0a's
matrix (TP ≈ PP ≈ 1.55x single card at every valid setting).

## 0b. Cold long-context prefill: TP needs a warm ladder, the split does not (open)

fixtest11 measured TP at 380k with a two-run protocol (380k warmup, then a different 380k doc): BOTH runs
took 272/274 s — 1 385 tok/s, and the second run did not warm at all. The split under the same protocol:
270 s then **129 s** — it converges after one pass. fixtest12 ran TP through a 36k → 124k → 380k ladder
instead: 380k then took **126.9 s (2 993 tok/s)**, full parity. So the 1 385 was not a length cliff — it is
a WARM-UP asymmetry, and it is real and reproducible, not noise.

Leading hypothesis (not proven): the auto expert cache sizes the same 6 338 slots per card either way, but
TP spreads them over all 48 layers (**132 experts/layer**) while the split gives each stage 24 layers
(**264/layer**). A 380k-token document routes to nearly the whole 512-expert set per layer, so TP's cold
hit rate is roughly half the split's and it keeps streaming; a shorter ladder first pins the document
family's hot set. What was NOT tested: three back-to-back 380k runs on TP (does it converge eventually?),
cache sizes forced equal per layer, and whether a bigger `--expert-cache` cap closes the gap. If you pick
this up, that is the first experiment.

## 0a. One latent bug found while probing the ceiling

The N-5 defect is NOT fully dead - the default settings hide it, four switches expose it:

- `--pool-workers 5` (instead of 11): reproduced `non-finite logits (#879) at position N-5` three runs in a row.
- `--adapt-async 1`: reproduced it DETERMINISTICALLY (4 requests in a row, both ranks, positions 110-111 of a
  115-token prompt) - and the tier's own line says `0 rounds, 0 experts swapped in`, so the trigger is the
  asynchronous tier's PRESENCE (its exchange buffers / wait flags / extra allocations), not the swapping.
  The gate accepts `--adapt-async` under TP (`adaptive tier asynchronous ... arena: copy-in only`) - do not
  use it on a TP build until this is understood.
- `--spec 4` WITHOUT `--mtp` (suffix drafter): reproduced it 2/2 (78-token prompt, position 73, row 0; that
  boot's cache was 6 424 slots - no draft head). So the no-MTP decode path cannot be measured on TP at all
  until this is fixed.
- Once, with the DEFAULT settings, the very first post-boot request that ran a decode (112-token prompt,
  512 generated) hit #879 twice and returned 503; the next boot's first such request was clean, and a retry
  after the 503 always succeeded. So the exposure is timing/memory-layout sensitive, not worker-count
  specific.

Decode, short context (count 1→300, streamed, per-250-token buckets), spec 4 + MTP: TP **196.4** vs PP
**188.1** vs single card **123.3** - TP ≈ PP ≈ 1.55x one card (the ~220 max is MTP at full acceptance on a
counting sequence; the ~116 min is the thinking-to-answer transition). Without MTP (suffix drafter): PP
drops to **73.5**, TP cannot run (N-5 above). `--spec 0` is INVALID on this native IQ pack (the engine
requires `--spec T (T >= 2)` and falls back to a different mode - the spec-0 numbers measured that fallback,
not native decode). The "PP wastes half on bubbles" intuition is wrong in steady state: the split overlaps
consecutive windows (and, at spec 0, consecutive tokens) across stages, so it gets the same ~1.5x as TP.

At 180k context with 3000 generated tokens (pt=180 127, streamed, per-250-token buckets) TP median **94.1**
/ min 85.4 / max 100.9 vs PP **95.5** / 84.9 / 103.7 - a match, distributions overlapping; the author's
99.3 is the split at 36k - the same curve. One regime is NOT settled: on the short explanatory prompt, the
slope method measured TP **140-155** (two warmed samples) vs PP **191.8** (one sample, and its 64-token
endpoint was the first request after boot, which biases the PP figure UP). So short-context prose decode is
the one place where TP measured below the split, on thin evidence - re-measure both sides warmed before
believing either number. First-request decode after boot pays
~120 s of window graph capture, and the first large prompt pays ~90-140 s of prefill-graph capture - more
on the TP build than on the split, which is why first-request wall times differ (228.9 s vs 90.6 s at
180k+3000) while the warm steady state matches (fixtest7: 2 994 vs 2 987 at 123.6k). Not investigated
further (this session's budget); the first thing to try for 0a is `STRATA_DBG_GDN=1` under `--adapt-async 1`
to catch the first non-finite step.

**Still untested:** the Q4_0 and K8V4 batched appends (they still take the model's shapes), HIP, SYCL,
long multi-turn sessions.

---

## 1. What this is

Strata today runs multi-GPU by **splitting layers** (`--pipeline-windows 2`, `layer_split: auto`): each card
owns whole layers and never duplicates an expert. This work adds a second mode, `--tp 2`, where **only the
dense/attention weights are split** and the MoE is replicated on both ranks. The point of the experiment was
to find out what that buys on this hardware. The answer so far: correct, but not faster.

## 1a. Why this exists, given #156

Upstream **declined** this feature already: issue **#156 "[Feature Request] support tensor parallelism for
multi gpu setup"** was closed on 2026-09-30 as *"not planned for now: decode is limited by the experts, so
tensor parallelism of the dense part gains little here."* `docs/MULTI_GPU.md` at the base commit says the same
in one line: *"This is pipeline (layer) parallelism, **not tensor parallelism**"*, and the base commit
`296b59d2` contains no `--tp`, no `TpRuntime` and no `shard.hpp` at all — the whole path is new code, not a
modification of an existing one.

**This report is therefore not a feature proposal. It is the measurement that #156 never had.** I built the
thing to find out, and the numbers below **support the maintainer's reason for closing it**:

- end to end, the correct TP path is about **2 %** of the layer split's prefill throughput (37–52 tok/s vs
  2 114.8 tok/s on the same machine, same pack, same day);
- and the phase timing says why: **host grouping is 47.8 %** of the TP prefill — both ranks route their
  experts through the same CPU pool — with `gdn out proj` (the TP reduce) taking another 16.2 %.

So the interesting outcome is a negative result with evidence attached, plus the code that produced it.
If the maintainers still do not want it, the useful parts are the two real bugs in section 8.

## 2. Hardware and software (what every number below was measured on)

| | |
|---|---|
| CPU | AMD Ryzen 9 5900X, 12 cores / 24 threads, `--pool-workers 11` |
| RAM | 80 GiB |
| GPU 1, 2 | 2x NVIDIA GeForce RTX 3080, 20 480 MiB each, PCIe 4.0 x8 |
| GPU 0 | Quadro P1000 4 GB — vision encoder only (`vision.cuda_device: 2`) |
| OS | Windows 11 (10.0.26200) |
| Strata | engine **0.1.44.1**, base commit **296b59d2**, CUDA source build |
| Model | Qwen3.8-Flash-Next abliterated **IQ3_S** pack `sc117-iq3_s`, native experts |
| Settings | `--max-context 524288 --kv int8 --kv-resident 32768 --rope-scaling yarn --rope-scale 2 --expert-cache auto --expert-profile <profile> --vram-reserve-mib 64 --vram-reserve-later-mib 8 --spec 4 --prefill auto:16384 --vision` |
| TP env | `STRATA_SPLIT_MTP_BATCH=1`, `STRATA_GDN_KEYHEAD=0` (the latter is **required**, see 6) |

## 3. What works (measured)

`--tp 2`, prompt read through the two-rank verify windows (the default for TP in this patch):

| prompt tokens | read | prefill | decode |
|---|---|---|---|
| 108 | 108 of 108 | 43.6 tok/s | 37.4 tok/s |
| 3 458 | 3 458 of 3 458 | 37.7 tok/s | 45.9 tok/s |
| 6 856 | 6 856 of 6 856 | 52.1 tok/s | 58.4 tok/s |
| 14 060 | 14 060 of 14 060 | 41.3 tok/s | 43.4 tok/s |

- No non-finite logits at any of those lengths; `12*7` returns `84`; answers are produced end to end.
- VRAM per card: **19 909 / 19 867 MiB (19.44 / 19.40 GB)** — the same as the layer split on this pack
  (19 399 / 19 957 MiB), which was one of the goals.
- Vision, MTP draft layer, `--kv-resident`, `--spec` all work in this mode.

**For comparison, the existing layer split on the same machine and pack** (measured earlier in the same
session, same day): prompt **2 114.8 tok/s** at 36 329 tokens, decode **99.3 tok/s**, VRAM 19 399 / 19 957 MiB.
So the correct TP path is about **2 % of the layer split's prompt throughput**. That is the honest summary:
this patch does not advance speed.

## 4. What does not work

> **Resolved 2026-10-19 — see section 0.** Kept below as the record of the search.

The fast path (`STRATA_TP_FASTPROMPT=1`, one batched prefill instead of 6-token verify windows) reaches
**860–990 tok/s** on the same prompts but produces non-finite logits:

```
strata serve: non-finite logits (#879) in a prompt window at position 6851 (row 0)
strata serve: prompt 6856 tokens = 0 reused + 6851 of 6856 read in 7 972 ms (859.4 tok/s)
```

It is correct for prompts that fit in a single KV page and breaks beyond that. Because of this the patch
leaves the **verify-window path as the TP default** and makes the batched path opt-in:

```cpp
if (o.tp > 0 && std::getenv("STRATA_TP_FASTPROMPT") == nullptr) return true;   // windows
```

## 4a. Two stages: a usable 990 first, then the layer split's 2900+

**Nothing here is a speed optimisation: 1000 tok/s already happens (860-990 measured, 1012 peak). The work is
making it correct.**

**Where the fast path differs.** The default reads the prompt in 6-token verify windows; the fast path runs one
batched prefill. The only real difference is *how the KV is written*: the batched path writes N cells at once
(`kv_append`), the window path writes 5-6 cells per step (`kv_append_q8_step`), while the decode and the window
have **one** read algorithm. Two writers with different page rows plus one reader cannot all agree.

**Why it fails now.** After the batched append writes N-5 tokens, the first read (the short-read window or the
first decode token) does not get those cells: `qsa attn 583872 of 635904 non-finite (max |x| 2.288e+06)`, while
the `qsa q(in)` probe is clean — **the queries are fine, the KV read is not.**

**Repair order, cheapest first.**
1. **Compare the two numbers.** Print `row_id` for the first 8 cells inside `kv_append_kernel`, and `r` for the
   same cells where `srow` is filled in `src/kernels/cuda/qsa_prompt_attn.cu`. If they differ, that is the defect.
2. **Check `ident_table`.** `prefill.cpp:1204` builds it; when `staged=1` the reader is
   `pools_of(m.stage, m.ident_table)`. Confirm `ident[i] == i` rather than some permutation.
3. **Remember `page_size == 4`.** Block count is cells/4; anything derived from 64 has to be redone — a whole
   family of my earlier deductions died on this.
4. If step 1 shows the numbers *agree* yet the output is still bad, the defect is in how the stage pool is
   **filled** (`kv_stage_from_host` and the `stage` argument of `kv_append`), not in the row arithmetic. I
   changed that path but never proved it.

**Two things that do not work (do not repeat them).**
- Writing `page*1` instead of `page*2` alone: measured 92% non-finite -> **100%** non-finite (`max |x| 0`). The
  reversal says the reader is not addressing what I assumed; the write side is not the lever.
- Changing only the reader's `QsaShapes`: no behavioural change (the counts and offsets were already right).

### Stage 2: from ~990 to the layer split's 2900+ (990 is only 34 % of it)

**Calibration.** The author measures the original layer split (PP) reaching **2900+ tok/s** on prefill; this
report measured **2114.8 tok/s** on the same machine and pack at 36 329 prompt tokens (different length,
different reading — both are recorded here). One card on the same 3 458-token prompt: **397 tok/s**. So:

- **~990 tok/s is one card's worth**, the second card buys almost nothing on prefill;
- against 2900, 990 is **34 %**; against this report's 2114.8 it is **47 %**.

**Why.** The MoE is replicated: both ranks compute the *same* experts and feed the *same* 11-thread CPU pool.
Phase timing (6 856 tokens, 7 161 ms GPU):

| phase | share | duplicated? |
|---|---|---|
| host grouping | **47.8 %** | yes — identical on both ranks |
| gdn out proj (TP reduce) | **16.2 %** | no — TP-only cost |
| qsa proj | 7.4 % | split |
| gemm gate/up + down | 6.9 % | yes |
| gdn (conv + recurrence) | 5.4 % | split |
| combine | 4.4 % | yes |
| qsa attn | 3.3 % | split |
| hc read | 4.7 % | split |
| router / gather / dequant / ple / kv | ~4 % | mixed |

**About 60 % of the prefill is duplicated work**, plus 16.2 % for the reduce.

**The arithmetic, stated honestly.** Even a perfect expert-id split that halves the duplicated 60 % only brings
the time to about 0.70x, i.e. **990 -> ~1400 tok/s (1.4x)**. Reaching **2900 needs 2.9x**, which requires the
duplicated work to go to nearly zero *and* the 16.2 % reduce to be absorbed — **not achievable while the MoE is
replicated.**

**So stage 2 is a design decision, not the tail of this patch:**

1. **Let rank 1 reuse rank 0's grouping** (low risk, biggest single win). Both ranks route to the identical
   expert set; grouping twice is pure waste. Not measured.
2. **Merge the two reduces of a layer** (low risk) — 16.2 % together; batch them or overlap with the next layer.
3. **Split experts by id with all-to-all routing** (high risk; the only thing that halves the duplicated 60 %).
   Lands around **1400-1800**.
4. **To reach 2900-class numbers** each card's per-token expert work must also halve — i.e. the expert set is
   genuinely cut in two and a token is only computed where its experts live. That is a third parallel strategy
   alongside the layer split, **not a patch to TP**.

**In one line:** fixing the KV takes you from 0 to 990; routes 1-3 reach ~1400-1800; **2900 needs route 4,
which is a redesign, not a bug fix.**

**What it would give you.** The batched path measured **860-990 tok/s**, i.e. **41-47%** of the layer split's
2114.8. Beyond that you must attack expert scheduling — **host grouping is 47.8%** of the TP prefill because
both ranks route the same experts through the same 11-thread CPU pool — not the KV.

## 5. The evidence chain (all of it measured)

1. **Single-card control, same binary, same prompt.** 3 458 tokens on one card: 3 458 of 3 458 read,
   40 tokens generated, no error. Dual rank: 3 453 of 3 458, then the trailing window fails. One card has no
   "batched write -> window read" hand-off at all; TP does.
2. **Forcing the windows on TP** (`STRATA_TP_WINDOWS=1`) gives 3 458 of 3 458 — everything except the batched
   prompt path is sound (reduce, GDN, MoE, VRAM).
3. **`STRATA_DBG_NAN=1`** (existing) reports the MoE of layer 1/2 all non-finite (`GU` 3453x12800 NaNs). The
   MoE is downstream, so that is a symptom.
4. **`STRATA_DBG_GDN=1`** (added by this patch) gives the real cross-section:

   ```
   layer 0/1/2  GDN block        clean
   layer 3      qsa q(in)        clean            <-- the queries are fine
   layer 3      qsa attn         583872 of 635904 non-finite (max |x| 2.288e+06)
   layer 3      qsa bo(proj)     527360 of 529920 non-finite
   layer 3      qsa bo(reduce)   527360 of 529920 non-finite
   layer 4      GDN block        fully non-finite (downstream)
   ```

   GDN's own counts and offsets are correct (`ghv=24 gch=5120 ghk=8 tpr=2`, `state_off` an exact multiple of
   408 576), so the GDN state slices do not overlap.
5. **Measured plumbing, printed from the running kernel path**:
   ```
   layer 3  staged=1  kv_mode=1  page_size=4  s.n_head_kv=2
   ```
   Two of those matter and cost me most of the debugging time:
   - **`page_size` is 4, not 64.** A whole family of my earlier deductions assumed 64 and was wrong.
   - **`staged=1`**: the prompt attention reads the *staged* pool with `m.ident_table`, not the session pool.

**Conclusion I am confident in**: the queries are fine, the failure is in how the QSA layer reads KV cells,
and it is specific to the batched append + staged pool combination.

### What I already tried and reverted (so nobody repeats it)

Changing the page stride in either direction, alone or together with the reader, does not fix it: writer
`page*2` / reader `page*1` gives 92 % non-finite (`max |x| 2.29e6`), writer `page*1` / reader `page*1` gives
**100 %** non-finite (`max |x| 0`). That reversal is the strongest clue I have: it means the reader is not
addressing the rows I thought, so the write side is not the lever.

## 6. The one thing that must not be forgotten

`STRATA_GDN_KEYHEAD=0` is **required** in TP. `gdn_rec_kh_kernel` was not parameterised by this patch and
still uses the model's full `HV=48 / HK=16` while a rank owns 24 / 8. Without that env the GDN writes out of
its own state slice. That kernel needs the same treatment as the ones below before this can be default.

## 7. Files this patch touches, and why

| file | what changed |
|---|---|
| `include/strata/prefill/kernels.hpp` | `gdn_gates` / `gdn_conv` / `gdn_recurrence` / `split_q` / `gate_attn` / `kv_append` take runtime head/channel counts, defaulting to the model's |
| `src/prefill/kernels.cu` | the same kernels use them. **`kv_append` still writes the model's 2-head page rows (`page*2`) on purpose** - the rank-local row (`page * heads + kvh`) is one of the things that was tried and made the first QSA layer's attention 100 % non-finite, so it was reverted. `CONV_TILE` unchanged. |
| `src/prefill/prefill.cpp` | GDN block passes the rank's counts; QSA attention uses rank-local `QsaShapes` and a `qh*256` query stride; `ssm_out` and `attn_output` each get one TP reduce; KV staging legs take rank-local shapes; `STRATA_DBG_GDN` probes |
| `src/program/generate.cpp` | TP prompt path selection (windows by default, batched opt-in); expert-cache retry fills to `device_free_bytes()` instead of a fixed 3/4 — **this is the only reason VRAM went 18.4 -> 19.44 GB** |
| `include/strata/core/layout.hpp` | `qsa_shapes` was tried with a rank-count parameter and is **reverted**: the file as handed over has the original one-argument signature, so it may not need to change at all. |
| `src/core/verify.cpp` / `src/core/layer.cpp` / `src/core/session.cpp` | rank-local GDN/QSA shapes on the verify side |

**Warning for the reviewer:** the base commit `296b59d2` contains no tensor-parallel code at all, so this
branch is a **snapshot of an uncommitted working tree**, not a reviewed series of commits. It carries the whole
TP effort — 19 modified files plus 20 new ones (`include/strata/core/{shard,native_shard,tp_runtime,tp_host_runtime}.hpp`,
`include/strata/kernels/allreduce.hpp`, `include/strata/program/{tp_pair,tp_rules}.hpp`,
`src/kernels/cuda/allreduce.cu`, and ten `tests/core/tp_*` / `tests/program/tp_rules_test.cpp`) — of which this
document only describes the parts I touched. Scratch artifacts were left out (crash dumps, `.bak` config
copies, local expert profiles), so the branch is not guaranteed to build standalone. Do not review the whole
diff as if it were this work.

## 8. Bugs found and fixed on the way (worth keeping even if the rest is dropped)

1. **QSA query stride.** `split_q_kernel` packs the rank's query heads tightly (`q[i] = qf[t*H*512 + h*512 + d]`,
   `i = t*H*256 + h*256 + d`), and `rope` / `rms_rows` / `gate_attn` all address that buffer with `qh*256`.
   The two attention call sites still used the model's `n_head` (24 -> 6144). Fixed with a rank-local
   `QsaShapes` and `qstride = sq.n_head * head_dim`. This moved the GDN probe's non-finite count by exactly
   one token's worth of channels, so it was a real defect, just not the last one.
2. **Expert-cache sizing.** The retry shrank by a fixed 3/4 and read a pre-context `g_tp_free_bytes`, so with
   13.29 GiB actually free it allocated 11.46 GiB — 1.83 GiB thrown away. It now fills to
   `device_free_bytes()`, which is what puts the card at 19.44 GB.
3. **GDN v-band split** (in `shard.hpp`, already in the base commit at the time of writing): a contiguous half
   split breaks the `v % 16` v-head/q-head pairing and changes the generated text.

## 9. What I did **not** test

- **Only CUDA was built.** `AGENTS.md` asks for CUDA + HIP + SYCL on any file a change touches. The kernels
  here are in shared `.cu` files, so the HIP and SYCL builds are unverified.
- **No other GPU or driver.** One machine, one pack, one model.
- **No 50K–100K prompt measurement.** The correct path runs at about 40 tok/s, so a 50K prompt is ~20 minutes
  per run; I measured up to 14 060 tokens.
- **No quality evaluation.** I checked `12*7 -> 84`, that the token counts come out whole, and that no row is
  non-finite. I did **not** run a recall/needle suite or a perplexity comparison against the layer split.
- **No bitwise single-card regression against HEAD.** Every new parameter defaults to the model's own
  geometry and `tp_ranks_current()` is 1 on one card, so the intent is a no-op on one card — but I did not
  diff single-card output before and after.
- **The batch/spec path harder than `--spec 4`**, and no long-running soak.

## 10. Prior art: llama.cpp and ggml

This patch's tensor-parallel rules were written against llama.cpp's tensor mode, which Strata already
carries as a pinned dependency and cites inline. Nothing else is claimed.

- **Licence: MIT**, verbatim — `third_party/llama.cpp/LICENSE` reads
  `MIT License / Copyright (c) 2023-2026 The ggml authors`, with the standard permission notice, the
  "shall be included in all copies or substantial portions" condition and the `AS IS` disclaimer. The same
  text is in `third_party/ggml/LICENSE`. Neither is a modified or dual licence; there is no added
  non-commercial, copyleft or patent clause.
- **Not redistributed:** `.gitignore:31` lists `/third_party/llama.cpp`, and `CMakeLists.txt` exposes
  `STRATA_ORACLE_SOURCE_DIR` for the parity tests that run against a pinned copy. The pinned tree is
  downloaded, not committed, so a clone of this repository does not contain llama.cpp at all.
- **What was used:** the tensor-mode rules, cited in Strata's own comment style with file and line —
  `shard.hpp` cites llama.cpp's `handle_mul_mat:600-603` (splitting the expert axis with a replicated
  activation), `GGML_BACKEND_SPLIT_AXIS_PARTIAL:591-593` (which axes must be added across cards),
  llama.cpp's rule for `MUL_MAT_ID`, and the qwen4exp shapes at `qwen4exp.cpp:242-247`.
- **Obligation, if anything here is an adaptation rather than a reimplementation:** MIT requires the
  copyright and permission notice to travel with substantial portions. The repository's `LICENSE` already
  satisfies Strata's own MIT terms, and the inline `file:line` citations identify the origin. If review
  finds a function that was copied rather than rewritten, keep the ggml notice beside it.

## 11. Licence and attribution

- Strata is **MIT**, Copyright (c) 2026 Niko1221 and the Strata contributors. Any redistribution of this
  modified source must keep the `LICENSE` file and its copyright + permission notice intact. Do not
  relicense it.
- The MIT text makes the software **"AS IS", without warranty of any kind**. Nothing in this report is a
  fitness-for-purpose claim: the fast path is broken and the correct path is slower than the existing
  layer split.
- Model weights, the native expert pack, and the expert profile are **separate artifacts, not part of
  Strata's MIT-licensed source**. Their terms are their own; I am not asserting them here, and they should
  not be committed into the repository.
- No credentials are included. The local config's `api_key` has been removed from this report; if you paste
  a log or a config, strip it too.
- Per `AGENTS.md`: this is one change, it is opt-in (`--tp`), and the default path is meant to stay
  byte-identical. The measurements above say what they were measured on.

## 12. The three steps I would try next

1. Print, from inside `kv_append_kernel`, the `row_id` it writes to the stage for the first 8 cells, and
   print from `prompt_attn_kernel` the `r` it computes for the same cells
   (`src/kernels/cuda/qsa_prompt_attn.cu`, the `srow` fill). If those two disagree, that is the defect.
2. Dump `m.ident_table` (`prefill.cpp:1204` builds it) and confirm `ident[i] == i` rather than some other
   permutation.
3. Remember `page_size == 4`, so block counts are `cells / 4`; re-derive anything that assumed 64.

## 13. Reproducing

```
cmake --build . --config Release --target strata
strata.exe --serve --tp 2 --pack <pack> --native <gguf> --ple-gguf <gguf> \
  --expert-profile <profile> --expert-cache auto --max-context 524288 --kv int8 \
  --kv-resident 32768 --prefill auto:16384 --pool-workers 11 --vision
# required in TP:
STRATA_GDN_KEYHEAD=0
# diagnostics:
STRATA_DBG_GDN=1        # per-step finiteness inside the GDN and QSA blocks (added by this patch)
STRATA_DBG_NAN=1        # existing; MoE only
STRATA_TP_WINDOWS=1     # force the verify-window prompt path (the correct one)
STRATA_TP_FASTPROMPT=1  # the batched fast path (currently broken past one page)
```
