# Prefill: model and large-chunk design

# DS41 prefill: TTFT model, ranked options and a large-chunk prefill design

I worked read-only and used no GPU. The repo is at HEAD `9c8dccb9`; it moved while I worked, so each line number below is paired with its symbol in case it drifts. The scratch scripts are in `/home/as/.claude/jobs/635603c1/tmp/prefill/` (`t1`–`t7*.py`, single-thread, `python3 -I`). Every number is marked as either measured or modelled.

## 0. Bottom line

- **Today's prefill is limited by the drive, and long prompts get no cheaper per token.**
  - A 16-token pass costs **114 ms plus 2.70 ms for each missed expert**, summed over the ranks. This was fitted on 40 requests (mean error 2.4%).
  - 2.70 ms per 18.8 MB expert is 6.96 GB/s: the NVMe and the mirror working together.
  - On the held-out prompts that is 670 ms a pass, 49 ms a prompt token, or 0.83 of a decode step. About 83% of each pass is miss reads.
  - Modelled for long prompts: about 34–50 ms a token. That is 35–52 s for 1,024 tokens and 4.7–6.9 min for 8,192.
- **The big lever is large chunks with the RAM cache read-only during each chunk.**
  - Each expert is then read at most once per chunk, and the cache does not thrash.
  - With the CPU computing every non-resident expert at 1,024-token chunks, TTFT improves ×3–4.6 for prompts of 1,024 tokens or more.
  - With the card taking the experts that have many rows, in VRAM lent by decode residents, at 2,048-token chunks: ×5–7.7.
  - With whole-prompt chunks: about ×20 at 8,192 tokens.
- **Larger chunks alone do not help.** With today's admit-everything LRU, passes of 64–256 tokens thrash the cache and gain little (section 5).
- **The cheapest exact route keeps every op on its decode-band kernel at every pass height.**
  - Only three ops switch arithmetic with pass height: the MoE GEMM above 16 rows, and sparse attention and the indexer above 48.
  - If the engine is batch-invariant with those three pinned, replies stay byte-identical to **today's base engine at 16-token chunks**.
  - The existing CPU kernels then stay exact, and no new CPU arithmetic is needed. Whether batch invariance really holds is the P0 gate.
- **Prefix caching is exact by design** (`--prefix-resume exact`, `core/sched/geometry.h:63-71`). It removes the prefill of any repeated prefix; its value depends on the workload.

## 1. Calibration (measured)

**FP8 window `ab-20261029T135754Z`, arm F8:**
- 312 prompt tokens (46/40/39/44/35/32/40/36) in 23 passes.
- TTFT sum 15.42 s: 0.67 s a pass, 49.4 ms a token.
- Decode 59.8 ms a step (`step_ns_decode` / 481).

**Per-pass law** (`t2_regress.py`), fitted on the 40-request trace window `ab-20261029T115523Z`, whose trace gives every pass's misses:
- `prefill_ms = 114.4·passes + 2.70·misses`, mean |error| 2.4%.
- With the cached experts ("hits") as a third term: 76 ms a pass, 2.67 ms a miss, 0.06 ms a hit. The CPU work on hits hides under the asynchronous miss reads.
- Applied to the F8 window, the law gives 4,741 misses: 206 a pass, 1.29 a pass per layer and rank.
- Cross-check against the counters: 9,360 misses in all, minus about 4,680 in decode (from 4,457 decode layers that missed), leaves about 4,680 for prefill. The law agrees.

**Shape of a pass:**
- In 16-token passes, 99.8% of (pass, layer) pairs have some rank reading the drive.
- 82% of layer-rank pairs miss at least once.

**Distinct non-resident experts touched, per layer and rank** (today's 712/720 residents; `t4_unions.py`):

| Tokens | 16 (one pass) | 26 (one prompt) | 53 | 105 | 210 | 319 | 631 |
|---|---|---|---|---|---|---|---|
| Experts touched (of ~78) | 7.05 | 9.95 | 15.2 | 22.6 | 32.7 | 39.0 | 52.1 |

- The 53–631 columns are prompts concatenated, so several topics; a single coherent document would touch fewer.
- Coherent decode text touches fewer still: 5.8 over 16 tokens, 14.2 over 64.
- Residents carry 54.6% of the profile's prefill selections, against 67.2% of its decode selections.
- The RAM cache holds 1,424 expert pairs a rank, 45.6% of the non-resident experts.

**Extrapolation model** (`t3`, `t6`):
- Independent draws from the profile's 614 prefill tokens, with a uniform floor α of 0.02–0.05.
- An effective-token factor of 0.79 at 16 tokens (fits 7.05), rising to 1 at 600 tokens and above (fits 52.1).
- Predicted touched experts: 1,024 tokens → 57–65; 2,048 → 64–73; 8,192 → 76–78 of 78.

**Other constants:**
- Drive in bulk: 2.45 ms an expert (7.7 GB/s; the ideal is 2.36).
- CPU pool: 0.20 ms for an expert's first row (94 GB/s). Each further row costs 0.074–0.092 ms today at 2–4 rows; 0.04–0.09 is assumed at 8–32 rows (not benchmarked).
- Card, tiled MXFP4 MoE: 12.7 + ~7 ms for 96 experts at 4 rows each, about 5.7 T MAC/s raw. Zero-copy reads from host memory: 19.4 GB/s. (Both from `ds41-kernel-perf-v2`.)
- Exact attention kernel: 367 µs at 48 rows (32 heads × 640 keys; `r4d_ds41_sparse.hip:1741`).
- Activation arena: 7.41 MiB at 16 tokens. I estimate 0.2–0.45 MiB a token at larger chunks; this is unmeasured.
- A VRAM resident is worth about 0.054% of decode: the file-buffer cap gave +176 residents and +9.5%.

## 2. TTFT against prompt length (modelled)

Times are in seconds. "Today" uses 1.0–1.6 misses per pass, layer and rank; the 1.29 measured on held-out prompts sits inside that range. The large-chunk columns use the read-only cache, α 0.02–0.05 and a CPU cost of 0.06 ms per further row.

| Prompt tokens | Today, 16-token passes | P0+P1: 1,024-token chunks, CPU only | + P2 drive pipelining | P3: 2,048-token chunks, card takes 24 experts a layer-rank | P4: whole prompt in one chunk |
|---|---|---|---|---|---|
| 64 | 2.2–3.2 (2.7 measured in-family) | 2.1–2.8 | same | same | same |
| 256 | 8.7–13 | 6.5–7.7 | 6.0–7.2 | 6.5–7.7 | same |
| 1,024 | 35–52 | 11–14 | 10–13 | 11–14 | same |
| 2,048 | 70–103 | 22–28 (17–18 with 2,048-token chunks) | 20–26 | 13–16.5 | same |
| 4,096 | 140–206 | 44–55 | 40–51 | 27–33 | 16–18 |
| 8,192 | 280–412 | 89–111 | 81–103 | 54–66 | about 18 |

What sets each column:
- **P0+P1 at 1,024-token chunks:**
  - Per chunk: about 4,050–5,080 misses, so 9.9–12.5 s of drive time.
  - The CPU needs 8.2–8.6 s for the same chunk, so the two are balanced.
  - If a further row really costs 0.09 ms, the CPU becomes the limit at 12.6 s.
- **P0+P1 at 2,048-token chunks, CPU only:** limited by the CPU at 15.6–16 s a chunk.
- **P3 at 2,048:** drive-bound at 12–15 s. The CPU needs only about 3 s, the card 2–5 s and the uploads 0.7 s.
- **P4 at 8,192:** about 16 s of drive.
- **Decode cost:**
  - Until P3 exists, the larger activation arena is permanent: about −0.6% to −1.3% decode at 1,024-token chunks and −1.2% to −2.6% at 2,048.
  - P3 lends that VRAM back to residents, at the cost of about 1 s to refill them after each long prompt.

## 3. Ranked list

| # | Item | TTFT gain | Decode | Effort | Exactness |
|---|---|---|---|---|---|
| 1 | **P0+P1: batch-invariant tall passes + CPU path up to 1,024 rows + read-only cache** (section 4) | ×3.1–4.6 at ≥1,024 tokens; ×1.3–2 at 256; 0–22% at ≤64 | −0.6% to −1.3% (arena) until P3 | ~7–9 days | Identical to today's base at 16-token chunks if the P0 gate passes |
| 2 | **Prefix caching** (`--no-prefix-cache` off, `--checkpoint-interval` around 512) | Removes a repeated prefix's whole prefill (35–50 ms a token today). Hits resume only at checkpoint positions (`core/mem/prefix.cpp:186-216`) | Small: KV is 13.8 KB a token a card; snapshots about 7 MB a card each in pinned RAM | 1–2 days + 1 window | Exact by `--prefix-resume exact`; check cold against hit |
| 3 | P2: next layer's likely misses queued behind the current layer's | −5% to −10% on top of #1 | 0 | 2–3 days | Exact (only changes when bytes are read) |
| 4 | **P3: card takes high-row experts, VRAM lent by residents, 2,048-token chunks** | ×1.3–1.65 on top of #1 at ≥2,048 tokens. Essential once the drive is faster (second NVMe, more RAM) | Restores #1's arena cost | ~9–12 days | Exact (same bytes, same kernel) |
| 5 | 32–64-token passes on the CPU path (= #1 run at `--max-num-batched-tokens 64`) | −5% to −20% on in-family content, about 0% on diverse content (`t7`) | 0 | Comes with #1 | As #1 |
| 6 | Read-ahead for 16-token passes (only if #1 is not built) | −5% to −8% (the drive idles about 17% of each pass) | 0 | 3–4 days | Exact |
| 7 | Mirror share sweep for prefill (0.12 / 0.15 / 0.18 / 0.21) | ±2%; 0.18 already matches the bandwidth ratio | 0 | 1 window, no code | Exact |
| 8 | Card-time items: gemv staging 2+ rows (bit-neutral), FP8 tiled row (commit 10), tiled BM heuristic | 1–3% while drive-bound | 0 | 0.5–1 day each | Exact |
| 9 | Whole-prompt chunks (P4) or layer-major execution | About ×3 over P3 at 8,192 tokens | P4: a large lend | Extra large | — |

Hardware for comparison:
- **Second NVMe:** today's prefill −38% (114 + 1.45 × 206 ms a pass). With #1 built, the CPU becomes the limit, so #4 is needed to gain further.
- **256 GB RAM:** misses largely vanish. Today's 16-token path rises to about 65–90 tok/s, held back by fixed costs and DRAM. #1 plus #4 would reach about 400+ tok/s.

## 4. Large-chunk prefill mode (LCP)

### 4.1 Exactness, and which reference to compare against

Two ops change their arithmetic with pass height:

- **MoE GEMM.**
  - `moe_gemm_ggml` uses the gemv row only up to 16 tokens (`r4d_ds41_mxfp4.inc:22-31` `cDs41MxMoeGemv`; launcher `r4d_ds41_mxfp4.hip:805` `if (gemv && tokens <= 16)`).
  - Above 16 it uses the tiled row: a sequential f32 chain per output (`:458-459`, `:579-594`).
  - The CPU kernels copy the gemv's lane order and butterfly, so they are exact only at 16 rows or fewer.
- **Sparse attention and the indexer.**
  - `attn_sparse_sink` and `index_score` leave their scalar, libref-exact forms above 48 rows (`kWmRows`, `r4d_ds41_sparse.hip:22-25`, `:1227`).
  - The tall-pass form is a WMMA kernel that is "not bit-exact".

Dense GEMMs are already independent of M: the split ignores MB, and the bf16 tiled row is byte-identical to m64 (`r4d_gemm_args.h:121-127`). The FP8 skinny row serves every M (`r4d_rows.cpp:4719-4722`).

**E2 (recommended):**
- One environment switch, `RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1`, read in libr4d's launchers the way `RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL` is.
- With it, the tiled MoE row launches the gemv kernel. The gemv already handles any number of rows: an expert's rows go in passes of XLM, each output keeps its lane and block order (`r4d_ds41_mxfp4.hip:347-348`, `:383-389`).
- The sparse attention and index launchers keep their scalar forms at every height. Those forms are documented as giving the same bits for any row count (`:1739-1742`).
- If the end-to-end gate holds, LCP's output equals today's base at 16-token chunks, and the existing CPU kernels stay exact at any number of rows.

**E1 (fallback if invariance fails):**
- A new CPU kernel that copies the tiled chain (`acc += x_k·w_k` for k ascending, rounded multiply then add, no FMA). It is easy to vectorise across outputs.
- The reference becomes the base engine at the same `--max-num-batched-tokens`. Replies then differ from today's 16-token replies.

**Card cost of E2 per 1,024-token chunk** (estimate; fine while the drive is the limit):
- Resident MoE about 0.7 s: gate_up re-reads the weights once per row (XLM=1).
- About 0.2–0.3 s wasted computing aliased rows for the CPU's experts.
- Exact attention 0.2–0.5 s; dense FP8 0.2–0.4 s.
- Total about 1.5–2 s, against 10–12 s of drive time.
- At 8,192-token chunks the gemv nears the drive time; E1's tiled row is about 3× faster there.

### 4.2 Phases, file by file

**P0: batch-invariant tall passes (gate)**

- `libr4d/r4d_ds41_mxfp4.hip:805` (`mx_moe`): when the switch is on, the condition becomes `(gemv && tokens <= 16) || exact_tall`. Log the first tall launch.
- `libr4d/r4d_ds41_sparse.hip` (the `kWmRows` switch at `:1227`, and the attention and index-score emit paths at `:1208`, `:1425`): when the switch is on, never take the WMMA forms.
- `tools/ds41-host-experts/ds41_mxfp4_device_check.hip`: new cases.
  - The gemv at M ∈ {17, 64, 256, 1024}, with ds41 shapes and skewed routing, byte-compared with the same rows run at 16 rows or fewer.
  - Attention and index score at 49–1,024 rows byte-compared with launches of 48 rows or fewer.

**P1: CPU path at tall passes**

- Row limits:
  - `core/place/ep_route_counts.h:9` `kEpRouteMaxRows` 16 → 1024. The witness in `Ctx::note_ep_route_producer` (`core/runtime/ctx.cpp:651`) follows the constant.
  - `core/place/stager.h:357` `kRoutedRows` 64 → `kEpRouteMaxRows`.
  - `stager.cpp:673` (`routed_`) and `before_op` (`:2467`, the `ep` test) follow.
- Pinned buffers grow with the constant: `kHxRows` (`stager.h:455`) and the `hx_x_` allocation (`stager.cpp:1103`). That is about 63 + 10.5 MB a rank.
- `FileStager::hx_init` (`stager.cpp:991`, the refusal near `:1010`): when `max_tok > 16`, also require `EXACT_TALL`. Check that every band of the routed gate_up and down ops resolved to the gemv; refuse with a message otherwise.
- Job size:
  - `core/place/host_expert_pool.h:43` `kMaxRows` 4 → 32. The slot arrays at `:122-123` grow to about 57 MB in all; `job_cost` (`:64`) moves to the benchmarked row cost.
  - `tools/ds41-host-experts/ds41_host_experts.h:106` `kGemvRows` 4 → 32. Each round is decoded once and rows run in register blocks of 4 with the same per-row operations.
  - `FileStager::hx_jobs_for` (`stager.cpp:1659`) cuts jobs of up to 32 rows.
- Read-only cache for passes of 256 tokens or more:
  - `Mover::borrow_host_unit` (`core/place/mover.cpp:873`) gains a `readonly` flag.
  - That needs a `borrow_packed` read-only variant using `slab_cache_.borrow_no_touch`, like `borrow_readonly` (`native_mapped_file_cache.h:289-308`).
  - Misses are not queued for insertion (`stager.cpp:1843`, `hx_bg_q_` push) except on the prompt's last chunk.
  - The last-chunk flag comes from `Engine::arena_step`'s `more` (`core/engine.cpp:1626`) through a new `FileStager::set_last_chunk`.
- Miss reads:
  - `Mover::read_unit_planes` (`mover.cpp:913`) gets per-unit completion, so `hx_submit_misses_now` (`stager.cpp:1786`) submits each unit as it lands.
  - Cap the planes held in flight: about 0.5 GB a rank at 1,024-token chunks.
- Scheduler: no change. `--max-num-batched-tokens C` sets the chunk, decode stays at 1 token, and `--busy-prefill-chunk auto` covers mixed steps.

**P2: keep the drive busy through tall passes**

- At layer L's decision, queue layer L+1's non-resident, uncached units behind L's misses, ordered by prior touch probability (the profile's prefill share and the chunk length).
- Reuse the whole-unit read-ahead (`hx_pf_target` / `hx_pf_loop`, and their take in `hx_choose` at `stager.cpp:1243`).
- Cap the queue at about 1.3 × the expected misses; drop unselected units at L+1.
- The piece reader (`readahead.h`) does not fit here: it yields to demand reads, and in tall passes those never stop.

**P3: the card takes high-row experts, in VRAM lent by residents**

- **Arena levels for deepseek41.**
  - Set `shape_probe_ok` and declare correctly at smaller step sizes; every buffer must derive from `g.max_tok` (`arch/deepseek41/deepseek41.cpp:890`, `:928-939`, `:2036-2045`).
  - That turns on `Engine::probe_arena_levels` (`core/engine_bringup.cpp:1527`) and the per-step reclaim in `Engine::arena_step` / `arena_tick` (`core/engine.cpp:1613`, `:1663`).
- **Lending under a static plan.**
  - `Mover::lend_from` returns early for static plans (`mover.cpp:1966`), and the arena loan requires the heat engine (`engine_bringup.cpp:4128`).
  - Change: the planner (`core/place/planner.cpp`, after the fill at `:793-795`) names lendable residents: the next units by decode share.
  - At startup they fill the lent slots. A reclaim drops them to the file tier (published as null; the mover's existing quarantine).
  - The refill reads from the container, or from the RAM cache through a packed→block conversion.
  - The FileStager and the CPU path treat a unit as resident only while it holds a slab slot (`start_ep_routed`, `stager.cpp:717`; `host_planes_eligible`).
- **Card share in tall passes.**
  - `hx_choose` gives the card the top-S cached units by rows (from `counts_host_`). `stage_file_units` (`mover.cpp:1758`) stages them into the reclaimed slots.
  - Packed hits are widened back to block form inside `read_native_file_impl` / `stage_relaid` (`:1544`, `:1653`) instead of being refused.
  - The rest stay as aliases computed by the CPU.
- **Cost at 2,048-token chunks:** arena 0.4–0.9 GB plus 50 slots of 18.8 MB, so 1.3–1.8 GB a card (about 70–98 residents a rank). Refilling them after the long prompt reads 5.3–7.4 GB, about 0.7–0.95 s.

### 4.3 FP8 dense weights above 64 rows

**Nothing is needed for correctness.**
- The skinny row `gemm_bf16f8_nt_m64` serves every M byte-identically (`r4d_rows.cpp:4719-4722`).
- An op must read a single stored form, so a bf16 band cannot sit beside it.

**Speed:**
- The skinny row re-reads the activation for each N tile (`r4d_gemm_bf16_nt_tiled.hip:6-8`). That costs about 0.2–0.4 s per 1,024-token chunk, 2–4% of a drive-bound chunk.
- Commit 10 of FP8-DENSE-DESIGN, `gemm_bf16f8_nt_tiled`, is already designed (4.5 h): it changes only the weight staging and is device-checked against `gemm_bf16_nt_tiled`. It is a P4 item.
- A bf16 fallback would need either bf16 copies (+2 GB a card, about −105 residents) or a per-layer FP8→bf16 expansion into scratch. Both are worse than commit 10.

### 4.4 Switches

All are off by default.

| Switch | Effect | A/B part |
|---|---|---|
| `RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL=1` | Decode-band MoE, attention and index kernels at every pass height | `Xt` |
| `--max-num-batched-tokens C` (existing) | Chunk size | `BT<C>` |
| `RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_MISSES=C` (existing; clamp follows the constant) | Misses on the CPU in passes up to C tokens | — |
| `RADIANCE_EXPERIMENTAL_DS41_TALL_READONLY=N` (default 256) | Passes of N+ tokens borrow read-only and insert nothing, except the last chunk | `Ro<N>` |
| `RADIANCE_EXPERIMENTAL_DS41_TALL_PIPELINE=1` | P2 | `Tp` |
| `RADIANCE_EXPERIMENTAL_DS41_TALL_CARD=S` | P3 | `Tc<S>` |
| `--max-model-len` (existing) | Long prompts | `ML<N>` |

`hx_init` refuses `max_tok > 16` without `EXACT_TALL`. `ab_window.sh:139-141` hard-codes `--max-model-len 512`, `--max-num-batched-tokens 16` and `--no-prefix-cache`; these need parameters.

### 4.5 Commit plan (effort)

**P0 (~3 days):**
1. libr4d MoE launcher switch (0.5 d).
2. libr4d sparse attention and index switch (0.5 d).
3. Device-check cases (1 d).
4. ab_window parameters and a long-prompt receipt builder (0.5 d).
5. Invariance window (0.5 d).

**P1 (~5–6 days):**
6. Row constants and the `hx_init` refusal (0.5 d).
7. 32-row jobs and `kGemvRows=32`, with host tests for 5–32 rows in every layout: packed, raw rounds, NaN rows, subnormal scales (1.5 d).
8. `hx_jobs_for` and buffers (0.5 d).
9. Read-only tall passes and last-chunk admission (1 d).
10. Streaming miss submission and the in-flight cap (1 d).
11. README and counters (0.5 d).

**P2 (2–3 days):**
12. Prior-ordered next-layer reads in tall passes.

**P3 (~9–12 days):**
13. deepseek41 smaller-step declares (2 d).
14. Static-plan lending with tier flips (4–5 d).
15. Card share with the packed→block conversion (2–3 d).
16. place_test, stager tests and the window (1–2 d).

**P4 (optional):** gemv staging 2+ rows (bit-neutral), commit 10, whole-prompt chunks.

### 4.6 Test plan

- **Host:** `gemv_avx2_rows` must equal `gemv_avx2` row by row at 1–32 rows in every layout. Pool tests with 32-row jobs.
- **Device (P0):** tall launches byte-equal to short ones, for MoE, attention and the index score.
- **P0 gate (engine):** base at 16-token chunks against base + `Xt` at 64 and 1,024. Prompts: 8 held-out, 16 chat, and 256 / 1,024 / 2,048-token prompts. Replies must be identical, and logits too if `engine_probe` can dump them. **If this fails, switch to E1.**
- **P1 identity:**
  - LCP against base at 16-token chunks on the same prompts.
  - Prompts longer than 1,024: compare LCP with and without read-only/pipelining, plus one verify arm with the cap removed.
  - Decode after a long prompt: identical replies at the same speed.
- **Fault, memory and stress:**
  - Mirror fault (`Z` part) during a tall pass: fallback works, replies identical.
  - MemAvailable stays at 8 GB or more during an 8,192-token prompt.
  - 8,192-token prompt three times on one server: drive temperature, no leak.

### 4.7 Risks

- Batch invariance may break in an op not yet audited, for example the scatter path above 170 tokens (exercised by the P0 gate). The fallback is E1.
- The CPU row cost at 8–32 rows is unbenchmarked. At 0.09 ms or more per row, 1,024-token chunks become CPU-bound (+13%).
- Until P3, the arena's VRAM is permanent. Measure it in W1.
- About 2.5 GB more host RAM is in use (planes held in flight, pinned buffers); about 14 GB is free today.
- A 1,024-token chunk takes about 11 s and stalls other sequences' decode; with `--max-num-seqs` > 1, cap the chunk or interleave.
- Read-only chunks leave the prompt's experts uncached for the decode that follows. A replay measured +17% decode misses when prefill experts are not admitted; admitting the last chunk mitigates it.
- P3 changes residents' tier at run time under a static plan. That is the largest correctness risk, in the mover's invariants.
- The drive runs at 100% for minutes; watch its temperature.

## 5. Smaller items, evaluated

- **32–64-token passes on the CPU path** (`t7_cpath.py`, keeping today's admit-everything LRU):
  - Long prompts: −5% to −20% at α 0.02, about 0 at α 0.05.
  - Short prompts: −3% to −10%, from fewer passes.
  - At 256 tokens a pass, the LRU thrashes (22–25 misses per pass, layer and rank) and gains nothing.
  - Exactness above 16 rows needs P0 or E1. Raising `kEpRouteMaxRows` / `kMaxRows` alone would give the wrong bits.
- **Read-ahead for 16-token passes:** the drive idles about 17% of each pass, so the ceiling is 17%. With roughly 50% precision, expect −5% to −8%. The piecewise reader is decode-only (`hx_choose`, `stager.cpp:1251`, `ra = ... pass_tok_ == 1`).
- **Mirror share for prefill:**
  - 0.18 matches 1.45 / (6.5 + 1.45). The 2.70 ms per miss is 88% of the bulk ideal; the gap is small batches and the mirror's 10 ms per-read latency.
  - A one-window sweep is worth ±2–3%. For bulk reads in LCP, keep 0.18.
- **Prefix caching:**
  - ds41's paged groups are prefix-cacheable. The window and linear/conv state ride on checkpoint snapshots (`prefix.cpp:186-216`). A hit therefore resumes only at a checkpoint on the chunk grid: every 2,048 tokens by default, so lower it for chat.
  - Exactness is by construction (`geometry.h:63-71`). Compare base and best both with the cache on, or check that a cold reply equals the hit reply.
  - It does nothing on the single-shot held-out set.

## 6. The first GPU measurement

**Before the window, on CPU only** (follow AGENT-LOAD-RULES):
- Microbenchmark `gemv_avx2_rows` at 4, 8, 16 and 32 rows, shaped like the pool at 40 threads.
- This sets the CPU cost per further row (0.04–0.09 ms assumed), which decides between 1,024 and 2,048-token chunks.

**Window W1, "the prefill curve".** It needs only small `ab_window.sh` parameters for `MAX_MODEL_LEN` and `MAX_BATCHED`, plus a receipt of distinct real-text prompts of 64, 256, 1,024, 2,048, 4,096 and 8,192 token ids built with `/tokenize`, with `AB_MAX_TOKENS=8`.

| Arm | What to run | What to read |
|---|---|---|
| A | Today's best arm with `--max-model-len 8448` and the access trace (`x`), lengths in ascending order; repeated in a second server lifetime | TTFT(P); misses per pass against pass index (does 1.29 hold or grow?); re-reads per 1,024 tokens (sum of misses ÷ distinct experts touched); NVMe busy; pool and async-miss waits |
| B | Base engine at 16-token chunks on 64 and 256 only | Reply identity; base engine's prefill rate |
| C | Base engine started at `--max-num-batched-tokens` 64, 1,024 and 2,048 | The "activation arena" startup line (VRAM per token, which sets the decode price of chunk size); one 1,024-token request at 1,024 for the base engine's large-chunk TTFT and reply |
| D (optional) | Base engine at 64-token chunks on the 8 held-out prompts | Whether greedy replies change when only the MoE row switches above 16 tokens |

- The window takes about 35–45 minutes.
- Keep the base engine away from long prompts: about 4.9 tok/s means 28 minutes for 8,192 tokens, and `ab_requests.py` times out at 600 s.
- **Decision gate:** build P0 and P1 if long prompts measure 30 ms a token or more and the trace shows each expert re-read twice or more per 1,024 tokens. The arena line then sets C: if its decode price is 1% or less, run 1,024-token chunks before P3.

## 7. Scratch scripts

All in `/home/as/.claude/jobs/635603c1/tmp/prefill/`:
- `t1_trace_stats.py`: per-pass statistics.
- `t2_regress.py`: the per-pass law.
- `t3_profile.py`: residents' share of selections and the distinct-expert model.
- `t4_unions.py`: measured distinct experts against token count.
- `t5_sim.py`: replay of long prompts against the LRU.
- `t6_model.py`: the TTFT table (arguments: α, CPU row cost).
- `t7_cpath.py`: CPU path at larger passes.
