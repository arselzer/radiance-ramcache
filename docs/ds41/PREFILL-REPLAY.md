# Prefill trace replay: consumer skip and large-chunk prefill

# DS41 prefill: trace replay of the consumer-suffix skip and large-chunk prefill

This is PREFILL-CRITIQUE item 0(b) and the survey's replay of idea #1, run offline on the long-prompt trace. I worked read-only and used no GPU. Every step is a single-thread Python script in `/home/as/.claude/jobs/635603c1/tmp/prefill-replay/`; together they run in under 3 minutes. The harness does not let this subagent write a report file, so REPORT.md does not exist and the report is returned here; the scripts and their JSON outputs are in that directory.

**Data:** window `ab-20261029T164814Z`, arm `all-t40-f-M16-H3-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-x-Lm8192-Ulong`.
- Five documents of 82, 272, 933, 1,890 and 3,086 tokens, served in order in one process, each followed by 16 output tokens.
- Inputs: the trace, the server log, the results JSON, and the sealed decode profile the engine planned from.
- The window's `Ulong8k` arm re-ran the 272-token prompt in a fresh process. All 2,720 of its prefill records carry the same unit sets as the long arm's run of that prompt, so I use it as a sixth timing point.

**Labels:** "measured" means read from those files; "replayed" means my cache model run over the trace; "modelled" means a fitted law or the critique's cost constants applied to replayed counts.

## 0. Bottom line

- **The replay is exact enough to trust.**
  - The cache model reproduces the engine's drive reads in 74,968 of 75,040 layer-pass records, and every prompt's prefill and decode totals within 0.7%.
  - The model is:
    - an LRU of 1,424 units per rank;
    - warmed at startup with the profile's 1,424 hottest units;
    - 2 second chances for profile-hot units;
    - six units (over the four ranks) that the packed cache never admits.
  - **The second chances matter.** Without them, one prompt's reads are off by 25%.
  - `cache_sim.py`'s plain LRU without warm-up matches the window total (+0.2%) only by coincidence. Per prompt it is off by up to +91%.
- **Today's per-pass time has a flat part and a read-driven part.**
  - The requested linear law (a + b·hits + c·reads a pass) either misses the 82-token prompt by 19% or needs a negative cost per hit.
  - The law that fits is per layer-pass (one layer in one pass): t = −1.69 + max(3.20·M, 0.43·U) ms. M is the drive reads and U the selected file-tier units, both summed over the 4 ranks.
  - It fits all six TTFTs within 2.4% (leave-one-out mean 1.6%).
  - Up to about 4 reads per layer-pass, a layer-pass costs a flat 11.9 ms. Above that, each read adds 3.2 ms.
  - The 272-token prompt ran twice: once after the 82-token prompt, once in a fresh process. The second run cost 2.31 ms more per hit that became a read: 1.79 s for 776 reads. The law predicts 1.71 s of it.
- **The consumer-suffix skip passes the survey's gate (≥15% at about 3K tokens) by a wide margin.**
  - I re-derived its pass-granular schedule from the graph code. I checked it against row-level dependencies for every prompt length from 1 to 16,384 tokens: no failures.
  - **Replayed effect:**

    | Prompt (tokens) | MoE layer-passes removed | Drive reads | TTFT |
    |---|---|---|---|
    | 3,086 | 30.1% | −56% | 99.2 → 70.9 s (−28.6%) |
    | 1,890 | 19.5% | −43% | −24.8% |
    | 933 | 10.2% | −20% | −10.4% |
    | 272 and 82 | 3.5% and 2.1% | −4% and −3% | about −3% |

  - 40–46% of the removed reads come from a cache effect: with fewer layers competing, the layers that still run hit more often.
  - At 7.5K–8K tokens: −40% to −46%, and −55% for a document as read-heavy as the 1,890-token one.
  - Across the L2-type laws (constant free or held at 0–6 ms) and L3, the 3,086-token result runs from −28.3% to −34.5%. The poorly fitting linear law L1 gives −40%.
  - Decode reads in the 15 tokens after each prompt rise 6%.
- **Large-chunk prefill (LCP) is CPU-bound** at C ≥ 1,024 for every prompt from 933 tokens on, at any CPU cost per row from 0.06 to 0.10 ms.
  - **Why:** the CPU pool computes 170.7 non-resident rows per prompt token. The critique counted 158 because it left out the missed experts' rows.
  - **Gain** (central, with the recommended cache policy):

    | Prompt (tokens) | Speed-up | TTFT |
    |---|---|---|
    | 933–3,086 | ×1.9–2.0 | 3,086: 52.7 s (40.9–64.6) |
    | about 7.5K | ×2.0 | 130 s, against 260 s modelled for today |

  - **C = 2,048 buys about 2% over 1,024** (6% with the skip), because the CPU, not the drive, sets the time.
  - **Short prompts** (82 and 272 tokens) run as one chunk and stay drive-bound: ×1.1–1.6.
- **Cache policy inside tall chunks.**
  - **Admit-all thrashes.** At C = 512 it reads 10% more than today's 16-token passes.
  - **Read-only** halves the reads (−48% at C = 1,024), but decode after the prompt reads 24% more. After a one-chunk short prompt it reads 84% more.
  - **Bounded admission** (`ROB128`) admits only the units the prompt's last 128 tokens touch.
    - It reads about as much as read-only.
    - Decode after the prompt reads 12% more than today, against +23% to +25% for read-only, admit-all and final-chunk admit-all.
  - While LCP is CPU-bound, the policy barely moves long-prompt TTFT (≤1%). It decides decode after the prompt and the speed of short prompts.
- **LCP plus the skip:**
  - ×2.6–2.7 at 1,890–3,086 tokens (3,086 tokens: 37.6 s).
  - ×3.2–3.4 at 7.5K–8K tokens (about 80 s against 260–286 s).
- **Order this suggests:**
  1. Build the skip first: exact, no tall-pass work, −29% at 3K tokens.
  2. Benchmark the 32-row CPU kernel (critique 0a). At 3,086 tokens, LCP is ×2.4 at 0.06 ms a row but ×1.5 at 0.10.
  3. Build LCP at C = 1,024 with bounded admission.
  4. Then move rows off the CPU (the card-share item). LCP leaves the drive about half used: at 3,086 tokens, 24 s of drive time against 45 s of CPU.

## 1. Summary per prompt length

TTFT in seconds.
- **Today:** measured for the five documents, modelled for the two synthetic prompts.
- **Skip, 16-token passes:** the replay plus law L2, scaled to each prompt's measured TTFT.
- **LCP columns:** the critique's cost constants on replayed counts, with bounded admission (`ROB128`).
  - Central value: 0.08 ms per extra CPU row and 0.9 s of card time per 1,024 tokens.
  - Range: from 0.06 ms with 0.6 s to 0.10 ms with 1.2 s.
- **7.5K and 8K rows:** synthetic prompts (§4.3, §5.4).

| Prompt tokens | Today (s) | Skip, 16-token passes | LCP C=1,024 | LCP C=2,048 | LCP + skip C=1,024 | LCP + skip C=2,048 | LCP bound |
|---|---|---|---|---|---|---|---|
| 82 | 5.8 measured | 5.7 (-3%) | 5.1 (5.1–5.1), ×1.14 | 5.1 (5.1–5.1), ×1.14 | 5.0 (5.0–5.0), ×1.17 | 5.0 (5.0–5.0), ×1.17 | drive |
| 272 | 11.0 measured | 10.7 (-3%) | 8.8 (8.8–8.9), ×1.25 | 8.8 (8.8–8.9), ×1.25 | 8.6 (8.5–8.7), ×1.28 | 8.6 (8.5–8.7), ×1.28 | drive |
| 933 | 30.9 measured | 27.7 (-10%) | 15.8 (12.3–19.4), ×1.95 | 15.8 (12.3–19.4), ×1.95 | 14.3 (11.1–17.5), ×2.16 | 14.3 (11.1–17.5), ×2.16 | CPU |
| 1,890 | 68.1 measured | 51.2 (-25%) | 33.3 (25.9–40.8), ×2.04 | 33.1 (25.6–40.5), ×2.06 | 26.5 (20.7–32.4), ×2.56 | 26.2 (20.3–32.1), ×2.60 | CPU |
| 3,086 | 99.2 measured | 70.9 (-29%) | 52.7 (40.9–64.6), ×1.88 | 52.3 (40.5–64.2), ×1.90 | 37.6 (29.3–46.0), ×2.64 | 37.1 (28.8–45.5), ×2.67 | CPU |
| ≈7.5K (7,511, synthetic) | 260 modelled (241–268) | 143 (-45%; -40% to -54%) | 130 (103–159), ×2.00 | 129 (99–158), ×2.02 | 81 (65–96), ×3.22 | 76 (59–93), ×3.41 | CPU |
| ≈8K (8,201, synthetic) | 286 modelled (263–293) | 155 (-46%; -40% to -55%) | 142 (112–174), ×2.02 | 141 (109–173), ×2.03 | 84 (66–102), ×3.40 | 84 (66–103), ×3.39 | CPU |

## 2. The trace and the replay (task 1)

**The trace** (format: `stager.cpp` `trace_layer`):
- 18,760 records per rank: 469 passes × 40 routed layers.
- Each rank's drive reads in the trace (drive-read weights ÷ 2) equal its shutdown line "misses on the CPU": 12,869, 12,875, 13,251 and 13,255 units.

**Split into passes:**
- Prompts in order: 5×16+2, 17×16, 58×16+5, 118×16+2 and 192×16+14 tokens, so 6, 17, 59, 119 and 193 prefill passes.
- Each prompt is followed by 15 one-token decode passes. That makes 394 prefill steps and 75 decode steps, as the server's `steps_prefill` / `steps_decode` counters say.

**The engine's cache, reproduced** (`replay_lib.py`):
- **Plan.**
  - Trace unit ids are u = 11 + 97·layer + e, and rank r's local expert e is global expert 96r + e.
  - Each rank's VRAM / pinned / SSD split is the planner's ranking: profile share descending, then expert, then layer. It is filled greedily with 710/719/719/719 VRAM units and 3 pinned.
  - This reproduces the logged "decode calibration mass" of all 4 ranks to 6 decimals, and rank 0's tier list exactly.
  - No traced unit is resident or pinned.
- **Capacity.** 1,424 units per rank. The shutdown lines show 1,424 slots each in the gate_up and down classes, and a unit borrows and inserts both.
- **Policy** (`mapped_expert_slab_cache.h`):
  - A hit moves the unit to the front and resets its chances.
  - A miss is inserted at the front.
  - The victim is the LRU end. The exception: a protected entry there with fewer than 2 chances moves to the front instead (`victim_for`).
- **Protected units.** SSD units with share ≥ 6/4,326: 1,051, 1,045, 1,075 and 1,017, exactly the logged counts.
- **Warm-up** (`hx_warm`). The 1,424 hottest SSD units, inserted coolest first.
- **Never-cached units.** The packed cache refuses over-budget units (`insert_unit_packed`), but the log gives only their counts, not their ids. I inferred them from the replay residuals:
  - Rank 0: (39,7) and (39,87); rank 1: (39,92) and (37,27); rank 2: (38,58); rank 3: (39,94).
  - Their accesses plus their warm-up attempts equal the logged refusals: 110+65+2 = 177, 25+34+1 = 60, 76+1 = 77 and 70+1 = 71.

| Prompt | Phase | Unit accesses | Engine's drive reads | Exact model | Diff | Without second chances | Plain LRU, no warm-up (cache_sim.py) |
|---|---|---|---|---|---|---|---|
| 82 | prefill | 6,330 | 1,913 | 1,910 | -0.2% | -1.9% | +91.0% |
| 82 | decode | 2,029 | 174 | 175 | +0.6% | -23.6% | +51.1% |
| 272 | prefill | 20,766 | 3,291 | 3,295 | +0.1% | -24.6% | -15.6% |
| 272 | decode | 1,908 | 146 | 145 | -0.7% | +11.0% | +9.6% |
| 933 | prefill | 73,058 | 7,218 | 7,219 | +0.0% | -3.0% | -3.1% |
| 933 | decode | 2,075 | 224 | 224 | +0.0% | +7.1% | +6.7% |
| 1,890 | prefill | 154,322 | 18,606 | 18,600 | -0.0% | +0.4% | -0.9% |
| 1,890 | decode | 1,414 | 194 | 194 | +0.0% | +13.4% | +13.9% |
| 3,086 | prefill | 243,960 | 20,353 | 20,351 | -0.0% | -4.1% | -4.2% |
| 3,086 | decode | 1,864 | 131 | 131 | +0.0% | +6.1% | +6.1% |
| **all** | | 507,726 | 52,250 | 52,244 | -0.01% | -3.4% | +0.2% |
| layer-pass records reproduced exactly (of 75,040) | | | | 74,968 | | 58,650 | 57,764 |

**Calibration:**
- **Capacity.** I swept the exact model's capacity around 1,424 (`s1b_capsweep.py`): 1,424 is the only fit. ±10 slots moves the total by about ±1.7%.
- **Second chances.** The replay makes 44,027 second-chance moves and 51,859 evictions. The logs, which count both classes, give 88,000 / 2 = 44,000 and 103,694 / 2 = 51,847.
- **The no-second-chance column** is plain LRU with the warm-up and the never-cached units. It misses per prompt by −25% to +13%.
- **The last column** is the original `cache_sim.py` LRU. Its total looks right only because two errors cancel: missing the warm-up (+91% at 82 tokens) and missing the protection.

## 3. A per-pass time law (task 2)

**Data:** prefill counts from the trace, summed over the ranks; TTFT from the results JSON.

| Prompt | Passes | Hits | Drive reads | Reads a pass | TTFT (s) | ms a pass |
|---|---|---|---|---|---|---|
| 82 | 6 | 4,417 | 1,913 | 319 | 5.82 | 970 |
| 272 | 17 | 17,475 | 3,291 | 194 | 11.03 | 649 |
| 933 | 59 | 65,840 | 7,218 | 122 | 30.91 | 524 |
| 1,890 | 119 | 135,716 | 18,606 | 156 | 68.07 | 572 |
| 3,086 | 193 | 223,607 | 20,353 | 105 | 99.24 | 514 |
| 272 (fresh process) | 17 | 16,699 | 4,067 | 239 | 12.82 | 754 |

**Fits:**
- The per-pass rows are unweighted least squares.
- L1–L3 are per layer-pass: H = hits, M = reads, U = H + M, summed over ranks.
- L1–L3 minimise the squared relative error: a grid over the nonlinear coefficients, least squares for the rest.
- The "a held at" rows refit L2 with a physical constant.
- Sources: `s2_timelaw.py`, `s2b_maxlaw.py`, `s3c_sens.py`.

| Law | Coefficients (ms) | 82 | 272 | 933 | 1,890 | 3,086 | 272 fresh | Mean abs error | Leave-one-out mean abs |
|---|---|---|---|---|---|---|---|---|---|
| Per pass: a + c·reads | a 376, c 1.29 | -18.8% | -3.5% | +1.9% | +1.0% | -0.4% | -9.2% | 5.8% | 7.8% |
| Per pass: a + b·hits + c·reads | a 765, b -0.318, c 1.09 | -9.3% | +0.2% | +3.9% | +0.2% | -0.4% | -5.3% | 3.2% | 7.1% |
| L1, per layer-pass: a + b·H + c·M | a 21.8, b -0.439, c 1.28 | -1.3% | +3.0% | +2.6% | -0.3% | -3.3% | -1.0% | 1.9% | 4.0% |
| **L2, per layer-pass: a + max(cd·M, cu·U)** | a -1.69, cd 3.2, cu 0.43 | -0.4% | +0.0% | -1.1% | +2.4% | -0.5% | -0.5% | 0.8% | 1.6% |
| L3: L2 + g·(position/1000) | a -2.92, cd 3.35, cu 0.48, g -0.21 | -0.3% | +0.2% | -0.6% | +2.2% | -1.0% | -0.5% | 0.8% | 2.7% |
| L2 with a held at 0 | cd 3, cu 0.37 | -0.4% | +0.9% | -1.0% | +2.4% | -1.1% | +0.2% | 1.0% | — |
| L2 with a held at 2 | cd 2.72, cu 0.3 | -1.5% | +1.5% | -0.6% | +2.5% | -1.4% | +0.2% | 1.3% | — |
| L2 with a held at 4 | cd 2.45, cu 0.22 | -2.5% | +2.0% | -0.5% | +2.6% | -2.6% | +0.4% | 1.8% | — |
| L2 with a held at 6 | cd 2.15, cu 0.135 | -4.3% | +2.7% | +0.6% | +3.5% | -2.5% | +0.5% | 2.3% | — |

**Reading the fits:**
- **The requested form cannot be fitted cleanly.** In 16-token passes the units per layer-pass hardly vary (26–32), so hits and the per-pass constant are collinear.
  - The two-term form misses the read-heavy 82-token prompt by 19%.
  - The three-term form needs a negative cost per hit.
  - A token term does no better: it is collinear with passes, and its fit is unstable (condition number about 650–690).
- **L2 fits with physical structure.**
  - Up to about 4 reads (all ranks), a layer-pass costs about 11.9 ms flat. That covers the card, the cross-rank sync, the pool, and one read batch's latency.
  - Beyond that, each read adds 3.2 ms: bulk drive about 2.5–2.7 ms, then the CPU computes the unit after it lands.
- **Where the long documents sit:**
  - The 3,086-token document averages 2.64 reads per layer-pass, with 16% of layer-passes in the read-driven regime.
  - The 1,890-token document averages 3.91 reads (37% read-driven), which is why it costs 572 ms a pass against 514.
- **L2's negative constant** comes from the same collinearity. With a held at 0–6 ms the fit stays within 1.0–2.3%. The a = 2 ms variant gives 2.72 ms per read, matching the design's 2.70 ms per miss.
- **Context:** L3 finds no growth with position up to 3,086 tokens (−0.21 ms per layer-pass per 1,000 positions).
- **Log cross-check** (shutdown lines), rank 0's issuing thread over 219.8 s of steps:
  - 73.6 s waiting at the routers;
  - 86.0 s waiting on reads: 10.2 ms per layer that read, over 8,409 layers;
  - 47.4 s waiting on the pool's rows.
  - The time is a per-layer latency chain: a layer with any read pays about one read batch. L2 prices this as the flat part.
- **What this means for the skip:** L2 credits the skip's cache effect only above the breakpoint. If a layer that stops reading altogether saves its whole read latency, the skip gains more (L1 gives −40% at 3,086 tokens). L2 is therefore the conservative choice, and I use it below.

## 4. Consumer-suffix skip (task 3)

### 4.1 The schedule, re-derived

**What the graph code shows** (`arch/deepseek41/deepseek41.cpp` at `f4e3aa91`):
- **Topology** (`:262-276`):
  - Compression ratio 0 at layers 0–1, 2 at layers 2–19, 1 at layers 20–39.
  - Compressed and index keys are written only at the kv sources {2, 8, 14, 20} (`L.src`, `:1413-1474`).
  - Layers 20–39 read layer 20's ratio-1 groups (`kv_compressed_r1` and `kv_index_r1`, one layer each in the KV report).
  - Index layers {20, 24, 28, 32, 36} score their own rows against layer 20's index keys.
  - The layers between them reuse the last index layer's per-row selection `m.sel` for the same rows.
  - Layers 24–36 also read layer 20's per-row candidate mask `m.keep`.
- **Window key:** every layer computes it from its own input (`wkv` → `norm_rope_rt` → `kv_store`, `:1393-1412`).
- **Attention:** each layer attends to its row and the 127 rows before it, plus the top-k compressed entries.

**Row-level requirement:**
- Layer L ≥ 21 reaches earlier positions only through its own window and layer 20's keys.
- The last row therefore needs layer L fully on the last n_L = 1 + 127·(39 − L) rows (2,414 at layer 20). It needs that layer's window keys on the 127 rows before those.
- Decode then needs each layer's last 127 window keys, which are already inside that set.
- Layer 20 must write its compressed and index keys for every row.
- The same-row reads (`m.sel`, `m.keep`) are covered, because each layer's full rows nest inside those of the layer below it.

**At 16-token granularity:** if layer L runs fully from pass s, its window keys are needed from row 16s − 127, which falls in pass s − 8. So:
- Layer L runs fully on passes from F − 8·(39 − L) on, where F is the prompt's last pass.
- On the 8 passes before that it runs only its window-key ops (collapse norm, `wkv`, norm/rope/round trip, `kv_store`), with no MoE.
- On earlier passes it runs nothing.
- Layer 20 runs fully from F − 152. On every earlier pass it runs its key ops: window key, ratio-1 compressor, index keys and stores.

**Check** (`skip_lib.check`, every prompt length 1–16,384, `s3a_check.py`):
- The schedule computes every needed row and key.
- No computed row ever reads a window entry that was not written.
- It runs at most 22 more consumer layer-passes than a minimal schedule that tolerates garbage rows (10.7 on average).

**Example at 3,086 tokens (F = 192):**
- Layer 20 runs fully on passes 40–192.
- Layer 39 runs fully on pass 192 only, with key-only work on passes 184–191.
- In all: 5,400 full, 192 key-only and 2,128 skipped layer-passes.

### 4.2 Replay

**Method:**
- Records of key-only and skipped layer-passes are dropped, since they run no MoE.
- The exact cache model replays the whole window: all five prompts and their decode, in one process. Fewer reads therefore also change the cache for later passes, later prompts and decode.
- TTFT is law L2 over the layer-passes that still run, plus each key-only layer-pass at κ·t0:
  - t0 = 11.91 ms, the flat layer-pass;
  - κ = 0.05 for layers 21–39, and 0.10 for layer 20's larger key set.
  - Both κ values are estimated from the kernel count: about 5 small kernels, no EP sync, no all-reduce.
- Each prompt's measured TTFT is scaled by the law's skip / base ratio.
- κ from 0.02 to 0.25 moves every result by under 1 point.

| Prompt | Passes | Layer-passes: full / key-only / skipped | MoE layer-passes removed | Unit accesses removed | Drive reads (base → skip) | Removed reads: in dropped layer-passes / cache effect | Decode reads after (base → skip) | TTFT (s): measured → estimate | Change, L2 | Change, other laws |
|---|---|---|---|---|---|---|---|---|---|---|
| 82 | 6 | 235 / 5 / 0 | 5 (2.1%) | 171 (2.7%) | 1,910 → 1,857 (-2.8%) | 44 / 9 | 175 → 177 | 5.8 → 5.7 | **-2.7%** | L2 a 0–6 ms: -2.5% to -2.7%; L1: -2.1% |
| 272 | 17 | 656 / 16 / 8 | 24 (3.5%) | 727 (3.5%) | 3,295 → 3,161 (-4.1%) | 120 / 14 | 145 → 151 | 11.0 → 10.7 | **-3.3%** | L2 a 0–6 ms: -3.3% to -3.6%; L1: -3.8% |
| 933 | 59 | 2,120 / 58 / 182 | 240 (10.2%) | 7,293 (10.0%) | 7,219 → 5,771 (-20.1%) | 818 / 630 | 224 → 239 | 30.9 → 27.7 | **-10.4%** | L2 a 0–6 ms: -10.4% to -13.3%; L1: -14.2% |
| 1,890 | 119 | 3,830 / 118 / 812 | 930 (19.5%) | 32,564 (21.1%) | 18,600 → 10,554 (-43.3%) | 4,322 / 3,724 | 194 → 212 | 68.1 → 51.2 | **-24.8%** | L2 a 0–6 ms: -24.8% to -29.4%; L1: -29.1% |
| 3,086 | 193 | 5,400 / 192 / 2,128 | 2,320 (30.1%) | 71,430 (29.3%) | 20,351 → 8,890 (-56.3%) | 6,703 / 4,758 | 131 → 140 | 99.2 → 70.9 | **-28.6%** | L2 a 0–6 ms: -28.6% to -34.5%; L1: -40.4% |
| **all five** | 394 | | | | 51,375 → 30,233 (-41.2%) | | 869 → 919 | 215.1 → 166.1 | -22.8% | |

**Reading the table:**
- **Cache effect.**
  - At 3,086 tokens, the dropped layer-passes held 6,703 reads. The layer-passes that still run lost another 4,758, because their units now fit better in 1,424 slots.
  - Under L2, the time saved is 12.2 ms per removed layer-pass at 3,086 tokens and 18.6 ms at 1,890. The mean layer-pass costs 12.8 and 14.6 ms.
  - The 1,890-token document is read-heavy, so its cache effect pays more.
- **Decode after the prompt:** +6% reads (869 → 919). The consumer layers' units are touched only in the prompt's last passes.

### 4.3 Extrapolation to 7.5K and 8K

There is no trace at these lengths. I give three estimates:
- **Like the 3,086-token document:** that document's measured cost per pass, and its replayed saving per removed layer-pass.
- **Like the 1,890-token document:** the same, for a read-heavier document.
- **Synthetic multi-document prompts:**
  - The five documents' prefill passes in the order 1,890, 933, 3,086, 272, 82, repeated until the length is reached.
  - Served first in a fresh process.
  - Replayed and costed with law L2.
  - Several topics touch more distinct experts than one long document would.

**Shared assumption:** cost per pass does not grow with context beyond what L3 saw up to 3,086 tokens. Index scoring does grow with position at layers 20–36, but that growth is not measured past 3K.

| Prompt | Passes | MoE layer-passes removed | Today (s) | Skip (s) | Change | Basis |
|---|---|---|---|---|---|---|
| 7,500 | 469 | 7,840 (41.8%) | 241 | 146 | -39.5% | the 3,086-token document's measured 514 ms a pass and its replayed saving of 12.16 ms a removed layer-pass |
| 7,500 | 469 | 7,840 (41.8%) | 268 | 122 | -54.4% | the 1,890-token document's measured 572 ms a pass and its replayed saving of 18.61 ms a removed layer-pass |
| 8,192 | 512 | 8,700 (42.5%) | 263 | 158 | -40.2% | the 3,086-token document's measured 514 ms a pass and its replayed saving of 12.16 ms a removed layer-pass |
| 8,192 | 512 | 8,700 (42.5%) | 293 | 131 | -55.3% | the 1,890-token document's measured 572 ms a pass and its replayed saving of 18.61 ms a removed layer-pass |
| 7,511 (synthetic) | 472 | 7,900 (41.8%) | 260 | 143 | -44.8% | replay of a multi-document prompt (fresh process), law L2; drive reads 62,548 → 15,240 |
| 8,201 (synthetic) | 516 | 8,780 (42.5%) | 286 | 155 | -45.7% | replay of a multi-document prompt (fresh process), law L2; drive reads 69,470 → 15,278 |

## 5. Large-chunk prefill (task 4)

### 5.1 Method (`lcp_lib.py`, `s4_lcp.py`)

**Chunks:**
- A chunk of C tokens covers C/16 of today's 16-aligned passes.
- For each (layer, rank), a chunk touches the union of its passes' selected units and reads each at most once.
- I used C = 512, 1,024 and 2,048. The 82- and 272-token prompts are a single chunk at every C; the 933-token prompt is one chunk at C ≥ 1,024.

**Policies during the prompt** (decode passes always use the engine's policy):

| Policy | Hits | Misses |
|---|---|---|
| AA (admit-all) | touch | insert, as today |
| RO (read-only) | no touch | no insert |
| ROF | as RO, except the final chunk, which admits all (PREFILL-DESIGN's plan) | as RO, except the final chunk inserts |
| ROB*N* | as RO, except units the prompt's last *N* tokens touch, which are touched | as RO, except those units are inserted when read |
| ROB128, layer-aware victims | as ROB128 | as ROB128, but while a chunk is at layer L, an insert may evict only units of layers ≤ L, which this chunk no longer needs |

**Costs** (the critique's constants, `r1_review_check.py` `lcp_chunk`):
- **Drive:** 2.45 ms per read.
- **CPU, per (layer, rank):** jobs × 0.20 ms + (rows − jobs) × c_row.
  - jobs = D·max(1, ⌈rows/D/32⌉), with D the units the chunk touches for that layer and rank.
  - c_row = 0.06 / 0.08 / 0.10 ms.
- **Card:** 0.6–1.2 s per 1,024 tokens.
- **Chunk time:** card + 1.10 × max(drive, CPU) + 0.15 s.

**Rows:**
- The trace has no row counts. For each rank I took its logged CPU rows (hits + misses, shutdown lines), removed decode's one row per access, and spread the rest over its prefill accesses: 2.03–2.26 rows per unit access.
- That gives **170.7 non-resident rows per prompt token, 71% of 240**.
- The critique's 158 (66%) left out the misses' rows, which the CPU also computes.

### 5.2 Drive reads and decode reads by policy

**C = 512** (cells: drive reads in prefill / decode reads in the 15 decode passes after the prompt; 1 read = 18.8 MB)

| Prompt | Units touched (chunk unions) | Today, 16-token passes | AA | RO | ROF | ROB256 | ROB128 | ROB128, layer-aware victims |
|---|---|---|---|---|---|---|---|---|
| 82 | 3,654 | 1,910 / 175 | 1,810 / 212 | 1,532 / 322 | 1,810 / 212 | 1,810 / 212 | 1,810 / 212 | 1,532 / 217 |
| 272 | 5,817 | 3,295 / 145 | 3,330 / 176 | 2,451 / 187 | 3,330 / 176 | 3,281 / 193 | 3,139 / 261 | 2,120 / 228 |
| 933 | 13,245 | 7,219 / 224 | 8,406 / 241 | 5,747 / 182 | 6,310 / 249 | 4,910 / 215 | 4,681 / 175 | 4,451 / 254 |
| 1,890 | 28,606 | 18,600 / 194 | 18,221 / 231 | 12,599 / 232 | 13,999 / 226 | 13,776 / 209 | 10,467 / 183 | 10,440 / 205 |
| 3,086 | 40,472 | 20,351 / 131 | 24,995 / 148 | 17,523 / 158 | 16,878 / 197 | 14,998 / 129 | 12,988 / 138 | 13,416 / 169 |
| **all** | | 51,375 (966 GB) / 869 | 56,762 (1067 GB) / 1008 | 39,852 (749 GB) / 1081 | 42,327 (796 GB) / 1060 | 38,775 (729 GB) / 958 | 33,085 (622 GB) / 969 | 31,959 (601 GB) / 1073 |

**C = 1,024** (cells: drive reads in prefill / decode reads in the 15 decode passes after the prompt; 1 read = 18.8 MB)

| Prompt | Units touched (chunk unions) | Today, 16-token passes | AA | RO | ROF | ROB256 | ROB128 | ROB128, layer-aware victims |
|---|---|---|---|---|---|---|---|---|
| 82 | 3,654 | 1,910 / 175 | 1,810 / 212 | 1,532 / 322 | 1,810 / 212 | 1,810 / 212 | 1,810 / 212 | 1,532 / 217 |
| 272 | 5,817 | 3,295 / 145 | 3,330 / 176 | 2,451 / 187 | 3,330 / 176 | 3,281 / 193 | 3,139 / 261 | 2,120 / 228 |
| 933 | 8,058 | 7,219 / 224 | 5,456 / 261 | 3,695 / 182 | 5,456 / 261 | 5,221 / 213 | 3,893 / 175 | 3,310 / 233 |
| 1,890 | 17,085 | 18,600 / 194 | 10,890 / 258 | 7,909 / 232 | 9,495 / 242 | 11,208 / 188 | 7,691 / 183 | 7,203 / 207 |
| 3,086 | 24,478 | 20,351 / 131 | 15,255 / 161 | 11,028 / 158 | 10,954 / 199 | 13,451 / 129 | 9,913 / 138 | 9,212 / 162 |
| **all** | | 51,375 (966 GB) / 869 | 36,741 (691 GB) / 1068 | 26,615 (500 GB) / 1081 | 31,045 (584 GB) / 1090 | 34,971 (657 GB) / 935 | 26,446 (497 GB) / 969 | 23,377 (440 GB) / 1047 |

**C = 2,048** (cells: drive reads in prefill / decode reads in the 15 decode passes after the prompt; 1 read = 18.8 MB)

| Prompt | Units touched (chunk unions) | Today, 16-token passes | AA | RO | ROF | ROB256 | ROB128 | ROB128, layer-aware victims |
|---|---|---|---|---|---|---|---|---|
| 82 | 3,654 | 1,910 / 175 | 1,810 / 212 | 1,532 / 322 | 1,810 / 212 | 1,810 / 212 | 1,810 / 212 | 1,532 / 217 |
| 272 | 5,817 | 3,295 / 145 | 3,330 / 176 | 2,451 / 187 | 3,330 / 176 | 3,281 / 193 | 3,139 / 261 | 2,120 / 228 |
| 933 | 8,058 | 7,219 / 224 | 5,456 / 261 | 3,695 / 182 | 5,456 / 261 | 5,221 / 213 | 3,893 / 175 | 3,310 / 233 |
| 1,890 | 9,749 | 18,600 / 194 | 6,814 / 265 | 4,735 / 232 | 6,814 / 265 | 6,790 / 207 | 5,597 / 183 | 4,675 / 212 |
| 3,086 | 16,607 | 20,351 / 131 | 11,238 / 213 | 7,741 / 158 | 9,276 / 216 | 10,648 / 145 | 8,755 / 134 | 6,842 / 166 |
| **all** | | 51,375 (966 GB) / 869 | 28,648 (539 GB) / 1127 | 20,154 (379 GB) / 1081 | 26,686 (502 GB) / 1130 | 27,750 (522 GB) / 970 | 23,194 (436 GB) / 965 | 18,479 (347 GB) / 1056 |

**Reading the policy tables:**
- **AA** inserts about 2,000 units per rank per 1,024-token chunk into 1,424 slots, in layer order. At C = 512 it reads 10% more than today's 16-token passes.
- **RO** has the fewest reads of the policies with the engine's victim rule at C = 2,048. The decode after the prompt pays for it: +24% overall, and +84% after the 82-token prompt (175 → 322).
- **ROF** saves reads only in non-final chunks and does not help decode.
- **ROB128** matches RO's total reads at C ≤ 1,024 and has 10% fewer decode reads.
  - Total reads: 17% fewer at C = 512, 0.6% fewer at C = 1,024.
  - It reads more on the one-chunk short prompts.
- **ROB256** has the fewest decode reads at C ≤ 1,024: +8% against today at C = 1,024. It pays with more prefill reads.
- **Layer-aware victims** cut ROB128's reads by a further 3–20%.
  - Decode reads rise 8–11%, because eviction no longer follows recency.
  - The cut matters most for one-chunk prompts: the 272-token prompt drops from 3,139 to 2,120 reads.

### 5.3 TTFT and the bound

| Prompt | Today (s) | C | Chunks | Drive s, RO / ROB128 | CPU s at 0.06 / 0.08 / 0.10 ms a row | Card s | TTFT s, RO: central (range) | TTFT s, ROB128: central (range) | Bound (c_row 0.06 / 0.10) | Speed-up vs today, ROB128 central (range) |
|---|---|---|---|---|---|---|---|---|---|---|
| 82 | 5.8 | 512 | 1 | 3.8 / 4.4 | 1.3 / 1.5 / 1.7 | 0.0–0.1 | 4.4 (4.3–4.4) | 5.1 (5.1–5.1) | drive / drive | ×1.14 (×1.14–×1.15) |
| 82 | 5.8 | 1,024 | 1 | 3.8 / 4.4 | 1.3 / 1.5 / 1.7 | 0.0–0.1 | 4.4 (4.3–4.4) | 5.1 (5.1–5.1) | drive / drive | ×1.14 (×1.14–×1.15) |
| 82 | 5.8 | 2,048 | 1 | 3.8 / 4.4 | 1.3 / 1.5 / 1.7 | 0.0–0.1 | 4.4 (4.3–4.4) | 5.1 (5.1–5.1) | drive / drive | ×1.14 (×1.14–×1.15) |
| 272 | 11.0 | 512 | 1 | 6.0 / 7.7 | 3.5 / 4.3 / 5.0 | 0.2–0.3 | 7.0 (6.9–7.1) | 8.8 (8.8–8.9) | drive / drive | ×1.25 (×1.24–×1.26) |
| 272 | 11.0 | 1,024 | 1 | 6.0 / 7.7 | 3.5 / 4.3 / 5.0 | 0.2–0.3 | 7.0 (6.9–7.1) | 8.8 (8.8–8.9) | drive / drive | ×1.25 (×1.24–×1.26) |
| 272 | 11.0 | 2,048 | 1 | 6.0 / 7.7 | 3.5 / 4.3 / 5.0 | 0.2–0.3 | 7.0 (6.9–7.1) | 8.8 (8.8–8.9) | drive / drive | ×1.25 (×1.24–×1.26) |
| 933 | 30.9 | 512 | 2 | 14.1 / 11.5 | 11.3 / 14.1 / 17.0 | 0.5–1.1 | 16.8 (16.3–20.1) | 16.7 (15.0–20.1) | drive / CPU | ×1.85 (×1.54–×2.06) |
| 933 | 30.9 | 1,024 | 1 | 9.1 / 9.5 | 10.5 / 13.5 / 16.5 | 0.5–1.1 | 15.8 (12.3–19.4) | 15.8 (12.3–19.4) | CPU / CPU | ×1.95 (×1.60–×2.52) |
| 933 | 30.9 | 2,048 | 1 | 9.1 / 9.5 | 10.5 / 13.5 / 16.5 | 0.5–1.1 | 15.8 (12.3–19.4) | 15.8 (12.3–19.4) | CPU / CPU | ×1.95 (×1.60–×2.52) |
| 1,890 | 68.1 | 512 | 4 | 30.9 / 25.6 | 23.9 / 29.9 / 36.0 | 1.1–2.2 | 36.8 (35.7–42.4) | 36.3 (31.1–42.4) | drive / CPU | ×1.88 (×1.61–×2.19) |
| 1,890 | 68.1 | 1,024 | 2 | 19.4 / 18.8 | 22.3 / 28.5 / 34.8 | 1.1–2.2 | 33.3 (25.9–40.8) | 33.3 (25.9–40.8) | CPU / CPU | ×2.04 (×1.67–×2.63) |
| 1,890 | 68.1 | 2,048 | 1 | 11.6 / 13.7 | 22.1 / 28.4 / 34.7 | 1.1–2.2 | 33.1 (25.6–40.5) | 33.1 (25.6–40.5) | CPU / CPU | ×2.06 (×1.68–×2.66) |
| 3,086 | 99.2 | 512 | 7 | 42.9 / 31.8 | 37.1 / 46.7 / 56.4 | 1.8–3.6 | 55.9 (50.1–67.4) | 55.2 (45.2–66.7) | CPU / CPU | ×1.80 (×1.49–×2.20) |
| 3,086 | 99.2 | 1,024 | 4 | 27.0 / 24.3 | 35.0 / 44.9 / 54.9 | 1.8–3.6 | 53.4 (41.6–65.3) | 52.7 (40.9–64.6) | CPU / CPU | ×1.88 (×1.54–×2.43) |
| 3,086 | 99.2 | 2,048 | 2 | 19.0 / 21.4 | 34.9 / 44.8 / 54.8 | 1.8–3.6 | 52.3 (40.5–64.2) | 52.3 (40.5–64.2) | CPU / CPU | ×1.90 (×1.55–×2.45) |

**Bound by prompt length:**
- **82 and 272 tokens:** drive-bound at every C (one chunk).
- **933 tokens and up:** CPU-bound at C ≥ 1,024, at both 0.06 and 0.10 ms a row.
- **C = 512:** balanced. Drive-bound at 0.06 ms a row (except 3,086 tokens under ROB128), CPU-bound at 0.10.

**Cost floor:** at C = 1,024 and 0.08 ms a row, the CPU's floor is 14.6 ms per token.
- The rows alone cost 13.7 ms per token.
- The first rows of about 50 units per layer-rank per chunk add the rest.

**Variant:** summing each layer's max(drive, CPU), instead of taking 1.10 × the larger total, gives 5–9% lower times. At 3,086 tokens and C = 1,024: 48.2 s against 52.7 s.

### 5.4 Synthetic 7.5K and 8K prompts

The same synthetic prompts as §4.3, in a fresh process. Each cell gives the central time, its range, the bound at 0.08 ms a row, then the drive reads.

| Prompt | Today, L2 (s) | Skip, L2 (s) | C | LCP RO | LCP ROB128 | LCP AA | LCP+skip RO | LCP+skip ROB128 | LCP+skip AA |
|---|---|---|---|---|---|---|---|---|---|
| 7,511 | 260 | 143 | 512 | 141 (134–164), CPU; 47,254 reads | 143 (135–165), CPU; 47,254 reads | 187 (185–190), drive; 66,283 reads | 84 (79–98), CPU; 27,681 reads | 84 (73–99), CPU; 24,013 reads | 91 (80–103), CPU; 18,512 reads |
| 7,511 | 260 | 143 | 1,024 | 131 (103–159), CPU; 30,626 reads | 130 (103–159), CPU; 31,540 reads | 135 (120–161), CPU; 42,319 reads | 78 (62–94), CPU; 18,618 reads | 81 (65–96), CPU; 17,575 reads | 83 (71–97), CPU; 17,202 reads |
| 7,511 | 260 | 143 | 2,048 | 129 (99–158), CPU; 18,584 reads | 129 (99–158), CPU; 20,968 reads | 129 (100–158), CPU; 26,814 reads | 76 (59–93), CPU; 11,780 reads | 76 (59–93), CPU; 11,003 reads | 76 (62–93), CPU; 13,986 reads |
| 8,201 | 286 | 155 | 512 | 155 (147–182), CPU; 51,724 reads | 154 (139–180), CPU; 48,438 reads | 205 (203–208), drive; 72,539 reads | 91 (86–108), CPU; 29,877 reads | 88 (73–106), CPU; 21,765 reads | 98 (86–112), CPU; 18,648 reads |
| 8,201 | 286 | 155 | 1,024 | 144 (112–175), CPU; 32,691 reads | 142 (112–174), CPU; 32,405 reads | 146 (128–176), CPU; 45,034 reads | 85 (67–104), CPU; 19,363 reads | 84 (66–102), CPU; 16,561 reads | 86 (71–104), CPU; 15,488 reads |
| 8,201 | 286 | 155 | 2,048 | 143 (111–175), CPU; 19,912 reads | 141 (109–173), CPU; 20,710 reads | 144 (112–175), CPU; 29,216 reads | 84 (66–103), CPU; 12,402 reads | 84 (66–103), CPU; 11,177 reads | 85 (67–104), CPU; 11,911 reads |

## 6. LCP plus the skip (task 5)

**Method:**
- Inside each chunk, layer L's MoE covers only the 16-row groups where the schedule runs it fully (the per-layer row suffixes).
- Unions, rows and reads shrink accordingly.
- Card time scales with the layer-rows computed, counting key-only rows at κ = 0.05.

**Result:**
- Since LCP is CPU-bound, the skip's share of removed rows carries straight through: −29% at 3,086 tokens, about −40% at 7.5K.
- At 3,086 tokens and C = 512 with the skip, AA reads less than RO (10,542 against 12,516). With fewer layers active, coherent text reuses the previous chunk's units.

**C = 512** (cells: drive reads in prefill / decode reads in the 15 decode passes after the prompt; 1 read = 18.8 MB)

| Prompt | Units touched (chunk unions) | Today, 16-token passes + skip | AA | RO | ROF | ROB256 | ROB128 | ROB128, layer-aware victims |
|---|---|---|---|---|---|---|---|---|
| 82 | 3,549 | 1,857 / 177 | 1,768 / 210 | 1,508 / 322 | 1,768 / 210 | 1,768 / 210 | 1,768 / 210 | 1,508 / 213 |
| 272 | 5,644 | 3,161 / 151 | 3,217 / 187 | 2,379 / 187 | 3,217 / 187 | 3,178 / 208 | 3,047 / 266 | 2,049 / 233 |
| 933 | 12,143 | 5,771 / 239 | 7,675 / 235 | 5,285 / 182 | 5,863 / 244 | 3,970 / 219 | 4,288 / 179 | 4,123 / 244 |
| 1,890 | 23,414 | 10,554 / 212 | 14,690 / 235 | 10,299 / 232 | 11,904 / 225 | 9,569 / 188 | 8,525 / 191 | 9,246 / 208 |
| 3,086 | 29,054 | 8,890 / 140 | 10,542 / 159 | 12,516 / 158 | 12,981 / 196 | 9,949 / 137 | 9,367 / 132 | 10,580 / 171 |
| **all** | | 30,233 (568 GB) / 919 | 37,892 (712 GB) / 1026 | 31,987 (601 GB) / 1081 | 35,733 (672 GB) / 1062 | 28,434 (535 GB) / 962 | 26,995 (508 GB) / 978 | 27,506 (517 GB) / 1069 |

**C = 1,024** (cells: drive reads in prefill / decode reads in the 15 decode passes after the prompt; 1 read = 18.8 MB)

| Prompt | Units touched (chunk unions) | Today, 16-token passes + skip | AA | RO | ROF | ROB256 | ROB128 | ROB128, layer-aware victims |
|---|---|---|---|---|---|---|---|---|
| 82 | 3,549 | 1,857 / 177 | 1,768 / 210 | 1,508 / 322 | 1,768 / 210 | 1,768 / 210 | 1,768 / 210 | 1,508 / 213 |
| 272 | 5,644 | 3,161 / 151 | 3,217 / 187 | 2,379 / 187 | 3,217 / 187 | 3,178 / 208 | 3,047 / 266 | 2,049 / 233 |
| 933 | 7,562 | 5,771 / 239 | 5,082 / 278 | 3,455 / 182 | 5,082 / 278 | 4,834 / 219 | 3,621 / 179 | 3,091 / 231 |
| 1,890 | 14,481 | 10,554 / 212 | 9,709 / 253 | 6,646 / 232 | 8,485 / 248 | 7,287 / 198 | 6,527 / 191 | 6,442 / 212 |
| 3,086 | 18,299 | 8,890 / 140 | 11,410 / 164 | 8,143 / 158 | 9,091 / 196 | 7,589 / 149 | 7,125 / 132 | 7,288 / 166 |
| **all** | | 30,233 (568 GB) / 919 | 31,186 (586 GB) / 1092 | 22,131 (416 GB) / 1081 | 27,643 (520 GB) / 1119 | 24,656 (464 GB) / 984 | 22,088 (415 GB) / 978 | 20,378 (383 GB) / 1055 |

**C = 2,048** (cells: drive reads in prefill / decode reads in the 15 decode passes after the prompt; 1 read = 18.8 MB)

| Prompt | Units touched (chunk unions) | Today, 16-token passes + skip | AA | RO | ROF | ROB256 | ROB128 | ROB128, layer-aware victims |
|---|---|---|---|---|---|---|---|---|
| 82 | 3,549 | 1,857 / 177 | 1,768 / 210 | 1,508 / 322 | 1,768 / 210 | 1,768 / 210 | 1,768 / 210 | 1,508 / 213 |
| 272 | 5,644 | 3,161 / 151 | 3,217 / 187 | 2,379 / 187 | 3,217 / 187 | 3,178 / 208 | 3,047 / 266 | 2,049 / 233 |
| 933 | 7,562 | 5,771 / 239 | 5,082 / 278 | 3,455 / 182 | 5,082 / 278 | 4,834 / 219 | 3,621 / 179 | 3,091 / 231 |
| 1,890 | 8,878 | 10,554 / 212 | 6,084 / 247 | 4,258 / 232 | 6,084 / 247 | 6,139 / 206 | 4,753 / 191 | 4,304 / 213 |
| 3,086 | 13,565 | 8,890 / 140 | 9,078 / 211 | 6,233 / 158 | 7,883 / 214 | 6,517 / 156 | 6,192 / 132 | 5,746 / 168 |
| **all** | | 30,233 (568 GB) / 919 | 25,229 (474 GB) / 1133 | 17,833 (335 GB) / 1081 | 24,034 (452 GB) / 1136 | 22,436 (422 GB) / 999 | 19,381 (364 GB) / 978 | 16,698 (314 GB) / 1058 |

**TTFT and bound, LCP + skip:**

| Prompt | Today (s) | C | Chunks | Drive s, RO / ROB128 | CPU s at 0.06 / 0.08 / 0.10 ms a row | Card s | TTFT s, RO: central (range) | TTFT s, ROB128: central (range) | Bound (c_row 0.06 / 0.10) | Speed-up vs today, ROB128 central (range) |
|---|---|---|---|---|---|---|---|---|---|---|
| 82 | 5.8 | 512 | 1 | 3.7 / 4.3 | 1.3 / 1.5 / 1.7 | 0.0–0.1 | 4.3 (4.3–4.3) | 5.0 (5.0–5.0) | drive / drive | ×1.17 (×1.16–×1.17) |
| 82 | 5.8 | 1,024 | 1 | 3.7 / 4.3 | 1.3 / 1.5 / 1.7 | 0.0–0.1 | 4.3 (4.3–4.3) | 5.0 (5.0–5.0) | drive / drive | ×1.17 (×1.16–×1.17) |
| 82 | 5.8 | 2,048 | 1 | 3.7 / 4.3 | 1.3 / 1.5 / 1.7 | 0.0–0.1 | 4.3 (4.3–4.3) | 5.0 (5.0–5.0) | drive / drive | ×1.17 (×1.16–×1.17) |
| 272 | 11.0 | 512 | 1 | 5.8 / 7.5 | 3.4 / 4.1 / 4.9 | 0.2–0.3 | 6.8 (6.7–6.9) | 8.6 (8.5–8.7) | drive / drive | ×1.28 (×1.27–×1.29) |
| 272 | 11.0 | 1,024 | 1 | 5.8 / 7.5 | 3.4 / 4.1 / 4.9 | 0.2–0.3 | 6.8 (6.7–6.9) | 8.6 (8.5–8.7) | drive / drive | ×1.28 (×1.27–×1.29) |
| 272 | 11.0 | 2,048 | 1 | 5.8 / 7.5 | 3.4 / 4.1 / 4.9 | 0.2–0.3 | 6.8 (6.7–6.9) | 8.6 (8.5–8.7) | drive / drive | ×1.28 (×1.27–×1.29) |
| 933 | 30.9 | 512 | 2 | 12.9 / 10.5 | 10.2 / 12.8 / 15.3 | 0.5–1.0 | 15.3 (15.0–18.1) | 15.3 (13.7–18.1) | drive / CPU | ×2.02 (×1.70–×2.26) |
| 933 | 30.9 | 1,024 | 1 | 8.5 / 8.9 | 9.5 / 12.2 / 14.9 | 0.5–1.0 | 14.3 (11.1–17.5) | 14.3 (11.1–17.5) | CPU / CPU | ×2.16 (×1.77–×2.78) |
| 933 | 30.9 | 2,048 | 1 | 8.5 / 8.9 | 9.5 / 12.2 / 14.9 | 0.5–1.0 | 14.3 (11.1–17.5) | 14.3 (11.1–17.5) | CPU / CPU | ×2.16 (×1.77–×2.78) |
| 1,890 | 68.1 | 512 | 4 | 25.2 / 20.9 | 19.0 / 23.7 / 28.5 | 0.9–1.8 | 29.7 (29.2–33.8) | 29.0 (24.8–33.7) | drive / CPU | ×2.35 (×2.02–×2.74) |
| 1,890 | 68.1 | 1,024 | 2 | 16.3 / 16.0 | 17.7 / 22.6 / 27.6 | 0.9–1.8 | 26.5 (20.7–32.4) | 26.5 (20.7–32.4) | CPU / CPU | ×2.56 (×2.10–×3.29) |
| 1,890 | 68.1 | 2,048 | 1 | 10.4 / 11.6 | 17.5 / 22.4 / 27.4 | 0.9–1.8 | 26.2 (20.3–32.1) | 26.2 (20.3–32.1) | CPU / CPU | ×2.60 (×2.12–×3.36) |
| 3,086 | 99.2 | 512 | 7 | 30.7 / 22.9 | 26.3 / 33.1 / 39.9 | 1.3–2.5 | 40.1 (36.0–48.2) | 39.4 (31.7–47.5) | CPU / CPU | ×2.52 (×2.09–×3.13) |
| 3,086 | 99.2 | 1,024 | 4 | 20.0 / 17.5 | 24.9 / 31.9 / 39.0 | 1.3–2.5 | 38.3 (30.0–46.7) | 37.6 (29.3–46.0) | CPU / CPU | ×2.64 (×2.16–×3.39) |
| 3,086 | 99.2 | 2,048 | 2 | 15.3 / 15.2 | 24.7 / 31.8 / 38.8 | 1.3–2.5 | 37.1 (28.8–45.5) | 37.1 (28.8–45.5) | CPU / CPU | ×2.67 (×2.18–×3.45) |

## 7. Assumptions and limits

- **Time law.**
  - L2 is fitted to six TTFTs from one window and one configuration; its constant cannot be pinned down separately.
  - The skip estimates are scaled to each measured TTFT.
  - Across L2 variants with a physical constant, the 3,086-token result moves from −28.6% to −34.5%.
- **Key-only layer cost** is estimated, not measured.
- **The LCP constants are the critique's and remain unbenchmarked:**
  - c_row at 32-row jobs, and the 0.20 ms first row;
  - card time of 0.6–1.2 s per 1,024 tokens;
  - the 1.10 factor.
  - c_row alone moves LCP's time by ±25%. The 32-row kernel benchmark (critique item 0a) settles it.
- **Rows** are a per-rank average, spread uniformly over each prompt's accesses.
- **Short prompts.** The LCP model assumes the 82- and 272-token prompts run as one tall chunk, which needs P0's exactness work. They may instead stay on 16-token passes.
- **Synthetic 7.5K / 8K prompts:**
  - multi-document and partly repeated;
  - served in a fresh process;
  - costed with L2 extrapolated beyond 3,086 tokens.
  - The skip's share of removed layer-passes (41.8% / 42.5%) is exact; its cache effect depends on the text.
- **Decode after the prompt** covers only 15 tokens per prompt. Its time effect is small: each prompt's decode took 0.84–1.10 s.
- **Async inserts.** The replay inserts each miss immediately; the engine relays misses on background threads. The 99.9% record match shows this does not matter at 16-token passes.
  - For bounded admission, inserting at the end of a chunk rather than when read would mean holding the planes of up to about 900 units per rank (about 17 GB). I did not model it; the layer-aware variant is the feasible alternative.
- **The skip needs** what the survey lists:
  - an owned prompt cut in `RadBatch`;
  - no periodic checkpoints before the cut;
  - every rank deriving E from the batch.
  - Layer 20's key-only passes must also run the ratio-1 compressor and the index-key ops, not only the window key.

## 8. Files

All in `/home/as/.claude/jobs/635603c1/tmp/prefill-replay/`; run each with `python3 -I`. Most step scripts write `<name>_out.json`, and their printed output is in `<name>_out.txt`.

| Script | What it does |
|---|---|
| `trace_lib.py`, `replay_lib.py` | Trace loader; plan, cache model, warm-up, protection, pass split |
| `s0_peek.py`, `s0b_units.py`, `s0c_map.py`, `s0d_plan.py` | Trace structure, unit numbering, expert mapping, plan reconstruction |
| `s1_validate.py`, `s1b_capsweep.py` | Replay against the trace, never-cached units, capacity calibration |
| `s2_timelaw.py`, `s2b_maxlaw.py`, `s3c_sens.py` | Time laws and their sensitivity |
| `skip_lib.py`, `s3a_check.py` | Skip schedule and its row-level check for 1–16,384 tokens |
| `s3b_skip.py` | Skip replay and TTFT |
| `lcp_lib.py`, `s4_lcp.py`, `s4b_layeraware.py` | LCP and LCP + skip, all policies and the synthetic prompts |
| `s6_extrap.py` | Analytic extrapolation |
| `s8_tables.py` | Prints every table in this report from the JSON outputs (a template on stdin) |

# Critique

# DS41 prefill trace replay: adversarial review of the consumer-suffix skip and large-chunk prefill

I worked read-only and used no GPU. All my work is single-thread Python in `/home/as/.claude/jobs/635603c1/tmp/prefill-replay/critique/` (c1–c10). Each script took at most about 32 s and ran behind the load-rule wait. I rewrote the parser, the cache model, the skip check, the time-law fits and the LCP replay myself rather than importing the report's code. The report's directory is unchanged.

## Verdict

- **Trace parsing, prompt boundaries and the cache replay are confirmed independently.** The six never-cached units are corroborated by independent evidence, not just fitted to the residuals.
- **The skip schedule is confirmed** by a forward simulation that propagates garbage values.
- **The skip's gain is confirmed. Build it first.**
  - At 3,086 tokens: −29% central, −26% to −33% across all laws that fit the TTFTs within 2%.
  - At 7.5K–8K tokens: about −45% (−42% to −50%).
  - It passes the survey's ≥15% gate under every law I tried.
- **LCP: every replay number reproduces, but two corrections change the conclusions:**
  1. **Bounded admission `ROB<N>` uses future routing.** It admits units in every chunk based on what the prompt's last N tokens will select, which the engine cannot know until it reaches them. A causal version reads more. TTFT is unchanged while LCP is CPU-bound.
  2. **The central CPU cost of 0.08 ms per row is below what this box measures.** The marginal cost of the 4th row is 0.101 ms with the `dec1` prototype and 0.137 ms with the production kernel. At 0.10 ms, LCP gives ×1.56–1.69 at 933–3,086 tokens; at 0.12 ms, ×1.33–1.44. That makes LCP alone roughly equal to the skip alone at 3K–8K tokens. The ×1.9–2.0 holds only if the 32-row kernel beats today's measured marginal by about 20%.

## 1. Independent checks

| Check | Result |
|---|---|
| Trace parse (`c1`) | 18,760 records per rank, 469 passes, layers 0–39 in order, no odd read weights. Reads per rank 12,869 / 12,875 / 13,251 / 13,255 equal the log. Accesses 126,020 / 122,788 / 131,813 / 127,105 equal the log's "units probed" (hit units plus misses), so the trace is exactly the CPU's unit stream. |
| Prompt boundaries | 5×16+2, 17×16, 58×16+5, 118×16+2 and 192×16+14 tokens, each followed by 15 one-token passes. Confirmed. |
| Cache model (`c2`) | My own linked-list cache, written from `mapped_expert_slab_cache.h` (`borrow` touches and zeroes chances; `victim_for` moves at most 64) and from `stager.cpp`'s protection and `hx_warm`. Result: 74,968 of 75,040 records exact, 52,244 reads against 52,250, 44,027 second chances, 51,859 evictions. Protected units 1,051 / 1,045 / 1,075 / 1,017 match the log. |
| Never-cached units | 2 / 1 / 1 / 1 of them are in the warm set. That equals logged refusals minus accesses on every rank (177−175, 60−59, 77−76, 71−70). |
| Rows per token | `hx_submit` adds only cached units' rows to `hx_stats_.rows`; misses go to `miss_rows`. So the report's 170.7 non-resident rows per token is right, and the critique's 158 undercounted. |
| Skip schedule (`c4`) | Forward simulation of window keys, layer 20's compressed and index keys, and the per-pass `m.sel`, `m.keep` and `cs_b` buffers, at 1,138 lengths from 1 to 16,384. Every needed output is valid: the last row's logits, every layer's last 127 window keys, and layer 20's keys for every row. No computed row reads an unwritten window entry. Four deliberately broken schedules all fail, so the check has teeth. |
| Skip replay (`c3`) | Prefill reads 1,910 / 3,295 / 7,219 / 18,600 / 20,351 become 1,857 / 3,161 / 5,771 / 10,554 / 8,890. Decode reads 869 become 919. Confirmed. |
| L2 refit | a = −1.55, cd = 3.18, cu = 0.425. Same errors, leave-one-out mean 1.6%. Confirmed. |
| LCP and synthetic (`c5`, `c6`) | Every AA / RO / ROF / ROB128 / ROB256 read, decode and TTFT figure at C = 512 / 1,024 / 2,048, with and without the skip, reproduces exactly. The synthetic prompts reproduce too: 62,548 → 15,240 reads, 260 → 143 s. |

## 2. Findings that change numbers

### 2.1 Bounded admission is an oracle

In `s4_lcp.run` and `s4b_layeraware.run`, the admission set `lastN[(L, r)]` (the units the prompt's last N tokens select) is applied in every chunk. In the causal version (`ROBc`), a chunk admits at (L, r) only the units selected by its own rows inside the last N tokens; earlier chunks are read-only.

Cells: prefill drive reads / decode reads after the prompt.

| Case | Report ROB128 | Causal ROBc128 | RO |
|---|---|---|---|
| C = 512 | 33,085 / 969 | 40,946 / 961 | 39,852 / 1,081 |
| C = 1,024 | 26,446 / 969 | 29,045 / 961 | 26,615 / 1,081 |
| C = 2,048 | 23,194 / 965 | 23,258 / 968 | 20,154 / 1,081 |
| C = 1,024 with skip | 22,088 / 978 | 23,892 / 979 | 22,131 / 1,081 |
| C = 1,024, layer-aware victims | 23,377 / 1,047 | 25,108 / 1,055 | — |

What this corrects in the report:
- **"ROB128 reads about as much as read-only" and "17% fewer at C = 512" are wrong.** Causally it reads more than RO: +3% at C = 512, +9% at 1,024, +15% at 2,048. Its real advantage survives: decode after the prompt reads 961, against 1,081 for RO and 869 today.
- **The bound at C = 512:** the exception "except 3,086 tokens under ROB128" disappears. All five prompts are drive-bound at 0.06 ms a row.
- **Drive use:** "24 s of drive against 45 s of CPU" becomes 28.4 s against 44.9 s (63%).
- **C = 2,048 against 1,024:** it buys ≤ 0.8% on the measured documents (≤ 1.6% with the skip) and 2–4% at 7.5K, not "2% / 6%".
- **One-chunk short prompts.** With immediate victims, bounded admission evicts units that later layers of the same chunk need. The 82- and 272-token prompts read 1,810 / 3,139 units against RO's 1,532 / 2,451. That gives 5.1 / 8.8 s against 4.4 / 7.0 s (×1.14 / ×1.25 against ×1.34 / ×1.58). Layer-aware victims recover RO's reads (1,532 / 2,120).
- **The decode advantage is small in time.** RO's extra 120 decode reads cost about 0.3 s in total (one-unit read ≈ 2.8 ms, per the log). If the kernel turns out fast enough that LCP is drive-bound, RO wins: at 3,086 tokens and 0.04 ms a row, RO takes 32.1 s against 33.7 s.
- **TTFT at ≥ 933 tokens is unchanged** because LCP is CPU-bound: 3,086 tokens at C = 1,024 is 52.7 s either way.

### 2.2 The central CPU row cost is optimistic

NIGHT-LOG 07:15 (`mrbench`, 40 threads) measured:
- First row: 0.201 ms.
- Each extra row, as an average at 2 / 3 / 4 rows: 0.49 / 0.56 / 0.60 of a row today, 0.36 / 0.41 / 0.44 with `dec1`.
- So the marginal cost of the 2nd / 3rd / 4th row rises: 0.098 / 0.127 / 0.137 ms today, 0.072 / 0.092 / 0.101 ms with `dec1`.

A calibration check (`c10`): applying the critique's cost model with today's production row costs to today's 16-token passes explains 81–89% of the measured TTFTs. The CPU pool alone accounts for 72–78% of TTFT at ≥ 933 tokens. Today's prefill is therefore mostly pool-bound, and LCP gains only the pool work it saves plus the per-layer latency chain.

LCP at C = 1,024 with causal bounded admission (`c8`), TTFT in seconds:

| Prompt | 0.08 ms | 0.10 ms | 0.12 ms | 0.137 ms |
|---|---|---|---|---|
| 933 | 15.8 (×1.95) | 19.1 (×1.62) | 22.4 (×1.38) | 25.1 (×1.23) |
| 1,890 | 33.3 (×2.04) | 40.3 (×1.69) | 47.2 (×1.44) | 53.0 (×1.28) |
| 3,086 | 52.7 (×1.88) | 63.7 (×1.56) | 74.6 (×1.33) | 84.0 (×1.18) |
| 7,511 (synthetic) | 132 (×1.97) | 160 (×1.63) | — | — |

At 0.10 ms, LCP alone is no better than the skip alone: 63.7 s against 70.9 s at 3,086 tokens, and 160 s against 143 s at 7.5K.

One effect is not modelled at all. A 32-row job runs for about 100 ms on one thread, and each layer of each chunk waits for its slowest job. The benchmark should measure jobs split across threads.

### 2.3 The time law: L2 is mid-range, not the conservative end

Fits to the six TTFTs (`c3`). The skip columns give the TTFT change for 933 / 1,890 / 3,086 / 7.5K tokens.

| Law | Mean fit error (leave-one-out) | Skip change |
|---|---|---|
| L2 (the report's) | 0.8% (1.6%) | −10.4 / −24.9 / −28.6 / −44.9% |
| L2 plus 20 ms a pass the skip cannot remove | identical fit | −10.0 / −24.2 / −27.4 / −43.4% |
| L2 plus 40 ms a pass | identical fit | −9.6 / −23.6 / −26.3 / −41.9% |
| Hinge: a + cu·U + 2.84·max(0, M − 4) | 0.7% (3.0%, worst 9%) | −11.0 / −25.2 / −29.8 / −46.0% |
| Soft max, power 4 / power 3 | 1.6% / 2.0% | −11.5 / −27.1 / −30.8 / −47.2% and −12.3 / −28.3 / −32.5 / −48.9% |
| Linear, no negative hit cost | 3.0% (4.8%) | −14.8 / −32.2 / −41.5 / −58.5% |

- **A per-pass constant cannot be identified** from this data: every pass has exactly 40 layer-passes, so it is collinear with L2's constant. Decode bounds it, though. A decode step costs 63.2 ms for all 40 layers, mostly pool work and reads, which leaves only a few ms of per-step overhead. That is worth ≤ 1 point.
- **L1's −40% is unreliable:** it rests on a negative cost per hit.
- **The fresh 272-token run** cost 1.79 s more for 776 more reads. Every law predicts 1.58–1.75 s; L2 is closest at 1.73 s.

### 2.4 Why the skip's gain grows with prompt length

Each rank has 1,552–1,583 file-tier units in layers 0–19, against 1,424 cache slots (`c9`). In passes where the skip runs only layers 0–19, that set nearly fits in the cache. In the synthetic 7.5K replay, reads per such pass fall from 94–167 today to 44, then 13, 2.3 and 1.0 in successive 80-pass stretches.

The 3,086-token document has only 40 such passes, so its saving of 12.16 ms per removed layer-pass underestimates longer prompts. The report's "like the 3,086-token document" figure of −40% is therefore a floor, not a central estimate.

### 2.5 Smaller corrections

- **Warm-up set:** it is the 1,424 hottest *file-tier* units, not the profile's hottest units overall.
- **Checkpoints under the skip.** The report says "no periodic checkpoints before the cut", but the constraint is stronger. Layer 39 has window keys only from pass F − 8, so every checkpoint more than about one pass before the prompt's end is incomplete. With the skip on, the only resume point is the prompt's end. Shared-prefix reuse at mid-prompt checkpoints is lost, which matters if prefix caching (critique item 1) is built with a 128–256-token interval.
- **What layer 20's key-only work must include:** the whole ratio-1 compressor chain (`c_gemm`, `c_norm`, `c_tab`, `l_rope`, `l_q`, `l_st`) and the index-key ops (`k_gemm` to `k_st`). The consumers' window-key rope reads the angles table `cs_b` that `c_tab` writes. That is not binding in this schedule, because layer 20 runs fully on every pass where any consumer layer does work.
- **Block-granular kernels are safe.** Window blocks are 16 rows and every key-only band starts on a 16-row boundary, so no kernel reads an unwritten entry.
- **Synthetic prompts:** they contain four short passes in the middle (472 passes for 7,511 tokens), so their 8-pass bands are not exactly 128 rows. The counting effect is under 0.5%.

## 3. Corrected expected TTFT (seconds)

LCP columns use C = 1,024 and causal bounded admission. Cells give TTFT at 0.08 / 0.10 / 0.12 ms a row.

| Prompt | Today | Skip, 16-token passes: central (range) | LCP | LCP + skip |
|---|---|---|---|---|
| 82 | 5.8 | 5.7 (−3%) | 5.1, drive-bound (RO or layer-aware: 4.4) | 5.0 (RO: 4.3) |
| 272 | 11.0 | 10.7 (−3%) | 8.8, drive-bound (RO: 7.0; layer-aware: about 6.1) | 8.6 (RO: 6.8) |
| 933 | 30.9 | 27.7 (−10%; −10 to −12%) | 15.8 / 19.1 / 22.4 | 14.3 / 17.2 / 20.2 |
| 1,890 | 68.1 | 51.2 (−25%; −24 to −28%) | 33.3 / 40.3 / 47.2 | 26.5 / 32.0 / 37.4 |
| 3,086 | 99.2 | 70.9 (−29%; −26 to −33%) | 52.7 / 63.7 / 74.6 | 37.7 / 45.4 / 53.1 |
| 7,511 (synthetic) | 260 (modelled) | 143 (−45%; −42 to −49%) | 132 / 160 / — | 79 / 95 / — |
| 8,201 (synthetic) | 286 (modelled) | 155 (−46%; −43 to −50%) | 144 / 176 / — | 86 / 104 / — |

## 4. Which to build first

1. **The skip, first. I agree with the report.**
   - It is exact by dependency with today's kernels of 16 rows or fewer, so it needs no P0 work.
   - It gives −29% at 3K and about −45% at 7.5K–8K tokens, and clears the ≥15% gate under every law (the lowest is −26%).
   - Gate it on bytes: last-row logits and KV hashes.
   - Accept that mid-prompt checkpoints disappear while it is on.
2. **The 32-row CPU kernel benchmark, now and in parallel.** It is CPU-only and needs no window. Benchmark with jobs split across threads. It decides LCP:
   - ≤ 0.08 ms a row: LCP alone ×1.9–2.0; with the skip ×2.6 at 2K–3K tokens.
   - About 0.10 ms: ×1.55–1.7, no better than the skip alone at ≥ 3K tokens.
   - ≥ 0.12 ms: ×1.2–1.4.
3. **LCP at C = 1,024, only if the kernel reaches about 0.09 ms a row or less.**
   - Read-only for non-final chunks.
   - In the final chunk, admit the last 128 tokens' units with layer-aware victims, or victims chosen after the layer's work.
   - C = 2,048 adds at most about 4%.
4. **If the kernel lands at 0.10 ms or more, move the card-share item ahead of LCP (or into it).** Both today's passes (pool about 75% of TTFT) and LCP are bound by the CPU pool.

## Files

All in `/home/as/.claude/jobs/635603c1/tmp/prefill-replay/critique/`. Run with `python3 -I` from that directory; each script's printed output is in `c<N>_out.txt`, and c1–c8 also write `c<N>_out.json`.

| Script | What it checks |
|---|---|
| `c1_trace.py` | Parse, log equalities, prompt boundaries, never-cached unit counts |
| `c2_replay.py` | Independent cache replay and its variants |
| `c3_timelaw.py` | Law fits, leave-one-out, skip TTFT under each law |
| `c4_sched.py` | Forward validity simulation of the schedule and the broken-schedule controls |
| `c5_lcp.py` | LCP with every policy, including causal `ROBc` |
| `c6_synth.py` | Synthetic 7.5K / 8K prompts |
| `c7_layeraware.py` | Layer-aware victims with causal admission |
| `c8_crow.py` | Sensitivity to the CPU row cost |
| `c9_capacity.py` | Producer-layer capacity and reads per pass |
| `c10_today_model.py` | Cost model applied to today's passes |
