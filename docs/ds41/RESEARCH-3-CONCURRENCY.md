# concurrency: research

The best configuration starts with `--max-num-seqs N` but never batches decode rows, so N>1 is just serial service today. I built and committed an opt-in scheduler switch that fixes this. The model then gives +38–55% aggregate decode at N=4 (+16–32% on the 64-token held-out set once prefill is counted), with each stream slowing to about 1/N. The single NVMe is the limit. Nothing was run on the GPUs: decode-row batch invariance on the card and the multi-row step cost are both unmeasured, so window 1 below is the real test.

## Why N>1 does not batch today
- **The 16-token quantum blocks admission.** The best config needs `--max-num-batched-tokens 16` (`hx_init` refuses more under B1/K, `core/place/stager.cpp:944`). The chunk quantum is also 16 (the attention block). One decode row leaves 15 tokens of budget, `admit_one` returns `RAD_E_FULL` (`budget < quantum && budget < remaining`), and admission stops at the head of the queue. A running prompt behind a decode row starves the same way. This matches the 05:33 window: TTFTs of 3.5, 11.0, 19.6 and 28.2 s.
- **16 rows is also the exactness limit, so a 16-token prefill chunk can never share a pass with decode rows, and N ≤ 16.** Above 16 tokens `moe_gemm_ggml` leaves the gemv path for the tiled one, attention leaves the exact kernels above 48 rows, and the CPU path is capped at `kEpRouteMaxRows=16`.
- **KV pool:** the planner gives the cache one `--max-model-len` sequence (`vram_budget.cpp`, "no lender"). At 512 that is 494 tokens in total, enough for four ~110-token requests. Eight need `--max-model-len 1024` and sixteen need 2048 (+22 / +65 MiB a card, about 1.2 / 3.5 fewer VRAM experts a rank). Preemption has to be avoided: recompute re-prefills output tokens through the chunked path.
- **CPU path limits are not binding.** `kMaxRows=4` hardly matters: only 0.04% of file-tier units get more than 4 rows at N=16. The 128 job slots compare with about 50 jobs a layer at N=16.
- **Batch invariance up to 16 rows is designed in:**
  - bf16 m64 GEMM in the decode band: MB=1 and BR_FLAT, so the split never depends on M.
  - f32 router: "all forms byte-identical".
  - moe gemv: each row's dot product is independent of the other rows.
  - `moe_gather`: sums in ascending slot order.
  - attention and indexer: exact scalar kernels up to 48 rows, so verify reproduces decode.
  - all-reduce: fixed rank order.
  - CPU rows: identical for 1–5 rows, and the verify arms found 0 differences in passes of 1–16 rows.
- **N-row decode passes take the multi-row settings** (async misses, mirror share 0.18). That suits layers with several misses: 0.18 is close to the bandwidth-balanced 0.177.

## Model
Scripts are in `/home/as/.claude/jobs/635603c1/tmp/research2/conc/`.
- It replays the 40-prompt trace with the RAM cache as LRU plus 2 second chances at 1,424 packed units a rank, warm start. The replay reproduces the engine's misses (rank 0 decode: 8,624 = 8,624).
- Timing per layer is the card work plus the larger of the card's routed work and the CPU completion including miss reads.
- Two calibrations both reproduce N=1 (held-out 64.4 ms, chat 73.7 ms):
  - **A:** misses overlap as at N=1 (1.9 ms a miss, card+host 32 ms).
  - **B:** misses at drive bandwidth (2.36 ms a miss, 24.4 ms card+host). B fits the measured 16-row prefill passes better.

Decode only, 24 distinct prompts, continuous batching:

| N | ms a step (B / A) | aggregate tok/s (B / A) | a sequence, tok/s | misses a token | drive busy (B) |
|---|---|---|---|---|---|
| 1 | 70.9 / 70.7 | 14.1 / 14.1 | 14.1 | 12.5 | 46% |
| 2 | 116 / 108 | 17.2 / 18.5 | 8.6–9.3 | 13.1 | 59% |
| 4 | 207 / 183 | 19.4 / 21.9 | 4.8–5.5 | 15.1 | 75% |
| 8 | 429 / 364 | 19.0 / 22.3 | 2.3–2.7 | 18.3 | 86% |
| 16 | 938 / 778 | 17.3 / 20.8 | 1.1–1.3 | ~22 | 90% |

- **With prefill counted** (prefills serialized, 2.08 s a prompt): 64-token replies go from 9.7 (N=1) to 11.9–12.8 tok/s at N=4. 256-token replies go from 12.7 to 16.7–18.6.
- **What a step with N rows costs:**
  - Card fixed work amortizes: 24–32 ms a step plus about 0.4 ms a row.
  - Misses do not amortize and rise with N, because the sequences compete for the cache.
  - Different prompts rarely share file-tier experts: 88–98% of a step's file-tier units serve one row, and CPU work a token falls only 1–6%.
  - VRAM experts are shared more: distinct experts are 0.58 of N rows at N=16.
- **Sensitivity:**
  - 0.8 ms a row instead of 0.4: about −0.5%.
  - A 20 ms excess per multi-row step (like DSpark's unexplained one): N=4 drops to +26% (B).
  - A second NVMe: N=4 reaches 28–29 tok/s.
  - 256 GB of RAM: N=4 reaches 32–34, N=16 about 41.

## Bottlenecks and changes, ranked
1. **Done: the prefill-alone switch.** Without it there is no batching at all.
2. **KV sizing for N sequences.** `--max-model-len` works as a stopgap; a proper planner setting is the clean fix.
3. **Alternate prefill chunks with decode steps** (a follow-up, value 2 of the switch). Today every admission stalls all decoders for its prefill: about 2 s for a 40-token prompt, and roughly 100 s for a 2,000-token one. This is throughput-neutral but bounds the stall.
4. **Two-batch overlap.** Offset two micro-batches by half a layer so one's miss reads overlap the other's card work. Upper bound +15–35% on top at N=2–4; large effort.
5. **Decode rows beside a prompt's final chunk:** +1–2%.

Not worth it: sharing-aware job batching (nothing to share), cache policy tweaks (Sc3, Sc4 and T4 are equal or worse at every N; Belady's 57–62% is an oracle), passes over 16 rows (not exact, and N>8 does not pay on this drive), and retuning the mirror share.

## Prototype (branch `claude/ds41-conc`)
- **`ff482e2f`:** `RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE=1`.
  - When decode rows run and a chunk is due, the step carries that one chunk alone, cut exactly as an idle server cuts it. Decode rows of all sequences run together in the steps between.
  - If no chunk could be planned after all, the step is replanned as without the switch.
  - `step_next` issues nothing ahead of a step that left rows out.
  - Unset, the scheduler is unchanged.
  - New `sched_test` case passes (176/176): batching happens, every step is pure, chunks are 16, 16, 8 either way, nothing is issued ahead.
  - It cherry-picks cleanly onto decode-impl `54ff6dde`.
- **`8e32b661`:** new `ab_window.sh` arm parts: `Pf` (the switch), `NK` (this arm's `--max-num-seqs`), `CK` (this arm's AB_CONCURRENCY) and `LK` (this arm's `--max-model-len`). It conflicts with upstream's F8 parts, but trivially.
- The binary is built at `/home/as/ds41-claude-build/aql-conc/bin/radiance`; `abi/` is unchanged, so the home and home2 plugins work with it.

## Arms to run first
Window 1 (about 12 min; run under both research locks, with the parts above). With `B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2`:
```
/home/as/.claude/jobs/635603c1/tmp/research2/conc/launch_conc.sh conc1 "$B $B-Pf-N2-C2 $B-Pf-N4-C4 $B-Pf-N8-C8-L1024 $B-Pf-N4-C4-r2" MIRROR=/srv/bulk/source-copies/ds41-native-mxfp4/d8b2e3c20f5f4b6d5044d37f20b8d53f260f83e466f72259572be76d7e756bf3-main
```
`conc_sum.py` runs at the end of the window. It prints identity against the night's base engine, agg/s (tokens over wall time), rows per decode step, ms per decode step, steps-based tok/s, prefill steps and mixed steps.

Gates:
- **Identity: kill on any difference.** Every reply 8/8 identical, 0 preemptions, 0 mixed steps, 23 prefill steps. If identity fails, run `base` against `base-Pf-N4-C4` to separate the card kernels from the CPU path.
- **Batching:** rows per step of at least 3.5 at C4 (model 3.82).
- **Go:** at C4, step-based tok/s ≥ 19 and agg/s ≥ 11.6 (model 19.6–24.6 and 11.9–13.5, against 15.3 and 10.1 serial).
- **Kill or profile:** if step-based tok/s < 17 or agg/s < 10.7, profile one C4 arm with `o`.
- **Choice of N:** if C8 is no better than C4, use `--max-num-seqs 4`.

Window 2, if window 1 passes:
- `AB_MAX_TOKENS=256` with `$B $B-Pf-N4-C4-L2048` (model: +32–47% including prefill).
- `PROMPTS=/home/as/ds41-claude-build/work/extra-prompts/chat-receipt.json AB_REQUESTS=16` with `$B $B-Pf-N8-C8-L1024 $B-Pf-N16-C16-L2048`, to measure the curve at N=8 and 16 with distinct prompts.

Everything is in `/home/as/.claude/jobs/635603c1/tmp/research2/conc/`; the main entry points are `model.py`, `sched_sim.py`, `conc_sum.py`, `launch_conc.sh` and `build_conc.log`.

# concurrency: adversarial assessment

**Concurrency track (`--max-num-seqs N` with the prefill-alone switch): adversarial assessment**

## Verdict
The diagnosis is correct, the switch is correct for the A/B, and the trace replay is sound. The claimed gain is too high at the top end:

- **Calibration A is not physical.** It prices one 18.8 MB miss at 1.9 ms, which is 9.9 GB/s. The NVMe and the SATA mirror together deliver about 7.5–8 GB/s, and a lone miss was measured at 2.5–2.7 ms. Calibration B fits both measurements, so the "+55%" figure from A is not a valid upper bound.
- **Corrected gain at N=4 against N=1:**
  - decode-only aggregate: **+20 to +38%, central +32%**;
  - end to end on the 64-token held-out set: **+13 to +20%**;
  - each stream drops to 4.6–5.2 tok/s, or 3.6–4.3 tok/s counting the stalls while other requests prefill.
- **The same ratios hold on today's FP8 base.**
- **Scope.** This is exact and only raises aggregate throughput when 2 or more requests are in flight. It does not raise the single-stream rate the night optimized. If "parallelizing experts" meant running some experts on the CPU while the GPU moves on (KTransformers' expert deferral), that idea is lossy and already in the lossy track (`research2/lossy/lossy_ab.py`).

Re-runs and my scripts are in `/home/as/.claude/jobs/635603c1/tmp/research2/conc-assess/`: `assess.py` (model with the missing terms, called B′ below, and the FP8 placement), `tbo_sim.py` (event simulation of two pipelined lanes), `gate_pred.py` (what the trace arm should see) and `run_guarded.sh`. I did not modify any worktree or use the GPUs, and the heavy replays waited for GPU windows as the load rules require.

## What I verified (wt-conc @8e32b661)
- **Today's serial service at N>1 is real.**
  - `admit_one` refuses an admission when the budget is under both the 16-token quantum and the remaining prompt (scheduler.cpp:1267).
  - `hx_init` refuses a batch budget over 16 tokens (stager.cpp:944).
  - This matches the 05:33 window's time-to-first-token values.
- **The switch behaves as described.**
  - It is inert at `--max-num-seqs 1`.
  - The automatic chunk cap never cuts a chunk below the quantum (scheduler.cpp:1185), so prompts are still cut 16/16/8.
  - It replans only when an admission is refused.
  - ff482e2f applies cleanly to 54ff6dde (checked with a scratch index). 8e32b661 conflicts there, as the researcher said.
- **Batch invariance up to 16 rows holds in the design.**
  - The bf16 m64 GEMM forces MB=1 in the decode band, so its split does not depend on M (r4d_gemm_args.h:132). There is no tune cache in home or home2, and the FP8 kernel reuses this body.
  - Sparse attention and the indexer stay on their scalar forms up to 48 rows (r4d_ds41_sparse.hip:1227).
  - The one-shot and two-shot all-reduces sum in the same rank order (r4d_ar_entry.hip:965).
  - The gather sums per token, in slot order.
  - Key quantization works on 32-value blocks within a row.
  - The attention prefetch still runs for N-row decode passes (deepseek41.cpp:2591).
- **The replay reproduces the engine.**
  - Rank 0 decode misses: 8,624 replayed against 8,624 recorded. Ranks 1–3 are within 0.1%.
  - Every table and harness figure the researcher reported reproduces exactly.
  - At N=4, 97.7% of file-tier experts serve a single row.
  - The best cache-policy variant cuts misses by only 0.9% at N=4, so I agree cache policy is not worth pursuing.

## Errors and missed risks
1. **Calibration A** is non-physical, as above. Drop it as the upper end of the range.
2. **Unmodelled costs that grow with N.** Each is zero at N=1, so the calibration still holds:
   - more miss DMA and cache-insert traffic slows the CPU pool (×1.04 to ×1.10);
   - card cost per extra row is 0.6 ms, not 0.4 (attention goes from 17 to 44 µs a call between 1 and 4 rows);
   - host decisions add about 0.3 ms per extra row;
   - cold n-gram page faults on a busier NVMe add about 0.4 ms per extra row (the table is on disk);
   - single-miss layers in multi-row passes get the prefill mirror share (item 3).

   With these terms, aggregate decode against 14.1 tok/s at N=1:

   | N | tok/s | Gain |
   |---|---|---|
   | 2 | 16.4 | +16% |
   | 3 | 17.9 | +27% |
   | 4 | 18.6 | +32% |
   | 8 | 18.5 | +31% |

   If multi-row passes carry a 20 ms excess like DSpark's verify passes, N=2 gains nothing and N=4 gains +21%.
3. **The multi-row settings do not suit 2–3 row decode passes.** The mirror share and async misses are chosen by row count (mover.cpp:926, stager.cpp:1415).
   - So a 2-row decode pass gets the prefill mirror share of 0.18, although its miss layers mostly have one miss (26.4 misses in 19 layers at N=2).
   - For lone misses, 0.18 measured about −4% against 0.10–0.13. The DSpark arms used min-rows 7 to avoid exactly this.
   - Fix: pick the share from the number of miss reads in flight (2 or more → 0.18). Worth +1.5 to +3.5% at N=2–4.
4. **The exactness evidence is thinner than "designed in" suggests.**
   - No pass with rows from different sequences has ever run on ds41 in this engine. The 05:33 window served them one by one, and DSpark's multi-row passes were one sequence and checked on text only.
   - The per-sequence paths (compressor state and block tables, Engram n-gram windows) run for the first time.
   - 8 prompts × 64 greedy tokens rarely flip on a one-bit logit change, so text identity is a weak test.
   - The server can return logprobs (oai.cpp:1777) from a one-block-per-row, fixed-order kernel. Each logprob includes the log-sum-exp over the whole vocabulary, so exact logprob equality between the serial and N-row arms is a near-complete bit test.
5. **The base has moved.** FP8 measured at 14:13 (ab-20261029T135754Z): 16.97/16.94 tok/s against 15.11/14.98, replies identical, 107 more VRAM experts a rank. The conc binary is pre-FP8. On the FP8 placement (with 2.8 ms less card time, which reproduces the measured ratio), the model gives batched N=4 at 20.6 tok/s against 15.6. On the held-out waves at C4 that is 23.5 decode tok/s and 13.8 tok/s overall, against 17.5 and 11.5 serial.
6. **`--max-model-len` confounds the N8 arm.** The indexer's score and top-k kernels scan the whole plane, max_ctx/ratio + 1 wide (deepseek41.cpp:894; r4d_ds41_sparse.hip:1592 and 1695), not the live context. So the L1024 arm pays extra indexer work that has nothing to do with N. The effect is small, but every arm in a sweep, the serial one included, should use the same value. The KV pool at 512 is 680 tokens (1.3 sessions), not 494, so four held-out requests fit.
7. **Item 3 needs no new code.** The existing `--prefill-interleave N` flag already works with the switch:
   - a held prefill turns prefill-alone off (scheduler.cpp:1636);
   - the held chunks are skipped (scheduler.cpp:1672);
   - the hold re-arms after each non-final chunk (scheduler.cpp:1999).

   It limits each stall to one ~0.7 s chunk, but it lowers aggregate throughput when requests arrive in bursts.
8. **The gain depends on the workload.** Prefill (about 53 ms a prompt token) is untouched by batching. End-to-end gain at N=4 by prompt/reply length:

   | Prompt tokens | Reply tokens | Gain |
   |---|---|---|
   | 40 | 64 | +19% |
   | 100 | 1000 | +29% |
   | 2000 | 256 | +3% |
9. **Smaller points.**
   - Issuing the next step ahead is refused whenever a request waits (scheduler.cpp:2193), so under a standing queue every step loses it, about 1–3 ms a step.
   - The gates are absolute, but today's serial base ran 15.0 tok/s against the 15.5 they assume. They should be relative to the same window's serial arm.

## Verdict per proposal
| Proposal | Verdict | Claimed | Corrected |
|---|---|---|---|
| Diagnosis: N>1 is served serially | strong | — | confirmed |
| 1. Prefill-alone switch | strong as a fix; gain overstated | +38–55% decode, +16–32% end to end | +20–38% decode (central +32%); +13–20% end to end (64-token), about +27% (256-token). N=3 gives 84% of the N=4 gain at 6.0–6.6 tok/s a stream |
| 2. KV sizing via `--max-model-len` | plausible, trivial | — | fits; keep the value equal across arms |
| 3. Alternate prefill chunks with decode steps | strong as a latency fix; already exists | throughput-neutral | 0 to −5% overall on bursts; bounds the stalls |
| 4. Two-batch overlap | plausible, likely the strongest long-term option | +15–35% on top | see below |
| 5. Decode rows beside a prompt's final chunk | weak | +1–2% | 0 to +1% |
| Not worth: sharing-aware batching, cache policies, passes over 16 rows, N>8 | agree | — | — |
| Not worth: retuning the mirror share | partly refuted | — | +1.5–3.5% at N=2–4 from the in-flight rule |
| Hardware scenarios (second NVMe, 256 GB) | plausible direction; top values inherit A | — | — |

**Two-batch overlap, from my event simulation** (idealized, calibration B; with one stream it reproduces the batched model exactly: 14.1 / 17.2 / 19.4):

| Lanes × rows | Aggregate tok/s | Per stream | Busy: card / CPU / drive | On the FP8 base |
|---|---|---|---|---|
| 2 × 1 | 21.8 | 10.9 | 56% / 52% / 75% | 24.4 |
| 2 × 2 | 23.2 | 5.8 | drive 89% | 25.9 |

- Two 1-row lanes beat batched N=2 by 27% and batched N=4 by 12%, while each stream runs 2.2× faster than at batched N=4. Expect perhaps 60–80% of the simulated gain in practice.
- With 1-row lanes every pass is an ordinary N=1 decode pass, so the replies are exact by construction and the multi-row risks above do not apply.
- The effort is large. Each rank needs two issue lanes, each with its own stream, all-reduce instance, activation rows and file stager. The CPU pool, RAM cache and readers are already shared. It must not add a fourth hardware queue per card, which costs about 13 ms a step (tierexec.cpp:258).

## Recommendation, ranked
1. **Window 1, revised (about 14 minutes, under both locks).**
   - Preferably on the FP8 base: cherry-pick ff482e2f onto 54ff6dde, merge the 8e32b661 arm parts by hand, rebuild.
   - Make two small harness changes: send `"logprobs": 5` in `ab_requests.py` and compare logprobs exactly in `conc_sum.py`.
   - Arms, with `B=all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2[-F8]`:
     ```
     $B-L1024  $B-L1024-Pf-N4-C4-x  $B-L1024-Pf-N2-C2  $B-L1024-Pf-N4-C4-Rm5  $B-L1024-Pf-N8-C8  $B-L1024-Pf-N4-C4-r2
     ```
   - Gates, measured against the serial arm in the same window:

     | Gate | Condition | Model expectation |
     |---|---|---|
     | Kill | any text difference from the night's base or any logprob difference from the serial arm; or any preemption, any mixed step, or a prefill step count other than 23 | — |
     | Rows a step at C4 | at least 3.5 | 3.82 |
     | Misses a C4 step (trace arm `-x`) | within ±10% of 48.7 base / 41.4 FP8 (trace miss weights ÷ 2, summed over ranks); otherwise recalibrate before trusting any N>4 or two-lane figure | 48.7 / 41.4 |
     | Go | decode tok/s at C4 at least 1.22× serial, and overall tok/s at least 1.13× (the researcher's 19 / 11.6, made relative) | 1.34× / 1.20×; 1.21× / 1.13× with a 20 ms excess |
     | Profile (`o` arm at C4) | below 1.10× or below 1.03× | — |
     | Mirror rule | `Rm5` at least 2% better | — |
     | Choice of N | if C8 is within 5% of C4, cap N at 4, or use 3 when per-stream speed matters | — |
2. **If window 1 passes:** add the in-flight mirror-share rule (small). Run a `--prefill-interleave 1` arm to check identity and stalls; showing its effect needs staggered arrivals in `ab_requests.py`. Size the KV pool for the production N.
3. **Two-lane pipelining, only if multi-client use is real** and window 1 matches the model within ±10% at N=2 and N=4.
   - First, read the per-layer busy time at N=1 from existing counters; the simulation says card, CPU and drive are only 36%, 34% and 46% busy.
   - Then do a design spike, and gate it on a two-lane HOST-backend test with identical outputs before a GPU window comparing 2×1 lanes with batched N=2 and N=4.

Sources:
- [MoE-Lightning (CPU-GPU-I/O pipelining across micro-batches)](https://arxiv.org/abs/2411.11217)
- [MoE-Lens (critique of MoE-Lightning's GPU utilization)](https://arxiv.org/html/2504.09345v1)
- [SGLang two-batch overlap: −27% at 32 tokens a device, +25.5% at 256](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
- [DeepSeek V3/R1 inference system (dual micro-batch decode pipeline)](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- [Thinking Machines on batch invariance](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [KTransformers expert deferral (lossy)](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/)
