# DeepSeek V4.1 Decode Plan

How to turn the review's findings into working speedups on this machine: the order of work, what each step changes, how each step is proven, and where you need to decide.

- Date: 2026-10-28
- Based on: `REVIEW.md` in this folder (round 2)
- Code: `/home/as/radiance-ds41-development`, branch `dev/ds41-bank-head-current-20261028`, HEAD `a86b587e`
- Runs and receipts: `/home/as/radiance-setup/research/deepseek-v41-1017-20261024`
- Sizes: S is 1–3 days, M is 1–2 weeks, L is 2–4 weeks of focused work. Your agent loop may move faster; the order matters more than the dates.

The plan keeps every output bit-identical unless a step says otherwise. Each change lands default-off behind a flag, is proven on HOST first where it can be, and becomes the default only after a model A/B with identical token IDs.

## Progress, 29 Oct 2026 (overnight)

Decode on the loop's eight held-out prompts went from 5.03 to about 12 tok/s (2.4×), and time to first token summed over them from 64.0 to 19.4 s (−70%), with every reply byte-identical to the base engine's (also on 16 new chat-framed prompts and on the eight repeated three times). The measured steps, the configuration to run and what is left are on the results page, [DS41 Overnight Results](https://claude.ai/artifact/Y2mdhzJzGnwaK5zBFJRm7s); every window and its arms are in `NIGHT-LOG.md` in this folder. Branch `claude/ds41-decode-impl`, HEAD `bbf48615`, local only.

| Item | Result |
|---|---|
| Phase 2, host-resident experts on the CPU | Done. Every expert the RAM cache holds goes to the CPU (policy `all`); the hybrid CPU/link scheduler measured no better. |
| P1.2, CPU off the miss path | Superseded: the CPU computes misses from the container's planes, with no relayout or copy on the critical path (decode +49%). |
| P2.6, lend the decode file buffers | Done as a cap on the VRAM file buffers once the CPU takes every routed unit: 176 more VRAM experts a rank (+9.5%). |
| Phase 3, lookahead | Predictor built and measured: right 78% of the time for its top pick and 38% across its top 6. A one-layer lead is shorter than a drive read, so read-ahead stays off. |
| Phase 4, prefill | libr4d GEMV whole sub-passes (prefill +31%), prefill misses on the CPU, the plane mirror on /srv/bulk (TTFT −11%), a cache warmed at startup (first request's TTFT 17.5 → 4.6 s). |
| Not in the plan | The pinned host pool given to a 96 GiB RAM cache (+9%), slab pre-grow, a fix for read windows whose cache leases blocked every admission, 40 CPU threads using the SMT siblings (+4%), and prefill misses read while the card's ops are issued (TTFT −25%). |

The targets table below is the plan's estimate from 28 Oct; the measured result is above Phase 3's estimate without any lookahead.

Three results from the morning of 29 Oct:

- A CPU job of several token rows now decodes each MXFP4 round once for all of them (`bbf48615`). Jobs of 2–4 rows are 10–17% cheaper, and the replies are identical; decode is unchanged.
- Moving the shared expert ahead of the routed gather was already done: the block issues it before the gather, so it runs while the card waits for CPU rows.
- Speculative decoding was re-estimated from the night's traces. Even with the CPU path it comes out at 0.86–1.2× of today's decode (`NEXT-STEPS.md` section 3).

## Progress, 28 Oct 2026

Work is on branch `claude/ds41-decode-impl` in `/home/as/radiance-ds41-claude`, a separate clone that shares your repository's objects read-only. Your dev checkout and the research directory are untouched, and nothing is pushed. GPU logs: `/home/as/radiance-ds41-claude/tools/ds41-host-experts/build/window-20261028T203510Z/`.

| Item | Result |
|---|---|
| P2.2 bit-exact CPU expert chain | Proven on the card against your measured `libr4d.so`: 24 trials, 48 expert rows, zero differing values in gate/up, SwiGLU, down and the host-only chain, edge cases included. Scalar and AVX2 agree on llama.cpp blocks and on the release's planes. |
| SwiGLU, exhaustive | 2.2 billion (gate, up) pairs; 8,257,410 differ. That matches 126 signaling-NaN `up` patterns × 65,536 gates, inputs the GEMV can never produce because it quiets every NaN it writes. To be confirmed with signaling NaNs excluded. |
| P2.3 CPU throughput | 38.8 GB/s at 16 threads and 43.9 at 20 after prefetching (from 26.4 and 31.2). Compute-bound at about 3.4 GB/s per thread. Below the 50 GB/s gate. |
| P1.1 GEMV work split | Bit-identical; gate/up −14% and down −25% at one live expert, about +3% at three. The decode GEMV takes 33–97 µs per call, about 0.1 ms per layer rather than the 0.7 ms the review assumed, so the whole change is worth about 0.2% of decode. |

What changes:
- Phase 2 goes ahead with a hybrid scheduler instead of waiting for the 50 GB/s gate. Bit-exactness lets each RAM-resident expert go to the CPU pool or to its rank's copy link, whichever is free sooner. Model at the measured 44 GB/s: +24% (scaled) to +56%; with 50% prefetch, +58% to +138%.
- Two kernel optimizations remain open: transposing two blocks per 256-bit register and computing the scales in vectors. Using SMT siblings for the CPU pool is untested.
- P1.1 and P1.3 move to "Not planned": the decode GEMV is not a bottleneck.
- My 3-second memory-bandwidth test (20:04:04–20:04:07 UTC) overlapped the `cap-1` arm of your loop's fixed-verify sweep (about 20:00–20:05). Treat that arm's timings with caution. Every heavy step since holds `research-compiler.lock` then `research-heavy-runtime.lock`.

## Targets

Estimates come from the review's cost model and are indicative only. Each milestone is measured against the one before it.

| Milestone | Decode, tok/s | What limits it next |
|---|---|---|
| Today (Tail1 configuration) | 5.4 | one layer at a time: sync, stage, copy, compute |
| Phase 1 done | ≈6.0–6.8 | copying host-resident experts to the cards, and the busiest rank |
| Phase 2 done (hybrid) | ≈6.7–8.4 | NVMe misses on the critical path |
| Phase 3 done | up to ≈9–10 | drive bandwidth: ≈9.3 tok/s at 1.5× read amplification |
| With 256 GB RAM (hardware) | ≈13–20; ≈19–27 with DDR4-3200 | CPU memory bandwidth and GPU work |

Prefill has no long-prompt baseline yet; Phase 4 starts by measuring one.

## Phase 0: Foundations

Nothing later can be judged without these. Most of it is small, and P0.4 starts the measurement that decides Phase 3.

| ID | Work | Size | Needs a GPU window | Depends on |
|---|---|---|---|---|
| P0.1 | Build and pin a baseline from the dev branch; turn the best run's environment switches into flags | S | one sweep | |
| P0.2 | Phase-split and per-layer counters | S–M | with P0.1 | P0.1 |
| P0.3 | Runtime-switchable flags and an A/B/A/B harness, plus a long-prompt set | M | one sweep | P0.1 |
| P0.4 | Predictor shadow mode | M | one sweep | P0.2 |
| P0.5 | NVMe and CPU power A/B | S | one sweep | the poll-based harness fix |

### P0.1 Baseline

- Finish the pending incremental validation of `a86b587e` and build it. The best runs so far used frozen builds with environment switches that the branch no longer reads (`RADIANCE_EXPERIMENTAL_DS41_M1_DIRECT_ACTIVATION`, `RADIANCE_EXPERIMENTAL_DS41_MXFP4_PERM_DECODE`), so measured binaries and source have drifted apart.
- Turn the switches the Tail1 run set into command-line flags in `core/config.cpp`: `NATIVE_CACHE_MAPPED_SLABS`, `NATIVE_CACHE_MAPPED_SOURCE`, `NATIVE_FILE_WORKERS`, `NATIVE_CACHE_MMAP_REUSE`, `NATIVE_CACHE_CONCURRENT_COPY`, `NATIVE_CACHE_PREFILL_READ_ONLY`, `DS41_LIVE_TAIL`, `DS41_MXFP4_PACK_SSE2`. Upstream removed its own environment switches, so this is also needed before anything goes upstream.
- Done when one binary and one command line reproduce about 5.4 tok/s on the eight held-out prompts with the same token IDs as Tail1.

### P0.2 Counters

- Split the slab cache and Mover counters by phase (prefill, decode): hits, misses, bytes, NVMe bytes, relayout time. Today's counters mix both, which is why the decode miss rate is only an estimate and the cost model's calibration factor is 0.42.
- Add cheap per-layer decode timings from host timestamps: time waiting on the counts sync, time in `stage_relaid`, time from the copy event to GEMM start, and all-reduce wait. Avoid `--profile-ops`, which adds syncs.
- Done when decode NVMe bytes per token from the counters agree with the OS read counters.

### P0.3 Faster A/B

- One A/B arm costs 7–10 minutes today, almost all fixed: stopping and restoring QFN, checks, and about 21 s of model startup.
- Let the flags under test change between requests through an admin call, so one model process can serve alternating arms. Where a flag can't change at runtime, run A, B, A, B as separate model processes inside a single QFN stop and restore.
- Every comparison: at least three alternations, identical token IDs, decode tok/s, prefill seconds, and the P0.2 counters.
- Add a long-prompt set (2K, 8K and 16K tokens) beside the eight held-out prompts, for Phase 4.

### P0.4 Predictor shadow mode

- On a side stream at layer L's router input, apply layer L+1's and L+2's routers (sqrt-softplus plus the correction bias) and write their top 12 to pinned memory. The `xp_logits` and `xp_plan` kernels in `core/runtime/xprefetch.*` show the shape; the DeepSeek router needs its own scoring.
- At the next counts sync, compare predictions with the real routes. Report recall at 6, 8 and 12, the share of real misses that were predicted, and how many predicted experts are non-resident (the reads a prefetcher would issue).
- No reads and no change to outputs. This measurement decides whether Phase 3 happens.

### P0.5 Power settings

- Today the drive may enter sleep states with up to 100 ms exit latency (`default_ps_max_latency_us=100000`), the governor is `schedutil`, and decode leaves the drive idle while the GPU computes.
- A/B: drive PM QoS latency tolerance 0 plus the `performance` governor, against the current settings. Use the `select.poll` fix already written up in `DS41_NVME_POWER_HIGH_FD_CORRECTION_V2_PLAN_20261028.txt`; the first attempt failed before measuring anything.
- If it wins, set it per run in the window scripts. A persistent system change is your decision.

Gate 0: the baseline reproduces, the counters agree with the OS, and predictor recall is measured.

## Phase 1: Quick wins

| ID | Work | Size | Needs a GPU window | Depends on |
|---|---|---|---|---|
| P1.1 | MoE GEMV work split | S–M | numeric check and A/B | P0.3 |
| P1.2 | CPU off the miss path | M | after HOST tests | P0.3 |
| P1.3 | Sparse GEMV on by default | S | numeric screen and A/B | |

### P1.1 GEMV work split (review #2)

- Today `libr4d/r4d_ds41_mxfp4.hip:791-819` sizes the single-token grid for 6 expert slots, but a rank usually has about 1.5 live experts, so about a quarter of the blocks do work.
- After the device-side live count, spread `n_live × nrg` row groups evenly over the launched blocks. hipEngine's `tile_map` kernels show a static-grid version. Keep one wave per output row in the same order, which keeps outputs bit-identical.
- Proof: the existing MXFP4 numeric checker at M = 1–16, all 96 experts, zero local experts, and live-tail shapes; a kbench timing; then a model A/B.

### P1.2 CPU off the miss path (review #1)

- Add `reserve`, `commit` and `abort` to the slab cache (`core/place/mapped_expert_slab_cache.h`, `core/place/native_mapped_file_cache.h`). On a miss, pack straight into the reserved slot and copy to the card from the slot, as hits already do. Keep victim choice and LRU order as they are.
- Do not hold `writer_mutex_` across any payload copy (`mapped_expert_slab_cache.h:238` and `:290`).
- In `read_relaid` and `stage_relaid` (`core/place/mover.cpp:806`, `:984`), split each read into 2–4 MB row ranges and pack range k while range k+1 is in flight.
- Keep the window path as the fallback when no slot is free.
- Proof: HOST tests showing identical bytes, hits, misses and victim order against the current path, a TSan run for the new concurrency, then a model A/B. Prefill gains too, because several misses per window no longer serialize.

### P1.3 Sparse GEMV

- Finish the broader numeric screen of the existing sparse-GEMV variant (measured −44% to −57% VRAM event interval at 64 rows) and turn it on if token IDs stay identical. Do this after P1.1, since both touch the same kernel.

Gate 1: each change shows a gain beyond the spread of its A arms, with identical token IDs.

## Phase 2: Host-resident experts on the CPU (review #7)

This is the largest single gain in the model and the biggest piece of work. It starts with HOST-only steps that can run in parallel with Phases 0 and 1, and it has a go/no-go gate before any integration.

| ID | Work | Size | Needs a GPU window | Depends on |
|---|---|---|---|---|
| P2.1 | Write down the exact operation order | S | | |
| P2.2 | AVX2 kernels for both layouts, proven bit-identical | M | one capture window | P2.1 |
| P2.3 | Throughput gate | S | | P2.2 |
| P2.4 | Integration prototype | L | yes | P2.3, P1.2 |
| P2.5 | A/B and tuning | M | yes | P2.4 |
| P2.6 | Lend the decode file buffers to experts (review #4) | M | yes | P2.5 |

### P2.1 Exact contract

Specify, from the kernels the baseline actually runs, every FP32 operation of the decode expert chain:

- gate/up GEMV: dequantize `kv[nibble] × 2^(e−128)` (exact), multiply by the BF16 activation widened to FP32 without FMA, accumulate per lane over blocks `lane, lane+32, lane+64, …` in `mx_row`'s order, with the in-block order of `mx_unit`, then close with `r4d_wave_sum`'s butterfly;
- clamped SwiGLU, including where the routing weight is applied and where values round to BF16 (check the DeepSeek graph, not the GLM path);
- down GEMV, the same way.

Pin the libr4d build these were read from, so a later kernel change shows up as a contract change.

### P2.2 CPU kernels

- AVX2 kernels for two inputs: ggml 17-byte blocks, for cache hits, and the release's planar codes plus E8M0 scales, for fresh misses. Unaligned loads are cheap on the CPU, so the 17-byte layout is not a problem there.
- Reproduce the contract exactly: 32 lane partials per row in AVX2 registers, the same block order, no FMA, the same butterfly.
- Proof: bit-identical to GPU outputs captured once in a GPU window on real expert rows, and to the GPU on hand-made cases (scales 0, 1 and 255, subnormals, signed zero, cancellation, NaN and Inf).

### P2.3 Throughput gate

- On HOST with real 18.8 MB experts: 8, 12, 16 and 20 threads, pinned per CCD (cores 0–5 and 24–29 share an L3, and so on).
- Go if 16 threads sustain at least 50 GB/s of expert weights; plain reads measured 68 GB/s at 16 threads. Otherwise stop here and keep the GPU path.

### P2.4 Integration prototype

- At the existing counts sync (`core/place/stager.cpp:645-647`), also copy the layer's normed activation row to the host. That is about 10 KB per token, and every rank holds the same row.
- Add one process-wide pool of CPU expert workers, shared by the four rank threads. Each rank submits its selected experts that are not in VRAM (pinned pool, slab hits and misses) and waits for its own results.
- The GPU computes the VRAM experts meanwhile. Skip host experts in the GPU GEMMs with a per-expert mask in the live-range code (`libr4d/r4d_ds41_live_tail_policy.h`, `range()`). For a first prototype, pointing them at a dummy alias and overwriting the rows also works.
- Copy each CPU result row into the rank's sorted expert-output buffer before `moe_gather`. The gather and all-reduce stay unchanged, which is what keeps outputs bit-identical.
- Misses: an O_DIRECT read of the planes, computed directly from the planar buffer, then packed into a slab slot off the critical path for later hits and for prefill.
- Large prefill chunks stay on the cards. Fiddler's rule, choosing CPU or GPU per expert by how many rows it serves, fits here.

### P2.5 A/B and tuning

- Identical token IDs on the eight held-out prompts and the long-prompt set; at least three alternations against the Phase 1 build.
- Tune worker count and placement so the four issuing threads and the NVMe readers keep their own cores.
- Gate 2: at least +15% decode over Phase 1, with identical IDs. Otherwise leave it off.

### P2.6 Lend the decode file buffers

- Once host experts no longer travel to the cards in decode, the two 1.61 GiB file buffers per card matter only for prefill.
- Lend them to about 170 more hot experts per card during decode and recall them before a prefill or full-layer read. Today's lending covers only pinned-pool units, so slab-backed units need a new residency state.
- Held-out upper bound: +16% of decode selections move into VRAM.

## Phase 3: Lookahead into RAM (review #3)

Start only if P0.4 shows useful recall one layer ahead, about 50% or better at 1.5× read amplification or less. As layers get faster, one layer of lead shrinks below the ≈2.9 ms an 18.8 MB read needs, so P0.4 also measures two layers ahead.

| ID | Work | Size | Needs a GPU window | Depends on |
|---|---|---|---|---|
| P3.1 | Background NVMe → slab prefetcher | M–L | yes | P0.4, P1.2 |
| P3.2 | Bandwidth budget | S | with P3.1 | P3.1 |
| P3.3 | A/B | S | yes | P3.2 |

- P3.1: read predicted experts that are not in VRAM, the pinned pool or the slab cache into reserved slots (P1.2's API). Demand reads always go first; cap reads per layer; check the file epoch before publishing; cancel at the end of a pass.
- P3.2: stop prefetching when drive utilization or wasted bytes pass a threshold, and report useful against wasted bytes per token.
- Gate 3: at least +10% decode with identical IDs.

## Phase 4: Prefill and time to first token

A parallel track; it does not block Phases 1–3.

| ID | Work | Size | Needs a GPU window | Depends on |
|---|---|---|---|---|
| P4.1 | Long-prompt baseline: 2K, 8K, 16K | S | yes | P0.3 |
| P4.2 | Large chunks for long prompts | M | yes | P4.1 |
| P4.3 | Exact decoder suffix (CED) | L | yes | P4.1 |
| P4.4 | Prefix reuse across turns | M | yes | |

- P4.2: above a prompt length to be measured, switch to chunks of 1,024–2,048 rows that stream whole layers with read-ahead and read-only admission. Check activation and KV memory at that size first.
- P4.3: follow your CED notes. Layers 21–39 need only the last ≈2,560 prompt positions, but the scheduler's single commit cursor must first be split into encoder and global-key completion plus decoder-window validity (`DS41_CED_EXACT_REPLAY_CRITIQUE_AND_MINIMUM_SEQUENCE_20261028.txt`). Prove the last-row logits and the following decode tokens identical.
- P4.4: validate upstream PR301 with this model's mixed KV groups, so a second turn does not prefill the conversation again.

## Not planned unless you decide

| Option | Why it waits | What would change that |
|---|---|---|
| GEMV work split (P1.1) and sparse GEMV (P1.3) | Measured: the decode GEMV is about 0.1 ms per layer, and the split is worth about 0.2% of decode | Many more live experts per rank, as in verification passes |
| TP-sliced experts (review #5) | Changes rounding; after Phase 2 the busiest-rank problem remains only for VRAM experts | Counters showing that VRAM-side imbalance still matters |
| Planar layout in the GPU kernels (review #6) | Phase 2 reads planar data on the CPU already | The two-layout split becoming a maintenance burden |
| Rebalancing experts across ranks | Phase 2 replaces it | |
| DSpark/MTP and prompt lookup | Slower at exact routing here and in DS41Engine. Re-estimated on 29 Oct with the CPU path at 0.86–1.2×: a verify pass of k tokens needs 0.70–0.87 of the experts of k steps, and each extra CPU row costs about 0.4 of a row | Acceptance well above DSpark's 3.8 tokens a pass, or different hardware |
| 256 GB RAM | Hardware cost | Your budget; the model's biggest single lever |
| A second NVMe drive | Hardware cost | Doubles the drive ceiling |
| A lower-bit model such as UD-IQ1_S | Changes quality | A KLD and accuracy comparison on your own tasks |

## Decisions for you

1. Is a hardware change possible? Eight 32 GB DDR4-3200 modules (check the board's memory support list first) or a second NVMe drive change the targets more than any software step.
2. Are changes that alter rounding acceptable? If not, TP slicing stays off the table.
3. Should a lower-bit model get a quality comparison?
4. Who runs this: the existing Root and agent loop? The work packages are shaped for it: agents write source, HOST tests and reviews; Root runs builds, HOST suites, GPU windows and commits.

## Risks

- CPU contention: Phase 2 workers, four issuing threads and the NVMe readers share 24 cores. Pin by CCD and measure with P0.2's timings.
- Memory bandwidth: CPU experts, NVMe writes and prefill copies share about 80 GB/s of DRAM bandwidth.
- Exactness drift: a libr4d kernel change silently breaks the CPU kernels' bit-identity. Pin the contract to a kernel checksum and rerun P2.2's proof on every build.
- Noise: single sweeps cannot resolve a few percent. Keep at least three alternations per comparison.
- Source and binary drift: P0.1 fixes it; keep measured builds on the branch afterwards.

## Critical path

1. Weeks 1–2: Phase 0. P2.1 and P2.2 start in parallel on HOST.
2. Weeks 2–4: Phase 1.
3. Weeks 4–8: P2.3 through P2.5, then P2.6.
4. Weeks 7–9: Phase 3, if Gate 0 recall allows.
5. From week 3, in parallel: Phase 4.
