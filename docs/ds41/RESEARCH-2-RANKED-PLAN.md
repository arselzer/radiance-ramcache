# DS41 decode: ranked plan for the next optimizations

*29 Oct 2026, 11:15 UTC.*

- Repo paths are relative to `/home/as/radiance-ds41-claude`: branch `claude/ds41-decode-impl`, HEAD `25973b49`, plus the uncommitted engram split and DSpark/hx fixes.
- A/B windows are in `tools/ds41-host-experts/build/`. `research/...` means `/home/as/.claude/jobs/635603c1/tmp/research/`.
- "Decode" is tok/s over the 8 held-out prompts, 64 greedy tokens each. Every arm cited gave 8/8 replies identical to the base engine.

## 0. Baseline and today's measurements

**The baseline is 13.5 tok/s (74 ms a token), not 12.3.**
- HEAD measured 13.42 / 13.54 (ab-20261029T100728Z) and 13.59 / 13.57 (ab-20261029T105844Z). TTFT sum is about 18.0 s.
- The brief's figure predates four changes: kept readers on 2 MiB pages (+4.6%), decode mirror share 0.13 (+0.6%), direct slot insert (+1.1%) and the embedding in host RAM (+1.5%).

**Window ab-20261029T105844Z** (it ran while this was written; I only read its logs):

| Arm | Decode | vs base | Notes |
|---|---|---|---|
| Base, 96 GiB cache (two run homes) | 13.59 / 13.57 | — | |
| **engram_kv rows split over the ranks (`Es`, uncommitted)** | **14.05** | **+3.4%** | TTFT 17.57 vs 18.07 s. Static weights 5.23 → 4.79 GiB. VRAM experts 573/590 → 598/615 a rank |
| DSpark bank loaded, drafting off (`Ds0`, 90 GiB cache) | 12.76 | −6.0% | The 90 GiB cut alone gave 12.95 / 13.08 (−3.4%; misses +8-10% a rank) in the 10:27 and 10:43 windows. The rest: 32 fewer VRAM experts a rank, static weights 5.80 GiB |
| DSpark, verify cap 1 / 2 / 3 | 10.76 / 11.07 / 11.12 | −21 / −18.5 / −18% | 1.80 / 2.27 / 2.58 tokens a pass at 167 / 205 / 232 ms a pass. 81 / 64 / 53% of verified drafts accepted |
| DSpark, adaptive window | 10.60 | −22% | 2.79 tokens a pass at 263 ms |

**Where a 74 ms token goes:**
- **~28 ms: drive-miss stalls.** 13.6 misses a token; in about 11 layers a token some rank reads the drive. The drive is about 43% busy.
- **~34 ms: card work and host overheads.**
  - Before each layer's counts sync there is a serial chain of 19 dispatches, about 15 ms a token.
  - The critical dense GEMVs are attention (7.4 ms) and engram (1.4 ms).
  - The shared expert's 2.3 ms hides under the CPU waits.
- **~9-12 ms: waiting for CPU rows.**

**Calibration and noise.**
- 1% fewer critical misses is worth about 0.37-0.48% decode (from the 102, 96 and 90 GiB arms).
- Same-config runs differ by ±0.5-1%, so anything under ~1% needs 3+ repeats or counters.

## 1. Ranked plan

Ranked by expected gain per day of work, then by how much of the gain is already measured.

| # | Item | Decode gain | Effort | Exactness | Next step |
|---|---|---|---|---|---|
| 1 | engram_kv row split; then router and compressor weights read as BF16 | +3.4% measured; router +0.8-1.2% | S | exact by construction | repeat, profile, commit |
| 2 | Keep the FP8-sourced dense weights as e4m3 + E8M0; rebuild the exact BF16 fragments inside the m64 GEMV | +7 to +14% on top of #1 (central ~10%) | M (3-5 d) | exact once all 610 tensors pass the bit check | VRAM-half arm (no code); full bit-check scan |
| 3 | Prefetch the next layer's attention weights into the 64 MB Infinity Cache | +1 to +3.5% (central ~2%) | S | exact (read-only) | queue count; 2-hour microbench |
| 4 | RAM cache in a CPU-only packed-scale layout; reclaim idle pinned staging | +2.5 to +4.5% (central 3.5%) | M | exact if verified | full scale scan (CPU only) |
| 5 | Two-ahead read-ahead that yields to demand reads, with a cheap predictor | +2.5 to +5.5% (central 4%) | M | exact (hints only) | 1-day prototype; one window |
| 6 | Card's serial chain: hazard-checked overlap on the live path, then fusions | +1.2 to +3.5% (central 2.3%) | M | exact if verified | dump-only arm + live stamps |
| 7 | Small exact items | about +2% together | S each | exact | gated arms |
| 8 | Stage the next layer's predicted cached experts in each CCD's L3 | +0.5 to +4% (central ~2%) | L | exact | quiet-box burst bench |
| 9 | DSpark: bring-up, verify-length control, drafter residency | −18 to −22% measured | M-L | text identical (32/32); logprobs unchecked | park it |

### #1 engram_kv row split, then BF16 router weights

**What.** Each rank holds and computes 6,400 of engram_kv's 25,600 rows (layers 1 and 14). A row-interleaved `all_gather` rebuilds `ekv`. Flag: `RADIANCE_EXPERIMENTAL_DS41_ENGRAM_SPLIT=1`, arm part `Es`.

**Why.** The weight had no shard (`arch/deepseek41/deepseek41.cpp:1028` before the diff). So every card stored 600 MiB and ran the same 315 MB GEMV: 711-719 µs a call, 1.43 ms a token, serial on all four ranks.

**Gain.** +3.4% and +3.5% against both same-window bases, one pair each. TTFT −2.8%, and +25 VRAM experts a rank. The assessed range was +2.7 to +3.5%.

**Exactness.** Exact by construction:
- `pick_sk` gives SK=1 at both 1,600 and 400 N-tiles for K=6144 (`libr4d/r4d_gemm_args.h:91-104`).
- Each wave owns a 16-column tile and walks K in the same order.
- The gather only copies bytes. Keep it a copy: a zero-padded all-reduce turns −0 into +0.
- Pin SK if a tune cache ever appears.

**Next.** Run two more pairs and one `--profile-ops` arm. Expect the N=6400 GEMM at ≤250 µs and the gather at ≤40 µs. Then commit.

**Then the router (S, +0.8-1.2%).**
- `glm_router.weight` is declared F32 (`arch/common/rad_block_glm_router.h:46`): 7.86 MB a layer, about 300 MiB a card, replicated. The release ships it as BF16 [384, 5120], and the low halves are all zero.
- A `gemm_nt_f32` form that loads BF16 and widens it exactly gives the same bits from half the bytes. Lane `l` must keep k = l, l+32, …, with fmaf and the same xor butterfly.
- The router is partly latency-bound: 19.7 µs at 399 GB/s, against 519 GB/s at N=1024. Expect about 13.5 µs, so 0.25-0.35 ms a token, plus 150 MiB freed (+8 experts).
- The three compressor kvg GEMMs (N=1024, also BF16 in the release) take the same change, for about +0.15%.
- Gate: a kbench at N=384, K=5120, M=1 must come in at ≤14 µs.
- `GlmRouter` is shared with GLM. Switch on the source dtype, and refuse any tensor with a nonzero low half.

### #2 FP8-sourced dense weights kept as FP8

**What.**
- 610 non-expert projections are exact BF16 casts of block FP8: e4m3 codes with one E8M0 scale per 32×32 block. The conversion receipt counts `fp8_e4m3->bf16 = 610`.
- They cover attention q_a/q_b/kv/o_a/o_b, indexer q_b, the shared expert and engram_kv.
- Store codes plus scales, which is half the bytes.
- A new libr4d gemm_nt row rebuilds the identical BF16 bits in registers and feeds them to the unchanged `r4d_gemm_bf16_nt_m64` body: same `pick_sk` SK, same k order, same fixed LDS split-K order.
- Decode already runs that body: M≤16 forwards to m64 (`libr4d/r4d_gemm_bf16_nt_m16.hip:14-16`).

**Why it pays here: two halves.**
- **VRAM (about two thirds of the gain).**
  - It frees about 1.85-2.0 GiB a card after #1, worth about +105-113 resident experts a rank.
  - The assessor's independent replay, starting from 573/590 experts: decode misses −13.2%, CPU hits −10.5%. It holds from a cold start (−11.3%), in steady state (−12.8%) and with a 102 GiB cache (−14.3%).
  - Calibrated on the measured resident and cache arms and scaled for #1: −3 to −4 ms of stalls and −1.4 to −1.9 ms of CPU waits a token.
- **Card.**
  - Critical-path dense bytes fall from 3.54 to 1.77 GiB a card a token. The ideal saving is 2.87 ms for attention plus about 0.18 ms for engram after #1.
  - Expect −1.6 to −3.1 ms. It could be 0 if decoding the codes makes the loop latency-bound; use KU=16 or b128 loads in fragment order to avoid that.

**Gain.** +7 to +14% on top of #1, central about 10%. The floor is about +6.5% if the FP8 row is no faster than BF16. The assessment gave +7 to +15% standalone.

**Exactness.** Exact by construction, provided three things hold:
- The load decodes every block again and refuses to load on any mismatch.
- The decoder handles −0 and e4m3 subnormals. Every checked tensor has 137-170 negative zeros and 3.5k-4.8k subnormal codes, so an integer formula for normal codes alone is wrong. Use `v_cvt_pk_f32_fp8`, multiply by 2^(s−127), then take the high halves (4 VALU ops per 2 codes).
- The row steps aside above M=64, because the tiled row reads the same stored bytes (`core/build/rad_builder.cpp:2197-2206`).

No release scales are needed: the smallest legal E8M0 per block can be derived from the container's BF16. Three whole tensors pass the bit check (1,328 strips); 607 remain.

**Blockers.**
- A new row with layout, relayout and unrelayout hooks (`gemm_w4a16_nt_m64` is the pattern).
- A flag in `lw()`, a load-time self-check and selftests.
- Build with the pinned clang 21.1.8 / HIP 7.1.
- No planner change: the builder stores the hook's byte count (`rad_builder.cpp:2240-2256`, `planner.cpp:89-92`).
- Production with longer contexts may want this VRAM for KV instead.

**First steps.**
1. **One window, no code.** Run the best arm (with `Es`) against the same arm at `--gpu-headroom-mib` ≈ 2,200, two alternations. `ab_window.sh:136` hard-codes 256, so make it a per-arm setting.
   - The planner should show about 110 fewer VRAM experts a rank.
   - Expect decode to drop 6-8%.
   - If it drops less than 3.5%, the VRAM half is overstated. The project then shrinks to its card half (about 3%), and #3 is the better buy.
2. **Outside windows, CPU only.** Run `research/fp8dense/certify.py` over all 610 tensors (about 12.7 GiB read). Any failing block narrows the plan.
3. **Only then build.**
   - Byte-compare against `gemm_bf16_nt_m64` at the 9 shapes × M = 1, 2, 4, 16, 17, 64.
   - kbench both rows in one window.

### #3 Infinity Cache prefetch of the next layer's attention weights

**What.**
- Declare per-layer `prefetch` ops (`decl_prefetch`) for the next layer's q_a, kv, q_b and o_a. That is 56 MB in BF16; after #2 the whole attention set (about 46 MB) fits.
- Issue them on lane 1 from a new hook after `sh_down` (`arch/common/rad_block_moe_fp8.h:1831`; the gather is at :1878).
- Decode passes only, gated the way gemma3 does it (`arch/gemma3/gemma3.cpp:1708`).
- Read in use order, including the compressor and indexer weights. In the Engram layers, start the read after the engram GEMM.

**Why.**
- The dense GEMVs fit t ≈ 6.5 µs + bytes/620 GB/s: no Infinity Cache hits today.
- The card idles about 325 µs a layer waiting for CPU rows while `hx_finish` blocks (`core/place/stager.cpp:1739`).
- deepseek41 declares no prefetch op.
- On the same die, libr4d's prefetch took a cold 29 MB layer from 84.0 to 70.1 µs (`libr4d/r4d_prefetch.hip:1-8`), and gemma3's prefetch gave +13% decode.
- Starting at `sh_down` gives a 40-55 µs head start even when there is no CPU gap.

**Gain.**
- 1.2-2.6 ms a token (+1.5 to +3.4%), assuming 1.0-2.0 TB/s of Infinity Cache bandwidth. No public figure exists for Navi 48.
- About +1% if the cache's replacement policy evicts the read itself.
- 1.0-2.1 ms after #2.

**Exactness.** The op only reads: an `asm volatile` sink, no stores. The decode-only gate avoids the prefill all-reduce race recorded at `rad_arch.h:630-632`.

**Risks.** Lane 1 must be the third hardware queue. KFD reports `num_cp_queues=4`, and a fourth queue costs about 13 ms a step (`core/runtime/tierexec.cpp:259-261`). A lane-0 variant would serialize.

**First steps.**
1. In the next window, while ds41 decodes, count the queues per card under `/sys/class/kfd/kfd/proc/<pid>/queues`. Expect 2.
2. A microbench of about 2 hours' work, extending `r4d_selftest`'s bf16_perf. Time the chain q_a → kv → q_b → o_a ×2 → o_b:
   - cold;
   - after a second-stream prefetch of 16 / 32 / 48 / 56 / 64 MB, with gaps of 0, 300 µs and 3 ms;
   - with back-to-back repeats, which give the cache hit rate.
3. Kill it if 48 MB saves under 20 µs a chain; go if it saves 35 µs or more.

### #4 Packed-scale CPU-only RAM cache; idle staging reclaimed

**What.**
- Under the file-buffer cap the card never reads the RAM cache: the logs say "0 cached experts went over the copy link". `hx_init` already requires POLICY=all, MISSES=16 and passes of at most 16 tokens (`stager.cpp:932-951`).
- So the slots can hold the codes plus losslessly packed E8M0 scales, in one of two layouts:
  - 2 bits a block with one base per weight (codes for b−1..b+2); at most 0.065% of rounds are flagged;
  - 1 bit a block with an 8% exception budget.
- Either gives ×1.044-1.047 slots: 1,364 → about 1,424-1,428 expert pairs a rank. Each CPU hit also streams 4.4% fewer bytes.
- Separately: under the cap, each rank's 4 stage slots and 4 × 32 MiB read windows sit idle. That is 0.78 GiB of pinned RAM in all (`planner.cpp:100-103, 1263-1305`; `mover.cpp:461-468`). Skip them.

**Evidence.**
- 99.975% of scales are 119, 120 or 121.
- gate_up is stored as [w1; w3], so both halves must share one base. With 1 bit, flagged rounds have a median of 1.38% and a maximum of 7.95%.
- The value of capacity has been measured three times: 102 GiB gave +3.3% and +4.3%, and 90 GiB gave −3.4%.
- Replay at 1,426 pairs: decode misses −6.5%.

**Gain.** Capacity +2.1 to +3.0%, fewer bytes +0.3 to +0.8%, reclaimed slack +0.4 to +1.0%. Together +2.5 to +4.5%, central 3.5%.

**Exactness.** Four requirements:
- At insert, round-trip every weight's scales and refuse on a mismatch or an overflow.
- The packed one-row and multi-row kernels must be bit-identical to the planar kernel on:
  - every sampled weight;
  - synthetic exception rounds at every position;
  - down's 8-block tail;
  - the general path.
- A hard layout tag, so that no card path ever reads a packed slab.
- POLICY=verify cannot run under the cap (`stager.cpp:944-950, 1271`); the kernel tests plus reply identity replace it.

**Blockers.**
- Slot sizes are compile-time constants in several places:
  - `mapped_expert_slab_cache.h:20, 67, 319`
  - `native_mapped_file_cache.h:60`
  - `host_expert_pool.h:36-37`
  - `stager.cpp:1130, 1360, 1633`
- Packed slabs make the cache CPU-only. Greedy or verify policies, passes over 16 tokens and a DSpark head tail would all need the block layout back.
- Synergy: reading missed codes straight into reserved slots would remove the HostPlanes free list (about 1.15 GiB, +0.4-0.6%).

**First steps.**
1. Scan every expert's scales: about 17 GB from `/srv/bulk`, CPU only, adapting `research/adv_packed/scale_layouts.py`. Kill the idea if a robust fixed slot gives less than ×1.035.
2. Run packed one-row and four-row kernels on the 40-thread pool bench. Continue only if they are bit-identical and at least 3.5% faster per expert.

### #5 Read-ahead that yields to demand reads

**What.** Keep the two-ahead gate-reuse predictions from `eee14f49`, but change how they are read:
- Read predicted uncached units from the NVMe in 512 KB pieces, at most 4 in flight across the whole process.
- Before every piece, block on `hx_demand_reads`: no polling, no 2 ms timeout.
- Stop reading a unit as soon as its layer does not select it.
- Read ahead the top 4 to top 6 predictions.

**Why whole units failed, and why pieces should not.**
- A 2.1 ms lead is shorter than a 2.4 ms read.
- In 53.7% of miss layers a demand read in L−1 or L−2 waited behind a whole 18.8 MB read-ahead.
- With pieces, the gain comes from units that are partly read by the time they are needed.
- piecebench on `model.rad`: QD4 × 512 KB reaches 6.77 GB/s (QD1 × 2 MB only 4.67 GB/s). A yielding stream adds +0.075 ms to an 18.8 MB demand read.
- A byte-level timeline model at HEAD gives +3.4-3.7% gross at top-2 and +5.9-7.1% at top-6. Calibrated against `eee14f49`, the model predicted about +1.7% gross; the engine measured +0.2% net plus the predictor's ~1.5% cost.

**Gain.** Net +2.5 to +5.5%, central 4%. That is after the predictor (−0.2 to −0.8), DMA traffic into the DRAM-bound pool (−0.4 to −1.0) and wake-up overheads. It composes with capacity (top-4: +5.2% with a 102 GiB cache, against +5.5%), but it shrinks as misses fall.

**Exactness.** Predictions are hints only. Never compute a partly read unit: use per-plane completion counters and epoch checks. Wrongly predicted units are never inserted.

**Pitfalls.**
- `NativeReadWorkers::run` holds `call_mutex_` for a whole read (`native_read_workers.h:38`).
  - A read-ahead through `plane_workers_` (`mover.cpp:929`) therefore serializes the rank's demand reads. Any re-run of PREFETCH=on at HEAD is biased negative for this reason.
  - Use `read_source_file` (`mover.cpp:166`) instead.
- Read-aheads must read every byte from the NVMe; the RAID mirror is too slow for their tails.
- The predictor's cost lands on rank 0 only: decision time 46.3 vs 34-35 µs a layer, gather waits +0.7 s. Three fixes:
  - score only L+2;
  - move `post()` after `hx_submit` (`stager.cpp:738` → after :777);
  - wake the reader when the prediction is ready instead of polling every 50 µs (:1528).
- Each +0.1 ms of mean delay to demand reads costs 1-1.5 points.
- The drive goes from 43% to 63-79% busy. Watch its temperature.

**First step.** Build a one-day prototype at HEAD, without the partial hand-off yet. Then one window: control ×2, K=4 ×2, K=6 ×2. Log rank 0's decision time and gather waits, and the drive's busy time. Kill it if it gains less than 2%.

### #6 The card's serial chain: overlap on the live issue path

**What.**
- Armed passes are issued live and never recorded (`core/runtime/ctx.cpp:1397-1398`).
- Live dispatches keep their barrier bit ("chained launches: off"), so `haz_mark` never runs.
- So the overlap that DS41's issue order was written for (`deepseek41.cpp:2253-2262, 2283-2289, 2313-2316`) never happens. The loop's audit counted played = recorded = unordered = 0.
- Fix: port the range test to the first live dispatch of each op.
  - Stager copies, CPU-row blits, events and packets from other paths count as ordered points.
  - Collectives are opaque.

**Evidence.**
- A plain layer has 19 serial dispatches from pcn_a to scatter: about 290-320 µs unprofiled, about 15 ms a token.
- 6, 9 and 17 dispatches could start without waiting in plain, indexer and source layers. That is worth 0.7-2.4 ms a token.

**Then fusions, worth another 0.5-0.7 ms:**
- q_rope into the q_b epilogue;
- o_f into the attention epilogue;
- topk + scatter, which needs a new sqrtsoftplus kernel (`rad_block_moe_fp8.h:867, 1561`);
- mix_a as one launch, with its arrival counter in op scratch.

GEMVs that share an input need a split count per segment: kv alone picks SK 4, but a fused q_a|kv (N=1792) picks 2 and would change kv's bits.

**Gain.** +1.2 to +3.5%, central 2.3%. It grows as misses shrink: about 4-5% after a RAM upgrade.

**First steps.**
1. **Dump-only arm.** With `--debug-tape-dump`, record the armed pass and drop the tape without playing it.
   - Expect about 6 / 9 / 17 'U' lines per plain / indexer / source layer.
   - 2 or fewer a layer means the overlap is dead.
2. **Live stamps.** Extend `RADIANCE_AQL_STAMP` to live dispatches on card 0 and run `scripts/stampgaps.py` on them. Do not use `--profile-ops`: its events order everything.
3. **Decide.** Build it if the gap between dependent launches is ≥2.5 µs and the chain is ≥250 µs per plain layer.

### #7 Small exact items

Each is S effort and each needs its own gate, since one arm cannot resolve gains under 1%.

| Item | Gain | Evidence and gate |
|---|---|---|
| AVX2 swiglu in `ds41hx::swiglu` (`research/swv/swiglu_vec.cpp`) | +0.3-0.5% | Cuts the last job's serial step from 11.5-12.1 to 1.8 µs a row. Bit-identical on all 2³² input pairs except a signalling-NaN `up`, which the GEMV's NaN-quieting output never produces. Gate: kernel and pool tests plus one verify arm without the cap |
| Hold inserts while pool jobs run (an active-jobs counter, relay in ~256-row slices, sleep in 20 µs steps, cap 3-5 ms, decode only), plus non-temporal stores in the relayout | +0.4-0.8% | An insert displaces 37 MB of the pool's stream per miss (25 MB with NT stores). The same timeline model predicted +1.2% for the direct insert, and +1.1% was measured. Never spin |
| Split the decode mirror tail over 3 readers when K=1 (`mover.cpp:895-905`) | +0.6-1.9% | The mirror's copy takes 0.13-0.65 ms inside the read path. Gate: add counters for which read leg ends last; drop the item if the mirror ends last in under 25% of reads and the copy takes under 0.15 ms |
| Hand miss reads to the readers as soon as `hx_choose` finds them | +0.3-0.6% | Saves 20-40 µs per critical layer. Needs post/join in `NativeReadWorkers`. Build if dispatch-to-first-read plus the caller's return exceeds 40 µs |
| One dispatch for the 4 D2H copies (`rad_memcpy_ranges_async`, `device.h:54-59`) | +0.2-0.7% | Only if an instrument arm shows the 4-copy block costs ≥10 µs a layer |
| Second-chance eviction from the profile prior (T=6, 1 chance) | +1-1.5% on new task families; +2.8-3.5% on the benchmark's families | Set a protect bit in `hx_warm`, reset the chance in `touch()`, add a bounded skip at the victim sites (`mapped_expert_slab_cache.h:378-386, 437-445`). The held-out prompts share task families with the calibration prompts. Gate: trace the 16 chat prompts first (arm flag `x`, no cost) and drop it if critical misses fall by less than 3% |

### #8 CCD-L3 staging (speculative, L)

**The mechanism works in isolation.**
- Staged lines survive gaps of 0.3-3 ms with the CCD in C2.
- One CCD's top-2-sized burst runs 15% faster with staged tiles first, and 24% faster with staged and DRAM tiles interleaved (30% at f=0.53).

**In the engine it is small.**
- Top-2 predictions cover only f=0.31 of hit bytes.
- Only layers without a miss (about 70%) benefit.
- Real bursts run at about 70 GB/s effective.
- The predictor is off in the best configuration.
- Net: about +1.5% as designed, about +2.5% with interleaving at top-2, up to about 4% at top-4.

**Dependencies and hazards.** It needs #5's predictor fixes. It must avoid both measured hazards: hard pinning (−3.5%) and runnable idle workers (−7 to −30%).

**First step.** Run `research/ccdl3/burst4` on a quiet box: 40 workers, the three commands in that directory, about 30 s each. Kill the idea if the interleaved arm is less than 20% faster than base.

**Price an alternative first.** Copy each rank's best-predicted cached expert into its 1-unit VRAM file buffer during the gap. Two things are unmeasured: whether a slab → VRAM DMA needs a bounce buffer, and what an 18.8 MB H2D copy does to all-reduce latency.

### #9 DSpark: measured negative on this box; park it

**Measured today.**
- Speculation itself costs 13-17% against the bank-loaded, drafting-off arm (12.76 tok/s). Loading the bank costs another 6%.
- Pass costs, in single steps of 78.4 ms (the bank-loaded step):

  | Arm | Steps a pass | Tokens a pass | Replay's estimate (incl. a 17.5 ms draft) |
  |---|---|---|---|
  | Cap 1 | 2.1 | 1.80 | ~1.7 |
  | Cap 2 | 2.6 | 2.27 | ~2.2 |
  | Cap 3 | 3.0 | 2.58 | ~2.6 |
  | Adaptive window | 3.4 | 2.79 | — |

  After its rejected-row costs, the replay leaves about 20-30 ms a pass unexplained. Two suspects:
  - "ordinary target pipelining restored" appears only in the drafting-off arm's log, so the drafting arms may lose issue-ahead;
  - the draft stage's zero-copy experts, re-read per row with XLM=1.
- Acceptance is lower than the replay assumed: about 0.81 / 0.57 / 0.67 by position, against about 0.88 / 0.77 / 0.74.
- Rejected rows add misses: +6 / +14 / +21% at caps 1 / 2 / 3 (rank 1).

**Verdict.**
- Even with the excess removed, the replay's own pass model at this acceptance gives about −5% with the bank's costs included, and about 0% without them.
- Verify-length control (+1 to +6% over the adaptive window) and drafter residency (+2 to +4%) cannot close that gap.
- Two startup refusals were fixed along the way. The second fix, which stops `hx_init`'s gather scan at the draft head's `moe_gemm`, is uncommitted in `stager.cpp`.

**Only worth doing.** One `--profile-ops` cap-2 arm, plus a bit-for-bit logprob comparison. The same per-pass excess decides whether MTP pays after a RAM upgrade (§3).

## 2. Dropped or refuted

| Idea | Why |
|---|---|
| DSpark at depth 2 for +10% (#31's claim), its RAM headroom plan, commit-gated admission, the hard −inf mask on drafter routing | Measured −18.5%, and the follow-ups can't recover it (#9). The admission gate's replayed gain came from an oracle gate; an implementable gate gives −1.7 to +0.8%. At N=8 the mask drops about 25% of stage-40 rows' top-1 pick |
| Keeping the pool awake (MWAITX, longer spin), polling the counts, nice or RT priority for rank threads | Measured: MWAITX −7% / −30%, a 2.5 ms spin −8.5%, counts polling ±0.6%. Cause: the kernel runs lazy preemption, so a thread woken onto a CPU held by a user-mode PAUSE/MWAITX waiter waits about 1 ms (p50), at any nice level (`research/schedx/schedx2.cpp`). At most one "polite" arm remains: a 10 µs MWAITX timer with sched_yield every round |
| Pre-waking the pool for a pipelined miss | With the mirror on w3, gate_up lands last (2.69 vs 1.55 ms, `research/mrp/land_run1.txt`). Even with the mirror moved to w2: −0.3 to +1% |
| Retuning the decode mirror share | Already harvested: 0.13 is the default (+0.6%) |
| Static pinning of the profile's hottest experts | +2.4-3.0% in the benchmark's task families, but leaving each request out of the ranking gives +1.9% critical misses, worse than LRU. Second chance is kept instead (#7) |
| N-gram row gathers off the issuing thread | When warm, the card idles only 24-39 µs a call (0.1-0.3%). Only cold n-grams pay (0.4-0.9%), and the A/B harness cannot show that case |
| Per-card host epochs to avoid system-scope acquires | ≤0.2%: the CPU-row blits force their own system acquire anyway |
| Issuing the routed card ops before the counts sync | ≤0.5% at L effort; the shared expert already fills that gap |
| Trimming the HostPlanes free list; glibc malloc tunables | The plane buffers are never freed, so the tunables cannot return them. Removing the free list properly means filling packed slots directly (#4) |
| Triage drops | Embedding to host RAM is done (`22993c93`, +1.5%). THP for the remaining buffers is done (`ee90326a`). A second NVMe is hardware (§3) |
| Still dead | Power settings, pool pinning, spin 0, insert thread count and nice, prompt lookup, chunked reads, other cache policies (TinyLFU, SLRU, 2Q, LFU, one shared LRU, prefill at the LRU tail), one-ahead read-ahead, async decode misses, all-reduce |

## 3. How far software alone goes, against hardware

**Software savings by component, after #1** (central values, raw):

| Component (ms a token) | Now | Saved | From |
|---|---|---|---|
| Drive-miss stalls | ~27.5 | −8 | FP8's extra residents −3, packed scales −1.6, second chance −0.7, read-ahead −2, read-path items −0.5 |
| Card and host overheads | ~33 | −5 | router −0.3, FP8 −2.3, Infinity Cache prefetch −1.3, serial chain and fusions −2, less overlap between them |
| CPU-row waits | ~11 | −2.8 | FP8's extra residents −1.6, packed bytes −0.4, swiglu −0.35, insert hold −0.4 |
| **Total** | **71.5** | **−16 raw; −11.5 after a 30% discount on everything unmeasured** | **≈ 60 ms** |

**Scenarios:**

| Scenario | ms a token | Decode tok/s | Basis |
|---|---|---|---|
| Today, HEAD `25973b49` | 74 | 13.5 | measured |
| + engram split (#1) | 71.5 | 14.0 | measured, one pair |
| Software plan #1-#7 | ~60 | **~16.5** (15.5-18.5) | table above |
| Software ceiling on this box | ~52-55 | ~18-19 | The CPU path runs at 92-96% of the DRAM ceiling. The ideal (Belady) policy at this capacity has only ~37% fewer misses than LRU, mostly compulsory. The card stays at ≥25 ms with these kernels |
| + DSpark, as measured | | −18 to −22% | today |
| + DSpark with its pass excess removed | | about −5 to 0% | replay's pass model at the measured acceptance |
| A second NVMe in the free M2_1 slot, holding a byte-identical container copy, with each miss read split between the drives (a few hours' work, generalizing the plane mirror) | ~61-63 | ~16-16.3 | A lone miss falls from ~2.4 to ~1.5 ms; stalls from ~28 to ~17 ms |
| … + the software plan | ~50-53 | ~19-20 | |
| 3-4 NVMe drives, striped (M2_1, OCuLink, or a bifurcated x16 card if a slot is physically free) | ~55-57 | ~17.5-18 | stalls fall to ~10-12 ms |
| … + the software plan | ~45-48 | ~21-22 | |
| 8 × 32 GB DDR4-3200 RDIMM (256 GB; the EPYC 7443 runs 3200 at one DIMM per channel; check the ROMED8-2T's memory support list) | ~45-47 | ~21-22 | A ~200 GiB cache holds 88-91% of the file tier, so stalls fall to 2-6 ms. 1.5× the DRAM bandwidth cuts CPU waits to 7-8 ms |
| … + the software card and CPU items | ~38-40 | ~25-26 | The capacity items lose most of their value |
| … + DSpark with its pass excess removed | ~31-33 | ~30 (speculative) | Without drive stalls, verify passes amortize the card and CPU work: ~1.2-1.25× at the measured acceptance |
| 8 × 64 GB DDR4-3200 (512 GB): every expert in RAM | ~42-44 | ~23-24 | |
| … + software | ~35-37 | ~27-28 | |

**Bottom line.**
- **Software alone:** realistically about 16.5 tok/s (+22%) after 2-3 weeks of work, about 18-19 at best. About half of that comes from #1 and #2.
- **MTP:** loses 18-22% today. At the measured acceptance it cannot pay on this box even with its pass costs fixed. It becomes a lever only after a RAM upgrade, and only if a profile finds and removes the ~20-30 ms per-pass excess.
- **Second NVMe:** the cheapest step. About +18-20% on its own, and it composes with the software plan (about 19-20 tok/s).
- **RAM upgrade (256 GB DDR4-3200):** the largest single jump, about 21-22 tok/s alone and about 25 with the card-side software. If it is likely within weeks, do the card-side items (#1, #2, #3, #6) first and the capacity items (#4, #5, second chance) last, because more RAM makes most of the capacity items moot. #2 keeps its CPU-wait saving either way.

## 4. Suggested order

1. **Now:** repeat `Es` twice, profile it, commit it. kbench the BF16 router.
2. **Next window:**
   - the FP8 VRAM-half headroom arm ×2;
   - the KFD queue count;
   - an access-trace arm on the 16 chat prompts;
   - optionally, one profiled DSpark cap-2 arm.
3. **CPU only, outside windows:**
   - `certify.py` over all 610 tensors;
   - the full scale scan;
   - the AVX2 swiglu;
   - the read-ahead prototype;
   - the Infinity Cache microbench code.
4. **Then:**
   - build FP8 if both of its gates pass;
   - the Infinity Cache prefetch arms;
   - read-ahead at K=4 and K=6;
   - the serial-chain dump-only arm;
   - the small items, piggybacked on those windows.