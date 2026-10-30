# Prefill Phase 0: what sets Phase 1

Phase 0 of PREFILL-200-DESIGN.md (section 4): measure the links, the drive at layer batches and the
card kernels before Phase 1 is built. Two gated GPU windows (QFN stopped and restored by
`work/gpu_cmd_window.sh`, both verified answering afterwards), branch `claude/ds41-p0` in
`/home/as/ds41-claude-build/wt-p0` (base `7e9ed3b9` plus the lead's build fix `1aae0815`), build dir
`/home/as/ds41-claude-build/aql-p0`. Raw logs: `/home/as/ds41-claude-build/work/p0/gpu1/` (run 1,
10:55–11:05 UTC) and `gpu2/` (run 2, 13:45–13:47 UTC). M4 and M5 are arm lines for the lead's next prefill window
(section 5).

## 0. Summary

- **Links: FAIL (narrowly), no rebalancing needed.**
  - Each card's DMA lane does 27.4 GB/s alone, cards 2 and 3 included. Their 16 GB/s figure was the
    zero-copy rate.
  - Cards 2+3 together cap at 32.4 GB/s (root complex 00), and all four together at 54.3 GB/s:
    26.9 / 22.9 / **13.6 / 13.6**.
  - That fails the ≥ 14 GB/s gate narrowly but is above 10. Phase 1's own link traffic is about
    4 GB/s a card.
- **Drive: PASS.** Layer batches of 128 or 160 units, read the engine's way (O_DIRECT, 18% from the
  SATA mirror), give **8.1 GB/s, 433 units/s, 2.31 ms a unit**. That holds while every landed unit
  and as many RAM-cache-like units go to the cards on the DMA lanes. The design assumed 7.67 GB/s.
- **Dense half: PASS.** About 11.6 µs a row-layer at 256 rows (design: 18) and 10.9 at 2,048, plus
  the indexer: 0.9 G scores/s, 0.9–1.9 µs a row-layer on average at 8K–16K. Attention is 3.1–3.3 µs
  a row-layer at any context.
- **The twin: FAIL, badly. This decides Phase 1.**
  - **As the servers call it: 1,000–3,300 µs a row-layer.** `mx_range` gives the rank's last
    expert every slot routed to another rank, about 4.5 × M dead passes.
  - **Without that dead tail: 45–320 µs.** Real prefill routing puts up to 1,644 of a rank's 3,836
    rows on one expert, and that expert's rows run serially on a few blocks.
  - The non-exact tiled kernel does 8–16 µs at 2,048 rows.
- **Phase 1 changes:**
  1. Remove the dead tail at every height (extend the live tail past 256 rows) and split work items
     over an expert's rows. Both are exact. Gate them at ≤ 30 µs a row-layer on the real routing
     before building the stager.
  2. C = 2,048, end-aligned. 3,072 is above the bound, and its arena would be about 1.46 GiB.
  3. Cards compute every expert row. The CPU keeps I/O, packing and DMA lists.
  4. No root-complex rebalancing.
  5. The stager hands out a unit's reads together. The engine's mirror-first order loses 3–9%.
- **Expected tok/s once the twin is fixed** (Phase 1 at C 2,048, central / pessimistic):
  **1K 122 / 114, 2K 216 / 201, 3K 220 / 205, 8K ≈ 280 / 265.**
  - With the twin as it is, Phase 1 would do about 13–17 tok/s. With only the dead tail removed,
    66–98.
- **Caveat: both GPU runs had QFN restarted underneath them** by the lead's QFN watchdog, because
  `gpu_cmd_window.sh` does not claim its stop (section 8.1). Run 2's link and drive repeats are set
  aside. The kernel numbers agree across the two runs and with an offline model of the kernel's work
  split.
- **Side finding:** today's servers run the routed gemv without the live tail. A 16-token call
  costs 10.4 ms instead of 1.26 ms in this bench (section 8.2). This deserves an A/B arm for
  today's 16-row prefill and for LCP. Window H's LCP arms ran 2.2–3.4× slower than 16-row passes,
  with the NVMe mostly idle.
- **M4 and M5:** six arm lines, dry-run clean (section 5). The 16K receipt is new:
  `work/extra-prompts/long16k-receipt.json`.

## 1. Tools and method

| Tool | What it runs | Where |
|---|---|---|
| `ds41-p0-xfer m1` | M1. Host→card copies of expert units on each card's **DMA lane** (`rad_memcpy_list_async`, the card's own SDMA queue that `aql.cpp` opens), from `rad_dev_alloc` pinned host memory to a 12-slot VRAM ring, batches of 128 or 160 units. Each card alone, the root-complex pairs, all four at once. Then the backend's own blit (`rad_memcpy_async` card to card, 128 MiB, read + write counted) alone and beside the copies on the same card. | `tools/ds41-host-experts/ds41_p0_xfer.cpp` (CMake target `ds41-p0-xfer`, links `rad_core`) |
| `ds41-p0-xfer m2` | M2. Layer batches of expert units (128 or 160 a layer, 32 or 40 a rank, real container offsets) read the way `Mover::read_unit_planes` reads them. O_DIRECT, one `pread` an extent. The gate_up codes' 18% tail (3,383,296 B) comes from the plane mirror's safetensors on `/srv/bulk` through a per-reader bounce buffer. 8 readers a rank. Reads land in pinned staging slots. "Alone": slots are reused as units land. "Drained": each landed unit is copied to its card's VRAM ring on the DMA lane, and its slot is reused only behind the copy; `--extra 1` adds one packed RAM-cache-like unit copied from pinned memory per drive unit, as the design's mix has. Before the timed run, the first unit's mirror tail is checked against the container's bytes (equal). | same file; unit table from `p0_tables.py units` |
| `ds41_p0_bench` | M3. libr4d's own launchers, dlopen'ed from the libr4d.so built at this commit (sha256 in `gpu1/`), under `RADIANCE_EXPERIMENTAL_DS41_LCP=2048` as an Lc arm's server has it: the twin's bound at 2048, the sparse ops' scalar forms and the norms' pin at every height. hipEvent pairs around each launch on an idle card; the median of 5 (3 in run 2) after 2 warm-ups. Rank 0's real shapes. It includes `ds41_mxfp4_device_check.hip` and reuses its fixtures and sweep builders. | `tools/ds41-host-experts/ds41_p0_bench.hip` (Makefile `p0-bench`) |
| Routing for the twin | Real prefill routing: the 3,086-token prompt's first 2,048 rows at every layer, from window `ab-20261029T195838Z`'s Rt trace (`p0_tables.py routes`). Each rank's slots are grouped by local expert in token order, as `moe_scatter` writes them. | `work/p0/routes-long3086.bin` |
| Model | The critique's per-layer pipeline model (`c2_pipeline.py`) and replay (`c1_floor.py`), imported unchanged; only the measured parameters are put in. Without them it reproduces `c5_out.txt` exactly (109 / 165 / 194 / 198 / 218 / 242). | `work/p0/p0_model.py` |

GB = 10^9 B. The drive's unit is 18,800,640 B (block form). The RAM cache's packed unit is
18,001,280 B.

## 2. M1: host→card links on the DMA lane

The method is in section 1. Each figure is one card's bytes over its own batch time. "Together" is
every card's bytes over the slowest card's time. Run 1 gave the same results in the packed form (128
units of 18.0 MB) and the block form (160 units of 18.8 MB, lists with `RAD_COPY_YIELD`).

QFN was serving on the same cards during run 1 (section 8.1), yet the rates held within 0.5% from
batch to batch. Run 2's repeat overlapped QFN's restart and model load. Its alone rates matched
(27.0–27.3), but "together" fell to 37.7 GB/s, so it is set aside.

| Copies | Card 0 (c3:00.0) | Card 1 (83:00.0) | Card 2 (03:00.0) | Card 3 (06:00.0) | Together |
|---|---|---|---|---|---|
| each card alone | 27.4 | 27.4 | 27.4 | 27.4 | — |
| cards 0+1 | 27.4 | 27.4 | | | 55.1 |
| cards 2+3 (root complex 00) | | | **16.2** | **16.2** | **32.4** |
| cards 0+2 | 27.4 | | 27.5 | | 55.1 |
| cards 1+3 | | 23.3 | | 23.3 | 46.7 |
| **all four** | **26.9** | **22.9** | **13.6** | **13.6** | **54.3** |
| HIP runtime `hipMemcpyAsync`, all four (comparison) | 27.2 | 23.5 | 13.7 | 13.7 | ≈ 55 |

GB/s, best of 3 batches; the means are within 0.5%.

- **Cards 2 and 3 each reach the full 27.4 GB/s alone.** The design's 16 GB/s was the zero-copy
  kernel rate. On the DMA lane, the root complex they share caps the pair at 32.4 GB/s.
- **All four together cap at 54.3 GB/s.** That is a host-side ceiling (the IO die or DRAM reads), not
  the links: cards 1+3 alone already drop to 23.3 each. With all four busy, cards 2 and 3 get
  13.6 GB/s each.

**Beside a VRAM-bound kernel on the same card** (the blit runs 568 GB/s alone on every card):

| | Card 0 | Card 1 | Card 2 | Card 3 |
|---|---|---|---|---|
| copies beside the blit, one card at a time | 22.6 | 27.0 | 27.0 | 27.0 |
| the blit's loss | **−40%** | −5.8% | −5.9% | −5.8% |
| all four cards, copies + blits at once | 22.5 / −40% | 22.5 / −5.5% (once −45%) | 13.5 / −5.7% | 13.5 / −5.6% |

- **On cards 1–3 the DMA lane costs a VRAM-bound kernel about 6%.** The design assumed none ("at no
  cost to the kernels"). 6% is small.
- **Card 0 (rank 0) loses 40% of the blit's rate, and its copies drop to 22.6 GB/s.** This happened
  every time, in both runs. Card 1 showed it in one of the yielding all-card runs. The cause is
  unknown: same model, same Gen4 x16 link. QFN was serving on these cards at the time (section
  8.1), so this needs a clean window before Phase 1 relies on overlap on rank 0.

## 3. M2: the drive at layer batches

Six layer batches back to back in each run: 768 units (14.4 GB), or 960 units (18.0 GB) for the
160-unit batches. "Container" and "mirror" rates are each leg's bytes over the span from its first
read to its last.

| Run (run 1 unless marked) | Drive GB/s | Units/s (ms a unit) | Container leg | Mirror leg |
|---|---|---|---|---|
| alone, 128 a layer, unit order | **8.15** | 434 (2.31) | 6.73 | 1.47 |
| alone, 160 a layer | 8.14 | 433 (2.31) | 6.71 | 1.46 |
| alone, engine order (a batch's mirror reads first, as `Mover::plane_reads` lays them out) | 7.40 | 394 (2.54) | 6.85 | 1.33 |
| alone, 2 of the 8 readers mirror-only | 8.02 | 426 | 6.58 | 1.47 |
| alone, NVMe only (no mirror) | 6.17 | 328 (3.05) | 6.79 | — |
| **drained to the cards, 128 a layer** | **8.12** | 432 (2.32) | 6.66 | 1.48 |
| **drained + one RAM-cache-like unit copied per drive unit** (15.9 GB/s on the DMA lanes in all) | **8.14** | 433 (2.31) | 6.69 | 1.47 |
| drained + RAM-like, 2 mirror-only readers | 8.13 | 433 | 6.99 | 1.47 |
| drained, layer barrier (a layer starts once every rank's previous layer is on the cards) | 8.05 | 428 | 6.65 | 1.45 |
| drained, engine order | 7.87 | 418 | 6.76 | 1.43 |
| drained + RAM-like, 160 a layer, 24 slots a rank | 7.55 (8.07 after the first layer) | 402 | 6.61 | 1.46 |
| drained + RAM-like, NVMe only | 7.05 | 375 | 7.06 | — |
| *Set aside, run 2 (QFN loading its model from the same NVMe):* 20 layers drained + RAM-like | 5.80 | 309 | 4.76 | 1.05 |
| *Set aside, run 2:* 12 readers a rank, drained + RAM-like | 6.47 | 344 | 5.32 | 1.18 |

- **The drive gives 8.1 GB/s in layer batches, with copies to the cards draining it: 433 units/s,
  2.31 ms a unit.** The design assumed 7.67 GB/s and 2.45 ms. Copies of RAM-cache-like units at the
  same rate beside it cost nothing measurable. Neither does a layer barrier.
- **The 18% mirror share is balanced.** SATA gives 1.47 GB/s; the NVMe in this mix gives
  6.7 GB/s; 1.47 / 8.2 = 18%.
- **Order matters a little.** The engine lays out a call's mirror reads first. In layer batches that
  idles the NVMe at each batch's start: −9% alone, −3% drained. Phase 1's stager should hand out a
  unit's reads together (unit order).
- **Latency.** A unit's reads take a median 143 ms from slot grant to landing with 16 slots a rank
  (435 ms with 48). That is queueing behind the slots, not drive latency.
- **Pinned staging.** 16 slots a rank (1.2 GB in all) are enough for full rate.

## 4. M3: card kernels per row-layer (one card, rank 0's shapes)

### 4.1 The routed MoE: the twin `moe_gemm_ggml_mxfp4_gemv_tall`

The design assumed 31 µs a row-layer. As the servers call it, the twin is **30 to 100 times
slower**. Without the dead tail it is still 1.5 to 10 times slower. The cause is the call's shape
and the kernel's work split, not its arithmetic.

**A. As the servers call it.** The call takes `moe_scatter`'s slot array: T = M × 6 slots, the
other ranks' slots set to −1 past the local ones. Run 1, rank 0 and rank 2, four layers, real routing:

| M | µs a row-layer (min / mean / max over 8 layer×rank points) | uniform routing |
|---|---|---|
| 64 | 718 / 1,034 / 1,309 | 2,231 |
| 256 | 1,000 / 1,614 / 2,125 | 3,288 |
| 512 | 1,234 / 1,894 / 2,454 | 3,305 |
| 1,024 | 1,398 / 2,132 / 2,652 | 3,315 |
| 2,048 | 2,402 / 2,636 / 2,945 (4 points; the step's 420 s timeout cut the rest) | — |

**Why: the dead tail.**
- `mx_range` (`libr4d/r4d_ds41_mxfp4.hip:207`) ends the rank's last expert's run at T, not at
  `off[ne]`. So expert 95 owns every slot routed to another rank: about 4.5 × M of them.
- Each of those slots is one pass of the gemv's per-row loop. For gate_up (XLM = 1) a pass is a
  barrier, staging, and rgpi row groups of weights, and it writes a row of zeros.
- The live tail (`RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1`) exists to skip those slots. But its
  `eligible()` stops at 256 tokens, and the servers run without it (ab_window.sh never sets it, and
  neither does the engine).

**Checks of that explanation:**
- An offline model of the kernel's work split (`work/p0/twin_model.py`) explains run 1's 36 points
  within about 5%, mostly 1–2%: 256 blocks, live row groups split into items of rgpi groups, each
  item running every pass over its expert's rows, the last expert with the dead tail. It has one
  constant per projection: 5.1 µs (gate_up) and 4.3 µs (down) a row-group pass.
- Uniform routing costs about 0.55 ms of gate_up per dead slot at every M from 256 to 1,024.
- At 16 tokens the gemv row itself, as served, costs 653 µs a row-layer, against 79 µs with the
  live tail (run 2).
- `ptime` from the device check (live tail on, as that tool sets it) reproduces the earlier
  667–723 µs at 16 tokens.

**B. Without the dead tail** (run 2). Two variants:
- "Cut": the call sized to the local slots, T = local, which is what `mx_range` ending at `off[ne]`
  would give.
- "Live tail": at M ≤ 256, the existing switch.

Both give the same times. The single-point columns below are rank 0.

| M | cut, real routing, µs a row-layer (min / mean / max, 8 points) | layer 25 rank 0 (busiest expert's rows) | uniform | live tail at the same points |
|---|---|---|---|---|
| 64 | 30 / 70 / 112 | 76 (28) | 67 | 58–74 |
| 256 | 55 / 113 / 208 | 182 (194) | 54 | 54–179 |
| 512 | 51 / 118 / 215 | 215 (390) | 54 | — |
| 1,024 | 45 / 135 / 298 | 298 (813) | 52 | — |
| 2,048 | 52 / 139 / 320 | **320 (1,644)** | 50 | — |

- **What remains is routing skew.** Real prefill routing gives one expert of a layer most of the
  rows: up to 1,644 of rank 0's 3,836 local rows at layer 25. That expert's rows run serially on
  the few blocks that hold its row groups.
- The model predicted this before run 2: cut 38 / 314 µs at layer 3 / layer 25 (measured 52 / 320).
- **Over every layer and rank** (the model, from the real routing): cut costs a mean of 150–220 µs
  a row-layer, and a layer's slowest rank averages 290–440 µs.
- With uniform routing, cut is about 50 µs, the critique's 45–50 estimate.

**C. References.**
- **libr4d's tiled kernel** sums in another order, so it is not exact. Cut form, real routing:
  8–16 µs a row-layer at 2,048 rows, 10–21 at 1,024, 16–30 at 512, 27–47 at 256. That is the speed
  a row-batched kernel reaches today. Phase 3's 9–13 µs target is realistic.
- **Splitting an expert's rows over blocks** (work items over row ranges as well as row groups) is
  exact: each output row is still one wave's same lane and block order. The model puts it at
  5–20 µs a row-layer, a mean of about 11, if the weight re-reads hit L2 or the 64 MB MALL. If they
  come from VRAM, the floor is 1.41 rows × 12.5 MB at 568 GB/s ≈ 31 µs.

**Measurement conditions.** QFN was restarted during both runs (section 8.1). In run 1 the twin
ran while QFN loaded and then served. In run 2 it ran while QFN loaded. The same points agree
across the runs within 0–6%. The run-1 constants predict run 2's cut times. Run 2's `ptime` control
ran before QFN started and matched the earlier 667 µs within 8%. So the contention was small, and
the ratios above stand. Treat the absolute µs as upper bounds.

### 4.2 Attention and the indexer

`attn_sparse_sink`: 16 heads a card, 512 dims, window 128, top-k 512. `index_score`: 32 heads × 128
dims on every rank. The scalar forms are pinned under LCP. M rows end at the context.

| ctx | ratio | M | attention µs (µs a row) | index_score µs (µs a row) | scores/s |
|---|---|---|---|---|---|
| 8,192 | 2 | 256 | 1,007 (3.93) | 1,404 (5.48) | 0.74 G |
| 8,192 | 2 | 2,048 | 6,459 (3.15) | 8,057 (3.93) | 0.91 G |
| 8,192 | 1 | 256 | 824 (3.22) | 2,215 (8.65) | 0.93 G |
| 8,192 | 1 | 2,048 | 6,739 (3.29) | 16,494 (8.05) | 0.89 G |
| 16,384 | 2 | 256 | 825 (3.22) | 2,310 (9.02) | 0.90 G |
| 16,384 | 2 | 2,048 | 6,719 (3.28) | 17,539 (8.56) | 0.90 G |
| 16,384 | 1 | 256 | 823 (3.21) | 4,388 (17.1) | 0.95 G |
| 16,384 | 1 | 2,048 | 6,699 (3.27) | 34,470 (16.8) | 0.91 G |

The rows at 512 and 1,024 are in `gpu1/m3-sparse.log`.

- **Attention is 3.1–3.3 µs a row-layer, whatever the context.** It reads the window plus the top
  512. The design assumed 6 µs (4–8).
- **The indexer scores 0.9 G scores/s.** The design guessed 1 G (0.25–1.5). Each index layer costs
  4–17 µs a row at 8K–16K, and only 8 of the 40 layers have one.

### 4.3 The rest of the dense half

The median µs a launch, with µs a row in brackets. FP8 is `gemm_nt_bf16f8`.

| Op (rank 0) | 32 rows | 256 | 512 | 1,024 | 2,048 |
|---|---|---|---|---|---|
| q_a FP8 1280×5120 | 21 (0.67) | 169 (0.66) | 328 (0.64) | 615 (0.60) | 1,120 (0.55) |
| q_b FP8 8192×1280 | 24 (0.76) | 155 (0.60) | 322 (0.63) | 664 (0.65) | 1,263 (0.62) |
| kv FP8 512×5120 | 13 (0.40) | 50 (0.19) | 97 (0.19) | 185 (0.18) | 361 (0.18) |
| wo_a FP8 1024×4096 (×2 a layer) | 17 (0.53) | 131 (0.51) | 269 (0.53) | 410 (0.40) | 711 (0.35) |
| wo_b FP8 5120×2048 | 22 (0.69) | 170 (0.66) | 351 (0.69) | 698 (0.68) | 1,372 (0.67) |
| shared gate_up FP8 1536×5120 | 21 (0.65) | 160 (0.62) | 320 (0.63) | 602 (0.59) | 1,120 (0.55) |
| shared down FP8 5120×768 | 15 (0.48) | 58 (0.22) | 108 (0.21) | 216 (0.21) | 435 (0.21) |
| indexer q_b FP8 4096×1280 (index layers) | 17 (0.53) | 67 (0.26) | 141 (0.28) | 286 (0.28) | 597 (0.29) |
| router f32 384×5120 / its 96-column slice | 37 / 16 | 138 / 45 (0.17) | 244 / 75 | 457 / 139 | 888 / 243 (0.12) |
| compressor f32 1024×5120 (kv sources, ratio 2) | 80 (2.49) | 326 (1.27) | 636 (1.24) | 1,273 (1.24) | 2,502 (1.22) |
| mhc_mix (×2) | 26 (0.80) | 82 (0.32) | 146 (0.29) | 275 (0.27) | 529 (0.26) |
| mhc_post_collapse_rmsnorm (×2) | 15 (0.47) | 42 (0.17) | 72 (0.14) | 227 (0.22) | 403 (0.20) |
| rmsnorm 1280, norm-rope 512, swiglu 768, index projection and key, ratio-1 compressor | each ≤ 0.37 a row at 32 rows and ≤ 0.15 from 256 rows | | | | |

**The dense half a row-layer, averaged over the 40 layers:**

| | 256 rows | 512 | 1,024 | 2,048 |
|---|---|---|---|---|
| measured ops (GEMMs, router, norms, mix, shared expert, compressors) plus attention | 8.6 | 8.3 | 8.0 | 7.9 |
| plus the unmeasured small ops (rope, stores, top-k, scatter, gather; ~7 µs a call at 16 rows, so ≤ 1 µs a row) | ≤ 1 | ≤ 1 | ≤ 1 | ≤ 1 |
| plus two exact all-reduces of 10 KB a row (design's estimate; this bench has one card) | ~2 | ~2 | ~2 | ~2 |
| **total, without the indexer** | **≈ 11.6** | ≈ 11.3 | ≈ 11.0 | ≈ 10.9 |
| the indexer, averaged over 40 layers, at 8K / 16K context | +0.9 / +1.9 | | | |

The design assumed 18 µs (12.5–26) without the indexer. The single worst layer is an index layer
at 16K context with ratio 1: about 29 µs a row.

## 5. M4 and M5: arm lines for the lead's next prefill window

**The receipt.** `work/extra-prompts/long16k-receipt.json` (md5 `edcc619c…`) holds one request:
"Summarize the following text in two sentences." followed by the first 501 paragraphs of
`/usr/share/perl/5.40.1/pod/perldiag.pod`. That file is Perl's own documentation, GPL-1+ or
Artistic (`/usr/share/doc/perl-modules-5.40/copyright`, `Files: *`). The text is 61,629 characters,
about 15,960 tokens: `make_long16k_receipt.py` counts with the tokenizer's merges and gets 6,813 on
long8k, where the server counted 6,817. Max 16 tokens out. The part is `Ulong16k`, at `Lm16896`.
The 6,817-token document is `Ulong8k` at `Lm8448`, as in windows B and H.

**The arms.** With `P` and `q` from `work/int4-tmp/arms.sh`:

```bash
. /home/as/ds41-claude-build/work/int4-tmp/arms.sh
# M4: route traces, base and Cs (Mr16 only sizes CPU jobs; routing does not depend on it)
M4="$P-Mr16-Rt-Lm8448-Ulong8k $P-Cs-Mr16-Rt-Lm8448-Ulong8k $P-Mr16-Rt-Lm16896-Ulong16k $P-Cs-Mr16-Rt-Lm16896-Ulong16k"
# M5: the startup arena / VRAM lines at C = 2048 (8,448 context, comparable with window H's C 512 / 1024 lines), and at 16,896 context
M5="$(q 2048)-Cs-Lm8448-Ulong8k $(q 2048)-Cs-Lm16896-Ulong16k"
$W/launch_window.sh winP0 "$M4 $M5" RADIANCE_HOME_DIR2=$H4 MIRROR=$MIRROR \
  DEVICE_CHECKS=lcp TALL_CHECK=$AQL/dc/ds41_mxfp4_device_check TALL_LIBR4D=$H4/kernels/libr4d.so LCP_CHECK_C=2048
```

The arm names expand to:

- `all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr16-Rt-Lm8448-Ulong8k`
- `all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Cs-Mr16-Rt-Lm8448-Ulong8k`
- `all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr16-Rt-Lm16896-Ulong16k`
- `all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Cs-Mr16-Rt-Lm16896-Ulong16k`
- `all-t40-f-M2048-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-Cs-Lm8448-Ulong8k`
- `all-t40-f-M2048-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-Cs-Lm16896-Ulong16k`

**The dry run.** `work/int4-tmp/dryrun.sh` (ab_window.sh's own arm loop, every action stubbed) was
run against this branch's `ab_window.sh`, which is identical to main's, with
`RADIANCE_HOME_DIR2=home-int4 LCP_CHECK_C=2048`:
- All six arms parse, none is skipped, and each switch is the one named: the `ROUTE_TRACE` paths,
  `CONSUMER_SKIP=1` on the Cs arms, `LCP=2048` with `--max-num-batched-tokens 2048` and `MISSES=2048`,
  `--max-model-len 16896`, and the long16k receipt.
- **C = 3,072 is not allowed by the bound.** `$(q 3072)` is skipped: "Lc3072: a chunk is a multiple
  of 16 in 64..2048". libr4d caps the twin at `kR4dDs41LcpMax = 2048` too.

**What window H already gives for M5.** Window H (`ab-20261030T105535Z`) logged the arena at three
chunk sizes:

| C | context | activation arena | rank 0's VRAM residents |
|---|---|---|---|
| 16 | 8,448 | 11.8 MiB | 710 |
| 512 | 8,448 | 246 MiB | 690 |
| 1,024 | 8,448 | 488 MiB | 670 |

That is 0.47 MiB (480 KiB) a row. The extrapolation:
- **C = 2,048:** about 975 MiB and about 80 residents fewer than at 16 rows, about −4.3% decode if
  the reservation stays (at 0.054% a resident).
- **C = 3,072:** about 1.46 GiB and about 120 residents fewer.

This matches the critique's count (0.93 GiB at 2,048; 60–74 residents), not the design's
0.5–0.6 GiB. The Lc2048 arms above confirm the C = 2,048 line.

**Time.** Window B took 224 s (base) and 125 s (Cs) for the 6,817-token document. At 16K, expect
about 9 min base and about 4 min Cs, plus about 1 min of startup an arm. The Lc arms depend on the
twin (section 4.1). Window H's Lc1024-Cs arm took 388 s on Ulong8k. **Budget about 50–60 minutes
for M4 + M5**, or drop the Lc 16K arm.

## 6. The design's gates

| Gate (PREFILL-200-DESIGN §4, Phase 0) | Measured | Verdict |
|---|---|---|
| **Links ≥ 14 GB/s each, all at once** (below 10: rebalancing moves into Phase 1) | Every card 27.4 alone. All four at once: 26.9 / 22.9 / **13.6 / 13.6** (cards 2 and 3, root complex 00), 54.3 in all | **FAIL, narrowly** (13.6 < 14). Above 10, so no rebalancing in Phase 1 (section 7) |
| **Drive ≥ 7.2 GB/s beside the DMA copies** (critique: under 6.8, Phase 2 or a second NVMe is needed for 200 at 3K; under 6.3, Phase 1 tops out near 175) | **8.12–8.14 GB/s** drained to the cards, with as many RAM-cache-like copies again; 433 units/s, 2.31 ms a unit | **PASS** (+6% over the design's 7.67) |
| **Dense half ≤ 30 µs a row-layer at 256 rows** | ≈ 11.6 µs averaged over the layers (8.6 measured + ≤ 1 small ops + ~2 all-reduces), plus the indexer's 0.9–1.9 average at 8K–16K; the worst single layer (index, 16K, ratio 1) ≈ 29 | **PASS** |
| **Twin µs a row-layer** (≥ 25 → the row-batched kernel before Phase 2); critique's version: ≤ ~58 so MoE stays under the drive per producer layer | served: 1,000–3,300; dead tail removed: 45–320 (mean 139 at 2,048), 50 uniform | **FAIL by every reading.** The fixes are section 7.1 |

## 7. What this changes in Phase 1

### 7.1 The twin comes first; without it Phase 1 is slower than today

The model is `work/p0/p0_model.py`: the critique's pipeline with the measured u = 2.31 ms,
dense = 11.7 µs, the indexer at 0.91 G scores/s and the measured concurrent links. With the twin
as it is, Phase 1 gives:

| Twin | µs a row-layer (er) | 1K | 2K | 3K (C 2,048) | 6.3K |
|---|---|---|---|---|---|
| served (dead tail) | ~2,400 (1,600) | 12 | 13 | 15 | 17 |
| dead tail removed | ~400 for a layer's slowest rank (267) | 66 | 75 | 85 | 98 |
| fixed and rows split, or the Phase 3 kernel | ≤ ~30 (≤ 20) | **122** | **216** | **220** | **268** |

Today's CPU path does about 49 tok/s at 3K. Phase 1 therefore has to start with two exact kernel
changes and one gate:

1. **No dead tail at any height.** Either the live tail's `eligible()` covers M up to
   `r4d_ds41_tall_rows()`, or `mx_range` ends the last expert at `off[ne]` when the call is one
   whole routing (`ds41_live_tail::whole`). The bits do not change: the dead slots are written as
   zeros by `mx_tail_zero`, and the device check's tail mode already holds the live tail to the
   plain row.
2. **Work items over an expert's rows as well as its row groups.** Hot experts (up to 1,644 of a rank's
   3,836 local rows at 2,048 tokens) then spread over all 256 blocks instead of the `ipe` blocks that hold their
   row groups. This is exact by construction: each output row is still one wave's, in the same lane
   and block order. Model: 5–20 µs a row-layer, ≤ 31 if the re-reads come from VRAM.
3. **The gate.** Re-run `ds41_p0_bench twin --tail served` on the new libr4d. Want ≤ 30 µs a
   row-layer at every layer and rank of the real routing, M ≥ 512.

If (2) misses that, Phase 3's batched kernel moves into Phase 1. Its target of 9–13 µs is
realistic: the non-exact tiled kernel already reaches 8–16 at 2,048 rows.

### 7.2 Chunk size C

- **C = 2,048, end-aligned.**
  - At 3K it is 220 tok/s against 156 at C = 1,024.
  - The bound (`kR4dDs41LcpMax`, ab_window's `Lc` range) does not allow 3,072. Lifting it would
    buy 3K about 290 (the design's 292 floor), but it costs about 1.46 GiB of arena a card (120
    residents) against about 0.95 GiB (80) at 2,048 (section 5).
- **Decode cost of C = 2,048.** About −4.3% decode while the reservation stands. That is the
  critique's number, not the design's −1.5% to −2.4%. Phase 4's displacement or buffer unions are
  needed to keep decode whole.

### 7.3 CPU/card split

**Cards compute every expert row once 7.1 is done.**
- The CPU keeps the readers, packing (1–2 threads at 433 units/s) and DMA lists.
- The pool sleeps during chunks.
- No CPU share of expert math is needed. Links are not binding (7.4), and with 7.1 the cards
  finish a chunk-layer's MoE (≤ 30 µs × 2,048 ≈ 60 ms) inside the drive's time for it at 3K:
  5,830 units over 2 chunks × 40 layers × 2.31 ms ≈ 170 ms.

**Interim hybrid**, only if 7.1(2) slips: hand each layer's few hot experts (> ~256 rows) to the
CPU's many-row kernel. That is 0.05 ms per extra row, about 2 ms of the 40-thread pool for a
1,644-row expert. The twin with the dead tail removed keeps the rest.

### 7.4 Root complexes

**No rebalancing in Phase 1.**
- Cards 2 and 3 get 13.6 GB/s each when all four copy at once.
- Phase 1's own traffic at the measured drive rate is about 4 GB/s a card: drive units plus about
  as many RAM-cache units, 15.9 GB/s over the four lanes in M2.
- The links bind only if the drive does not:
  - **with a second NVMe:** about 8 GB/s a card, still under 13.6;
  - **with 256 GB of RAM:** RAM units at link speed. There the cap is the host's 54 GB/s aggregate
    as well as root complex 00's 32 GB/s, and the design's "cards 0/1 compute part of ranks 2/3's
    units" lever applies.

**Two scheduling notes:**
- **Rank 0's card** lost 40% of a VRAM-bound kernel's rate when its DMA lane was busy (cards 1–3:
  6%). Confirm this in a clean window. If it holds, rank 0's stager should pace copies during its
  dense half.
- **Rank 2 carries 19% more local rows** than rank 0 at every M (1.68 against 1.41 a row; routing
  statistics, `work/p0/routes-stats.txt`). The MoE's layer barrier waits for rank 2.

### 7.5 Expected exact tok/s (one NVMe, 96 GiB cache, 7.1 done)

| Prompt | Phase 1, C = 2,048 (central / pessimistic) | Phase 2 | Design (Phase 1) | Critique (Phase 1) |
|---|---|---|---|---|
| 1K (1,024 of the 3,086 doc) | **122 / 114** | 120 | 105–116 | 109 [99–116] |
| 2K (2,048 of it) | **216 / 201** | 213 | 165–206 | 194 [174–206] |
| 3K (3,086 doc) | **220 / 205** | 308 | ≈ 200 | 198 [179–210] |
| 8K | **≈ 280 / 265** (extrapolated from 4.1K and 6.3K) | ≈ 600–670 (extrapolated; needs Phase 3) | ≈ 280 | — |

- **Central** is u = 2.31 ms, dense 11.7 µs, idx 0.91 G/s, the twin ≤ 20 µs an expert-row, and
  measured links. **Pessimistic** adds +5% drive, +15% dense, −20% indexer and +10% twin.
- **1K is drive-bound at 122 whatever is built.** 200 at 1K needs a second NVMe or more RAM
  (design section 5).
- **200 is reached from about 2K tokens of prose.** Code-like text (the 1,890-token document)
  reaches 184 / 172.
- **The 8K and 16K numbers stand on extrapolated drive counts until M4's traces are replayed.**

## 8. Side findings for the lead

### 8.1 Both runs had QFN restarted underneath them

The lead's `work/qfn_watchdog.sh` (holder 930222) restarts QFN within 60 s of any stop that no live
process has claimed in `work/qfn_stopped_by`.
- `ab_window.sh` claims its stops (line 767). `gpu_cmd_window.sh` does not.
- So QFN was started at 10:55:46 (15 s into run 1) and at 13:45:54 (7 s into run 2), per
  `qfn_watchdog.log` and docker's StartedAt. It loaded its model from the same NVMe (about 50 s)
  and then served on the same four cards.

What that does to each measurement:

| Measurement | Effect |
|---|---|
| Run 2's M1 "together" (37.7 GB/s) | Set aside: overlapped QFN's load |
| Run 2's M2 (5.8–6.5 GB/s) | Set aside: overlapped QFN's load |
| Run 1's M2 | Ran after QFN's load. QFN reads no disk while serving |
| Run 1's M1 | Ran beside a serving QFN. Rates were steady within 0.5%. Card 0's blit loss may be QFN's |
| Kernel timings | Upper bounds; the contention looks small (end of 4.1) |

The other `gpu_cmd_window` users at 07:42 and 09:29 were hit the same way.
- **Fix:** have `gpu_cmd_window.sh` write `$$` to `qfn_stopped_by` before stopping QFN and remove it
  after the restore, as ab_window does.
- **Re-measure** the card-0 anomaly and M1/M3 in one clean window. Run
  `work/p0/run_p0_gpu1.sh`'s M1/M3 steps with `P0_OUT=<dir>`; about 9 minutes.

### 8.2 Today's servers run the routed gemv with the dead tail

No window sets `RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL`. In this bench, a 16-token call (gate_up +
swiglu + down, rank 0 with all 96 experts in VRAM) takes:
- **10.4 ms without the live tail;**
- **1.26 ms with it.**

In decode the dead tail is about 4.5 passes, roughly 70 µs a layer (estimated).
- The per-op profile in `ab-20261029T005842Z` (arm `all-f-M1-o`) shows 1.5–4.1 ms a gate_up call.
  That is consistent with a dead tail on fewer live experts.
- **If the served card call has the scatter form, `LIVE_TAIL=1` could be a large exact win for
  today's 16-row prefill.** It is worth one A/B arm pair: window env `RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1`
  against the same arm without it, with `Kh` to check the bytes.
- **The LCP arms in window H were 2.2–3.4× slower than 16-row Cs-Mr16, with the NVMe 11–20% busy:**
  Ulong 345 / 422 / 468 s at C 256 / 512 / 1,024, against 156 s. That is what an over-256-row
  dead tail on the cards would do, since the live tail is not eligible there. An `o` profile of one
  Lc arm would show the card's gate_up per call.

### 8.3 Smaller notes

- **The lock wrapper would deadlock.** `flock build.lock gpu_cmd_window.sh …`, as worded in this
  task, cannot run: the script takes build.lock itself on a second descriptor (tested on a scratch
  file). I called the script directly; it queues on the lock the same way.
- **The exit trap removes the window flag.** `gpu_cmd_window.sh`'s exit trap removes
  `WINDOW_PENDING` even when a lead window has set it in the meantime. After run 1 I put it back
  while window H ran; chain2 does the same.
- **Real prefill routing is very skewed.** At 2,048 rows a rank touches about 70 of its 96 experts
  a layer. The busiest takes up to 2,000 rows on some layers (`routes-stats.txt`). This matters
  for any batched kernel's work split, and for the twin gate.

## 9. Files and commits

**Branch `claude/ds41-p0`** (worktree `/home/as/ds41-claude-build/wt-p0`, not pushed):
- `24b99e18`: the lead's build fix `1aae0815`, cherry-picked. `7e9ed3b9`'s `rad_core` does not
  build without it.
- `d1a30f94`: the benches, `p0_tables.py` and `make_long16k_receipt.py`.
- `357500b2`: the twin's kernel and slot-form options, the offline routes mode, and the pipelined
  copier.

**Build:** `/home/as/ds41-claude-build/aql-p0`. Binaries used (sha256 prefixes):
- `work/p0/bin/libr4d.so` `783056d8…`
- `ds41-p0-xfer` `ef38c87b…`
- `ds41_p0_bench` `99b18b00…` (run 1), rebuilt for run 2 at `357500b2`

**Inputs:**
- `work/p0/units.txt`: 15,360 units, container and mirror offsets.
- `work/p0/routes-long3086.bin`.
- `work/extra-prompts/long16k-receipt.json`.

**Logs:** `work/p0/gpu1/*.log`, `work/p0/gpu2/*.log` (each with `window.log` and `cmd.log`).

**Models:**
- `work/p0/p0_model.py` (critique's c2 model with measured inputs; `m_*.json` the inputs).
- `work/p0/twin_model.py` (the twin's work split, fitted on run 1).

