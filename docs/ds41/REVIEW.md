# DeepSeek V4.1 Decode Review

Native-MXFP4 DeepSeek-V4.1-Flash on four RX 9070 XT cards with a RAM expert cache: what limits decode, what in the code causes it, which changes to make first, and what other engines do differently.

- Date: 2026-10-28 (round 2 added the same day)
- Code reviewed: `/home/as/radiance-ds41-development` at `a86b587e` (not yet built)
- Evidence: GPU receipts in `/home/as/radiance-setup/research/deepseek-v41-1017-20261024`, plus primary sources from Strata, DS41Engine, hipEngine, DeepInfer and Fiddler
- Method: read-only. No GPU, service or model access. Two CPU microbenchmarks and an offline cost model, all in this folder.
- Not reviewed in depth: `radiance-ramcache/radiance/arch/deepseek41` and `libds41`, an older reference-only stage with no serving graph.

Status, 29 Oct 2026: the review's top recommendation, computing host-resident experts on the CPU, is implemented and measured, together with misses computed on the CPU from the container's planes. Decode on the eight held-out prompts went from 5.03 to about 12 tok/s, with every reply byte-identical to the base engine's. See [DS41 Overnight Results](https://claude.ai/artifact/Y2mdhzJzGnwaK5zBFJRm7s) and the plan's progress section. The analysis below is unchanged from 28 Oct.

Round 2 changed three things. The cost model now treats the NVMe drive as shared (v1 let every rank read at full speed at once), so all gains are restated as ranges. Computing host-resident experts on the CPU, which the fastest DeepSeek V4.1 engines on consumer hardware do, enters the ranking near the top. And the prefetch estimate now carries an NVMe bandwidth ceiling and DS41Engine's measured prediction accuracy.

## Files in this folder

`/home/as/ds41-decode-review-20261028/`

| File | What it is |
|---|---|
| `REVIEW.md` | This review |
| `PLAN.md` | The implementation plan built on this review |
| `review.html`, `plan.html` | Both as pages; sources of the published copies |
| `build_review_html.py` | Turns `REVIEW.md` and `PLAN.md` into the two pages |
| `ds41_supply_model.py` | Decode expert-supply cost model, v2; reads the Tail1 run's log and routing histogram |
| `ds41_supply_model.out.txt` | Its output with 80 GB/s of CPU bandwidth, as quoted in section 4 |
| `ds41_supply_model.cpu60.out.txt` | The CPU rows rerun at 60 GB/s |
| `packbench.cpp` | CPU microbenchmark: SSE2 MXFP4 pack, memcpy, O_DIRECT read |
| `packbench.out.txt` | Its output |
| `membw.cpp` | Aggregate DRAM read bandwidth by thread count |
| `membw.out.txt` | Its output on this machine, with the production service running |

```sh
python3 ds41_supply_model.py [RUN_DIR]                  # default: the Tail1 held-out run directory
DS41_CPU_GBPS=60 python3 ds41_supply_model.py           # CPU expert bandwidth for the CPU rows
g++ -O2 -msse2 -o packbench packbench.cpp && ./packbench /path/on/the/model/disk/odirect.bin
g++ -O2 -mavx2 -pthread -o membw membw.cpp && ./membw 96   # MB per thread
```

## Summary

Decode runs at about 5.4 tok/s (≈183 ms per token) in the best measured configuration. Nearly all of that time goes to supplying experts one layer at a time, and the GPU, the read workers, the NVMe drive and the PCIe link take turns instead of overlapping.

With one drive, native MXFP4 decode is capped by NVMe reads. Decode still reads about 0.5 GB per token from disk, so at ≈7 GB/s the ceiling is 14 tok/s with perfect overlap. Any wasted speculative read lowers it, to 9.3 tok/s at 1.5× read amplification. Software can move decode a long way toward that ceiling; going past it needs fewer misses (more RAM) or more drive bandwidth.

The changes that look most valuable:

1. Compute host-resident experts on the CPU instead of copying them to a card (new in round 2). This is what Strata, DS41Engine, Fiddler and llama.cpp do. Model: +23–54% at 60 GB/s of CPU bandwidth, and outputs can stay bit-identical if the CPU kernel reproduces the GPU's summation order.
2. Take CPU work off the disk-miss path (+11–26%). This is the smaller, earlier step toward 1.
3. Fix the work split of the single-token MoE GEMV (+5–10%, bit-identical).
4. Prefetch predicted experts from NVMe into RAM (+8–20% hiding 25% of misses, +21–49% hiding 50%), capped by the drive.
5. Lend the VRAM that sits idle during decode to more experts (+5–10%).
6. Turn off NVMe power saving for the run. It costs nothing to test, and the earlier attempt failed on a harness bug before it measured anything.

For long prompts, the exact decoder-suffix skip your research already worked out (CED) is the main prefill lever; DS41Engine runs the same idea and prefills 371–493 tok/s at 8K–123K tokens.

| Figure | Value | Basis |
|---|---|---|
| Best measured decode | 5.36 tok/s, ≈183 ms/token | Tail1 run, 80 GiB slab cache |
| Expert weights touched per decoded token | ≈4.5 GB | 40 layers × 6 experts × 18.8 MB |
| NVMe reads during decode | ≈0.5 GB per token | estimate from per-request OS read counters |
| RAM hits among file-tier lookups | 77% | 129,612 hits and 38,534 misses, 4 ranks, whole run |
| Issuing-thread time in `stage_relaid` | ≈46 s per rank of ≈139 s of requests | Mover shutdown report |
| Non-VRAM experts per layer, busiest rank vs mean | 1.93 vs 0.85 | cost model sample |
| DRAM read bandwidth | 68 GB/s with 16 threads, 79–83 GB/s with 20–24 | `membw.out.txt`; 8 × 16 GB DDR4-2133 |

## 1. The system as measured

| Item | Value |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017, official release, native MXFP4 routed experts |
| Layers | 40 routed layers: 20 encoder and 20 decoder; compressed KV sources at layers 2, 8, 14, 20; index sources 2, 8, 14, 20, 24, 28, 32, 36 |
| Routing | 384 experts, top-6; EP4, 96 experts per rank |
| One expert | 18,800,640 B: gate/up 12,533,760 + down 6,266,880, stored as ggml `block_mxfp4` (17-byte blocks of 32 values) |
| All routed experts | ≈289 GB |
| Container | `model.rad`, 416 GB, on the Samsung 990 Pro (`/`); the 103 GiB Engram table is read from disk |
| GPUs | 4× RX 9070 XT, 15.92 GiB usable each, PCIe Gen4 x16, each on its own root port |
| CPU and memory | EPYC 7443 (24 cores, 4 CCDs, AVX2), 8 × 16 GB DDR4-2133 in all 8 slots of an ASRock Rack ROMED8-2T |
| VRAM per rank | about 375–395 resident experts (rank 0: 374); two file buffers of 1.61 GiB for staged layers |
| Host memory | 80 GiB mapped slab cache (20 GiB per rank), 4 GiB pinned pool per rank (about 123 experts), 8 GiB headroom |
| Decode selections, held-out | VRAM 43.5%, pinned pool 6.5%, file tier 50.0% |
| Run settings | TP4 exact wire, one sequence, context 512, batch cap 16, prefix cache off, `--deterministic` |
| Power settings now | NVMe APST allowed down to 100 ms exit latency (`default_ps_max_latency_us=100000`); CPU governor `schedutil`, C1 and C2 enabled |

Tail1 shutdown counters, per rank: 45.7–47.2 s of the issuing thread inside the file stager's read and relayout; 9,404–9,730 missed weights relaid (82–85 GiB); 31,640–33,202 cache hits (277–291 GiB); 362–373 GiB copied to the card.

Intrusive op profile (`--profile-ops`, mixed prefill and decode): all-reduce rows are 72.5–75.5% of summed op time, native expert GEMMs 19.5–22.1%, dense GEMMs about 2.7%, Engram gather about 0.4%. These are summed event rows that include waiting for other ranks, not a partition of wall time.

## 2. The decode loop

For each routed layer, on every rank:

1. The issuing thread issues attention, mHC and the router.
2. At the layer's first expert op, the file stager copies the 96-entry routing counts to pinned memory and blocks in `rad_event_sync` until the GPU reaches it (`core/place/stager.cpp:645-647`).
3. It selects the file-tier experts the counts name and calls `stage_file_units`, which runs `stage_relaid` (`core/place/mover.cpp:984`). For each 32 MiB window, workers either borrow a cache lease (hit) or read, repack and insert (miss).
4. Only after every job in the window finishes (`mover.cpp:1047`) does it queue the copies to the card (`mover.cpp:1070` and `:1074`) and record the window's event.
5. The compute stream waits on that event; the grouped GEMMs, gather and all-reduce run; the thread moves to the next layer and blocks again at its counts.

In this mode the stager never reads the next layer ahead (`stager.cpp:800`: `if (!routed_ && …)`). The GPU idles while the host stages, and the drive and CPU idle while the GPU computes.

<!-- figure:start -->

```
Illustrative layer on the slowest rank: one missed expert, one RAM hit.
Durations come from the cost model and the CPU microbenchmark, not from a trace.

ms             0    0.5      1.5       2.5       3.5       4.5       5.5
GPU            [rtr]......... idle, waiting on staging and copies .........[GEMM][AR]
Thread         [sync][------- blocked in stage_relaid until the join -------][issue]
Read workers        [ pread gate/up 12.5 MB ][ SSE2 pack ][ slab copy ]
PCIe link           ............ idle until the join ............[ H2D 2 x 18.8 MB ]
                                            \__ #1 removes __/
```

<!-- figure:end -->

## 3. Measurements this review relies on

All are single sweeps or single ordered pairs, so a difference of a few percent is within what one sweep cannot resolve. Files are in the research directory.

| Comparison | Result | Source |
|---|---|---|
| Decode-profile placement off → on | 2.08 → 2.53 tok/s (+22%); prefill −15% | `DS41_DECODE_PROFILE_HELDOUT_ACTUAL_20261028T050600Z.txt` |
| Retained mapped slabs (V6) vs per-entry cache (V5) | ≈3.19 → 5.13 tok/s | `DS41_COMPREHENSIVE_OPTIMIZATION_RESEARCH_20261028.txt` |
| Slab cache 64 → 80 GiB | misses −22%; 205.5 → 191.9 ms/token (+7.1%); prefill 70.6 → 60.5 s | `ds41-mapped-dev-window-v6-slabs64-heldout-20261028T080200Z`, `…slabs80-heldout-20261028T075200Z` |
| Live tail off → on, same module | 4.97 → 5.36 tok/s; prefill 60.1 → 40.9 s | `ds41-pack-tail-dev-window-tail0-heldout-20261028T084500Z`, `…tail1-heldout-20261028T085400Z` |
| Ready-copy off → on | 5.172 → 5.153 tok/s (−0.4%); prefill +4.1% | `DS41_READY_COPY_PAIR_ACTUAL_20261028T131400Z.txt` |
| EP64 selected prefill reads off → on, 64 rows | prefill 308.5 → 36.3 s; decode 4.19 → 5.17 tok/s; OS reads 2,040 → 316 GiB | `DS41_M64_EP64_COMPLETED_REQUEST_COMPARISON_20261028T113800Z.txt` |
| Read-only prefill admission, 128 rows | decode after a long prompt 2.01 → 4.57 tok/s | `DS41_M128_READONLY_CACHE_PROBE_ACTUAL_20261028T070900Z.txt` |
| Prompt lookup depth 0 → 3 | −2.7% on SQL, edit and prose; +37% on repetitive copy | `DS41_LOOKUP_DEPTH013_ACTUAL_20261028T093600Z.txt` |
| DSpark depth 5, head cache off → on | 4.62 → 4.70 tok/s; 94 of 165 drafts accepted | `DS41_HEAD_CACHE_MATCHED_PAIR_ACTUAL_20261028T182400Z.txt` |
| Sparse GEMV kernel screen, 64 rows, all 96 experts | gate/up −57% and down −44% event interval from VRAM; −28% and −30% from mapped memory | `DS41_KERNEL_FULL96_MATCHED_ACTUAL_20261028T183000Z.txt` |
| WMMA kernel screen, same shapes | VRAM −66% and −60%; mapped +45% and +51% | same file |
| NVMe APST off | no result: the harness failed on a `select()` file-descriptor limit before the model started | `ds41-nvme-power-window-lowlatency-20261028T072400Z/receipt.json` |

## 4. Cost model and microbenchmarks

Method (`ds41_supply_model.py`, v2):

- Draw six distinct experts per layer from the held-out decode histogram of the Tail1 run (+0.5 smoothing per expert), and take each expert's tier from that run's logged `EXPERT_PLACEMENT` lines.
- Each non-VRAM expert costs 18.8 MB over a 27 GB/s link, on the rank that supplies it. A RAM hit (probability 0.78) needs nothing else. A miss waits for the shared drive, where all of the layer's misses are read together at 6.5 GB/s, then 2.0 ms of CPU for the pack and slab copy.
- A layer costs the slowest rank's supply time, and a token costs 40 layers.
- The CPU variant computes host-resident experts at an effective 80 GB/s (or 60, with `DS41_CPU_GBPS=60`) shared by all ranks, reads misses straight into cache memory with no repack, and adds 0.1 ms per layer to move activations and results.

The model gives 172.4 ms per token of supply against 183 ms measured in total, so supply dominates.

Calibration: for the 64 → 80 GiB cache step (whole-run hit rate 0.706 → 0.771) the model predicts +16.8% where the runs measured +7.1%. So the model is about 2.4× too sensitive to misses; part of the gap is that the measured hit rates mix prefill and decode. The table gives both the model's gain and the gain scaled by 0.42. Read the pair as a range: the scaled figure is pessimistic for changes that are not about misses, such as rebalancing or slicing.

| Scenario | Model supply (ms/token) | Model gain | Scaled gain |
|---|---|---|---|
| Current | 172.4 | | |
| #1 CPU off the miss path, 0.3 ms left | 136.6 | +26% | +11% |
| #3 prefetch hides 25% of misses | 144.0 | +20% | +8% |
| #3 prefetch hides 50% of misses | 115.7 | +49% | +21% |
| RAM hit rate 0.78 → 0.85 | 137.0 | +26% | +11% |
| #5 TP-sliced experts | 115.5 | +49% | +21% |
| Host experts rebalanced across ranks | 152.2 | +13% | +6% |
| #7 host experts on the CPU, 80 GB/s | 105.0 | +64% | +27% |
| #7 host experts on the CPU, 60 GB/s | 112.1 | +54% | +23% |
| #1 + #3 at 50% | 95.2 | +81% | +34% |
| #7 at 80 GB/s + #3 at 50% | 70.1 | +146% | +62% |
| #7 at 60 GB/s + #3 at 50% | 78.7 | +119% | +50% |

The model counts prefetch as free. In practice every wrong prediction is also a disk read, so prefetch rows are capped by the drive:

| Drive | ×1.0 | ×1.5 | ×2.0 | ×3.0 |
|---|---|---|---|---|
| One 990 Pro, ≈7 GB/s | 14.0 tok/s | 9.3 | 7.0 | 4.7 |
| Two drives striped, ≈14 GB/s | 28.0 | 18.7 | 14.0 | 9.3 |

The columns are read amplification (all disk bytes ÷ useful disk bytes), with 0.5 GB of useful reads per token.

Limits: the model leaves out GPU work other than supply, and the hit rate it uses mixes prefill and decode.

CPU microbenchmarks (one thread unless noted; EPYC 7443):

| Operation | gate/up, 12.5 MB | down, 6.3 MB |
|---|---|---|
| SSE2 pack, planes → 17-byte blocks | 0.74 ms | 0.29 ms |
| memcpy | 0.65 ms | 0.17 ms |
| O_DIRECT pread, scratch file on the 990 Pro | 1.22 ms, likely drive-cached | |

| Threads | 1 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| DRAM read GB/s (`membw.out.txt`) | 23.7 | 55.9 | 50.3 | 58.8 | 67.9 | 83.3 | 78.9 |

Warm caches make the first table optimistic. The second was measured with the production QFN service running in the background.

## 5. Code findings

Paths are relative to `/home/as/radiance-ds41-development`.

### F1. Each disk miss runs about 2 ms of CPU on the critical path

- `read_relaid` (`core/place/mover.cpp:806`) reads a weight's planes with one O_DIRECT `pread`, then repacks them into ggml `block_mxfp4` on the same worker thread.
- `read_native_file` then inserts the packed bytes into the mapped slab cache (`mover.cpp:964`), which copies them from the window into a slab slot.
- `SlabCache::insert` holds `writer_mutex_` for the whole call, including the `memcpy` (`core/place/mapped_expert_slab_cache.h:238` and `:290`). The state mutex is released during the copy, but writers still serialize, so a rank's inserts run one at a time. This matters most in prefill, where one window holds several misses.
- The miss's copy to the card then reads from the window, not from the slab.

Direction: a two-phase insert (reserve a slot, pack straight into it, publish) and a copy to the card from the slot, the way hits already work. Keep the window as the fallback when no slot is free. Then pipeline each miss in 2–4 MB row ranges so packing range k overlaps reading range k+1.

### F2. `stage_relaid` joins the whole window before any copy starts

`mover.cpp:1040-1047` runs every job of a 32 MiB window, and only then (`mover.cpp:1065-1080`) queues all its copies, so hits wait for misses. Ready-copy targeted this and measured −0.4%, because the grouped GEMM waits for the whole layer anyway. Leave it unless the GEMMs are split per expert.

### F3. One blocking sync per layer, and no read-ahead in decode

`stager.cpp:645-647` queues a 384-byte copy behind everything issued so far and waits in `rad_event_sync`; `stager.cpp:800` skips read-ahead whenever routed reads are on. This is why the loop is serial, and changes #3 and #7 are the fixes. DS41Engine measures its own 40 per-layer host round trips at about 6 ms per token. A small extra: issuing the shared expert before the counts copy would let it run during staging, worth tens of microseconds per layer.

### F4. The single-token MoE GEMV starts about four times more blocks than have work

- `libr4d/r4d_ds41_mxfp4.hip:791` sets `L = min(n_expert, T)`, which is 6 for a one-token pass, and lines 808–819 size `rgpi`, `ipe` and the grid from `L`: 252 blocks for gate/up.
- The kernel (line 321) counts the live experts on the device, about 1.5 per rank in decode, so only `n_live × ipe ≈ 63` items do work, each streaming about 112 rows.
- Every 32-value block is 17 bytes, so it is read as five covering dwords and funnel-shifted (comment at line 15), then dequantized through an LDS table.
- Correction after measuring on the card (28 Oct, `PLAN.md` progress): the decode GEMV takes 33–97 µs per call, not the historical ≈0.38/0.32 ms per layer. Spreading the work is bit-identical and helps at one live expert (−14% gate/up, −25% down) but is about +3% at three, worth about 0.2% of decode. Not a bottleneck.
- Each output row is reduced by one wave in a fixed order, so spreading rows over the launched blocks by the live count changes no bits. hipEngine's `tile_map` kernels show one way: expand the grouped prefix on the device into `(expert | tile << 16)` entries plus a live count, and size the grid to a static bound.

### F5. A next-layer expert predictor exists and never runs for DS41

`core/runtime/xprefetch.h` describes it: apply the next layer's router to this layer's router input, copy the likeliest experts to the card, and redirect weight tables only once a copy has landed, so results never change. It is skipped under `--deterministic` (`core/engine_bringup.cpp:4041`), which every DS41 run uses. It also covers only the pinned pool, and the MoE block builds its descriptor only for non-sigmoid routers without BF16 activations (`arch/common/rad_block_moe_fp8.h:1639`; call site `:1705-1706`), which excludes native DeepSeek on both counts.

### F6. The measured binaries and the branch have diverged

The best runs used frozen builds. The Tail1 environment set `RADIANCE_EXPERIMENTAL_DS41_M1_DIRECT_ACTIVATION` and `RADIANCE_EXPERIMENTAL_DS41_MXFP4_PERM_DECODE`, which nothing in the dev tree reads, and HEAD `a86b587e` has not been built. There are 14 `RADIANCE_EXPERIMENTAL_*` references in 10 files under `core/` and `libr4d/`. Upstream's DeepSeek work removed its environment switches, so these need to become flags or go before anything is offered upstream.

## 6. Ranked changes

| # | Change | Est. decode gain | Effort | Outputs |
|---|---|---|---|---|
| 7 | Compute host-resident experts on the CPU; VRAM experts stay on the cards | +23–54% at 60 GB/s | large | identical if the CPU kernel follows the GPU's order |
| 1 | CPU off the miss path: pack into a reserved slot, copy to the card from it, pipeline in row ranges | +11–26% | small–medium | identical |
| 2 | GEMV work split by live experts, or validate and enable the sparse-GEMV variant | ≈0.2% (measured) | small | identical |
| 3 | NVMe → RAM lookahead driven by a next-layer router prediction | +8–20% hiding 25% of misses, +21–49% hiding 50%; drive-capped | medium | identical |
| 4 | Lend idle decode VRAM to about 170 more experts per card | +5–10% | medium | identical |
| 8 | NVMe APST off and the `performance` governor during runs | unmeasured | trivial | identical |
| 5 | TP-sliced experts instead of whole experts per rank | +21–49% | large | different rounding |
| 6 | Kernels that read the release's planar MXFP4, with no repack | adds to #1 and #2 | medium–large | identical if per-block order is kept |

Ranges run from the calibrated model to the raw model, except #2 and #4, which come from the kernel screen and the held-out placement data. #7 makes #1 unnecessary for decode and frees the file buffers that #4 lends. Rebalancing host experts across ranks without changing where they run gains only +6–13% and changes rounding, so it is not listed.

### #7 Host-resident experts on the CPU

- Where: at the stager's existing counts sync, also copy the layer's normed activation rows to the host (about 10 KB per token). CPU threads compute every selected expert that is not in VRAM, for all four ranks: gate/up GEMV, clamped SwiGLU, down GEMV. Each result row is copied into the slot of the sorted expert-output buffer that the owner rank's gather reads, so the weighted sum and all-reduce are unchanged.
- Bit-identical outputs: the GPU computes `kv[nibble] × 2^(e−128)` (exact), multiplies by the BF16 activation without FMA, keeps 32 lane partials per row in a fixed block order, and closes with a wave butterfly. An AVX2 kernel can repeat exactly that sequence of FP32 operations, and the clamped SwiGLU's rounding with it. Prove it with the existing numeric gates before any model run.
- The slab cache can then hold the release's planar layout. A miss is an O_DIRECT read straight into a slab slot, with no repack, insert copy or copy to the card.
- Budget: about 107 RAM-resident experts per token, 2.0 GB, which is ≈25–33 ms of CPU at 60–80 GB/s, shared evenly by all ranks. Today the same experts cost ≈1.3 ms per layer on the busiest rank's link.
- Threads: about 16 compute threads pinned per CCD, with the four issuing threads and the NVMe readers on their own cores.
- Keep large prefill chunks on the cards; Fiddler picks CPU or GPU per expert from how many rows it serves, and that rule fits here.
- Risks: CPU contention with the readers, memory bandwidth shared with NVMe DMA, and the effort of matching the GPU's order exactly.

### #1 CPU off the miss path

- Add `reserve(key, bytes)` returning a busy slot lease, plus `commit()` and `abort()`, to the slab cache. Pack directly into `lease->host()` and copy to the card from `lease->device()`. Keep the victim choice and LRU order as they are.
- Do not hold `writer_mutex_` across any payload copy.
- Split each `pread` into 2–4 MB row ranges and pack range k while range k+1 is in flight.
- Check: identical bytes, hits, misses and victim sequence against the current path on the existing HOST tests, then one GPU pair.

### #2 GEMV work split

- After the device-side live count, spread `n_live × nrg` row groups evenly over the launched blocks, or size the launch from the counts the stager has already read back.
- Check: the existing numeric gates should show bit-identical output.

### #3 Lookahead into RAM

- Prediction: on a side stream, apply layer L+1's router (sqrt-softplus with its correction bias) to layer L's router input, as xprefetch does, and write the top experts to a pinned mailbox.
- Accuracy to expect: DS41Engine's predictor, scored on the state entering a layer, covers 45.9–82.1% of the routes it later computes on the CPU with predicted sets of 4–16 experts (74.1% for 16 at 17K context). Predicting one layer earlier, which a disk read needs for lead time, will score lower.
- Wider predictions find more of the real misses but waste disk bandwidth: the ceiling table in section 4 shows 9.3 tok/s at 1.5× amplification on one drive. Prefetch only experts that are not already in VRAM, the pinned pool or the slab cache, cap reads per layer, and let demand reads go first.
- First, shadow mode: log predicted against actual experts with no reads, and report recall, the share of real misses that were predicted, and wasted bytes.
- DS41Engine and Strata both ship prefetch turned off. Neither had a disk tier as large as this one, so their choice does not settle the question here.

### #4 Idle decode VRAM

- Decode stages at most about six units per layer (≈113 MB), but each of the two file buffers is sized for a whole prefill layer (1.61 GiB).
- Lend the unused tail to hot experts during decode and recall it before a prefill or full-layer read. Today's lending covers only pinned-pool units, so slab-backed units need a new residency state.
- Upper bound from held-out data: +100, +170 and +250 experts per card move 11.5%, 16.2% and 20.7% of decode selections into VRAM. A profile built only from training prompts will get less.

### #8 Drive and CPU power settings

- APST lets the 990 Pro drop into states with up to 100 ms exit latency between bursts, and decode leaves the drive idle while the GPU computes. A miss that lands on a sleeping drive pays the wake-up.
- For a run: set the drive's PM QoS latency tolerance to 0 (what the failed V1 attempt tried), and set the CPU governor to `performance`. DS41Engine uses both, plus C3 off, for all its reported numbers.
- The fix for the harness is already written up (`DS41_NVME_POWER_HIGH_FD_CORRECTION_V2_PLAN_20261028.txt`: `select.poll` instead of `select.select`).

### #5 TP-sliced experts

- Each rank holds a quarter of every expert: 576 gate rows and 576 up rows, and a 576-wide slice of down's input (18 blocks of 32). The existing all-reduce sums the partial down outputs.
- Removes the straggler, where the busiest rank has 2.2× the mean non-VRAM experts, and reads each miss as four 4.7 MB pieces in parallel.
- Changes rounding (partial sums in a different order), grows metadata, and needs new slab classes.

### #6 Planar MXFP4 layout

- Kernels read codes and E8M0 scales as separate planes: aligned 16-byte loads, no funnel shifts, and misses go from disk to slab with no CPU pass.
- Same dequantization; keep the per-lane accumulation order to stay bit-identical.

## 7. What other engines do

| Engine | Setup | What transfers here |
|---|---|---|
| DS41Engine (vincalkr) | 2× RX 7900 XTX, Ryzen 9 9900X (AVX-512), 122 GiB RAM, PCIe 5 NVMe; IQ2_XXS pack of 138.2 GB | Host-resident experts run on the CPU (`--pcie 0`; staging them over PCIe was no better). CPU pool 7.4 ms for 35 experts per token at 47 GB/s, against a 61–62 GB/s streaming ceiling. Decode 23.2–23.7 tok/s, 18.4–19.2 at 17K context. Prefill 371–493 tok/s at 8K–123K with enc-skip. Prefetch off; DSpark slower at exact routing; NVMe APST and C3 off. |
| Strata (Niko1221) | 4× R9700 32 GB, 251 GB RAM | Runs the native MXFP4 set with 6,197 experts on the GPUs and 9,163 computed on the CPU, but reports no speed for it. 2-bit sets reach ≈48 tok/s (2.00 bpw) and ≈80 (2.25 bpw) on code with a hot-expert profile. Default prefill chunk 1,024 (8,192 on Strix Halo, 11–15% faster); a layer-major prefill path; saved conversation slots. |
| hipEngine (shisa-ai) | gfx1151 and gfx1100 | Device-side live-expert tile maps with a statically sized grid (pattern for F4); fused selected-expert FFN; down projection fused with the weighted sum (+1.06%); dual gate/up launches (+2.6%). Graph capture took eager decode from 6.5 to 11.5 tok/s, but that engine was launch-bound; host issue costs about 2% here. |
| DeepInfer (deepseek-ai) | SM100 and Ascend 950 | EP with EPLB, DeepEP and asynchronous CPU scheduling. Its YOCO prefill truncation keeps 140 decoder rows by default, which its own Q&A calls approximate; the exact version is the CED suffix of about 2,560 rows. |
| Fiddler (Kamahori et al., ICLR 2025) | Mixtral on limited VRAM | The principle behind #7: for a few rows, compute an off-GPU expert on the CPU instead of moving its weights, and choose per expert by row count. Reports 1.26× for single-batch decode. |
| llama.cpp | `--n-cpu-moe`, `-ot exps=CPU` | The common local setup keeps experts in RAM and computes them on the CPU. |

Quality figures worth knowing before any quantization decision: Strata measures its 2.00-bpw set at KL 0.170 and +15.5% perplexity against the original, 2.25 bpw at KL 0.107 and +7.3%, and IQ1_M at KL 0.217 and +21%.

## 8. Prefill

- Keep EP64 selected reads and read-only admission; both have large measured wins.
- For long prompts, the exact decoder suffix (CED) is the main lever. Decoder global keys come only from layer 20, so layers 21–39 need only the last ≈2,560 prompt positions. That removes about 40% of layer-token work at 16K and more as prompts grow. Your notes already identify the blocker: the scheduler's single commit cursor (`DS41_CED_EXACT_REPLAY_CRITIQUE_AND_MINIMUM_SEQUENCE_20261028.txt`).
- Beyond that, use large chunks that stream whole layers. A chunk then costs about one pass over the uncached experts, so the cost per prompt token falls as the chunk grows. Rough bound: one pass over ≈289 GB at ≈7 GB/s takes ≈41 s before RAM hits; at 2,048 rows per chunk that is ≈20 ms per prompt token. This is unmeasured, and activation and KV memory at that chunk size need checking.
- For multi-turn use, prefix reuse across turns (PR301 here, saved slots in Strata) removes repeated prefill outright.

## 9. Lower priority

- Ready-copy: −0.4%.
- Prompt lookup: wins only on repetitive text. DS41Engine saw the same: 16.4 vs 21.8–22.8 tok/s on code, 26.4 vs 21.2 on edits.
- DSpark/MTP: a depth-5 pass verifies about 5.9 tokens and accepts about 3.8, and costs roughly 4.3 single-token steps, because more tokens need more distinct experts. DS41Engine found the same at exact routing (17.6 tok/s off; 12.5 at depth 4 and 14.8 at depth 2); engines that report gains use a routing bias, which changes outputs.
- WMMA for experts read from mapped memory: 45–51% slower.
- A bigger RAM cache alone, within current RAM: +7% for 16 GiB more.
- Rebalancing host experts across ranks while they still run on the cards: +6–13%, and different rounding.

## 10. Hardware and strategy

- Native MXFP4 is ≈289 GB of experts against 64 GB of VRAM and about 100 GiB of usable RAM, so NVMe stays in the decode loop. The drive ceiling in section 4 applies to every software change above.
- More memory is the largest single lever for this model. All eight slots hold 16 GB DDR4-2133 modules; 8 × 32 GB would take the box to 256 GB, enough to hold nearly every expert between RAM and VRAM. DDR4-3200 modules would also raise bandwidth about 50%, which helps #7 directly. Check the memory support list for this board and CPU first.
- A second NVMe drive striped with the first doubles the disk ceiling.
- Or change the model: upstream PR266's UD-IQ1_S mix holds about 148 GiB of weights besides the 103 GiB Engram table, its kernels exist, and upstream reports HumanEval .945 and GSM8K .980 at TP4 (llama.cpp on the same GGUF: .939 and .985). Strata's KL figures above show what 2-bit costs in other measures. One KLD and accuracy comparison on your own tasks would settle whether it is acceptable. The IQ2_XS GGUF already on disk has no GPU kernels in PR266.

## 11. Faster testing

- One A/B arm takes 7–10 minutes of wall time: about 21 s of model startup, about 2.5 minutes of requests, and the rest stopping and restoring QFN plus checks.
- Make the experimental switches selectable at runtime, per request or by an admin call, so one model process can run alternating A/B/A/B arms inside a single window.
- Use at least three alternations per comparison. Most current verdicts (Ready-copy, head cache, live tail) are single ordered pairs.

## 12. Not verified here

- The decode-only miss rate. Hit and miss counters cover prefill and decode together, so ≈0.5 GB/token is a regression estimate, and the model's calibration factor inherits the same mix.
- Whether a CPU MXFP4 GEMV reaches 60–80 GB/s on this machine; only plain read bandwidth was measured.
- That `T = 6` in the decode MoE GEMV; it is inferred from the rows × top-k buffer shapes.
- The row count behind the historical ≈0.38/0.32 ms VRAM GEMM figures.
- Prediction recall one layer ahead for this router with mHC streams; shadow mode measures it.
- The effect of APST off on miss latency.

## Sources

- Code: `/home/as/radiance-ds41-development` at HEAD `a86b587e`.
- Run: `/home/as/radiance-setup/research/deepseek-v41-1017-20261024/ds41-pack-tail-dev-window-tail1-heldout-20261028T085400Z/` (`model-server.log`, `receipt.json`, `expert-routing.json`).
- Reports in the same research directory: the files named in section 3, plus `DS41_LIVEOP_ACTUAL_FINDINGS_20261028T052800Z.txt`, `DS41_NATIVE_RAM_CACHE_DECODE_PIPELINE_DEEP_REVIEW_COMPACT_V1.txt`, `QFN_FLASHNEXT_TO_NATIVE_DS41_MEMORY_PATH_PORTING_REPACK_V1.txt`, `DS41_CED_GATING_EXACT_SUFFIX_AND_GROUP_STAGING_REVIEW_20261028.txt`, `DS41_CED_EXACT_REPLAY_CRITIQUE_AND_MINIMUM_SEQUENCE_20261028.txt` and `AUTONOMY_AUTHORITATIVE_20261028T183725Z.json`.
- Model config: `/home/as/radiance-setup/research/deepseek-v41-1017-20261024/ds41-native-official-tokenizer-metadata-20261027T092900Z/config.json`.
- Upstream DeepSeek notes: `/home/as/radiance-ds41-development/docs/deepseek41-wip.md`.
- DS41Engine: https://raw.githubusercontent.com/vincalkr/DS41Engine/main/docs/ds4/V41_EXACT_FAST.md
- Strata: https://raw.githubusercontent.com/Niko1221/Strata/main/docs/DEEPSEEK41.md
- hipEngine: https://raw.githubusercontent.com/shisa-ai/hipEngine/main/docs/KERNELS.md
- DeepInfer: https://github.com/deepseek-ai/DeepInfer
- Fiddler: https://arxiv.org/abs/2402.07033
