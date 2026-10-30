# Exact prefill at 200+ tok/s: design and critique

# Exact prefill at 200+ tok/s: layer-major prefill with every expert on the cards

I worked read-only and used no GPU. Code references are to `/home/as/radiance-ds41-claude` at main `b0a84f7c`. The latest measurements come from prefill window B (`tools/ds41-host-experts/build/ab-20261029T200958Z`), which finished at 21:47 UTC while I was working.

The drive-read counts come from replaying the long-prompt access trace (`ab-20261029T164814Z`, the trace the PREFILL-REPLAY report validated to 74,968 of 75,040 records). The replay uses that report's exact cache model: SC-LRU with 1,424 slots per rank, warm-up, second chances, and refused units.

The scripts are single-thread Python, each under 2 s, in `/home/as/.claude/jobs/635603c1/tmp/prefill200/`. Every number is labelled as one of:
- **measured:** a log or a window;
- **replayed:** the trace through the cache model;
- **modelled:** costs applied to replayed counts.

## 0. Bottom line

- **Today's prefill (measured, window B):**
  - 31 tok/s at 3,086 tokens (99.5 s); 45 tok/s with the consumer skip (69.0 s).
  - 30 tok/s at 6,817 tokens (224.3 s); 54 tok/s with the skip (125.3 s).
  - Two things set the time:
    - The CPU pool computes 120–170 non-resident expert-rows per token.
    - 16-token passes re-read experts. For the five long documents (6,263 tokens), rank 0 alone read 12,843 units from the drive, which is 241 GB.
  - Taller passes alone make it worse: the exact 32-token arm took 108.4 s at 3,086 tokens against 99.5 s. Drive reads did not fall (12,888 units on rank 0), and every pass got longer.
- **200 tok/s is 5 ms per token.** The CPU path cannot reach it at any chunk size.
  - At 0.08–0.10 ms per extra row it costs 10–17 ms per token, a ceiling of 60–100 tok/s. That ceiling is why the large-chunk prefill (LCP) work in progress is modelled at 80–95 tok/s.
  - So every expert-row must run on the cards.
  - Each touched expert must cross the drive and PCIe once per prompt, not once per pass. That means layer-major execution: one layer at a time over every row of the prompt (or of a long segment), with each layer's experts streamed through the cards once.
- **The binding floor is the drive at every length up to 16K.** The bytes in question are the touched experts that are neither VRAM-resident nor in the RAM cache, read once (replayed, fresh cache, consumer skip on):

  | Tokens | Drive reads (units / GB) | Drive floor | Best exact (modelled pipeline) |
  |---|---|---|---|
  | 1K | 3,488 / 66 | 8.6 s | **116 tok/s** (floor 119) |
  | 2K | 3,874–4,219 / 73–79 | 9.6–10.4 s | **176–206** (floor 181–214) |
  | 3K | 4,051 / 76 | 10.1 s | **294** (floor 307) |
  | 5K | 4,585 / 86 | 11.5 s | **425** (floor 447) |
  | 8K | ~4,930 / 93 | 12.5 s | **~615** (floor 658) |
  | 16K | ~5,100 / 96 | 13.2 s | **~1,000** (floor 1,239; the cards need 11.6 s) |

  - **The exact target is met from about 2K tokens.** The crossover sits between 1.9K and 2.3K depending on the text.
  - **Below 2K the drive decides.** A 1K prompt touches about 3,500 experts that are not in VRAM and not in the 96 GiB RAM cache: 66 GB at 7.7 GB/s. With today's cache, no exact engine change can beat 119 tok/s there.
- **200+ at 1K needs one of these** (section 5):
  - a second NVMe: about 200 tok/s at 1K, 360 at 2K, 520 at 3K;
  - 256 GB of RAM: about 410 at 1K, or about 570 with ranks 2 and 3 rebalanced or one card moved off root complex 00;
  - an opt-in lossy prefill, such as cache-aware routing (about 350 at 1K) or dropping drive misses (about 685).
- **The design** (section 2):
  - Prefill runs layer by layer over a segment of up to 8K rows. Phase 1 uses very large tall chunks; Phase 2 uses true segments.
  - VRAM residents are computed in place.
  - RAM-cache units are copied from their packed slab slots by the card's own DMA queue (copy lists) into a 12-slot VRAM ring. The card decodes the packed form as `gemv_vp` already does.
  - Drive misses are read from the NVMe and the mirror into pinned staging, packed by one or two CPU threads, then copied to the ring.
  - The next layer's likely misses are read during the dense half.
  - Cards 2 and 3's slower links need about a quarter of the drive's time and do not bind.
  - The consumer skip keeps layers 21–39 at 24,150 rows whatever the prompt length.
  - The RAM cache stays read-only. The last 128 rows' experts are admitted afterwards, so decode after the prompt is not hurt.
- **Exactness:** every kernel stays on the decode-band arithmetic.
  - The norm pin, the sparse scalar forms and the 48-row twin already exist under `RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL`.
  - What remains: lift the twin's 48-row bound, add the packed decode to the tall MoE row, and widen the height sweep.
  - The gates are bytes: `--kld-record` directories and prefill KV hashes against the base engine at 16-token passes.
- **Plan** (section 4):

  | Phase | What | Gets to |
  |---|---|---|
  | 0 | 1 day + 1 window of measurements | — |
  | 1 | Very-large chunks with card experts, built on #62's tall-chunk executor; about 2.5–3 weeks | ≥200 tok/s from about 2K–3K tokens (≈200–275 at 3K) |
  | 2 | Layer-major segments with dense pieces; about 2 weeks | 3K about 290, 8K about 600, 16K up to about 1,000 |
  | 3 | Exact row-batched MoE kernel | Needed from about 16K, or once storage is faster |
  | 4 | Decode-neutral memory | Displaced residents instead of a reservation |

## 1. Floors per prompt length

### 1.1 Constants

| Constant | Value | Source |
|---|---|---|
| Expert unit | 18.8 MB in block form (gate_up 12.53 + down 6.27); 18.0 MB packed (11,999,296 + 6,001,984 B) | dims H 5,120, n_ff 2,304 (`deepseek41.cpp:286-299`); NIGHT-LOG 12:10 |
| VRAM residents | 710–719 units a rank (12.48 GiB of the 15.28 GiB weights a card) | server log, window B |
| VRAM slack | none: KV 32.7 MiB at 8,448 tokens (3.96 KiB a token), arena 11.8 MiB, headroom 256 MiB | server log, window B |
| RAM cache | 96 GiB, 1,424 packed units a rank. File-tier units a rank: 3,118–3,127. Uncached after warm-up: producers (layers 0–19) 801–842, consumers (20–39) 857–894 | replayed (`lm_floor.py`) |
| Drive in bulk | 2.45 ms a unit = 7.67 GB/s (NVMe + 18% from the SATA mirror); NVMe alone 6.2–6.4 GB/s; mirror 1.36–1.49 GB/s | PREFILL-DESIGN §1; NIGHT-LOG 02:17, 08:50 |
| Engram gathers | about 0.045 ms a prompt token of drive time (cold gather 370 ms a call at 16K) | survey, hipEngine section |
| Host to card | DMA lane 28.7 GB/s on gfx1201 (`aql.cpp:2699-2717`; runtime copy 22 GB/s and −8% to kernels; copy kernels 2.3 GB/s). Zero-copy reads 29 GB/s on cards 0/1, 16 GB/s on cards 2/3 | measured in the cap-1 profile, `draftcache/DESIGN.md` §0.2 |
| DRAM | about 100 GB/s practical (membw 24–48 threads); 95 used here | NIGHT-LOG 07:58 |
| Exact MXFP4 arithmetic on the card | tiled row (mul then add, no FMA) 5.7 TMAC/s raw: 12.8 ms for 96 experts' gate_up at 32-row tiles | `ds41-kernel-perf-v2` `sparse-b1-gu-vram-0.log` |
| Today's tall MoE | the twin `gemv_tall` takes an expert's rows in passes of XLM = 1 for gate_up (the 5,120-wide row fills the LDS), re-reading 12.5 MB a row (`r4d_ds41_mxfp4.hip:575-600`). Bounded at 48 rows (`r4d_ds41_tall.h`) | code |
| Exact attention | `attn_dec_emit`, 367 µs at 48 rows for 32 heads × 640 keys on an R9700. TP4 gives a card 16 of the 64 heads, so about 4–8 µs a row-layer | `r4d_ds41_sparse.hip:1748-1749` |
| Indexer | 32 heads × 128 dims, top-k 512, window 128. Ratio 2 at index layers 2, 8 and 14; ratio 1 at 20, 24, 28, 32 and 36. One score is 4,096 multiply-adds | config; `deepseek41.cpp:1575-1621` |

### 1.2 Drive bytes: touched experts not in VRAM or the RAM cache, each read once

**Method** (`lm_floor.py`, `lm_floor2.py`):
- The unit is a whole-prompt union of the file-tier units each layer and rank selects. With the consumer skip, only the passes where the layer runs its MoE count.
- It is set against the RAM cache as it stands at the prompt's start, with the cache read-only during the prompt.
- **Prompts:**
  - the five traced documents;
  - the first 1,024 and 2,048 tokens of the 3,086-token document (one coherent text);
  - the documents concatenated to 4,103, 5,127 and 6,263 tokens (several topics, so pessimistic).
- **8K and 16K are extrapolated.**
  - Producer layers: a per-unit, per-pass frequency model over the concatenation, plus a Good-Turing mass for unseen units.
  - Consumer layers: the concatenation's suffix reads, which do not grow with length under the skip.

| Prompt | Tokens | Touched non-VRAM units | from RAM | from the drive: producers / consumers = total (GB) | Without the skip |
|---|---|---|---|---|---|
| 3,086-doc prefix | 1,024 | 7,410 | 3,922 | 1,823 / 1,665 = **3,488** (65.6) | 3,718 |
| 933-token doc | 933 | 7,562 | 3,999 | 1,879 / 1,684 = 3,563 (67.0) | 3,780 |
| 3,086-doc prefix | 2,048 | 8,073 | 4,199 | 2,096 / 1,778 = **3,874** (72.8) | 4,292 |
| 1,890-token doc (code text) | 1,890 | 8,878 | 4,659 | 2,287 / 1,932 = 4,219 (79.3) | 4,709 |
| 3,086-token doc | 3,086 | 8,368 | 4,317 | 2,266 / 1,785 = **4,051** (76.2) | 4,625 |
| concatenated | 4,103 | 9,343 | 4,721 | 2,783 / 1,839 = 4,622 (86.9) | 5,613 |
| concatenated | 5,127 | 9,250 | 4,665 | 2,819 / 1,766 = **4,585** (86.2) | 5,685 |
| concatenated | 6,263 | 9,581 | 4,781 | 2,875 / 1,925 = 4,800 (90.2) | 5,787 |
| extrapolated | 8,192 | ~9,850 | ~4,910 | ~3,008 / 1,925 = **~4,930** (92.7) | ≤ 6,820 |
| extrapolated | 16,384 | ~10,180 | ~5,080 | ~3,173 / 1,925 = **~5,100** (95.8) | ≤ 6,820 |
| upper bound | any | | | every uncached producer (3,301) + consumers ≤ 1,925 | every uncached unit (6,820) |

- **The window order instead of a fresh cache moves reads by up to ±4%.**
  - Read-only: 1,508 / 2,379 / 3,455 / 4,258 / 3,971.
  - With admission of the last 128 rows' units after each prompt (section 2.7): 1,508 / 2,046 / 3,195 / 4,180 / 3,759.
- **Producer reads saturate.** The union over the producer layers' uncached units grows as 1,205 / 1,981 / 2,264 / 2,651 / 2,875 at 256 / 1,024 / 1,792 / 2,546 / 6,263 tokens, out of a pool of 3,301.
- **Consumer reads stop growing at about 2.4K tokens.**
- **So the drive floor in tok/s grows almost linearly beyond 3K.** It is short prompts that suffer: the 82- and 272-token documents need 1,508 and 2,484 reads (3.7 and 6.1 s), so 22 and 45 tok/s at best.

### 1.3 PCIe bytes to the cards, per root complex

- **Every touched non-resident unit crosses its rank's link once:** 1,830–1,890 units a rank at 1K, 2,075–2,125 at 3K, about 2,400–2,550 at 8K–16K.
  - That is 34–48 GB a card per prompt.
  - Root complex 00 carries cards 2 and 3, so twice that: 69–96 GB.
- **Time at 16 GB/s (cards 2 and 3):** 2.2 s at 1K, 2.5 s at 3K, 2.9–3.0 s at 8K–16K. Cards 0 and 1 take 1.2–1.7 s.
- **Fully balanced over all four links** (89 GB/s), the floor is 1.6–2.1 s.
- **This is 23–26% of the drive floor at every length, so it never binds with one NVMe.** It binds only once the drive is gone (256 GB of RAM).
- The packed form saves 4.4% on RAM-cache units.

### 1.4 Card compute (per card, modelled; `lm_model2.py`)

**Routed MoE**
- **Work:** 6/4 expert-rows per row-layer per card, at 35.4 M multiply-adds each.
- **The work, with the skip:**
  - 36,480 row-layers at 1K, 86,320 at 3K, 188,480 at 8K, 352,320 at 16K.
  - That is close to 40 layers × P rows below 2.4K tokens (36,480 at 1K).
  - Above that it is 20·P + 24,150: layers 0–19 run on every row, layers 20–39 on the 24,150-row triangle.
- **Today's twin (XLM = 1).** It is bound by gate_up's per-row weight re-read: about 31 µs a row-layer if the re-reads come from VRAM. The 64 MB Infinity Cache, which holds a 12.5 MB gate_up, may make it faster; that is unmeasured, so Phase 0 measures it. In throughput terms this is about 1.7 TMAC/s.
- **Exact row-batched kernel (Phase 3).**
  - The arithmetic is the gemv's: lane = block mod 32, blocks ascending, groups 0–3, values 0–7, mul then add, then the same wave butterfly.
  - Rows go in register tiles of 32: `acc[RPW][32]` in place of `acc[RPW][8]`.
  - Each 32-block K chunk is staged once per tile in LDS. A tile of 32 rows × 1,024 values in bf16 is 64 KiB; or 16 rows in f32.
  - The weights are decoded once per tile.
  - It is VALU-bound at 2 operations per MAC, like the tiled row, which reaches 5.7 TMAC/s raw on the same arithmetic. Target: 3–5 TMAC/s, or 9–13 µs a row-layer.
  - FMA under an exactness guard (a bf16 × E2M1 × 2^e product is exact in f32 unless it under- or overflows) could roughly double it. It is not needed for 200 tok/s.
- **At 200 tok/s the cards need only 1.7 TMAC/s of expert arithmetic in total**, about 0.43 per card.

**Dense half, central (range), per row-layer per card**
- Attention 6 µs (4–8).
- Dense FP8 GEMMs 7 µs (5–10). The skinny row re-reads the activation per N tile: 0.2–0.4 s per 1,024 rows (PREFILL-DESIGN §4.3).
- Norms, mHC (20-iteration Sinkhorn on 4×4), rope, round trips, router and compressor: 3 µs (2–4).
- Two exact all-reduces of 10 KB a row: 2 µs (1.5–3).
- Pieces of 32 rows add about 6 µs a row-layer of launch overhead: about 45 kernels, roughly 0.2 ms a piece-layer.

**Indexer at long context**
- 4,096 multiply-adds per score.
- Without the skip there are 3.25·P² scores (0.87 G at 16K). With the skip there are 0.75·P² from the producer index layers plus the consumer triangle × P (0.31 G at 16K).
- At about 1 G scores/s for the exact scalar form, a guess (range 0.25–1.5), that is 0.31 s at 16K with the skip (0.2–1.2 s) and 0.9 s without it.
- The score plane is bounded by the row parts (index runs): 512 rows × 16K × 4 B = 32 MiB.

| Tokens | Attention | Indexer | GEMMs + misc + all-reduces | MoE, Phase 3 kernel | MoE, today's twin | Card total (Phase 2+3) |
|---|---|---|---|---|---|---|
| 1K | 0.22 | 0.00 | 0.45 | 0.48 | 1.14 | 1.3 s |
| 3K | 0.52 | 0.02 | 1.07 | 1.15 | 2.70 | 2.9 s |
| 8K | 1.13 | 0.10 | 2.37 | 2.50 | 5.89 | 6.2 s |
| 16K | 2.11 | 0.31 | 4.42 | 4.68 | 11.0 | 11.6 s |
| 32K (skip) | 4.1 | 1.0 | 8.4 | 9.0 | 21 | 22.6 s |

With today's twin and 32-row pieces the cards stay under the drive floor up to about 8K (12.1 s against 12.5 s) and bind at 16K (22.7 s). With tall pieces and the batched kernel they reach the drive floor only at about 20K.

### 1.5 CPU, if it keeps a share

- **Measured marginal row cost:** the 2nd/3rd/4th row of an expert cost 0.098 / 0.127 / 0.137 ms today, and 0.072 / 0.092 / 0.101 ms with `dec1`. The many-row kernel cuts 10–17%. The Mr16 arm measured −3% to −8% TTFT.
- **The CPU's floor** is 120–171 non-resident rows a token × 0.08–0.10 ms, so 10–17 ms a token, a ceiling of 60–100 tok/s whatever the chunk size.
- **The CPU's other costs in this design:**
  - packing drive misses for the ring: about 2.5 ms of one thread an expert, so 1–2 threads at the drive's 410 units/s;
  - the I/O threads that already exist.
- A CPU share of expert math makes sense only where links bind (section 2.4).

### 1.6 DRAM

- Bytes per prompt: drive DMA writes + DMA-lane reads of every touched unit + the CPU pack (about 2 × 18.8 MB a miss).
- That is 205–290 GB, or 2.2–3.0 s at 95 GB/s, against 8.6–13 s of drive time. Average use is 20–25 GB/s. It never binds.

### 1.7 Per-layer overheads

- **The layer barrier.** A layer's MoE needs every row's routing, then the gather and all-reduce before the next layer. That is about 3 ms a layer: counts read back, lists built, the last unit's compute, the gather, the all-reduce of the layer's output. So 0.12 s per segment.
- **The real overhead is the drive idling during each layer's dense half**, before that layer's routing is known: 0.7 s at 1K (8% of the floor), 1.6 s at 3K, 3.6 s at 8K, 6.8 s at 16K.
  - Speculative reads of the layer's prior-ranked uncached units during the dense half recover it.
  - Precision is high from 2K up: 55–63% of a layer's uncached units are touched at 1K–2K, 87% at 6.3K, about 96% at 16K.
  - Speculation costs only host staging; wasted reads use drive time that would otherwise idle.
- The model below credits 50–90% of the dense half as covered.

### 1.8 Floors, the binding one, and the best exact tok/s

| Tokens | Drive | PCIe (card 2/3; balanced) | DRAM | Cards (Phase 2+3) | Binds | Exact floor | Pipelined (central [optimistic … pessimistic]) |
|---|---|---|---|---|---|---|---|
| 1K | 8.6 s | 2.2 (1.6) | 2.2 | 1.3 | drive | 119 | **116** [117 … 110] |
| 2K | 9.6–10.4 | 2.4–2.6 | 2.4–2.6 | 2.1–2.2 | drive | 181–214 | **176–206** [178–209 … 164–191] |
| 3K | 10.1 | 2.5 (1.8) | 2.5 | 2.9 | drive | 307 | **294** [300 … 267] |
| 5K | 11.5 | 2.7 | 2.7 | 4.2 | drive | 447 | **425** [436 … 375] |
| 8K | 12.5 | 2.9 | 2.9 | 6.2 | drive | 658 | **615** [639 … 504] |
| 16K | 13.2 | 3.0 | 3.0 | 11.6 | drive (cards at 88%) | 1,239 | **1,005** [1,166 … 611] |
| 32K | 14.2 | 2.9 | 3.1 | ~23 | cards | ~1,400 | 1,127 |

- **"Pipelined"** sums over the layers: dense half + max(drive − speculation, MoE, link) + 3 ms, plus the Engram gathers.
- **The pessimistic column is Phase 1–2 hardware use:** 32-row pieces, today's twin at 1.7 TMAC/s, an indexer 2× slower, and 50% speculation.
- **Segments of at most 4,096 rows** re-read producer experts per segment: 8K about 420 tok/s, 16K about 480. That is still at least 200.
- **Without the consumer skip** (KLD mode) drive reads rise 7–21%, and long prompts double their card rows: 3K about 260, 6.3K about 400.

## 2. The design

### 2.1 Execution order

- **A segment of N rows** (the whole prompt up to S_max) runs one layer at a time:
  1. For each piece of rows, in order: the layer's dense half up to the router. That is mHC collapse, norm, q/kv, compressor, indexer, sparse attention against this layer's KV (earlier pieces and segments already wrote it), output projection, all-reduce, `mhc_post`, FFN collapse and norm, router into segment-wide ids and weights, and the shared expert.
  2. `moe_scatter` over the segment's rows on each rank.
  3. The routed MoE for every row at once. Each touched expert is computed exactly once a segment.
  4. `moe_gather` in slot order, the all-reduce, and `mhc_post` per piece.
- Attention, the compressor ring and the window keys see rows in prompt order, exactly as 16-token passes do. Only the MoE's grouping changes, and each output row depends only on its own activation.
- **Phase 1 form: very-large chunks.**
  - This is #62's tall-chunk executor run at C = 2,048 (C = 3,072 opt-in), with the chunks aligned to the prompt's end. The final chunk then holds the consumer skip's 2,432-row suffix, and a short head chunk takes the remainder.
  - A tall pass already runs layer by layer over its rows, so within a chunk this is layer-major. Each expert is read once a chunk.
  - Replayed drive floors with end-aligned chunks (`lm_chunks2.py`):

    | Tokens | C = 2,048 | C = 3,072 | One segment |
    |---|---|---|---|
    | 3,086 | 214 | 292 | 307 |
    | 4,103 | 235 | 256 | 357 |
    | 5,127 | 244 | 314 | 447 |
    | 6,263 | 256 | 319 | 520 |

  - Start-aligned chunks lose up to 10%: at 6,263 tokens and C = 2,048, the 146-token tail chunk re-reads 2,067 consumer units.
- **Phase 2 form: segments with dense pieces.** The dense half runs in pieces of 32 rows (the heights the KLD gate already proved) or 256–512 rows (under the pins), and the MoE once per segment. VRAM then grows by about 80 KiB a row instead of a whole tall arena (section 2.6), so S_max can reach 8K–16K.

### 2.2 Expert supply per layer: a card stager

This is a new `CardExpertStager` in `core/place`. It is modelled on QFN's `PrefillStager` (`stager.h:1-115`: ring, pacing, free events, copy lists). It is armed per tall chunk or segment. Once a layer's routing counts are read back, it works per rank:

1. **Residents are computed in place.** They take the first MoE launch, with no transfer.
2. **RAM-cache units** are borrowed read-only (a packed variant of `borrow_no_touch`; no touch, no insert). Their two slab slots (gate_up 12.0 MB, down 6.0 MB) are copied by the card's DMA lane (`aql.cpp:2699-2717`, copy lists as in `Mover::stage_units`, `mover.cpp:792-830`) into ring slots. The slabs are host-mapped and pinned, so no CPU copy is needed.
   - The packed form is the CPU's own (HXP2: `pk_layout`). `gemv_vp` already decodes it on the card bit for bit (`mx_pk_open`, `mx_row_vp`; `r4d_ds41_mxfp4.hip:336-440`). The tall row gets the same decode.
3. **Drive misses** are read by the kept readers into pinned staging (`Mover::read_unit_planes`, `mover.cpp:1076`, with per-unit completion as #62 plans), with the mirror's 18% gate_up tail from `/srv/bulk`.
   - Order: round-robin over ranks, so all four finish a layer together; within a rank, units with more rows first.
   - One or two CPU threads pack each unit into HXP2 form in staging. This is the insert path's packer (`insert_unit_packed`, `mover.cpp:1300`), which checks every weight's round trip.
   - The DMA lane copies the packed unit to a ring slot.
   - The 7 units the packer refuses (raw-round budget) go in block form, relaid as the file path does today.
4. **Waves.** As ring slots land, the stager publishes their addresses in an expert table and launches the MoE row over the ready experts. This needs an explicit expert list per launch; today the kernel derives its live list from the offsets.
   - A slot is reused only behind its readers' "free" event, as in PrefillStager.
   - The ring has 12 slots of 17.9 MiB. The link moves a unit in 0.65 ms (cards 0/1) or 1.2 ms (cards 2/3). A unit computes in 0.5–3 ms. The drive delivers one unit per rank every 9.8 ms.
5. **Read-ahead.** When a layer's demand reads are all issued, the reader continues with the next layer's uncached units in prior order. The order is a prefill prior from route traces, or a card-side "router of L+1 on layer L's MoE input" (about 10 ms of GEMM at 16K rows).
   - The cap is 64 units of staging, about 1.2 GiB.
   - When the layer's counts arrive, unselected units are dropped.
   - From about 5K tokens, "every uncached unit of the next producer layer" reaches at least 87% precision.
6. **Engram.** The Engram rows of layers 1 and 14 for the whole segment are deduplicated and read at segment start, not per pass.

The 40-thread pool computes nothing during a segment and sleeps. Decode keeps today's CPU path unchanged.

### 2.3 Rows per expert and the kernel

- **Grouping.** `moe_scatter` (a stable counting sort) gives each local expert a contiguous run of its rows in token order.
  - Average rows per expert are P/64: 16 at 1K, 48 at 3K, 128 at 8K. Routing skew gives the hottest experts 5–10 times that.
- **Phase 1 runs the existing twin `moe_gemm_ggml_mxfp4_gemv_tall` at any M.**
  - Lift its 48-row bound (`kR4dDs41TallRows`, `kEpRouteTallRows`, the row table's constraint, `FileStager::tall_init`).
  - Add the packed entry decode and the explicit expert list.
  - Each output keeps its lane and block order in every pass of XLM rows, so it is exact today. It is just weight-traffic-bound.
- **Phase 3 kernel** (section 1.4).
  - Work items are (expert, 32-row tile, 8-wave × RPW output group). Hot experts spread over every CU; cold ones share a launch.
  - Gate_up's intermediate (silu·up, with the clamp table the CPU also uses) lives in per-expert scratch: rows × 2,304 × 2 B.
  - `down` writes local slots only.

### 2.4 Work split between the CPU and the cards, given the root complexes

- **Cards compute all expert rows. The CPU does I/O, packing, DMA lists and admission.**
- **Cards 2 and 3's links are busy about a quarter of the time at 16 GB/s** (section 1.3), so no CPU share is needed with one NVMe.
- **If links bind** (after a RAM upgrade), two exact levers, in order:
  1. **Cards 0/1 compute part of ranks 2/3's units and send the output rows peer-to-peer to the owner.**
     - That is about 10 KB a row against 18.8 MB a unit.
     - The owner's `moe_gather` and the rank-ordered all-reduce are unchanged, so the bits are the same.
     - It takes a 28% share at 1K: 2.2 → 1.6 s on the links, or 413 → 573 tok/s with 256 GB.
  2. **The CPU computes ranks 2/3's RAM-cache units that have 1–4 rows.** That costs 0.2 ms plus about 0.07 ms per extra row on the CPU, against 1.2 ms on the link.
- **The CPU computing drive misses in parallel does not help.** The drive's bytes are the binding resource whoever computes the expert. The CPU's 94 GB/s pass over each expert would also compete with the DMA lanes and the NVMe for DRAM.
- **Phase 0 measures the DMA lane on cards 2 and 3.** The 16 GB/s figure is a zero-copy kernel rate, and SDMA may differ.

### 2.5 Combining with the consumer skip

The skip (main `aad8ed79`, `4a1f6b01`, `8555ca6b`) becomes a per-layer row suffix inside the chunk or segment, on the same 16-row grid:

| Layers | Rows |
|---|---|
| 0–19 | every row |
| 20 | key ops on every row (window key, ratio-1 compressor, index keys, stores); full on the last 2,432 |
| L ≥ 21 | full on the last 16·(8(39−L)+1) rows; window-key ops on the 128 rows before |

- It removes 30% of MoE layer-rows at 3K and 42% at 8K.
- It also bounds consumer drive reads at about 1,700–1,925 units.
- Only the final chunk or segment holds consumer rows. Earlier ones run layers 0–19 plus layer 20's key ops.
- Mid-prompt checkpoints remain unavailable with the skip on; that is already accepted for the skip.

### 2.6 Memory budgets

**VRAM per card.** There is no slack today, so the room comes from residents. One resident is worth about 0.054% of decode.

| Item | Phase 1 (tall chunk C) | Phase 2 (segment N, pieces) |
|---|---|---|
| Expert ring | 12 × 17.9 MiB = 215 MiB (8 slots: 143) | same |
| Rows' state | ~100–170 KiB × C once the MoE buffers hold local slots only (today ~144 KiB a row of `egu`/`eff`/`edn` for all 6 slots, `deepseek41.cpp:1104-1106`): C 2,048 → 200–340 MiB, 3,072 → 300–510 MiB | ~80 KiB × N (streams 40 KiB bf16 `{T, 4×5,120}`, MoE input 10, local expert outputs ≤ 20, shared output 10): 4,096 → 320 MiB, 8,192 → 640 MiB. Streams in pinned host memory: −40 KiB a row |
| Expert scratch | ~35 MiB | ~35 MiB |
| Dense arena | inside the rows' state | 32-row pieces ~12 MiB; 512-row pieces ~150 MiB |
| KV | 3.96 KiB a token (16K: 64 MiB); window group of C + 128 rows | window ring of piece + 128 rows |
| Index score plane | 512 rows × width × 4 B: 32 MiB at 16K | same |
| **Total** | C 2,048: ~0.5–0.6 GiB; 3,072: ~0.6–0.8 GiB | N 8,192: ~0.95 GiB; with host streams ~0.6 GiB |
| **Residents taken** | 28–45 a card: −1.5% to −2.4% decode if permanent | 33–53 a card |

**Host RAM** (about 7.5–8 GiB available at the 96 GiB cache):
- Pinned staging: 1.2–1.5 GiB.
- Phase 4 adds the host streams (40 KiB × N: 0.33 GiB at 8K) and copies of displaced residents (0.5–0.9 GiB a card, transient). If those do not fit, trim the cache to 92–94 GiB.

### 2.7 Cache policy: decode after the prompt not hurt

1. **Read-only during prompt chunks and segments**, as the base engine's `NATIVE_CACHE_PREFILL_READ_ONLY` predicate does for pure prefill. This applies to pure prefill steps only. Admit-all thrashes: at C = 512 it reads 10% more than today.
2. **Bounded, causal admission on the final chunk.**
   - After each layer's MoE, insert the units the last 128 rows touched. In layer-major these are known at that layer.
   - Restrict victims to units of layers ≤ L, which the prompt no longer needs.
   - Cap inserts per layer.
   - Drive units are already packed in staging, so this is a direct slot insert.
3. **Second chances for decode-hot units stay.**
4. **Replay** (window order, one segment, skip on):

   | | Read-only | With admission | Today (16-token passes, admit-all) |
   |---|---|---|---|
   | Prefill reads | 15,571 | 14,688 (−5.7%; admitted units serve the next prompt) | — |
   | Decode reads in the 15 tokens after each prompt | 1,081 | 982 | 869 |

   - The +113 reads cost about 0.3 s over five prompts, against about 250 s of prefill saved.
5. **Residents** are untouched under a reservation. Under Phase 4's displacement they are restored before decode.

### 2.8 Scheduling

- **Threshold.** The card path serves every pure prefill above about 48 tokens. It is faster even there: 82 tokens 3.7 s against 5.9 s; 272 tokens 6.1 s against 11.1 s.
- **Other sequences.** They wait while a chunk or segment runs, as with `RADIANCE_EXPERIMENTAL_SCHED_PREFILL_ALONE`. The stall is 10–25 s. For several users, cap C or S_max.
- **Flags.**
  - `--checkpoint-interval ≥ C` (or `--prefill-span`). Otherwise chunks silently stay at 2,048.
  - `--max-model-len` for the 16K receipts.

## 3. Exactness

Every computed row and KV byte must equal the base engine's 16-token passes.

### 3.1 Per op

| Op | Height behaviour | What this needs | Proof today |
|---|---|---|---|
| Routed MXFP4 GEMM | gemv ≤ 16; twin 17–48 under EXACT_TALL; tiled above, in another order | Twin at any M (lift the bound); packed decode in the tall row; Phase 3's batched kernel in the same lane/block order with the same butterfly | Twin at 17–96 tokens equals gemv in passes of 16 and the host kernels: 2,392 rows, 0 differ. The tiled row differs in 3,361 values (`devcheck-tall.log`) |
| Norms (rmsnorm and the fused mHC/rope norms) | 1,024 threads ≤ 32 rows, 512/256 above | The pin under EXACT_TALL is already in main (`r4d_common.h:286-293`) | Without the pin, 55 of 3,872 rows differ above 32; with it, 0 up to 48 (`devcheck-sweep*.log`) |
| `attn_sparse_sink` | scalar ≤ 48, WMMA above | `attn_dec_emit` at every height under the switch (`r4d_ds41_sparse.hip:1870`; at most 65,535 rows a launch) | 17–100 rows, 16 and 32 heads, 0 differ |
| `index_score` | scalar ≤ 48, WMMA above | Scalar at every height; row parts keep each row's bits | Same check |
| Dense GEMMs (bf16 m16/m64/tiled, FP8 skinny) | One split law | Verify in the sweep up to C or the piece height | `tall_init`: only `gemm_nt`'s m16/m64 pair switches kernel, with identical bits |
| Exact all-reduce | One-shot ≤ 524,288 elements, two-shot above | Both fold from 0.0f in rank order; add a 16K × 5,120 check | Code audit (critique §3) |
| `moe_scatter` / `moe_gather` | 1 or 3 launches; slot order | Stable sort; gather after every expert of the layer | Audit; add a 98K-slot check |
| `index_topk`, `index_candidates` | Split or merge | Exact selection | Audit |
| Compressor, window `kv_store`, rope round trips | Per row, in order | Pieces and chunks start on the 16-row grid; window ring of piece + 127 rows | KLD identical at 32-row passes |
| Expert bytes | — | DMA copies are byte copies; packed entries round-trip checked at insert and at staging; a slot is published only after its fence | `gemv_vp` bit-identical on the card (window A) |
| Consumer skip | Pass cut | Row suffix inside a chunk or segment; key-only rows use the full layer's fused ops | Prefill KV hashes identical: 5/5 at 82–3,086 tokens and 1/1 at 6,817 |

### 3.2 Byte gates for the A/B (not replies)

1. **Device.**
   - The MoE row at M ∈ {17, 33, 48, 64, 100, 256, 1,024, 4,096}, block and packed, against gemv in passes of 16 and the host kernels, with skewed routing and NaN/Inf/subnormal rows.
   - Extend the height sweep from {1, 16, 17, 24, 25, 32, 33, 40, 48} to {49, 64, 102, 103, 170, 171, 256, 512, 1,024, 2,048, 3,072, 4,096}: every op, each row's bytes alone and in launches of 16.
   - All-reduce at 16K rows; scatter at 98K slots.
2. **`--kld-record`** (`core/config.cpp:318-325`), skip off. Use the 19-document corpus plus the 1K–7K documents. The record directories must be byte-identical to the base at 16-token passes, as the Xt-BT32 arm already showed for 1,512 positions.
3. **Prefill KV hashes** (`RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH=1`, part `Kh`): every KV group at each prompt's end, all ranks, skip on and off, 82 … 16K tokens.
4. **Greedy replies** with 64–256 decode tokens after each long prompt, the 8 held-out and 16 chat prompts. Decode speed after the prompt within the reservation's predicted cost.
5. **Standing refusals.** `tall_init` refuses any band not on the exact rows. The stager refuses a weight form it cannot decode exactly. An optional verify arm recomputes sampled expert rows with the host kernels.

## 4. Phased plan

| Phase | Build | Window and gate | Delivers (exact, one NVMe) |
|---|---|---|---|
| **0. Measure** (1–1.5 days of tooling; ~40 min; can ride on window C) | M1: DMA lane host→card per card, alone and 4 at once, beside a VRAM-bound kernel. M2: drive at layer batches (100–160 units back to back, kept readers + mirror) into pinned buffers while DMA lanes read them. M3: twin at M = 64–4,096; `attn_dec_emit`, `index_score` at 256–4,096 rows × 8–16K context; FP8 GEMMs at 256–4,096; norms pinned. M4: `Rt` route traces of the 6,817-token document and a new 16K one (base + `Cs`) for exact 8K/16K drive reads. M5: arena line at C 2,048 / 3,072 | Links ≥ 14 GB/s each concurrently (below 10, rebalancing moves into Phase 1); drive ≥ 7.2 GB/s beside DMA; dense half ≤ 30 µs a row-layer at 256 rows; twin's µs a row-layer (≥ 25 → do Phase 3 before Phase 2) | — |
| **1. Very-large chunks, experts on the cards** (~2.5–3 weeks with #62's shared parts) | From #62: EXACT_TALL bound 48 → C, skip row suffix inside a chunk, read-only packed borrow, per-unit miss completion, chunk flags. New: `CardExpertStager` (§2.2), twin at any M with packed decode and an expert list, local-slot MoE buffers, end-aligned chunks, VRAM reservation flag, last-128-rows admission, `ab_window` parts (card path, C, receipts 1K / 2K / 3K / 5K / 8K / 16K, `Lm16k`) | §3.2 gates; NVMe busy ≥ 85% inside chunks; decode within −2.5% | 1K 105–116; 2K 165–206; 3K ≈ 200 (C 2,048) / ≈ 275 (C 3,072); 5K ≈ 230 / 295; 8K ≈ 280 / 350; 16K ≈ 310 / 370 |
| **2. Layer-major segments** (~2 weeks) | Plugin: dense ops per piece (32 rows first, then 256–512 under the pins), segment-wide routing, MoE once, gather/post per piece; window ring at piece height; whole-next-layer read-ahead from 5K | Same gates; 8K and 16K receipts | 3K ≈ 270–294; 5K ≈ 375–425; 8K ≈ 500–615; 16K ≈ 610–1,000 (one segment) or ≈ 420–480 (4K segments) |
| **3. Exact row-batched MoE kernel** (~1–1.5 weeks; earlier if M3 says so) | §1.4: 32-row register tiles, 1,024-value K chunks in LDS, one decode a tile; block and packed forms; optional guarded FMA | Device checks §3.2.1; MoE ≤ 13 µs a row-layer | 16K ≈ 1,000; needed before any storage upgrade pays off |
| **4. Decode-neutral memory and polish** (~1–1.5 weeks) | Displacement instead of reservation: copy K residents to pinned host at segment start (~25 ms a card), serve them through the ring, copy back after. Host-resident streams. Root-complex rebalancing (§2.4) if links bind. Admission threshold from M4's replay | Decode on held-out equal to today's | Decode −1.5–2.4% → 0; S_max up to 16K |

**What happens to the work in progress (#62, `wt-lcp`, still at `b0a84f7c`).**
- Keep these parts; all are needed as they stand:
  - the bound from 48 to C (constants, `tall_init`, CPU-path tables);
  - the skip's row suffix inside a tall chunk;
  - the read-only packed borrow and causal admission;
  - streaming miss submission and in-flight caps;
  - chunk geometry flags and `ab_window` parameters.
- Make the routed-expert consumer pluggable (CPU pool or card stager) and target C ≥ 2,048 with end alignment.
- Defer the 32-row CPU jobs: they buy at most the CPU ceiling of 60–100 tok/s.
- LCP on the CPU stays the fallback when the card path is off.

## 5. Beyond the exact floor: hardware and opt-in lossy prefill

`lm_beyond.py` and `lm_model2.py`, modelled; exact unless marked lossy.

| Tokens | Today's box | + second NVMe (2 × 6.4 + 1.45 GB/s) | 256 GB RAM (no drive reads) | 256 GB + ranks 2/3 rebalanced or a card off RC 00 | Lossy: cache-aware routing (−70% misses) | Lossy: drop drive misses | Lossy: low-bit drive copies (0.53× bytes) |
|---|---|---|---|---|---|---|---|
| 82 | 20 | 36 | 65 | 88 | 62 | 102 | 37 |
| 272 | 42 | 75 | 143 | 195 | 131 | 240 | 77 |
| 1,024 | 112 | **202** | **413** | **573** | 354 | 685 | 207 |
| 2,048 | 201 | 362 | 763 | 836 | 633 | 836 | 370 |
| 3,086 | 288 | 518 | 970 | 970 | 902 | 970 | 530 |
| 8K | 615 | 970 | 1,164 | 1,164 | — | — | 978 |

- **Second NVMe** (slot M2_1 is free).
  - It needs a copy of the expert planes (about 270 GB) and the reader split by share: 0.45 / 0.45 / 0.10 with the mirror.
  - The drive time per unit falls to 1.33 ms. 1K reaches about 200; that is the cheapest way to 200 at 1K.
  - It also gives decode about +18–20% (RESEARCH-3-LOSSY).
- **256 GB RAM** (8 × 32 GB RDIMM in place of 8 × 16 GB).
  - The cache grows to about 220 GiB, which holds all of about 12,480 file-tier units (about 209 GiB packed).
  - Prefill becomes link- or card-bound: about 410 tok/s at 1K, 760 at 2K.
  - Decode misses vanish too. This is the largest lever at every length below 3K.
- **Moving card 2 or 3 off root complex 00** helps only once the drive is gone: 1K goes from 413 to about 573. Peer-to-peer rebalancing (§2.4) gives the same without moving hardware.
- **Opt-in lossy.** These never enter the prefix cache. Each needs a flag and the teacher-forced KL harness from RESEARCH-3-LOSSY.
  - **Cache-aware routing in prefill:** bias the selection toward VRAM- and RAM-held experts. The literature reports misses falling from 28% to 3% at +0.1–3% perplexity. Assuming 70% of drive misses go away: about 350 tok/s at 1K.
  - **Dropping or deferring drive-missed experts in prefill:** about 685 tok/s at 1K. The quality risk is moderate, because the prompt's KV carries the error.
  - **Low-bit copies:** these need 129–186 GB the 990 PRO does not have (20 GB free). With a second NVMe, the exact path already gives the same speed.
  - **A 128/140-row consumer replay** (DeepInfer's lossy default): only about 150 tok/s at 1K.
- **Below about 500 tokens** nothing short of the 256 GB RAM upgrade plus rebalancing, or a lossy drop, reaches 200. Each touched expert costs 18.8 MB however few rows use it.

## 6. Risks and unknowns

| Risk | Where it bites | What resolves it |
|---|---|---|
| 8K/16K drive reads are extrapolated: multi-topic concatenation plus a Good-Turing tail | Upper bound 5,226 units with the skip, so at most +6% at 8K and +3% at 16K | Phase 0 M4 replay |
| Dense-half cost at tall heights (15–32 µs a row-layer) and the twin's real rate (L2/MALL hits) | Only from 8K up | Phase 0 M3 |
| The DMA lane rate on cards 2/3 | If it is ≤ 10 GB/s, the links reach 3.5–4.5 s at 1K–3K, still under the drive | Phase 0 M1 |
| Pinned staging and host RAM | 1.2–1.5 GiB now, more with Phase 4 | Measure MemAvailable in every window (`.mem` files) |
| VRAM reservation | Decode −1.5% to −2.4% until Phase 4 | Measured in the Phase 1 window |
| An op not in the sweep changing with height | Exactness | Widened sweep, KLD and KV hashes on every phase |
| Drive temperature at 100% for 10–40 s | Measured 55–60 °C at 43–58% busy | `NVMe busy ... drive at N C` lines |
| Stalls for concurrent users | Segments of 10–25 s | Cap C or S_max; interleaving is out of scope for v1 |

## 7. Files

All in `/home/as/.claude/jobs/635603c1/tmp/prefill200/`. Run with `python3 -I -B <script>`; each takes under 2 s.

| File | What it does |
|---|---|
| `lib/` | Copies of the PREFILL-REPLAY report's validated loader, replay, LCP and skip libraries (unchanged) |
| `lm_floor.py` → `lm_floor_out.json` | Whole-prompt unions, RAM/drive split, fresh and window-order caches, read-only against last-128-rows admission, decode reads after |
| `lm_floor2.py` → `lm_floor2_out.json` | Per-layer reads, producer/consumer split, prefixes and concatenations, union growth and its extrapolation |
| `lm_model.py`, `lm_model2.py` → `lm_model2_out.json` | Floors, card components, pipeline model, hardware and lossy variants, 4K segments |
| `lm_chunks.py`, `lm_chunks2.py` | Phase 1 chunks, start- against end-aligned |
| `lm_beyond.py` | Short-prompt hardware and lossy table |

# Critique

## Verdict

- **The direction is right. For 200+ tok/s, the expert rows must move to the cards and each touched expert must be read once per prompt.** I rebuilt the drive-floor replay from scratch, without importing any of the design's code, and got every number the design reports:
  - one segment: 3,488 / 3,874 / 4,051 / 4,219 / 4,622 / 4,585 / 4,800 units;
  - the no-skip and chunk counts as well, e.g. 3,086 tokens at C 2,048 end-aligned is 5,832 units (214 tok/s floor), and at C 3,072 it is 4,254 (292).
- **I found nothing in the design that breaks exactness.** There are gaps, listed in the exactness section below.
- **Short prompts cannot reach 200 tok/s exactly on this box, whatever is built.** With one NVMe and 96 GiB of RAM cache, the drive alone caps 82 / 272 / 933 / 1,024 tokens at 22 / 45 / 106 / 119 tok/s.
  - The "met from about 2K tokens" headline is too optimistic. My model gives 2,048 tokens of prose 191–194 (range 170–206), and the 1,890-token code document 164 (146–176).
  - A robust 200 needs about 2.5K tokens of prose or about 3K of code-like text, and that is with Phase 2.
- **The VRAM budgets are about 2× too low, so Phase 1's "decode within −2.5%" gate will fail as planned.** Details in finding 1.
- **The best lever is missing from the plan's order.** A second NVMe plus Phase 1 alone gives 200+ from about 1.1K tokens (finding 3). The hardware decision should come before Phase 2.

## Findings that change numbers or decisions

**1. VRAM is under-counted.** (`c4_vram.py`)

- deepseek41 declares no `rad_op_reads`/`rad_op_writes`, so every buffer lives for the whole program (`core/runtime/issue.cpp` ~:695). The arena is therefore the sum of all declared buffers: about 429 KiB a row.
  - Check: this model gives 8.96 / 11.35 MiB at 16 rows, against the logged 9.38 / 11.77 MiB.
  - The largest parts: MoE `egu`/`eff`/`edn` 141 KiB, streams h0 + h1 80 KiB, `engram_kv` 50 KiB.
- Adding liveness declarations would not help much, because almost every buffer is used in every layer. Only explicit buffer unions inside the plugin, or piece-height execution, shrink it.
- Items the design leaves out:
  - the window group at C + 128 rows (85 MiB at C 2,048);
  - the score plane, sized `irows = min(C, …)`: 66–128 MiB at C 2,048, not 32 MiB;
  - in Phase 2, h1 must be held across the MoE, which makes the per-row state about 122 KiB, not about 80.

| Case | Design's claim | My count |
|---|---|---|
| Phase 1, C 2,048 | 0.5–0.6 GiB, −1.5% to −2.4% decode | 1.05–1.30 GiB, 60–74 residents, −3.2% to −4.0% |
| Phase 1, C 3,072 | 0.6–0.8 GiB | 1.45–1.82 GiB, −4.5% to −5.6% |
| Phase 2, N 8,192 | 0.95 GiB | 1.43 GiB; 0.80 with streams on the host |

**2. Phase 1 at 3K tokens is a coin flip for 200.**

- My per-layer model counts speculative reads at their precision, adds a per-layer tail (pack, DMA, last unit, gather/all-reduce/post) and a counts sync, and uses per-layer link time. It is within about 6% of the design's model for one segment.
- Results at 3,086 tokens:
  - Phase 1, C 2,048: **198 tok/s (179–210)**;
  - Phase 1, C 3,072: 267 (238–285), at about 1.5 GiB a card;
  - Phase 2, one segment: 277 (245–298).
- Phase 1 at C 2,048 clears 200 reliably only from about 4K tokens: 218 at 4,103 and 242 at 6,263.
- The claim "the final chunk holds the skip's 2,432-row suffix" is false at C 2,048. Layers 20–23 need 2,064–2,448 rows, so they spill into the previous chunk. The replays already account for this.

**3. A second NVMe makes Phase 1 enough.** With u = 1.34 ms a unit, Phase 1 at C 2,048 gives:

| Tokens | Central | Pessimistic |
|---|---|---|
| 1,024 | 199 | 176 |
| 1,890 | 298 | 261 |
| 2,048 | 343 | 294 |
| 3,086 | 357 | 308 |
| 6,263 | 442 | 380 |

It is also worth about +18–20% decode. Phase 2 is then needed only for very long prompts.

**4. The 256 GB RAM numbers are optimistic.** The design's `max(card, pcie) × 1.1` assumes link copies overlap the dense half perfectly, which cannot happen before routing is known. My model gives about 290–410 tok/s at 1K, not 413–573.

**5. The Phase 0 gate contradicts the design's own estimate.**

- The gate says "twin ≥ 25 µs a row-layer → Phase 3 before Phase 2", but the design's own estimate is 31 µs, so it already fires.
- The measured 16-token gemv with uniform routing (667 µs, about 450 GB/s) implies about 45–50 µs a row-layer.
- That is still not binding for 200 tok/s. MoE per producer layer stays under the drive's time per producer layer up to about 6K tokens if the twin is at most about 58 µs a row-layer.
- Gate on that comparison instead.

**6. Root complex.** The design is right that ranks 2 and 3 share root complex 00. The engine's own card table confirms it: rank 2 is 03:00.0 under port 00:01.1, rank 3 is 06:00.0 under 00:03.1, rank 0 is c3 (root complex c0) and rank 1 is 83 (root complex 80). The draftcache note saying the opposite is wrong. All four root ports run Gen4 x16 ("link-capped" lines in the log).

**7. Host RAM is less tight than stated.** MemAvailable was about 12.9 GB through window B, not 7.5–8 GiB. Phase 4's resident copies (4.4–5.6 GB by my count) fit only if they are pinned at startup: runtime pinning was measured at 2.6–3.6 s a slab.

## Exactness: nothing breaks; add these

- **Phase 2 cross-layer buffers.** These must be sized for the whole segment, because the next layers read them for the same rows: h1, `index_sel` (layers between index layers), `index_keep` (layer 20 → 24–36), and `pre_f`/`post`/`comb`.
  - I checked `cs_b`: `c_tab` rewrites the same angles as a function of position, so its order is safe.
- **RAM-cache slots.** Lease borrowed slots until their DMA completes. A read-only borrow does not stop a concurrent insert from evicting them.
- **Sweep coverage.** The 20:46 sweep lacks `gemm_bf16f8_nt`, the FP8 dense row production actually uses. Also add `gemm_nt_f32`, the Engram ops, the compressor, `kv_store` and `index_topk` forms, and `moe_scatter` at 98K slots, all at piece and segment heights.
- **Ring stress test.** Add an arm with a 2–3-slot ring, delayed DMA and the host verify arm; the real-prompt gates will not catch rare races.
- **Guarded FMA.** It is valid only with an exponent guard, as the design says (the product is exact in f32 otherwise).
- **Phase 3's rate basis.** The "5.7 TMAC/s raw" counts the padded 32-row tiles; the live rate in that perf log was 4 rows an expert, about 0.7 TMAC/s. It is a plausible ceiling, not a measured rate. Phase 3 is not needed for 200.

## Realistic exact tok/s (central [pessimistic–optimistic], one NVMe, today's RAM)

| Tokens | Drive floor | Phase 1, C 2,048 | Phase 2, one segment |
|---|---|---|---|
| 82 / 272 | 22 / 45 | 20 / 41 | same |
| 933 | 106 | 98 [89–104] | 97 [88–103] |
| 1,024 | 119 | 109 [99–116] | 109 [98–116] |
| 1,890 (code) | 181 | 165 [149–176] | 164 [146–175] |
| 2,048 | 214 | 194 [174–206] | 191 [170–205] |
| 3,086 | 307 | 198 [179–210] | 277 [245–298] |
| 4,103 | 357 | 218 [197–231] | 323 [285–349] |
| 5,127 | 447 | 228 [206–241] | 400 [339–438] |
| 6,263 | 520 | 242 [219–256] | 443 [359–511] |
| 8K | — | — | about 470–600 (cards begin to bind) |

## Order to build, and go/kill gates

1. **Phase 0, as designed, plus:**
   - the arena line at C 1,024 / 2,048 (I predict about 0.47 / 0.93 GiB at 8,448 context);
   - the twin with real routing at M 256–2,048;
   - SDMA on ranks 2 and 3 at the same time.
   - **Kill / redirect:** if the drive in layer batches beside DMA reads under 6.8 GB/s, exact 200 at 3K needs Phase 2 or a second NVMe. Under 6.3 GB/s, Phase 1 tops out at about 175.
2. **Decide on the second NVMe now.** It decides whether Phase 2 is needed at all.
3. **Phase 1: the card stager on #62's chunks at C 2,048.**
   - Budget plugin buffer unions or local-slot MoE buffers, or accept −3% to −4% decode until Phase 4.
   - **Go:** byte gates identical (`--kld-record` with the skip off; KV hashes with the skip on, 82–6.8K tokens; replies), 3,086 tokens in ≤ 16.0 s, 6,817 in ≤ 29 s, NVMe busy ≥ 85%.
   - **Kill / profile:** any byte difference stops the work. If 3,086 takes over 18 s, profile before building further.
4. **Phase 2, only without a second NVMe, or for ≥ 4K prompts.** Go: 3,086 in ≤ 12.3 s and 6,817 in ≤ 17 s.
5. **Phase 4** (decode-neutral memory) before the card path becomes the default.
6. **Phase 3** only for prompts of 8K and up.
7. **Also say plainly:** chat turns are short and stay bound by the drive at about 20–100 tok/s. Prefix reuse (#63) and joint segments across concurrent requests (exact; 4 × 1K prompts read about 4,600 units instead of about 14,000) are the levers there.

## Files

All in `/home/as/.claude/jobs/635603c1/tmp/prefill200/critique/`. Each is single-thread and runs in under 5 s.
- `c1_floor.py`: independent replay of the drive floor and chunk counts.
- `c2_pipeline.py`: my per-layer pipeline model.
- `c3_hw.py`: second NVMe and 256 GB variants.
- `c4_vram.py`: VRAM from the buffer declarations.
- `c5_p1nvme.py`: Phase 1 by chunk size with one and two NVMe drives.

Outputs are in `c1_out.txt`–`c5_out.txt` and `c1_out.json`/`c2_out.json`.
