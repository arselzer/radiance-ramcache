# Prefill design critique

**Review of the DS41 large-chunk prefill (LCP) design**

I worked read-only and used no GPU. The repo is at HEAD `9c8dccb9`, unchanged while I worked. The design's own scripts reproduce its tables: I re-ran `t6_model.py` and `t7_cpath.py`. My cross-check is `/home/as/.claude/jobs/635603c1/tmp/prefill/r1_review_check.py` (single thread). The main calibration source the design did not use is the long-prompt window `tools/ds41-host-experts/build/ab-20261029T145343Z`, which was still running when the design was written. It ran today's best configuration with FP8 and read-ahead on prompts of 82–3,086 tokens of coherent documents.

## Verdict

- **Direction:** right. The large chunks, the read-only cache and the exactness approach (E2) all hold up.
- **Exactness:** sound. My audit confirms that only three ops change arithmetic with pass height. Every code citation checks out, apart from minor line drift.
- **Gain:** overstated for real long documents. The calibrated gain is about ×1.8–2.8 (central ×2.2) on the measured documents, not ×3.1–4.6. Three reasons:
  - On those documents, residents carry 34% of prefill rows, not 54.6%. The CPU pool, not the drive, becomes LCP's bottleneck.
  - Today's LRU adapts to the prompt (88.6% hit rate). A read-only cache seeded for decode gives that up.
  - The `max(drive, CPU, card) × 1.1` model ignores the card's work at each layer before routing, which runs in series with that layer's reads.
- **Engineering gaps** (details below): the insert queue has no cap for an admitting chunk; read-only must apply to pure prefill passes only; the E2 switch can't be verified by the engine; window KV and the score plane need VRAM; lent residents become misses.
- **Order:** prefix caching should come first. Passes of up to 48 tokens are a cheap first step towards P0.

## 1. Numbers re-checked

| Design claim | Verdict | Evidence |
|---|---|---|
| Per-pass law: 114 ms + 2.70 ms a miss | Fits its own window; does not transfer to long documents | Long window: 394 passes, 51,375 prefill misses (0.815 a pass, layer and rank, against 1.29 held-out). The law predicts 183.8 s; measured 220.4 s (−17%). The implied fixed cost is 207 ms a pass |
| "Limited by the drive"; "83% of each pass is miss reads" | Partly wrong | NVMe busy 47% (`window.log`). Rank 0's issuing thread waits 87 s (40%) on miss reads, 50 s (22%) on CPU rows and 72 s (32%) at the routers (card work plus other ranks' misses). The pool is about 60% busy (7.1 units, 16 rows a layer-rank). It is a per-layer latency chain with drive and CPU both about half used |
| Long prompts: 34–50 ms a token | Low end right, high end too high | Measured 34.8 / 37.3 / 32.6 ms a token at 933 / 1,890 / 3,086 tokens. 8,192 tokens ≈ 4.5 min |
| Residents carry 54.6% of prefill selections | True for the profile only | Long documents: 16.0 of 24 rows a prefill layer-rank go to the CPU, so 34% resident (shutdown lines, all ranks) |
| CPU 0.04–0.09 ms per further row at 8–32 rows | Key unknown; probably optimistic | Today's marginal row cost rises with rows (2nd/3rd/4th row ≈ 0.36/0.46/0.53 of a row, NIGHT-LOG 07:15). No 32-row kernel exists yet (`ds41_host_experts.h:106`, `kGemvRows = 4`), so the "benchmark first" step needs a prototype first |
| LCP ×3.1–4.6 at ≥ 1,024 tokens | Corrected | Measured documents: 12–20 ms a token, ×1.8–2.8 (central 16 ms, ×2.2). In-profile text: 11–15 ms a token. CPU floor of 9.5–16 ms a token at 0.06–0.10 ms a row, whatever the chunk size (158 non-resident rows a token) |
| Card cost | Model inconsistent | `t6_model.py` `card_ms` uses the tiled kernel's rate, but E2 runs the gemv. The card is not the limit (≤ 1–1.5 s a chunk), but that phase runs in series with each layer's reads |
| Arena 7.41 MiB at 16 tokens; 0.2–0.45 MiB a token | Incomplete | Today's log shows 9.38 MiB. At 1,024 tokens: streams 80 MiB, routed MoE ≤ 141 MiB (before aliasing), index score plane 33 MiB at 8k context (128 MiB at 32k), window KV +40 MiB (`kv.cpp:1260-1262`). About 200–290 MiB a card, 11–16 residents, −0.6% to −0.9% decode |
| KV 13.8 KB a token a card | Minor error | The log shows about 3.1 KiB a token for the growing groups, plus a fixed window of about 7 MiB |
| Second NVMe: −38% | Workload-dependent | Held-out prompts −34%; long documents −26% |
| 256 GB RAM: 65–90 tok/s at 16-token passes | Optimistic | With no misses, about 330 ms of pool time a 16-token pass caps it near 40–50 tok/s |
| Read-ahead ceiling 17% (16-token passes) | Wrong premise | The NVMe is idle about 53% of prefill. The real limit is prediction: today's predictor costs 667 µs a token and layer (server log) |

## 2. Code claims

All checked; accurate:
- `r4d_ds41_mxfp4.hip:805`; `ep_route_counts.h:9`; `ctx.cpp:651`; `stager.h:357` and `:455`.
- `stager.cpp:673, 717, 991/1010, 1103, 1243/1251, 1659, 1786, 1843`.
- `host_expert_pool.h:43, 64, 122`; `mover.cpp:873, 913, 1544, 1653, 1758, 1966`; `native_mapped_file_cache.h:290-309`.
- `engine.cpp:1613/1626/1663`; `engine_bringup.cpp:1527, 4128`; `planner.cpp:793-795`.
- `geometry.h:63-71`; `prefix.cpp:186-216`; `r4d_gemm_args.h:121-127`; `r4d_rows.cpp:4719-4722`.

Drift:
- The WMMA switches are at `r4d_ds41_sparse.hip:1850` (attention) and `:1600` (index), not `:1208` / `:1425`.
- `ab_window.sh` already has an `LmN` part (`:348`). Only `--max-num-batched-tokens 16` and `--no-prefix-cache` are hard-coded (`:140-141`).

## 3. Exactness audit

**The three switches are enough.** I checked every other op whose kernel varies with height:
- **Dense GEMMs:**
  - m16 forwards to m64.
  - The tiled row is byte-identical by contract (`r4d_gemm_bf16_nt_tiled.hip:14-19`).
  - The split depends only on N tiles and K (`r4d_gemm_args.h:121-127`).
  - FP8 uses one row at every height (`r4d_rows.cpp:4719-4722`).
- **`gemm_nt_f32`** (router and compressor): its forms are byte-identical (`r4d_glm.hip:107-113`).
- **`mhc_mix`** split and fused forms: same sums (`r4d_ds41_elem.hip:700-704`). **`mhc_post`** vec and plain: same order (`r4d_mhc.hip:375-425`).
- **Exact all-reduce:**
  - One-shot up to 524,288 elements; two-shot above that, so passes over 102 tokens switch.
  - Both forms fold from `0.0f` in rank order (`r4d_ar_oneshot_Nrank_exact.hip:26-34`, `r4d_ar_twoshot_Nrank_exact.hip:18`).
- **`moe_scatter`**, 1 launch or 3 (above 170 tokens): both are a stable counting sort, so the output is unique (`r4d_rows.cpp:8196-8214`).
- **`moe_gather`**: ascending slot order in every form (`r4d_moe.hip:2103-2106`). ds41's f32 routing weights never take the wide form (`:3226`).
- **`index_topk`** split and merge: an exact selection either way.

**MoE gemv at any height.** Each output keeps its lane and block order whatever M, XLM or the K-chunk path (`r4d_ds41_mxfp4.hip:285-287`, `:376-389`). Removing the `tokens <= 16` guard fits in LDS: 36 KiB at XM 8, K 5,120.

**Attention.**
- Skipping WMMA at `:1850` falls through to the old scalar kernel (`attn_emit`). `attn_dec_emit`, which the 16-token reference uses, is only called for rows ≤ 48 (`:1861`).
- Both claim libref's bits, but E2 should send tall passes to `attn_dec_emit`. It is the reference's own kernel and faster (367 µs per 48 rows).
- The index score's scalar forms (`:1608`) claim the same bits as each other.

**Chunk boundaries and KV writes are invisible**, given row-invariant kernels:
- Window, compressed and index keys are rounded (`fp8`/`fp4` round trips) and written to the bf16 caches before attention reads them back from the cache (`deepseek41.cpp:1295-1307`, `:1362-1373`).
- The compressor's partial block lives in an f32 ring (`:1044-1052`, `:1374-1380`); Engram uses its conv state.
- The window group must hold window + step blocks a sequence (`kv.cpp:1260-1262`): 63 more blocks at 1,024 tokens.

**Robustness gap.** An environment switch read inside libr4d can't be checked by `hx_init`. With a stale `libr4d.so` in a run home, the card would silently run tiled/WMMA kernels for residents while the CPU computes gemv bits. Make the switch visible at declare time: a distinct row, an op parameter, or an exported symbol that `hx_init` checks and refuses without.

**Add to the P0 gate:**
- Final chunks of 1–15 and 17–48 tokens.
- Passes crossing 102 tokens (all-reduce form) and 170 tokens (scatter form).
- Prompt lengths that are not multiples of 16.

**E2 over E1.** E2 makes outputs independent of chunk size, so prefix resume points, mixed steps and final-chunk sizes all keep the bits. That is a strong reason to prefer it.

## 4. VRAM lending (P3)

The mechanics the design cites are right:
- `lend_from` returns early for static plans (`mover.cpp:1966`).
- The arena loan needs the heat engine (`engine_bringup.cpp:4128`).
- deepseek41 lacks `shape_probe_ok`, which lending needs (`:1526-1529`).

What the design misses:
- The smallest arena level is 256 rows (`lo = max(lo, 256)`, `engine_bringup.cpp` around `:1534`), so about 25% of a 1,024-row arena is never lent.
- Lent residents are not in the RAM cache, because `borrow_host_unit` only serves units on the SSD tier (`mover.cpp:878`). Every touch during the prompt becomes a drive miss: +1.75–2.45 misses a layer-rank a chunk for 70–98 lent units, about +7–10% drive time.
- The refill takes 0.7–0.95 s, and decode pays misses until it ends.
- Expert-table pointers can race with in-flight kernels.

**Verdict:** keep P3 last. Its cheaper variant P3-lite (§7, row 5) is the first card offload to try if LCP measures CPU-bound.

## 5. RAM cache interaction

- A read-only borrow that doesn't touch the LRU is sound and small: `borrow_no_touch` exists; a packed variant is needed.
- **Restrict it to pure prefill passes** (prefill phase with no decode rows). In a mixed step it would keep decode misses out of the cache, as the loop's earlier strategy note (`DS41_NATIVE_PREFILL_CACHE_STRATEGY_V2`) warned. The design keys only on pass height.
- **A read-only cache gives up today's adaptivity.** Out of the drive's 113,128 non-resident uses on rank 0, 100,278 hit (88.6%), thanks to reuse within the prompt.
  - LCP's miss reduction against today's 52 per 1,024 tokens per layer-rank is only about ×1.4–2.1 on out-of-profile documents (25–37 misses a chunk).
  - For coherent documents, admitting everything at C = 1,024 may beat read-only, because consecutive chunks share experts.
  - Replay the trace before choosing.
- **Final-chunk admission is unbounded.**
  - It inserts about 1,000–1,200 units a rank in one burst.
  - The demand path pushes to the insert queue with no cap (`stager.cpp:1840-1845`); only the warm-up waits for fewer than 24 (`:2199-2202`).
  - Each queued unit pins its 18.8 MB of planes, which can grow to several GB against about 13 GB free, and adds about 36 MB of DRAM traffic per insert.
  - Bound it: admit only experts touched in the chunk's last 128–256 tokens, cap per layer, and block above N queued.
- **The read-only threshold should equal the chunk size, not 256.** At 256 tokens a read-only pass misses about as often as today's LRU.

## 6. Simpler alternatives

- **Prefix caching:** the largest lever for chat and agent use. Each turn costs the new tokens plus up to one checkpoint interval, instead of the whole history (a 3,000-token history: about 105 s today, about 5–10 s with an interval of 128–256). It is exact by construction, but untested on ds41 with the CPU path, since `ab_window.sh` always passes `--no-prefix-cache`.
- **Passes of up to 48 tokens:** attention and index are already exact up to 48 rows (`:1861`, `:1600`). Only the MoE switch and `kEpRouteMaxRows = 48` are needed. That gives 5–20% on every prompt (the fixed cost of a pass is 114–207 ms) for about one resident. It also tests MoE invariance in production.
- **The base engine's prefill machinery is not simpler:**
  - Whole-layer reads need about 3.2 GiB of file buffers a card.
  - The card's copy path can't read the packed cache (startup line "the card's copy path neither reads nor fills it").
  - `PrefillStager` stages pinned-pool units only (`stager.h:118-121`).
  - Reusing it amounts to P3.
- **Hardware:** a second NVMe (slot M2_1 is free) gives prefill −26% to −34% today and roughly +20–30% decode. With LCP it leaves the CPU as the only bottleneck.

## 7. Corrected ranked plan

| # | Item | Gain (calibrated) | Decode | Effort | Gate |
|---|---|---|---|---|---|
| 0 | **Measure first:** (a) prototype a 32-row CPU kernel and benchmark it at 4/8/16/32 rows; (b) one traced re-run of the 14:54 documents plus one 6–8k document, then replay read-only, admit-all and bounded admission at C = 512 / 1,024 / 2,048 offline; (c) base-engine startups at 1,024 and 2,048 to read the arena line | Decides chunk size, cache policy and whether the card must take rows | 0 | 1–1.5 d + a ~20-min window | — |
| 1 | **Prefix caching** (`--checkpoint-interval` 128–256) | Multi-turn: TTFT about O(new + interval). Single-shot: 0 | ~0 (about 6 MiB pinned a checkpoint a card) | 1–2 d + 1 window | Cold == hit on ds41 TP4 with the CPU path |
| 2 | **P0 (E2) with a verifiable switch; first ship passes of ≤ 48 tokens** | 5–20% on all prompts | ~0 | 3–4 d | Device checks (tall == ≤ 16 rows) plus the end-to-end identity window above |
| 3 | **P1: LCP on the CPU at C = 1,024.** Read-only for non-final pure prefill chunks; bounded final-chunk admission; streaming miss submission; caps on planes and on the insert queue; 32-row jobs; keep `FETCH_LIST` off (its tables scale with `kEpRouteMaxRows`, `stager.cpp:1540`) | Measured documents ×1.8–2.8 (about 13–20 s per 1,024 tokens against 34–38 s). In-profile text about ×2.4–3.1. About 10–30% at 256 tokens | −0.6% to −0.9% permanent (more at ≥ 32k context) | 6–8 d | E2 + byte tests at 1–32 rows in every layout |
| 4 | **P2:** prefetch the next layer's likely misses inside LCP | −5% to −10% | 0 | 2–3 d | Exact |
| 5 | **Card offload, only if item 0 shows CPU-bound.** P3-lite: 2–4 permanent file-buffer units, top-N row-heavy units on the card, packed→block widening. Or full P3 (lending) | P3-lite −15% to −25% when CPU-bound. P3 ×1.3–1.6 at ≥ 2,048 tokens, minus the lent units' extra misses | −0.2% to −0.9% / 0 | 4–6 d / 9–12 d | E2 gemv on the card |
| 6 | Read-ahead for 16-token passes (needs a batched predictor) | −8% to −15% below about 512 tokens (unverified) | 0 | 3–4 d | Exact |
| 7 | Mirror share sweep; FP8 tiled row (commit 10) | ±2%; 1–3% | 0 | ≤ 1 d each | Exact |
| 8 | P4 / layer-major execution | Large, needs card offload | Large VRAM lend | Extra large | — |

**Decision gates:**
- If the 32-row kernel reaches ≤ 0.06 ms a row, keep C = 1,024 and the CPU alone.
- At ≥ 0.08 ms a row, LCP is CPU-bound, so larger chunks don't help; go to item 5.
- Pick the cache policy from the replay, including decode misses after the prompt.
