# Prefill (and decode) ideas from other engines: survey and synthesis

## Synthesis

# Addendum to PREFILL-DESIGN.md: prefill ideas from DeepInfer, DS41Engine, Strata, QFN-radiance, hipEngine and the offload engines

This addendum combines the five surveys with PREFILL-DESIGN.md and PREFILL-CRITIQUE.md. It uses the critique's calibration:
- Today: 33–37 ms per prompt token on real documents.
- Large-chunk prefill (LCP): 12–20 ms per token, with the CPU pool and the drive both near their limits.

I modified no repository and used no GPU. I re-checked the claims the plan depends on in the source. Local line numbers refer to `/home/as/radiance-ds41-claude` at `a566c066` (`9c8dccb9` plus one README commit).

Three single-thread scripts, each under a minute, are in `/tmp/addendum-calc/`:
- `tri.py`: the savings model.
- `toy_triangle.py`: a dependency check of the layer-skip schedule.
- `normwidth.py`: an emulation of the norm-width effect.

Link prefixes:
- DE = `https://github.com/vincalkr/DS41Engine/blob/d0db7270316c5015295fe1f2df273a54ff16c00f/`
- DEc = `https://github.com/vincalkr/DS41Engine/commit/`
- ST = `https://github.com/Niko1221/Strata/blob/46c152a1310f77cd2f0ce2c7cd213962df14fe83/`
- DI = `https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/`
- HE = `https://github.com/shisa-ai/hipEngine/blob/cc30876b/`

## 0. Bottom line

- **The design misses one large exact lever: run layers 20–39 only for the prompt rows that can reach the output.**
  - Layers 21–39 see earlier positions only through layer 20's compressed and index keys and through their own 128-row window. So only the last 1 + 127·(39 − L) rows need layer L: 2,414 rows at layer 20, down to 1 row at layer 39.
  - DS41Engine measured +39%, +61% and +74% prompt speed at 8K, 17K and 25.5K tokens. The greedy tokens were the same and the last-row logits byte-identical. Source: DEc/6aa1899d and DE:tools/v41/engine/ds41_generate.cpp:639-681.
  - It is on by default in DS41Engine and Strata. DeepInfer documents the exact 2,560-row bound and ships a lossy 128/140-row version (DI:doc/inference_deployment_qa.md:187).
  - Modelled here, in today's 16-token passes: 3,086 tokens go from 100 s to 72–77 s, and 8,192 tokens get −34% to −40%. On top of LCP the fractions are about the same. No P0 work is needed, and the effort is about a week.
- **A pass-granular form fits radiance without row offsets or provisional rows.**
  - Each 16-token pass runs layers 0..E fully, then layer E+1 only far enough to write its window key. On earlier passes layer 20 also writes its compressed and index keys.
  - Toy check against full execution: 476 of 476 cases matched (the last row, every layer's last 127 window keys, the global keys). No computed row ever read an unwritten window entry.
  - The window state at radiance's exact-mode tail checkpoint is complete in 354 of 354 cases.
- **P0 misses a fourth op whose arithmetic changes with pass height: the norms.**
  - All rmsnorms, and ds41's fused collapse, post-collapse and norm-rope norms, choose their thread count from M: 1,024 threads at 32 rows or fewer, 512 at 33–120, 256 above (`libr4d/r4d_common.h:278-282`; uses at `r4d_model_bf16.hip:1001` and `r4d_ds41_elem.hip:1936, 2028, 2075`). That changes how the sum of squares is grouped.
  - Emulated: 68–73% of row sums change, and 1.3–3.8% of rows get at least one different output bit per 5,120-wide norm. With about 80 such norms per token, nearly every row diverges, while greedy replies can still match.
  - So the norm width must be pinned too, and the critique's "passes of up to 48 tokens" first step must be **32 tokens or fewer**.
- **A flag trap affects every chunk size.** ds41 has linear state, so `--checkpoint-interval` (default 2,048) caps chunk length even with the prefix cache off. With the critique's 128–256 interval for prefix caching, LCP chunks would silently stay at 128–256 tokens unless `--prefill-span C/I` is set.
- **The other engines confirm LCP and P3, but none is exact, and their rates are not comparable to ours.** DS41Engine reaches 371–493 tok/s and Strata 200–431 tok/s with 2-bit packs, 1K–12K-row batches and every expert computed on the GPU. Their step beyond ours is layer-major execution, which should replace P4.
- **Decode: nothing large.** The power-settings idea was already measured here at +0.6%. CCD-L3 staging is already RESEARCH-2 item #8.

## 1. Ideas the design already covers (confirmations)

| Design item | Evidence from other engines (measured unless marked) | Adjustment |
|---|---|---|
| LCP: big chunks, each expert read once per chunk (P1) | <ul><li>Strata, 8K prompt: chunks of 1,024 / 2,048 / 4,096 / 8,192 gave 200 / 277 / 353 / 431 tok/s (ST:src/dsv41/ds4_model.cu:2206-2210). Its move from 16-token windows to 8K chunks gave ×4–10.</li><li>QFN on one card: 2,048 → 4,096-token chunks took it from 1,413 to 2,545 tok/s (compose/flashnext-tp1.yaml:6-9).</li><li>SeqMoE: chunked prefill "incurs repeated expert transfers across chunks".</li></ul> | None |
| RAM tier read-only during a prompt | <ul><li>DS41Engine rebalances only between requests.</li><li>Strata dropped its post-prompt refill, which cost a 4K prompt 4.7 s. The 300 tokens after an 8K prompt took 51.9 s without it against 48.6 s with it (ST:docs/release-notes-0.2.7.md:17).</li></ul> | <ul><li>Supports the critique's bounded final-chunk admission: the refill is worth about 7% of the decode that follows.</li><li>Reuse the existing pure-prefill predicate (`core/place/native_file_cache_policy.h:12-17`: no decode rows, no draft, more than 16 tokens).</li></ul> |
| Routed-only reads in tall passes (`kRoutedRows`) | DS41Engine's on-demand reads beat whole-layer prefetch: 8,000 tokens 33.6 → 32.4 s; +1,000 tokens 19.1 → 16.6 s (DE:tools/v41/engine/ds41_moe.cpp:986-993) | Read whole layers only in batches of 8K rows or more, where 97% or more of experts are touched |
| P2: next-layer prefetch | <ul><li>DS41Engine prestages the next layer during the dense half: +5.8% (DEc/3db46ed8).</li><li>HOBBIT: about −10%.</li><li>DuoServe and vLLM RFC 63666 read full layers without a predictor.</li></ul> | Keep the −5% to −10% estimate |
| P3: the card takes row-heavy experts | <ul><li>ik_llama.cpp offloads the MoE op when tokens·k ≥ 32·E, which is 2,048 tokens for ds41.</li><li>ktransformers switches at ≥2,048 tokens for DS-V4-Flash.</li><li>QFN's PrefillStager stages 225 GiB per 47K-token request with 0 overflows.</li><li>HybriMoE's scheduling alone gives 1.26×.</li></ul> | Change the mechanism (§2, idea #3) |
| P0, approach E2: batch-invariant kernels | <ul><li>DeepInfer's batch-invariant mode asserts bitwise-equal logprobs between teacher-forced decode and one 17K-token prefill (DI:tests/models/test_kl.py, KL bound 0.0).</li><li>hipEngine's chunked prefill is byte-identical to serial decode at 64- and 256-token chunks (HE:docs/campaigns/QWEN3.8-FLASH-NEXT.md:982-1000).</li></ul> | Feasible, but add the norm-width pin and gate on bytes (§4.1) |
| "gemv staging 2+ rows" (item 8 / P4) | <ul><li>Already in libr4d: the chunked-K path stages 8 rows (`kMxXM`, `r4d_ds41_mxfp4.hip:194`), and its bits do not depend on the path (`:284-286`). Today gate_up re-reads its weights once per row (`:376-381`).</li><li>hipEngine's row and output batching cut a 4K prefill from 291.6 to 231.8 s, byte-identical (HE:docs/KERNELS.md:275,277).</li></ul> | <ul><li>Move into P0 as a launcher heuristic (½–1 day).</li><li>Resident MoE drops from about 0.7 s to 0.1–0.2 s per 1,024-token chunk.</li><li>It removes E1's 3× advantage at chunks of 4K or more.</li></ul> |
| Layer-major execution (item 9) | <ul><li>DS41Engine: batch 8,192 → 12,288 rows gave 318 → 388 tok/s, and file reads fell from 157 to 114.5 GB (DEc/86879129).</li><li>Strata, 16K chunk on a V100: 461.8 → 321.3 s, and 736 → 506 GB read.</li><li>MoE-Gen: 841 tok/s against 77–109.</li></ul> | Replaces P4 (§4.2) |
| Prefix caching (item 2; critique #1) | <ul><li>QFN in production: exact resume, an 8 GiB host KV tier, 40 snapshots of 113 MiB.</li><li>DS41Engine resumes agent turns at 120–190K context in 0.6–10.5 s (DE:docs/ds4/V41_EXACT_FAST.md:79-80).</li></ul> | Add end-of-request snapshots (§2, idea #2) |
| Short prompts stay on small passes | <ul><li>DS41Engine runs up to 96 new tokens as decode passes: a 171-token turn went from 9.7 to 4.0 s.</li><li>QFN never holds a tall chunk below 1,024 tokens.</li></ul> | LCP threshold = chunk size (as the critique says) |
| CPU path: 32-row jobs, 4-row blocks, dynamic tile claiming | <ul><li>ktransformers' AVX2 MXFP4 kernel uses the same plan: each weight row loaded once and shared across tokens in 4-token blocks (kt-kernel/operators/avx2/mxfp4-moe.hpp:228-253).</li><li>Its dynamic task scheduling gives 1.83×.</li></ul> | Already planned |
| Power settings (APST off, governor, C-states) | DS41Engine benchmarks with them on | Already measured here: decode +0.6%, TTFT unchanged (`NIGHT-LOG.md:111`). Drop |

## 2. New ideas, ranked by expected prefill gain

TTFT is calibrated to the critique: today 33–37 ms per token; LCP 12–20 ms per token. "Model" means not measured.

| # | Idea | Source | Exact? | TTFT effect | Decode | Effort |
|---|---|---|---|---|---|---|
| 1 | **Exact consumer-suffix skip, pass-granular** (details below) | DS41Engine enc-skip; Strata YOCO (`keep = 20·127 + 128`, ST:src/dsv41/ds4_model.cu:9024-9038); DeepInfer Q&A:187 | Yes, by dependency. Passes stay 16 rows, so no P0 work | <ul><li>Today (model ×0.80–0.95 realisation, from DS41Engine's measured/ideal ratio): −9 to −10% at 1K; −17 to −20% at 2K; −24 to −28% at 3,086 (100 → 72–77 s); −34 to −40% at 8K.</li><li>With LCP and per-layer row suffixes: −6 to −11%, −17 to −21%, −24 to −30%, −34 to −43%.</li></ul> | 0 | About 1 week; +3–5 days for row suffixes inside LCP chunks |
| 2 | **End-of-request snapshot**; once E2 holds, exact resume at any 16-token boundary | DS41Engine conversation pool; Strata slots; DeepInfer Q&A:165. Radiance: "decode writes no snapshot" (`core/sched/scheduler.cpp:1337-1338`), and resume only on the chunk grid (`core/sched/geometry.h:63-72`) | Yes if decode rows equal prefill rows, which holds at 16 rows or fewer today. Gate: cold run == cache hit | <ul><li>A follow-up turn stops re-prefilling the reply: a 500-token reply costs 16–19 s today and 6–10 s with LCP.</li><li>Under LCP's 1,024-token grid, it also avoids re-prefilling up to 1,023 tokens per hit.</li><li>Only for agent and tool sessions and non-thinking chat: V4.1's encoder drops earlier reasoning unless tools are present (DE:tools/v41/dsv41_encoding.py:692, 785-788).</li></ul> | About 0 (a 5–7 MB device copy per rank every 16 decode tokens) | 2–4 days after prefix caching |
| 3 | **Card share through SDMA copy lists and a small permanent ring**, instead of lending 50 resident slots with tier flips | <ul><li>QFN PrefillStager (`core/place/stager.h:1-115`): ring at the top of the arena, 3 layers ahead.</li><li>SDMA moves 28.7 GB/s at no cost to kernels; the runtime copy moves 22 GB/s and slows kernels 8% (`core/device/aql.cpp:2699-2706`). ds41 never opens it today.</li><li>DuoServe uses a 2-slot ring; ktransformers 2 pinned slots.</li><li>HybriMoE: split chosen by a simulated timeline.</li></ul> | Yes: same bytes, E2 gemv | <ul><li>As the critique's P3-lite: −15 to −25% when LCP is CPU-bound.</li><li>With the row-batched gemv, the card beats the CPU from 8–19 rows per expert instead of 10–34.</li></ul> | −0.2 to −0.9% (4–8 slots per layer, two layers). No resident tier flips | 4–6 days |
| 4 | **FMA in the exact chains**: MXFP4 values have at most 2 significant bits, so a bf16 × MXFP4 product is exact in f32 and fma gives the same bits as multiply-then-add. Fall back for bf16-subnormal rows or extreme scales | hipEngine report; our kernels today use "no FMA" (`tools/ds41-host-experts/ds41_host_experts.h:14-18`) | Yes, under the guard; prove it with the host tests in every layout | <ul><li>Card: about half the VALU ops in the row-batched gemv.</li><li>CPU: Zen3 runs adds on their own pipes, so multiply-then-add already reaches FMA's peak. Expect 0–15% on the per-row cost, so 0–8% TTFT when CPU-bound.</li><li>Try it inside the critique's 32-row prototype.</li></ul> | About 0 | 1 day with tests |
| 5 | **Concurrency**: several prompts in one tall step; LCP threshold fed to `set_stage_rows` (`core/engine.cpp:2113-2117`); `--prefill-interleave` | QFN; DeepInfer sizes chunks from a time budget | Yes under E2 | Only with more than one sequence. Two 1K prompts take about 13–16.5 s together against 22–28 s in turn (model, drive-bound constants; less gain if CPU-bound) | Bounds decode stalls | 0.5 day |

**Not worth it here:**
- Engram gathers in chunk form: 1% at most, and only with the table on disk.
- All-reduce on a second lane: under 2% of a chunk.
- Slicing the MoE into row passes: saves only arena, and the CPU path refuses sliced passes.
- Zero-copy expert compute: 19.4 GB/s with a weight re-read every 8 rows; SDMA copies are faster.
- Exact tall index and attention forms: matter only at 32K and beyond.
- Exact FP4 KV storage: saves memory only.

**Idea #1, mechanism**
- **Why it is exact.**
  - The topology is checked at `arch/deepseek41/deepseek41.cpp:262-276`.
  - Compressed and index keys exist only at layers 2, 8, 14 and 20 (`L.src`, `:1314-1384`).
  - Layers 24, 28, 32 and 36 only score their own queries against layer 20's keys (`:1386-1430`).
  - Every layer's window key comes from its own input (`:1284-1312`).
- **The schedule.**
  - Let F be the index of the prompt's last pass. Layer L runs fully on passes from F − 8·(39 − L) on.
  - In the 8 passes before that, it runs only its window-key ops: collapse norm, `wkv`, norm/rope/round trip, `kv_store`. They must be the same fused ops as in the full layer, so the bits match.
  - On every earlier pass, layer 20 runs its key ops: window key, compressor at ratio 1, index keys, stores.
  - So each pass runs a contiguous prefix of the graph plus one key-only layer. In code this is an early exit in `step()`'s layer loop (`:2434`). The head already runs only when `n_out > 0` (`:2767`).
- **Work saved.** At 3,086 tokens, 5,400 of 7,720 layer-passes run (−30%).
- **Engine changes.**
  - `RadBatch` carries the prompt's end. Only pure prefill passes use it: no decode rows, speculation off, not in KLD or scoring mode.
  - The scheduler does not publish periodic checkpoints before the cut, because their consumer windows are incomplete. The tail checkpoint stays, because it is complete.
  - The FileStager must not read ahead for layers above E+1.
  - Every rank derives E from the batch.
- **No boundary capture needed.** The loop's CED critique asked for a boundary capture and separate encoder and decoder progress (DS41_CED_EXACT_REPLAY_CRITIQUE…:142-170). In-order passes need neither: each pass commits normally, and the window trim never drops an entry that a computed row reads.

**Lossy, opt-in only**

| Idea | Source | TTFT | Note |
|---|---|---|---|
| 128/140-row decoder replay | DeepInfer's default `KEEP_TOKENS=140`, which its Q&A calls approximate; Strata `DS4_YOCO=2` ("shifts the answers a little") | A further −30% at 1K, −24% at 3K and −13% at 8K on top of #1 | Never enters the prefix cache |
| Top-5 experts | Strata: prompts +5–7%, decode +4 to +13%, KL 0.10987 vs 0.10833 | About −8 to −15% here (estimate) | Covered by RESEARCH-3-LOSSY's drop arms |
| Low-bit copies of missed experts | HOBBIT | Up to ×1.8 on the drive-bound share | Quality unknown |
| WMMA or FP8 MoE, WMMA indexer, `wht6` wire | hipEngine, QFN | About 0 while the drive or CPU is the limit | hipEngine's screen missed 2 of 3,000 outputs under truncating WMMA |

## 3. Decode ideas that surfaced

| Idea | Source | Status here | Expected |
|---|---|---|---|
| NVMe APST off, `performance` governor, C-states off | DS41Engine `local/power-tune.sh` | Measured null here (`NIGHT-LOG.md:111`) | Drop |
| CCD-L3 staging of predicted CPU experts | DS41Engine: L3-hot experts compute 1.41–1.56× faster; next-layer prediction catches 59–72%; estimated +3–6%, never built | Already RESEARCH-2 #8: +1.5–4%, large effort | No change |
| Admission seeded from the whole prompt's routing counts | ktransformers dynamic expert update: +20% / +11% decode at 10% / 40% GPU experts, −8% at 80%; vLLM RFC 63666 | New; pairs with bounded final-chunk admission | +2–6% decode after long prompts (estimate); add as an arm in the critique's replay |
| Background promotion of recurring misses into VRAM | DS41Engine: 14.63 → 15.74 tok/s early on; later VRAM seeding gave nothing | Needs runtime tier flips under the static plan | 0 to +5%, low confidence; defer |
| PM4 command tapes / graph capture | hipEngine +8.1% on gfx1100; hipfire +1.1–7.9% on gfx1201 | Host issue costs about 2% here (`REVIEW.md:300`); task #47 | 2% at most |
| FMA in the gemv chain | hipEngine | Decode gemv is limited by memory | About 0 |

## 4. Changes to the plan

### 4.1 Corrections before writing code

1. **Add a fourth pin to P0: the norm width.**
   - With `EXACT_TALL` on, `r4d_act_threads` must return 1,024 at every M for every norm in the ds41 graph. The cost is a few ms per chunk.
   - Ship passes of 32 tokens or fewer first, not 48. That still gives about 4–15% on every prompt.
2. **Device check: sweep every op over pass heights, not just three ops.**
   - Run every op the ds41 graph declares at M ∈ {1, 16, 17, 32, 33, 48, 49, 72, 73, 120, 121, 170, 171, 1024}.
   - Compare each row's bytes with the same row launched in a pass of 16 rows or fewer.
   - Keep it as a standing test. hipEngine's exact path silently stopped accepting chunks over 256 tokens for a week (HE:docs/QWEN4EXP-STATUS.md:84-86).
3. **Gate P0 on bytes, not replies.**
   - Run `--kld-record` (`core/config.cpp:318-325`) on the long documents, once for the base engine at 16-token chunks and once for each arm, then compare the record directories byte for byte. They hold every position's f16 row and its exact f32 head.
   - Add hashes of every KV group at the prompt's end.
   - Add DeepInfer's teacher-forced test as a second case.
   - Matching replies can pass while nearly every row differs.
4. **Fallback if a pin fails: split the failing non-MoE ops into launches of 16 rows or fewer inside the tall step,** as QFN computes spans in 2,048-row pieces (GUIDE.md:1140-1143). This stays exact against today's reference and costs about 0.4–0.8 s per 1,024-token chunk. It replaces E1, which changes the reference.
5. **Flags.**
   - Give `ab_window.sh` parameters for the checkpoint interval and the span.
   - Without prefix caching, use `--checkpoint-interval ≥ C` for any C above 2,048.
   - With an interval of 128–256, use `--prefill-span C/I`. The span covers only a lone prefill and never its first interval (`core/sched/scheduler.cpp:983-1012`, `core/sched/geometry.cpp:125-136`).
   - Read the "chunk geometry" startup line (`geometry.cpp:190-203`) in every arm.
6. **W1 arm C.**
   - For C above 2,048, set `--checkpoint-interval ≥ C`, or the arm silently runs 2,048-token chunks.
   - To measure the base engine's whole-layer card path, also drop B1 and K and set `RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREFILL_READ_ONLY=1`. Otherwise the arm only gives the arena line.

### 4.2 Order and gates

| Step | Item | Gain (calibrated) | Effort | Gate |
|---|---|---|---|---|
| 0 | Measure first: the critique's item 0, plus a norm-width height check and a skip arm in the trace replay | Decides chunk size, cache policy, card share and the skip's real gain | 1.5 days + one window | — |
| 1 | Consumer skip in 16-token passes | 3K: 100 → 72–77 s; 8K: −34 to −40% | About 1 week | Base vs skip on 2.6–8K prompts plus the held-out set: last-row logits bitwise, every KV-group hash, 64 decode tokens |
| 1′ | Prefix caching plus end-of-request snapshot, in parallel | Multi-turn TTFT ≈ proportional to new tokens | 1–2 days + 2–4 days | Cold run == hit |
| 2 | P0 with four pins plus the row-batched gemv; ship passes of 32 tokens or fewer | 4–15% on every prompt | 3–4 days | §4.1, items 2–3 |
| 3 | P1: LCP at C = 1,024, with the skip's per-layer row suffixes | LCP ×1.8–2.8 times the skip: 3K 26–47 s; 8K 58–108 s | 6–8 days + 3–5 days | Byte gate; read-only only on pure prefill passes |
| 4 | P2: next-layer prefetch | −5 to −10% | 2–3 days | Exact |
| 5 | Card share (idea #3) if step 0 shows the CPU is the limit | −15 to −25%: 3K about 20–40 s | 4–6 days | E2 gemv on the card |
| 6 | Layer-major for prompts of 4K tokens or more (replaces P4) | 8K about 20–25 s if the card computes every non-resident expert through the ring; about 55–100 s if the CPU keeps them (model) | 2–3 weeks | As P1 |

Dropped:
- P3's static-plan lending, unless the card share falls short.
- P4's whole-prompt arena.
- E1.
- The power-settings item.

### 4.3 First experiment

- **Now, CPU only:** prototype the critique's 32-row CPU kernel and benchmark it at 4, 8, 16 and 32 rows. Include an FMA variant, byte-compared in the host test.
- **One GPU window of about 25 minutes, under both locks:**
  - (a) Run today's best configuration with the access trace on the 14:54 documents plus one 6–8K document, then replay offline:
    - read-only, admit-all and bounded admission at C = 512, 1,024 and 2,048;
    - the pass-granular skip: drop the skipped (pass, layer) accesses, replay the LRU, and apply the per-pass law.
  - (b) Start the base engine at C = 1,024 and 2,048 with `--checkpoint-interval ≥ C`, and read the arena and chunk-geometry lines.
  - (c) Run the height-sweep device check on the norms, attention, index score and MoE gemv. This needs a test-harness change only, no engine change.
- **Decision.**
  - Build the skip first if the replay shows ≥15% at about 3K tokens. The model says 24–28%.
  - Add the norm pin to P0 if (c) shows the norm bits change, as the emulation predicts.
  - If any other op differs at 17–32 rows, the first P0 step is 16-row passes only.

**Files:** `/tmp/addendum-calc/tri.py`, `/tmp/addendum-calc/toy_triangle.py`, `/tmp/addendum-calc/normwidth.py`; design `/home/as/ds41-decode-review-20261028/PREFILL-DESIGN.md`; critique `/home/as/ds41-decode-review-20261028/PREFILL-CRITIQUE.md`.

## Survey: deepinfer

I found one large exact prefill gain in DeepInfer that PREFILL-DESIGN.md does not have: run layers 20–39 only on the last ~2,400 prompt rows. In the model it cuts TTFT by 27–28% at 3,086 tokens and 39–43% at 8,192, with or without large chunks. None of this is measured.

# DeepInfer: prefill ideas for ds41 on 4× RX 9070 XT

I read `deepseek-ai/DeepInfer` at commit `553741557a57df44a46fb3ab72b6aafeaf945337` (2026-10-25), the same commit the research loop pinned. Links below are relative to `https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/`. Doc lines are `doc/inference_deployment_qa.md` (Q&A); use `?plain=1#L…` for its line anchors. My clone is in `/tmp/deepinfer-research-20261029/src/DeepInfer`. The models are single-thread, under a minute each: `/tmp/deepinfer-research-20261029/model/ced_model.py`, `ced_today.py` and `ced_twophase.py`. They reuse PREFILL-DESIGN's constants, routing profile and touch model from `t6_model.py`.

## Bottom line

- **DeepInfer is a datacenter engine. Its prefill uses no expert offload or caching.**
  - All experts sit in HBM; prefill runs EP4DP4 with 4K–64K tokens per rank per step and reaches 60–89K tok/s per GPU (`doc/images/prefill_pareto.png`, `examples/sweep/run_sweep.sh:41-48`).
  - Its EP scheduling, comm overlap, P/D disaggregation and FP8×FP4 MoE kernels do not transfer to a box limited by one NVMe (last section).
- **What transfers is a property of the model.** V4.1-Flash is a YOCO/"CED" model:
  - Layers 0–19 produce the KV; layers 20–39 consume it.
  - The compressed and index KV of layers 20–39 comes only from layer 20's normalized input.
  - Each consumer layer adds only its own 128-token sliding window.
  - So the exact consumer work for a prompt is bounded by about 2,541 rows, whatever the prompt length.
  - DeepInfer ships a lossy 140-row version of this as its default. Its Q&A states the exact bound.
- **Exact CED suffix, triangle form** (idea 1):
  - Consumer layer L runs only its last n_L = 127·(39−L)+1 rows: 2,414 at layer 20 down to 1 at layer 39.
  - Total consumer work is 24,150 rows instead of 20·P.
  - It is exact by dependency: every row it computes sees the same inputs as in the base engine.
  - Modelled TTFT:
    - Today's 16-token passes: −10% at 1,024 tokens, −28% at 3,086, −39% at 8,192.
    - On top of the large-chunk mode at 1,024-token chunks: −6%, −27% and −43%.
    - With the card-share phase P3 at 2,048-token chunks: −20% at 3,086 and −41% at 8,192.
  - Decode is unchanged.
  - This is not in PREFILL-DESIGN's ranked list. Earlier source audits in the research dir cover feasibility and blockers but give no speedup estimate. I would rank it 1b, next to the large-chunk mode (LCP).
- **DeepInfer's batch-invariant mode** (idea 2) is evidence that the P0 gate (E2) can hold.
  - DeepInfer asserts bit-identical logprobs between teacher-forced decode and a one-chunk 17K-token prefill (KL bound 0.0).
  - That test is a stronger gate than comparing a few greedy replies.
- **Lossy opt-in** (idea 3): DeepInfer's 140-row decoder replay gives about −40 to −45% in today's engine (−31 to −50% with LCP), but replies change.

## Idea 1: exact CED consumer suffix (exact, largest gain)

**What DeepInfer does**
- The Q&A line 187 says: "In the CED architecture the decoder layers' block KV cache is projected from the last encoder layer's hidden state… reconstructing it exactly would require running the last 2,560 tokens through 20 decoder layers. Instead, each prefill recomputes only the last 128 tokens… nearly halving total prefill computation." Line 183 calls the 128-token version approximate.
- Code:
  - The consumer range is `[n_prefill_layer, n_layer)`, with the first ratio-1 layer as the boundary (`v4_modules/yoco.py:14-21`, `hugging_face/config.py:158-161`).
  - For this checkpoint, consumer window KV comes from each layer's own input: `yoco_swa_share_hidden: False` (`config.py:47`, used at `attn.py:245-249`). That is what makes the window cascade 20 × 127 rows deep.
  - Compressed and index KV is written for every prompt row from the full boundary hidden (`attn.py:672-709` `_store_yoco_full_kv`, `decoder_layer.py:134-150`).
  - Only the kept query rows go on; hidden, mHC and candidate rows are compacted at the boundary (`yoco.py:93-123`, `deepseek.py:190-207`).
  - The kept window visibility is the sliding window intersected with the kept rectangle (`input_metadata/yoco.py:34-37`).
  - Keep must be at least window + backtracking (`worker/model_runner.py:107-113`, `environ.py:195-201`).
  - Constraints: Engram must sit in producer layers, and DSpark target layers must all sit on one side; DSpark inputs are compacted with the rows (`deepseek.py:126-138`, `:199-205`).
  - The published prefill benchmark runs with the 140-row setting (`run_sweep.sh:46-47`).

**Fit with radiance**
- The topology matches exactly:
  - `deepseek41.cpp:262-276` admits window 128, KV sources {2,8,14,20}, compress ratio 1 from layer 20, candidate layer 20 and Engram at layers 1 and 14.
  - DSpark taps are layers 37/38/39 (`DS41_CED_EXACT_REPLAY_CRITIQUE_…20261028.txt:167`).
- Nothing like it exists yet: `step()` runs every layer at `T = batch->n_tok` (`deepseek41.cpp:2377`, loop at `:2434`), and only the head is gated on `n_out` (`:2766-2771`).

**Why the triangle is cleaner than a constant suffix**
- Each consumer layer computes only rows whose inputs are exact, so it has no provisional rows.
- Layer 20 keeps its norm, compressor, index-K and window-key work for every row; this is cheap and has no routed experts.
- Layer L projects window keys on n_{L−1} rows but runs q, attention, output and MoE on n_L rows.
- The needed rows are always a suffix of the pass, so a single sequence needs row offsets, not gathers.
- A constant suffix of about 2,541 rows has provisional rows at its start. Those rows must not read window KV that was never written, so they need DeepInfer-style masking with a rectangle.
- The constant form is also weak below about 3K tokens.

**Exactness**
- No new arithmetic is introduced. The remaining dependency is that each op's per-row result must not depend on the other rows in the pass:
  - At heights of 16 or fewer, every op already runs its decode-band kernel.
  - With LCP, this is the P0 gate anyway.
- The engine work and risks are already listed in the prior critique (lines 142-199):
  - an owned prompt cut in `RadBatch`;
  - the scheduler must not publish consumer windows or checkpoints for rows before the cut;
  - the stager must not prefetch consumer experts for skipped rows;
  - the EP route witness for row subsets;
  - DSpark taps: they get 382/255/128 exact rows, at least the drafter's 128-key window, but scheduling them needs a separate proof.
- Prefix checkpoints: under the triangle, no checkpoint inside the prompt has complete consumer windows. Use end-of-request snapshots instead (smaller item a).
- DeepInfer only tests mask against compact, both lossy. The A/B gate here must compare off against the exact suffix, bit for bit.

**Modelled TTFT, today's 16-token passes** (seconds at m = 1.0 / 1.3 misses per pass-layer-rank; m = 1.0 reproduces the measured ~100 s at 3,086 tokens)

| P | Today | Constant suffix, K=2,541, whole passes | Triangle | Lossy 140 (opt-in) |
|---|---|---|---|---|
| 1,024 | 35 / 43 | 0% | −10% | −39% |
| 2,048 | 70 / 87 | 0% | −20% | −43% |
| 3,086 | 105 / 130 | −8% | −28% (76 / 94) | −44% |
| 4,096 | 140 / 173 | −17% | −32% | −44% |
| 8,192 | 280 / 346 | −32% | −39% | −45% |
| 16,384 | 559 / 692 | −39% | −42% | −45% |

**Modelled TTFT with LCP** (P0+P1, 1,024-token chunks; seconds for routing floor α = 0.02 / 0.05)

| P | LCP | + triangle | + two-phase (one consumer pass at the end) | Lossy 140 |
|---|---|---|---|---|
| 1,024 | 11.1 / 13.9 | 10.4 / 13.0 (−6%) | same | 7.7 / 9.5 |
| 2,048 | 22.1 / 27.7 | 18.4 / 23.0 (−17%) | 16.1 / 20.2 (−27%) | 12.9 / 16.1 |
| 3,086 | 34.1 / 42.5 | 25.0 / 31.2 (−27%) | 21.8 / 27.3 (−36%) | 18.9 / 23.6 |
| 8,192 | 88.5 / 110.8 | 50.2 / 63.5 (−43%) | 47.3 / 60.0 (−47%) | 44.1 / 55.9 |
| 16,384 | 177 / 222 | 92 / 117 (−48%) | 89 / 113 (−50%) | 86 / 109 |

- With P3 (card share S=24, 2,048-token chunks), the triangle takes 3,086 tokens from 24.5–30.4 s to 19.6–24.5 s, and 8,192 tokens from 53.5–66 s to 31.7–39.4 s.
- A constant 2,560-row suffix aligned to 1,024-token chunks gains 0% below about 3.6K tokens and −33% at 8K.
- Two-phase also needs about 100 MiB per rank of boundary HC state and a consumer pass of up to 2,414 rows (per the critique, line 97).
- Extra upside not in the model:
  - Index scoring at 8K tokens falls about 53% (consumer layers about 69%).
  - Consumer attention rows fall from 20·P to 24,150.
  - These matter once the drive is no longer the limit.
- Prompts of 128 tokens or fewer gain only about 1–2% (only layer 39 shrinks).
- Decode is unchanged.
- Effort: about 1.5–2.5 weeks for the triangle, scoped to one sequence, no speculation and no prefix cache. This is my estimate.

## Idea 2: batch-invariant mode as the gate for LCP's exactness (exact, enables idea 1 and LCP)

- `DEEPINFER_BATCH_INVARIANT=1` switches DeepGEMM and TileKernels to deterministic algorithms (`worker/runtime_config.py:67-69`).
- It puts FlashMLA in batch-invariant mode (`attn.py:666`, `:1002`; `layers/attention_backend.py:141`).
- It drops a fused prefill-only kernel (`attn.py:87-92`).
- It forbids multiple reduction in DeepEP while EPLB replicas move (`userbuffer.py:79`). It is documented as slower (`eval/run_eval.py:74`).
- Tests:
  - `tests/models/test_batch_invariant.py:52-105`: 64 prompts of about 2,500 tokens, batched against single, with tokens, logprobs and top-5 equal.
  - `tests/models/test_kl.py:30,95-206`: 1,024 teacher-forced decode tokens after a 16K-token prompt, with MTP 5, CP 2 and 32 extra experts, compared with one 32K-budget prefill of prompt + continuation. The KL bound is 0.0, which is bitwise because `deepinfer/testing.py:4-12` returns 0 only on equality.
- For us:
  - This shows a production V4.1-Flash graph can keep one reduction order from M=1 to 32K. That supports E2 (decode-band kernels at every height) over E1.
  - Adopt both tests as the P0 gate: teacher-forced decode logits against all-row logits from a one-chunk prefill at every position, plus batched against single.
- No speed effect by itself. It unlocks LCP's ×3–4.6 and makes decode-time snapshots (smaller item a) exact.

## Idea 3: DeepInfer's 140-row decoder replay (lossy, opt-in only)

- It is DeepSeek's production default: README `KEEP_TOKENS=140` and the benchmark sweep. The Q&A (lines 183-187) says it is approximate and never enters their prefix cache.
- It uses the same machinery as a constant suffix with K=140 plus masking by a visibility rectangle. It must mask: reading unwritten window KV is not the same thing.
- Gains are in the tables above. They are largest at 1–3K tokens, where the exact form gives least.

## Smaller items

- **a. End-of-request window snapshots (exact if P0 holds).**
  - DeepInfer copies the window to an on-device replica at each 128-token crossing during decode, and stores the latest copy at request end (Q&A line 165).
  - Radiance checkpoints only on prefill entries (`core/sched/scheduler.cpp:983-1012`, `:1897`). A chat's next turn therefore re-prefills the previous reply.
  - For a 500-token reply that costs about 17–25 s a turn today. This is also how to keep prefix caching compatible with idea 1.
- **b. Chunk size from a time budget, and prefill steps every N decode steps.**
  - DeepInfer sizes chunks with a fitted quadratic cost model (Q&A line 115) and interleaves prefill steps at intervals (`csrc/engine/scheduler.hpp:295-335`; chunks split at checkpoints, `:470-486`).
  - This only matters with `--max-num-seqs` above 1, to cap the stall of about 10 s per 1,024-token LCP chunk. Use the per-pass law for the budget.

## Not transferable (with evidence)

- **Expert offload or caching:** none. Every expert is resident in HBM; Engram tables are the only host-side weights. A grep for offload and pinned memory finds only staging.
- **EPLB and redundant experts** (Q&A lines 95-103, `model_executor/eplb/policy.py:11-60`): they balance per-GPU compute. Our prefill is bound by one shared NVMe and one CPU pool, and each expert has a single owner.
- **P/D disaggregation** (Q&A lines 43-67): needs separate machines. The single-box equivalent is P3's VRAM lending, which the design already has.
- **Mega MoE, DeepEP overlap and FP8 dispatch** (`v4_modules/mlp.py:461-511`, `:528-563`, `:600-610`; Q&A lines 229-231): NVLink kernels for SM100. Per-layer traffic here is about 10 MB per 1,024 rows (about 0.4 ms over PCIe), against about 10 s of drive time per chunk.
- **FP8×FP4 grouped GEMM** (`mlp.py:109-137`, `:274-290`, with e4m3/ue8m0 per-32 activations): radiance's routed activations are BF16, so this changes bits, and the card is not the bottleneck. At most a later lossy opt-in for P3/P4.
- **Context-parallel prefill and asynchronous scheduling:** attention compute and decode latency tools, not our bottleneck.

## Prior local work this builds on

These audits covered feasibility and blockers but give no TTFT estimate; one says "Nothing here predicts SSD reads, TTFT or a speedup" (line 74):
- `/home/as/radiance-setup/research/deepseek-v41-1017-20261024/DS41_YOCO_CONSUMER_REPLAY_DEPENDENCY_SOURCE_AUDIT_20261027.txt` (it also cites the DS41Engine and Project Maya exact-suffix implementations)
- `/home/as/radiance-setup/research/deepseek-v41-1017-20261024/DS41_CED_EXACT_REPLAY_CRITIQUE_AND_MINIMUM_SEQUENCE_20261028.txt`
- `/home/as/radiance-setup/research/deepseek-v41-1017-20261024/DS41_DECODER_PREFILL_SUFFIX_FEASIBILITY_REPACK_V1.txt`
- `/home/as/ds41-decode-review-20261028/REVIEW.md:301` (a one-line mention)

Sources:
- [DeepInfer repository](https://github.com/deepseek-ai/DeepInfer)
- [DeepInfer deployment Q&A, decoder bounded replay (line 187)](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md?plain=1#L187)
- [DeepInfer yoco.py](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/yoco.py)
- [DeepInfer attn.py, full-prompt KV store under truncation](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/attn.py#L672)
- [DeepInfer test_kl.py](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/tests/models/test_kl.py)
- [letsdatascience.com: DeepInfer release news](https://letsdatascience.com/news/deepseek-open-sources-deepinfer-its-production-inference-eng-bc78c60d)
- [braindetox.kr: DeepInfer scope and gaps](https://braindetox.kr/en/posts/deepseek_deepinfer_inference_engine_open_source.html)

## Survey: strata-ds41engine

# How Strata and DS41Engine prefill DeepSeek-V4.1-Flash, and what applies to radiance

I only read code and web pages. No repository was modified, no GPU was used, and no experiment was run. The only local check read a few sysfs values (power settings) and the radiance source.

Clones and extracts:
- DS41Engine, with full history: `/tmp/ds41-ext-research-2300348/ds41engine/repo`
- Strata: `/tmp/ds41-ext-research-2300348/strata/repo`
- An older DS41Engine doc version: `/tmp/ds41-ext-research-2300348/v41_exact_fast_6aa1899d.md`

Link prefixes used below:
- **DE:** = `https://github.com/vincalkr/DS41Engine/blob/d0db7270316c5015295fe1f2df273a54ff16c00f/` (HEAD of 2026-10-28)
- **DEc/** = `https://github.com/vincalkr/DS41Engine/commit/`
- **ST:** = `https://github.com/Niko1221/Strata/blob/46c152a1310f77cd2f0ce2c7cd213962df14fe83/` (HEAD of 2026-10-28). Strata's DeepSeek engine `src/dsv41` is a port of Project Maya's maya-ds4.

## Bottom line

1. **Neither engine prefills in small passes.**
   - Both run a prompt as large batches: DS41Engine 8,192 tokens; Strata 1,024–8,192 tokens by rule, at most 32,768.
   - Both compute every routed expert on the GPU and stream the non-resident ones through small double-buffered staging. The CPU computes experts only in decode and short-turn passes.
   - Both keep the RAM tier read-only during a prompt.
2. **Both skip layers 21–39 for every prompt token before the last ~2,560 ("enc-skip", "YOCO"). The skip is exact.**
   - DS41Engine measured +39% prompt speed at 8K tokens, +61% at 17K and +74% at 25.5K, with the same greedy tokens. It is on by default in both engines.
   - PREFILL-DESIGN.md does not include it. Root has it as "CED", but plans a two-phase encoder/replay design that these engines show is not needed.
3. **Both decouple the size of the MoE batch from activation memory.**
   - The dense half runs in sub-batches: 512 rows in DS41Engine, 256–2,048-row "attention slabs" in Strata.
   - The MoE runs once per layer over the whole batch, so each expert is read once per layer per batch.
   - For radiance, dense sub-batches on the decode band (16–48 rows) would keep attention, the indexer and norms bit-identical to today's base engine by construction. Only the MoE would need the tall-pass switch.
4. **Their measured prefill rates are 13–16× ours, on smaller packs.** They run 2-bit packs (8.6 MB experts, 121–155 GB) that mostly fit in VRAM plus RAM. Neither engine is bit-reproducible.

| Engine, box, pack | Prompt | tok/s |
|---|---|---|
| DS41Engine: 2× RX 7900 XTX, Ryzen 9 9900X, 122 GiB, PCIe 5 NVMe; 138 GB IQ2_XXS pack; 80 GiB pinned arena | 8,192 / 17,349 / 25,500 / 123,416 | 371–414 / 433–459 / 458–484 / 483–493 (DE:docs/ds4/V41_EXACT_FAST.md:64-78) |
| Same box, enc-skip off (2026-10-22) | 8,192 / 17,349 / 25,500 | 226–230 / 223–227 / 224–225 (DEc/6aa1899d) |
| Strata: 1× R9700 + 251 GB RAM, 8K prompt | chunk 1,024 / 2,048 / 4,096 / 8,192 | 200 / 277 / 353 / 431 (ST:src/dsv41/ds4_model.cu:2206-2210) |
| Strata (Maya-S, read in place): 7× RTX 3090 | 8K / 32K, windows of 16 → chunks of 8,192 | 115 → 461 / 99 → 1,010 (ds4_model.cu:2235-2241) |
| Strata: 1× R9700, 2.25 bpw | 8K / 32K | 286 / 444 (ST:docs/release-notes-0.2.7.md:17) |
| Strata: RTX 5070 12 GB + 64 GB + 3.6 GB/s NVMe | 4K / 8K | 170 / 268 (same file) |
| Strata: Strix Halo 128 GB | 4K / 16K / 32K | 214–234 / 326 / 342 (release-notes-0.2.7.md:16) |
| radiance today | long prompts | ~30 |

Strata's "windows of 16" (`kWin=16`, ST:src/dsv41/ds4_kernels.cuh:18) is our 16-token regime. Moving to large chunks gave them ×4–10.

## Ideas for radiance, ranked

### A. Exact decoder-suffix skip (enc-skip / exact YOCO)

**What it is.** The model's layers above layer 20 see earlier positions only in two ways:
- Layer 20's compressed KV and index keys. `kv_source_layer_ids` is [2, 8, 14, 20]. Index sources 24/28/32/36 only rank their own queries against layer 20's keys.
- Their own 128-token window.

Sources: DE:local/rco-iq2xxs-meta/geometry-keys.json, DE:tools/v41/engine/ds41_geometry.hpp:71-84. Radiance binds the same caches only at source layers (arch/deepseek41/deepseek41.cpp:1263-1279).

So prompt tokens before the last R positions need:
- layers 0–19 in full;
- layer 20's KV-writing half only: hc_pre/attn_norm, the window K/V row, the compressed-KV and index-key rows, and the ratio-2 compressor state;
- nothing from layers 21–39.

The last row and every layer's final 128 window rows come out identical.

**How they do it.**
- Neither engine has an encoder phase followed by a replay phase. Passes stay in prompt order.
- Passes wholly before the cut are simply truncated. Later passes run all 40 layers. The rows just after the cut read stale windows at layers 21–39; they are provisional and never reach the outputs that matter.
- DS41Engine:
  - The cut is at the last KV source, with `R = 128 × (40 − 20) = 2,560`, raised if smaller (DE:tools/v41/engine/ds41_generate.cpp:669-681).
  - The cut is rounded down to the 512-row sub-batch, and layer 20 runs `attn_router_span_kv` for the skipped columns (ds41_generate.cpp:1295-1318). The KV-only graph is at DE:tools/v41/engine/ds41_dense.cpp:861-948.
  - It is on by default. Its gate was byte-identical last-row logits at 8,192 tokens (ds41_generate.cpp:639-652).
  - Snapshots are taken only at the prompt's end. DSpark is seeded only from the last ≤128 rows (ds41_generate.cpp:1349-1366).
- Strata:
  - `keep = 20 × 127 + 128 = 2,668`, applied only when at least 1,024 rows would be skipped (ST:src/dsv41/ds4_model.cu:9024-9039, :910).
  - The skip logic itself is `Run::runs/kv_only/r0` (:8264-8282).

**Fit.**
- Phase-B passes are exactly today's 16-token passes, so this does not need the P0 tall-pass gate.
- Radiance needs:
  - a plugin graph variant that stops after layer 20's KV writes;
  - no prefix checkpoints before the cut;
  - DSpark context taken only from the last 128 rows;
  - the prompt length known at admission;
  - window pages for skipped positions at layers 21–39 allocated and finite.
- The CED critique's boundary capture at the 19→20 seam is not needed, and neither is a separate decoder-ready state (`/home/as/radiance-setup/research/deepseek-v41-1017-20261024/DS41_CED_EXACT_REPLAY_CRITIQUE_AND_MINIMUM_SEQUENCE_20261028.txt`). The window trim does not matter, because the only rows that read trimmed or unwritten windows are provisional.
- The gate: base engine against base plus skip on prompts of 2,600–8,192 tokens, comparing replies and last-row logits.

**Estimated effect (modelled).** A pass before the cut costs about 0.5 of a full pass. TTFT then scales by `0.5 + 1,280/P`:

| Prompt tokens P | TTFT factor |
|---|---|
| 3,086 | 0.92 (100 s → ~92 s) |
| 4,096 | 0.81 |
| 8,192 | 0.66 |
| 16K | 0.58 |
| 32K | 0.54 |

Under LCP, the same factor applies to chunks wholly before the cut: at 8,192 tokens, ×0.69 with 1,024-token chunks and ×0.75 with 2,048-token chunks. Decode is unchanged. Effort is days, independent of P0.

### B. Layer-major batches: MoE once per layer over B rows, dense in sub-batches

**How they do it.**
- DS41Engine runs B ≤ 8,192 rows layer by layer:
  - the dense half in 512-row sub-batches, in order, so KV, compressor and indexer state advance exactly as in sequential passes (DE:tools/v41/engine/ds41_dense.hpp:82-101);
  - then one MoE call per layer over all B rows;
  - then the finish step (ds41_generate.cpp:1266-1348).
- DS41Engine measured, on a 20K prompt: B 8,192 → 12,288 gave 318 → 388 tok/s, and file-tier reads 157 → 114.5 GB (3 → 2 sweeps), with the reader at 4.95–5.15 GB/s either way (DEc/86879129).
- Strata runs layer-major over several chunks on one GPU. It parks chunk states in host memory between layers and keeps each layer's file-read experts in a RAM layer cache (ds4_model.cu:8299-8303, :8906-8962, :3672-3683).
- Strata measured a 16K chunk on a V100: 461.8 → 321.3 s, and 736 → 506 GB read (:903-905).

**VRAM.** DS41Engine needs about `4.3 GiB + 0.95 MiB × B` (ds41_generate.cpp:1956-1975). It closes its whole 17.5 GiB expert cache during a prompt and re-seeds it afterwards in 0.6–2.9 s (ds41_generate.cpp:19-23, :1886-1887, :2079-2088).

**Fit.**
- In radiance, a batch's persistent per-token state would be the HC residual (about 40 KB in BF16) plus the current layer's MoE input and output (about 30 KB). That is about 0.6–0.9 GB a card at B=8,192, against 1.6–3.7 GB for an LCP arena of the same height.
- Choice of dense sub-batch:
  - 16 or 48 rows (decode band): bit-identical by construction, but the per-pass fixed cost (76–114 ms per 16 rows) makes 16-row sub-batches cost about 20–30 s per 8K tokens, and 48-row ones about 7–10 s.
  - 1,024 rows under E2: about 2 s.

**Estimated effect (modelled, drive-bound).** One sweep of the ~127 GB of non-resident, uncached experts takes about 18 s per batch, or about 9 s for batches wholly before the cut.

| Prompt | Layer-major estimate | Today | Other references |
|---|---|---|---|
| 8K | ~20–30 s | 280–412 s | P3 at 2,048-token chunks: 54–66 s |
| 32K | ~50–65 s | ~19–27 min | P3 plus A: ~115–140 s |

This is the design's "layer-major / P4" item; VRAM stays modest. Effort is extra large: the step executor has to be restructured.

### C. Cards compute (nearly) all experts in big batches

**How they do it.**
- DS41Engine orders experts as VRAM-resident, then RAM arena, then file tier (DE:tools/v41/engine/ds41_moe.cpp:4905-4917).
  - RAM-arena experts are DMA'd from pinned memory. The next layer's are prestaged into a 1.5 GiB region during the dense half, worth +5.8% (ds41_moe.cpp:4658-4757, :5331-5340; DEc/3db46ed8).
  - File-tier experts are read on demand into pinned bounce halves by 32 threads × 8 parts (3.6 → 5.3 GB/s, ds41_moe.cpp:5231-5239), then computed with MMQ.
- A 5,787-token prompt profile: the expert tier is 53% of the time, flash attention 22%, and GEMMs under 4% (DEc/cb5cac1b).
- Strata: "the VRAM ones in place, the CPU lane's streamed in batches" (ds4_model.cu:7806-7809).
  - Pinned RAM copies are DMA'd in place (:3660-3667; ST:src/dsv41/ds4_experts_rule.hpp:164-170).
  - Huge pages took an 8K prompt from 651 to 694 tok/s (release-notes-0.2.7.md:42).
  - The staging ring is 16–1,200 MB, and its decode cost was measured (ds4_experts_rule.hpp:123-145).

**Fit.** This extends P3. At B ≥ 8,192, the CPU path costs about 55–80 s per batch (0.06–0.09 ms per further row × ~0.9 M non-resident expert-rows) against 18 s of drive.
- Card: the E2 gemv at ~21 µs per expert-row (from the design's 0.7 s per 1,024-token chunk) is about 10 s a card per 8K batch.
- Transfer: host-to-device is about 59 GB a card per sweep (~3 s), and host DRAM traffic is about 235 GB (~2.5 s).
- RAM-cache entries are in the packed CPU layout. Widening them on the card after the DMA avoids `stage_relaid`'s CPU pass.

It is exact under E2 (the card gemv and the CPU kernels produce the same bits). It keeps batches above 2,048 drive-bound.

### D. Smaller items

- **Short turns.**
  - DS41Engine runs ≤96 new tokens as decode passes. Its batch has a floor of about 4 s, and the crossover is about 70–100 tokens: a +171-token turn went 9.7 → 4.0 s (old doc :251-255 at DEc/6aa1899d; DE:docs/ds4/V41_EXACT_FAST.md:61-62).
  - Uploading layer-independent index tensors once per batch, plus a routed-only prestage below 512 rows, made small turns 12% faster (DEc/292b7010).
  - Engage LCP or layer-major only above a few hundred tokens.
- **Read-only RAM tier during prompts.** Neither engine admits during a prompt.
  - DS41Engine rebalances only between requests (DEc/eaba7cc0).
  - Strata dropped its post-prompt store refill: it cost a 4K prompt 4.7 s, and the 300 tokens after an 8K prompt took 51.9 s without it against 48.6 s with it (release-notes-0.2.7.md:17).
  - This supports P1 with last-chunk admission.
- **Routed-only reads above 64 rows.** Once its reader was deep, DS41Engine's on-demand reads beat whole-layer prefetch: 8,000 tokens 33.6 → 32.4 s, +1,000 tokens 19.1 → 16.6 s (ds41_moe.cpp:986-993). Radiance reads whole layers above `kRoutedRows=64` (core/place/stager.h:353-357). At 1,024–2,048 rows that saves 6–27% of reads.
- **Exact prefix reuse per turn.**
  - DS41Engine snapshots each conversation's end state into a 4 GiB pinned pool (~340 MB for a 190K-token session) and a 32 GiB disk store. It rewinds up to 16 positions to an even position. Agent turns at 120–190K context resume in 0.6–10.5 s (DE:README.md:90-95; V41_EXACT_FAST.md:79-80, :327-338).
  - Strata keeps slots of 17 MB per 6.5K tokens and auto-pins the system prompt and tools that agents and sub-agents share (ST:docs/DEEPSEEK41.md:107-110).
  - Radiance's window ring already holds 128 + max_spec rows (deepseek41.cpp:1035-1041), so an end-of-request snapshot plus a short rewind is exact. With A, snapshots must sit at or after the cut.

### Lossy, opt-in only

- **128-row decoder replay** (DeepInfer's bounded replay; Strata `DS4_YOCO=2`): at 3,086 tokens, about −45% TTFT instead of −8%, and "shifts the answers a little" (ds4_model.cu:8984-8987, :9033-9038).
- **Top-5 experts:** prompts +5–7%, decode +4% to +13%; KL 0.10987 against 0.10833 (ds4_experts_rule.hpp:33-46).
- **WMMA prompt attention:** +5.4% (DEc/077a9c52). Small while prefill is drive-bound.

## Decode ideas

1. **NVMe APST and deep C-states off, `performance` governor.**
   - DS41Engine: one 9.7 MiB read every 10 ms took 2.77 ms, and one every 20 ms took 5.54 ms, against about 1.2 ms when the drive is kept busy (DE:local/power-tune.sh:2-10).
   - This box now: `pm_qos_latency_tolerance_us=100000`, governor `schedutil`, C2 enabled.
   - It is on REVIEW's list but unmeasured. It is exact; one window would settle it.
2. **L3 prefetch of predicted CPU experts during attention.**
   - DS41Engine measured that an expert already in L3 computes 1.41–1.56× faster than from DRAM, and that a next-layer prediction catches 59–72% of the CPU's experts. It estimated +3–6% and did not build it (V41_EXACT_FAST.md:244-275).
   - On the EPYC, L3 is 32 MB per CCD, so each CCD's threads must prefetch the rows they will compute, in a bounded burst. Estimated +1–3%; exact.
3. **Background promotion of recurring misses into VRAM.**
   - DS41Engine: 14.63 → 15.74 and 12.35 → 13.53 tok/s (old doc :120-126). Seeding VRAM from the conversation later gave nothing (V41_EXACT_FAST.md:191-212).
   - Radiance's ds41 plan is static. This is exact, but confidence is low: 0 to +5%.
4. **Exact FP4 KV storage.** Storing compressed KV as NVFP4 and index keys as MXFP4 matches the fake-quantised values bit for bit (DE:tools/v41/engine/test_ds41_kvfp4.cpp:1-3). Radiance stores them in BF16 (deepseek41.cpp:1053-1060). It saves memory and snapshot bytes only.

DSpark, lookup drafts, spinning syncs and a CPU/card split of decode misses all gave nothing at exact routing in DS41Engine, matching radiance's own results.

**Lead not followed:** antirez/ds4 (DS41Engine credits it for an exact candidate-block fast path, DEc/6aa1899d).

## Survey: hipengine

**hipEngine → ds41 prefill: research findings (no GPU used, no repository modified)**

hipEngine is at commit `cc30876b` (2026-10-29). A sparse shallow clone of its docs and gfx11 kernel sources is in `/tmp/hipeng-research-1029/src`. Every hipEngine `path:line` below is relative to `https://github.com/shisa-ai/hipEngine/blob/cc30876b/`. Our tree is `/home/as/radiance-ds41-claude` at `9c8dccb9`.

## What hipEngine is, and why little of it transfers directly
- **Hardware:** it runs one GPU per process, from a Python host with HIP kernels. Targets are gfx1100 (W7900, 48 GB) and gfx1151 (Strix Halo, 120 GB unified memory). It has no gfx12 kernels, and its README says other AMD architectures are "not automatically treated as compatible" (`README.md:178`). Tensor parallelism is not implemented (`docs/reference/TENSOR_PARALLEL.md:14`).
- **Memory:** every model is fully resident, with no CPU weight offload and no paged experts (`docs/campaigns/LAGUNA.md:55-56`). So nothing in hipEngine addresses our actual limit, which is NVMe expert reads per chunk.
- **Kernels:** its WMMA code uses the gfx11 builtins (105 uses of `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`). libr4d already uses the `_gfx12` forms (bf16, f16, fp8_fp8, iu8). None of its kernels are drop-in.
- **Where it helps:** it has already shipped the exact approach our P0/E2 proposes. On Qwen3.8-Flash-Next, a 111 GB MoE with DSA-like sparse attention, chunked prefill is byte-identical to serial decode:
  - 64-token chunks matched serial decode on all 687 teacher-forced rows.
  - 256-token chunks then reached 51–58 tok/s at a 508-token prompt, with byte-identical full logits (`docs/campaigns/QWEN3.8-FLASH-NEXT.md:982-1000`).
  - The method: each output keeps its logical reduction (the same partial sums, the same tree and the same fused multiply-adds). Only the physical schedule changes: rows batched, outputs batched, fewer physical threads, staging in LDS.

## Bottom line
- **Exactness costs hipEngine about 10–20× when it is compute-bound.** Its exact profile ran 51–58 tok/s in August; its F16-WMMA composition ran 1,028–1,562 tok/s in October (`docs/QWEN4EXP-STATUS.md:100-102`). The dates differ and other changes landed in between, so this is a rough ratio.
- **For us that cost is about zero.** From the design's own per-chunk card times (1.5–2 s per 1,024 tokens), the exact card path tops out at about 600 tok/s, or about 800–1,000 tok/s with idea 1. The drive allows 30–100 tok/s now and about 400 tok/s with 256 GB of RAM. Lossy prefill kernels therefore buy nothing on this box.
- **Two concrete changes to the design:**
  - Batch rows in E2's gemv without changing any bits, so large chunks never need E1.
  - Gate P0 on bitwise comparisons. My model shows that matching replies is a weak test here.

## Ideas, ranked

**1. Batch rows in the E2 gemv without changing bits (hipEngine's exact row-batch and output-batch kernels)**
- **What it is:** one decoded weight block is reused across many activation rows, and one staged activation across many outputs. Each output keeps its decode-band chain.
- **How hipEngine does it:**
  - Its input-amortizing variant swaps the loop nest. The code describes the swap as "layout only… the same 256-thread tree per output" (`hipengine/kernels/hip_gfx1100/quant/qwen4_exp_q5_1.hip:41-48`).
  - In another kernel, "128 physical threads materialize the same 256 logical partials" before the original reduction tree (`:458`). That cut its 512-token bucket from 3.470 to 2.534 s with exact output (`docs/KERNELS.md:277`).
  - Batching 8 rows in the Q4_K gate/up projection cut a natural 4K prefill from 291.6 to 231.8 s. Scheduling 4 or 8 outputs per workgroup took it from 235.8 to 228.6 s and from 237.1 to 222.2 s. All of these stayed byte-identical (`docs/KERNELS.md:275,277`; the general rule is at `docs/RDNA3-TUNING-GUIDE.md:689-704`).
- **Fit:**
  - Our gemv's 36 KiB LDS buffer holds a single 5,120-wide activation row. Gate/up therefore runs passes of one row (XLM=1), and each pass re-reads the expert's 12.5 MB of gate/up weights (`libr4d/r4d_ds41_mxfp4.hip:376-381,630-640`). The design's "Resident MoE about 0.7 s" comes from this (`PREFILL-DESIGN.md:137`).
  - The kernel already has a path that stages K in chunks, and its comment says the bits do not depend on how K is staged (`r4d_ds41_mxfp4.hip:286-288`).
  - For tall passes, take that path with 8 rows a pass whenever an expert has more rows than XLM. Optionally raise `kMxXM` (`:194`) to 16–32.
  - Add these heights to P0's device check.
- **Effect (modelled):**
  - Gate/up weight traffic falls 8× or more. Resident MoE drops from about 0.7 s to 0.2–0.3 s per 1,024-token chunk.
  - Time to first token is unchanged while the drive is the limit, because this work hides under the same layer's miss reads.
  - It removes E2's deficit at 4–8K-token chunks (`PREFILL-DESIGN.md:141`: "E1's tiled row is about 3× faster there") and in P3/P4.
  - Decode: no effect. Effort: about 0.5–1 day.
- **Related observation (exact, not from hipEngine):**
  - Decoded MXFP4 weights have at most 2 significant bits (`kMxValues`, `:47-48`), and activations are bf16. Every product is therefore exact in f32 unless it underflows or overflows.
  - In that case `fma` gives the same bits as today's multiply-then-add. This is the same argument `r4d_ds41_sparse.hip:19-20` already makes for its bf16 dot products.
  - With a per-row guard (no bf16-subnormal activations, otherwise fall back), fma would halve VALU instructions in the card's exact chain. The CPU kernels would need no change.

**2. Gate exactness the way hipEngine does (strengthens P0)**
- **How hipEngine does it:**
  - Arithmetic classes T0 (exact bytes) through T3 (`docs/EXECUTION-PROFILES.md:1034-1037`).
  - A batch-invariant gate that varies the slot, the neighbouring prompts and their lengths, the batch width, admission order, retirement and compaction (`:1288-1303`).
  - Exact routes are gated on teacher-forced full logits for every row, plus recurrent state and the full KV cache, run off/on/off (`docs/KERNELS.md:275,969-980`).
  - Its exact path silently stopped accepting chunks above 256 tokens for a week before anyone noticed (`docs/QWEN4EXP-STATUS.md:84-86`).
- **Why it matters here:** in my model, the gemv chain and the tiled chain round to a different bf16 in fewer than 1 of 3,000 outputs. That is still many perturbed elements per prompt. So W1's arm D (`PREFILL-DESIGN.md:308`) and reply identity on 8–30 prompts can pass even when the kernels are not batch-invariant.
- **Recommendation:**
  - Compare a per-row hash of the final logits or hidden state, plus hashes of the KV and indexer caches, for every prompt row at 16-token versus 64- and 1,024-token chunks.
  - Vary the neighbouring requests too; radiance packs several requests into one step (`core/sched/scheduler.h:465-468,794-796`). LCP's last-chunk admission flag must then be tracked per request.
  - Keep a standing device test at tall heights.
- **Effect:** no speed change; it prevents a false pass. About 1 day.

**3. Opt-in only: WMMA MoE and indexer, optionally with a rounding-risk screen and exact repair**
- **How hipEngine does it:**
  - **Screened routes:** grouped Q8 experts run on BF16 WMMA. Outputs within 4·2⁻²³·Σ|x·w| of a bf16 rounding midpoint are queued, and a sparse kernel recomputes them in strict order (`quant/gguf_q8_0_prefill.hip:681-693,698-747`). An integer-WMMA variant tracks a Kahan error bound and measured a 0.712% repair rate (`quant/gguf_q4_k_selected_prefill.hip:2101-2110,2378-2390,2517-2526`).
  - **Not proven exact:** the code itself calls the screen "empirical", and model-level gates remain binding.
  - **Unscreened routes:** a wide-tile F16 routed-expert GEMM cut routed-expert time from 4,621 to 695 ms per 4K prompt (`docs/QWEN4EXP-STATUS.md:137-141`). Grouped WMMA against the GEMV path runs from 0.97× at 32 tokens to 2.19× at 4K (`docs/LESSONS-LEARNED.md:302-310`).
  - **hipEngine's own warning:** reordering one projection's f32 sum moved mean KL to 2.09e-3 on prompts of 2.0–2.6K tokens, and a route change altered 9 of 12 continuations of 128 tokens (`docs/QWEN4EXP-STATUS.md:77-92`).
- **Fit:**
  - Our MoE output is one bf16 rounding of an f32 sum of exact products (`r4d_ds41_mxfp4.hip:9-13`).
  - Our tall-pass WMMA `index_score` changes only the dot's summation order (`r4d_ds41_sparse.hip:1423-1431`).
  - gfx1201's BF16 WMMA keeps both operands exact, so this falls in hipEngine's T2 class.
  - The existing `libr4d/r4d_moe_mxfp4.hip:1-4` runs MXFP4 experts on FP8 WMMA, but it quantizes activations to FP8, which is a larger change (T3).
- **Model result:** a provable bound flags 16–78% of outputs, which leaves no speedup. The empirical screen flags 0.4–3.2% of outputs. It caught every divergence when the WMMA unit is modelled as rounding to nearest, but missed 2 of 3,000 when it is modelled as truncating.
- **Verdict:** it cannot meet the byte-identical bar. Ship it only behind an opt-in flag, with a shadow mode that recomputes a sample strictly and counts misses.
- **Effect:** MoE compute would be about 5–10× faster (roughly 195 TFLOPS of dense bf16 WMMA, against a measured 5.7 T MAC/s for the exact chain). Time to first token gains about 0 on this box.

**4. Engram gathers in tall chunks (hipEngine's per-layer-embedding I/O work)**
- **How hipEngine does it:**
  - It adopted `MADV_RANDOM`, which made cold prefill 3.10×, 2.20× and 1.29× faster at 512, 1,024 and 4,096 tokens (`docs/QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md:33`).
  - Persistent `pread` workers were 6.3–6.7× faster on cold unique gathers but slower once warm, so they were not adopted (`:31`).
  - Deduplicating and sorting row ids won on realistic rows (`:30`).
- **Fit (only if the table is placed on disk, `--ngram-placement disk`):**
  - Ours issues 393k page hints, about 240 ms a rank, per 8,192 tokens. A cold gather took 370 ms per call on a 16K-token prompt (`core/runtime/rowprefetch.h:27-41`), on the same NVMe as the expert misses.
  - A chunk's Engram ids are known before its first block (`arch/deepseek41/deepseek41.cpp:1162`).
  - So in LCP: deduplicate the chunk's ids and issue both Engram layers' reads at chunk start. The bytes are the same, so this is exact.
- **Effect:** at most about 1% of LCP time to first token, plus less contention on the NVMe queue.

**5. Exact tall forms of `index_score` and sparse attention**
- **How hipEngine does it:**
  - Its tiled block scorer handles 16 queries × 16 blocks per workgroup. Queries are staged once, keys use a 129-float stride, and each row stops at its causal limit. Each (query, block) score runs the same statements as the original, so output is byte-identical (`attention/qwen4_exp_qsa.hip:1140-1216`; `docs/KERNELS.md:641-653`).
  - Two query heads share each K/V load, with an explicit fma that preserves the original fused multiply-add. This is exact and 2.54–2.61× faster (`docs/KERNELS.md:981-990`).
  - Its union-of-selections WMMA sparse attention is F16, so it is lossy (`:826-833`).
- **Fit:** P0 keeps our scalar forms above 48 rows (`r4d_ds41_sparse.hip:1227`). The same rule (change the staging, not the statements) gives faster exact tall forms.
- **Effect:** small. Exact attention costs 0.2–0.5 s per 1,024-token chunk (`PREFILL-DESIGN.md:139`), and it overlaps with drive reads once P2 lands. The indexer costs tens of ms per chunk at 8K tokens or fewer. This matters only for prompts of 32K tokens or more.

**6. Cautions for P3's card share**
- Grid shape does not control the order in which blocks run. An expert-sequential grid meant to keep weights in L2 was 59% slower (`docs/LESSONS-LEARNED.md:356-387`; `docs/RDNA3-TUNING-GUIDE.md:313-327`).
- Map host memory only for sparse access. In hipEngine, a direct file mapping corrupted outputs (`docs/RDNA3-TUNING-GUIDE.md:1322-1328`).
- So P3 should keep copying experts into lent VRAM slots over DMA. Running the gemv over zero-copy host memory (19.4 GB/s) would re-read the weights on every pass.

**7. Decode only (minor)**
- Retained PM4 command tapes gave +8.1% decode at one sequence on gfx1100 (`docs/reference/PM4.md:67,117-119`). hipfire's Redline reported +1.1% to +7.9% on gfx1201 (`docs/ROOFLINE.md:1288-1300`).
- Persistent kernels pay off only for glue whose working set fits in cache: 6–13× at 0.5–2 MB, but 0.93–0.96× beyond the last-level cache (`docs/RDNA3-TUNING-GUIDE.md:1445-1484`).
- For our ~60 ms decode step (task #47), my guess is +1–3%.

## Local model
Script: `/tmp/hipeng-research-1029/scratch/guard_sim.py`, run with `python3 -I`, single thread, about 40 s, on synthetic data.
- **Strict chain:** our gemv order (lane = block mod 32, xor butterfly), or the `index_score` order (in dimension order, d).
- **WMMA model:** steps of 16 terms, rounded once per step to nearest. The "truncating" column rounds each step toward zero instead.
- **Provable bound:** an error bound that holds for any rounding order, at 2u per addition.

Percentages in the first three result columns are the share of outputs whose bf16 differs from the strict chain.

| Case | Tiled order | WMMA | Truncating WMMA | Screen flags (misses) | Provable bound flags |
|---|---|---|---|---|---|
| gate/up, K=5,120 (3,000 outputs) | 0 | 0 | 0 | 3.17% (0) | 78% |
| down, K=2,304, SiLU·up inputs | 0 | 0.033% | 0.067% | 1.03% (0) | 33% |
| index dot, D=128 (20,000) | 0 | 0 | 0 | 0.70% (0) | 16% |
| gate/up, log-normal inputs (stress) | 0 | 0.033% | 0.37% | 0.43% (0; 2 under truncation) | 25% |

## Achieved numbers the brief asked for
- **W7900 (gfx1100):**
  - BF16 WMMA measured 84.8 TFLOP/s against a 123 spec; INT8 WMMA 75.3 TOPS (`docs/ROOFLINE.md:161-167`).
  - On RDNA3, INT8 is no faster than FP16 and there is no FP8 (`docs/RDNA3-TUNING-GUIDE.md:866-870`).
  - Effective active-weight bandwidth was about 150–292 GB/s against 864 GB/s (`docs/ROOFLINE.md:191,581`).
- **gfx1151:** 59.4 TFLOP/s WMMA peak; 4096³ GEMMs reach about 29–33 TFLOP/s (`docs/reference/ROOFLINE-gfx1151.md:21,603-612`).
- **Qwen3.8-Flash-Next prefill on gfx1151:**
  - Strict profile: 51–58 tok/s.
  - Original production profile (T2): 307–347 tok/s.
  - F16 composition: 769–1,562 tok/s.
  - Kernel time per 4K prompt: 12,186 ms before the F16 routes, 3,657 ms with them (`docs/QWEN4EXP-STATUS.md:59-61,133-141`).
- **gfx12:** hipEngine has no measurements of its own. For the RX 9070 XT: 48.7 TFLOPS FP32, 97.3 TFLOPS FP16 vector, 779 TOPS INT8 with sparsity, 640 GB/s. Dense INT8 is about 389 TOPS; about 195 TFLOPS for bf16 WMMA is my derivation, not a sourced figure ([geeks3d](https://geeks3d.com/20250305/amd-radeon-rx-9070-xt-and-rx-9070-launched), [club386](https://www.club386.com/?p=99745)).

## Pointers from hipEngine's own engine survey (`docs/reference/EXTERNAL-ENGINES.md`)
These are hipEngine's descriptions; I did not verify them. They may help with the dsv4engine and strata coverage.
- **DS41Engine** (`:358-387`): ds41 on 1–2 RX 7900 XTX.
  - Every routed expert is computed, from VRAM, a second card, 80 GiB of pinned RAM, or NVMe.
  - A tuning script keeps the NVMe and CPU awake between reads.
  - The expert ranking is rebuilt from observed router counts.
- **Xolotl** (`:646-689`): ds41 on one 24 GB card. Hot experts sit in VRAM, a pinned host arena is read by DMA, the rest stays on NVMe, and the next layer's experts are fetched a layer ahead.
- **lean-qwenflash** (`:926-967`): misses are read with O_DIRECT from NVMe straight into a bounded GPU cache, so all expert math runs on the GPU. This is the all-card form of our P3. Four PCIe 4.0 x16 links far exceed the 7 GB/s drive, so streaming misses to the cards would remove the CPU limit at chunks of 2,048 tokens or more.
- **Others:**
  - ds4 (`:1122-1169`): SSD streaming.
  - Strata (`:1079-1121`).
  - CapyFlash, fngine and FastLocalExperts (`:599-645,861-925`): the CPU computes experts held in RAM. FastLocalExperts reports prefill as its weak point.
  - Lucebox (`:457-491`): tiers across a discrete GPU, an iGPU and an SSD, with published prefill write-ups.

## Sources
- https://github.com/shisa-ai/hipEngine (commit `cc30876b`)
- [geeks3d: RX 9070 XT launch](https://geeks3d.com/20250305/amd-radeon-rx-9070-xt-and-rx-9070-launched)
- [club386: RX 9070 specs](https://www.club386.com/?p=99745)

## Survey: qfn-radiance

I wrote no report file; the full findings are below. No repository was modified, no GPU was used, and the only local computation was a few single-thread arithmetic lines.

# QFN-radiance as a source of prefill ideas for ds41

**Bottom line:**
- QFN prefills at about 4,500 tokens/s: about 1/150 of ds41's per-token cost.
- ds41 has the same engine but turns off the large-chunk machinery: `--max-num-batched-tokens 16`, `--deterministic` (static plan), no arena levels in the deepseek41 plugin, file-tier experts instead of a host pool, and `--no-prefix-cache`.
- The three most useful findings:
  1. **Chunk-size trap:** any chunk above 2,048 tokens silently runs as 2,048-token chunks unless `--checkpoint-interval` (or `--prefill-span`) is raised too.
  2. **PrefillStager pattern:** QFN copies non-resident experts into a VRAM ring at the arena's top, paced 3 layers ahead, sent as SDMA copy lists, with the ring lent to the experts during decode. This is the proven way to give the card all the experts that live in RAM.
  3. **Existing whole-layer path:** the base engine already reads a whole layer per pass and lets the card compute it. With 4 settings changed it should already run long prompts about 4–15× faster (modelled), though not byte-identical to the 16-token base.

**Path roots:**
- **T** = `/home/as/radiance-ds41-claude` at HEAD 9c8dccb9. It contains upstream 1.13 (commit 6c33ee4e) plus 195 commits. QFN's mechanisms are upstream code (https://codeberg.org/StillDeadcode/radiance, e.g. `PrefillStager` at upstream `core/place/stager.h:102`).
- **S** = `/home/as/radiance-setup`.
- **QFNLOG** = `S/benchmarks/20261024T123036Z-radiance-112-update/production-1.12.0-startup-container.log` (production image 1.12.0, `S/compose/.env:2`).
- **DSLOG** = `T/tools/ds41-host-experts/build/ab-20261029T150322Z/all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-r2.server.log`.

## 1. How QFN prefills

**Configuration** (`S/compose/flashnext-tp4.yaml:20-29`):
- `--tp 4 --tp-wire exact --max-num-seqs 8 --max-model-len 262144 --placement expert_tiered --host-pool-mib 10240 --kv-cache-dtype fp8 --prefix-cache-mib 8192 --num-speculative-tokens 3 --max-num-batched-tokens 4096 --checkpoint-slots 40 --prefill-span 2`
- The model is Qwen3.8-Flash-Next (qwen4exp, 48 layers): int8 W8A8 trunk and lm_head; routed experts are 4-bit codebook (w4nl64) with E4M3 activations.
- Each rank holds a slice of every expert (tensor parallel), not whole experts (expert parallel).

**Chunks:**
- QFNLOG:110-111: quantum 64, checkpoint interval 2048, `max_tok` 4096. A prefill alone in its step carries 2 intervals (4,096 tokens) from its second interval on; every other step is planned at 2,048.
- It spans only prompts of 16,384 tokens or more (QFNLOG:164; `T/core/engine.cpp:2081-2088`): a spanned step uses the full arena plan, which lends the experts nothing.
- A 47K-token prompt takes 13 steps (`analysis.json` `prefill_steps`).
- `--busy-prefill-chunk auto` never holds a chunk below 1,024 tokens, the stager's smallest pass (QFNLOG:163; `T/core/sched/scheduler.cpp:1167-1200`). `--prefill-interleave` is off.

**"Macro-batching":** radiance has no such term. The closest mechanisms:
- the span above;
- the MoE block runs a step in passes of 2,048 rows (`T/arch/common/rad_block_moe_fp8.h:20-30`; `T/arch/qwen4exp_fp8/qwen4exp_fp8.cpp:446-450`);
- at TP4 a spanned step is pipelined in 2,048-row pieces, with every all-reduce on a second lane beside the other piece's work (`qwen4exp_fp8.cpp:1447-1560`). All-reduces are about a quarter of a chunk's time at TP4 (`T/docs/GUIDE.md:1137-1141`);
- the shared token budget packs several prompts' chunks and decode rows into one step (`GUIDE.md:1260-1270`).

**Experts at large M: the PrefillStager** (`T/core/place/stager.h:1-115`):
- Experts that are not in VRAM sit in a pinned host pool (9.97 GiB a rank, QFNLOG:395-396). Small steps read them in place across PCIe.
- Steps above 1,024 tokens copy every routed layer's pooled experts into a VRAM ring (QFNLOG:134). The ring is 416 MiB, two layers' share, at the top of the activation arena (`T/core/mem/vram_budget.cpp:187-215`).
- Copies run up to 3 layers ahead (`stager.h:197`), paced on compute events. Unpaced copies measured −0.6%; one layer ahead −2.4% (`stager.h:57-66`).
- It is a ring rather than two buffers because layers' pooled parts differ about 2× (`stager.h:33-42`).
- At the end of a pass it stages the next chunk's first layers (`stager.h:67-80`; `T/core/engine.cpp:1625-1651`).
- Copies go as SDMA copy lists on a queue the process owns, and give way to collectives (`T/core/place/mover.cpp:841`; QFNLOG:133).
- `T/core/device/aql.cpp:2700-2706`: the SDMA queue moves 28.7 GB/s with no cost to running kernels. The runtime's copy moves 22 GB/s and slows kernels 8%. Copy kernels get 2.3 GB/s beside a saturating kernel.
- The bytes are the same, so output is bit-identical (`stager.h:16`).
- Why it exists: a 2,048-row gate_up takes 1.53 ms with every expert in VRAM and 4.34 ms with 13% read from host memory (`stager.h:7-12`).
- In production it stages 225 GiB per 47K-token request (about 4.3 GiB per rank-step), with 0 overflows (`S/benchmarks/20261024T123036Z-radiance-112-update/analysis.json`).

**VRAM lending:**
- The plugin re-declares at smaller step sizes (`shape_probe_ok`, `qwen4exp_fp8.cpp:1703`; `T/core/engine_bringup.cpp:1507-1599`).
- Arena levels of 256 / 1,024 / 2,048 tokens lend 1.31 / 1.00 / 0.18 GiB to the expert slab (QFNLOG:28, 154). The KV cache lends up to 5.48 GiB (QFNLOG:141).
- A prefill's first chunk reclaims the loan synchronously; lending resumes 2 s after large steps stop (`engine.cpp:1604-1678`).

**Kernels at large M:**
- W8A8 tiled GEMM (`T/libr4d/r4d_gemm_w8a8_tiled.hip:1-9`).
- `moe_gemm_w4a8`: 16-row tiles for decode, 32-row tiles for prefill, plus a prefill block map giving one run per block. The block map is worth +3.6% prefill and is bit-identical (`T/libr4d/r4d_moe.hip:14-22`).
- `attn_prefill_h256_gqa6_bs4_fp8kv`, the GDN chunk scan, and the QSA indexer scored in pass-tall windows.

**Prefix caching:**
- On, with exact resume (`T/core/config.cpp:195-199`).
- An 8 GiB host KV tier for finished conversations (QFNLOG:114).
- 40 linear-state snapshots of 113 MiB each, at a 2,048-token interval (QFNLOG:130).

**Exactness bar:**
- QFN is not invariant to chunk size: the split-K count is chosen from M (`T/spec.md:1658-1666`).
- It keeps output equal only (a) for prefix hits, through the resume grid, and (b) for spans, by computing in 2,048-row pieces (`GUIDE.md:1140-1143`).

## 2. QFN prefill rate

All runs are cold BetterBench with unique nonces and 0 cached tokens.

**1.12.0 production** (`S/benchmarks/20261024T123036Z-radiance-112-update/comparison.html`):

| Prompt length | Prefill rate |
|---|---|
| ~1.5K | 3,345 tok/s |
| ~6K | 4,131 tok/s |
| ~11.8K | 4,202 tok/s |
| ~23.6K | 4,493 tok/s |
| ~47.2K | 4,515 tok/s |
| 261,828 tokens | 59.8 s (≈4,380 tok/s) |

- That is ≈0.22 ms a token, against ds41's ≈33 ms.
- Decode is 221 tok/s single-stream and 491 tok/s at 8 clients.

**Lossy wht6 wire:** 7,178 tok/s at 47K (`S/benchmarks/20261023T163701Z-radiance-111-exact-review/comparison.html`).

**Chunk-size sensitivity:**
- **TP4 is compute-bound.** Compared with 2048/span1, 4096/span2 halves the steps and staged bytes (−43.8%) and changes throughput by under 1%. 4096/span1 (chunks still 2,048, arena bigger) is −13.5% (`S/benchmarks/20261023T182152Z-batch-span-investigation/findings.json`).
- **One card is link-bound.** Going from 2,048 to 4,096-token chunks took it from 1,413 to 2,545 tok/s on 26K-token prompts (`S/compose/flashnext-tp1.yaml:6-9`).
- The guide's table for two link-bound cards: chunks of 512 / 1,024 / 2,048 give 1,750 / 2,200 / 2,706 tok/s. In its words, "a chunk sweeps the whole expert plane once however many tokens it carries" (`GUIDE.md:1072-1086`).

## 3. What the ds41 configuration turns off

Sources: `T/tools/ds41-host-experts/ab_window.sh:135-141` and DSLOG.

- **Chunk:** `--max-num-batched-tokens 16` gives 16-token chunks (DSLOG:123). The CPU path is limited to 16 rows (`T/core/place/ep_route_counts.h:9`; `T/core/place/stager.cpp:1010`).
- **`--deterministic`:** sets `Plan::dynamic` false (`T/core/place/planner.cpp:1237`). That means no heat engine and no KV or arena loans. The PrefillStager is never armed, because it needs `plan.dynamic` and arena levels (`engine_bringup.cpp:3986`). The arena loan also needs the heat engine (`:4128`).
- **No arena levels:** the deepseek41 plugin does not declare `shape_probe_ok`. DSLOG:21: "activation arena: not lent". Any larger arena would be permanent.
- **Experts on the file tier, not in a host pool:** the FileStager runs instead of the PrefillStager.
  - With B1 its two VRAM buffers hold 1 unit each (DSLOG:153).
  - With K the RAM cache is "packed for the CPU only … the card's copy path neither reads nor fills it" (DSLOG:82).
  - File copies use `rad_memcpy_async`, i.e. copy kernels (`mover.cpp:1428-1431`; `aql.cpp:2628-2636`). The SDMA queue is never opened (DSLOG:419).
- **No prefix cache:** `--no-prefix-cache` (DSLOG:138), 0 checkpoint slots, no host KV tier.
- **Busy-prefill floor is 16 tokens** (DSLOG:174), because there is no stager.
- **Also:** `--max-model-len 512`, `--max-num-seqs 1`, no speculation.

## 4. Ideas for ds41, ranked

Effects are modelled with PREFILL-DESIGN's constants unless marked measured.

**I1. Raise `--checkpoint-interval` (or set `--prefill-span`) with `--max-num-batched-tokens` (flags only).**
- **What:** ds41 has linear state, so the interval is 2,048 even with the prefix cache off (DSLOG:123; `T/core/sched/geometry.cpp:131`). The effective chunk is the interval times the span (`geometry.cpp:195-196`; `GUIDE.md:1089-1102`).
- **Consequence:** chunks of 4,096 or 8,192 tokens silently run as 2,048 with a bigger arena. QFN measured −13.5% from exactly this. Setting `--checkpoint-interval ≥ C` costs nothing under `--no-prefix-cache`. The span never covers a prompt's first interval.
- **Effect:** needed for the design's P4 column (8,192 tokens in ≈18 s against 54–66 s at 2,048-token chunks, about ×3).
- **Note:** `ab_window` needs a parameter for this beside `MAX_BATCHED`.

**I2. The PrefillStager pattern for the card's share in LCP ("P3 generalised").**
- **What:**
  - The card computes the tall pass's experts that are in the RAM cache, from a VRAM ring at the arena's top, sized from what is left over.
  - Copies run 3 layers ahead, with the next chunk's first layers prestaged at the end of a chunk.
  - Copies go as SDMA copy lists straight from the pinned slabs, with no host memcpy bounce (`mover.cpp:1567`).
  - Overflow falls back to the CPU through the existing alias-and-overwrite path (`stager.h:403-418`).
- **Fit:**
  - The RAM cache must be in block form, or be widened back to it (K blocks the card today).
  - Exactness holds under E2: same bytes, same gemv. The bit-identity rationale is `stager.h:16`.
- **Effect now (one NVMe):** about the same as P3.
  - Moving the cached hits to the card is ≈102–111 GB per 2,048-token chunk, about 1.2–1.4 s with DRAM the limit. The CPU-only route needs 15.6–16 s per chunk.
  - So 2,048-token chunks become drive-bound at 12–15 s: ×1.05–1.3 over P0+P1 run at 2,048-token chunks.
- **Effect if misses vanish (more RAM):** DRAM to PCIe becomes the limit. A full sweep (235 GB) takes ≈2.5–3 s, giving ≈700–900 tok/s at 2,048-token chunks and ≈2,800–3,300 at 8,192. The design's CPU path tops out near 400 tok/s.
- **Decode cost:** the ring is 2.7–2.9 GB a card (2 × 73–78 units), about −8% decode if permanent. It must be lent (I3); then the only cost is one ≈1–1.6 s refill per long prompt.

**I3. Arena levels and lending.**
- **What:** deepseek41 declares at smaller step sizes, and a lending variant works under a static plan (design P3 items 13–14).
- **QFN policy to copy:** reclaim on a prefill's first chunk, regrow after 2 s (`engine.cpp:1613-1678`).
- **Effect:** recovers the 0.6–2.6% decode cost of a permanent LCP arena, and is what makes I2's ring affordable.

**I4. Slice the MoE into passes of `rows` tokens, as QFN does.**
- **What:** deepseek41 sets `s.cfg.rows = T` (`T/arch/deepseek41/deepseek41.cpp:1513`), so the MoE buffers are T×6 rows (`:1014-1017`): ≈0.30 GB a card at 2,048 and ≈1.18 GB at 8,192. With `rows` = 512 they are ≈74 MB.
- **Exactness:** exact, because routing is per token and the gemv's bits do not depend on M (`T/libr4d/r4d_ds41_mxfp4.hip:286-288`).
- **Constraint:** the FileStager keeps a layer staged across slices (`stager.h:238-242`). But the CPU (hx) path rejects sliced passes (`stager.cpp:686-692`; `stager.h:342-343`). So slicing needs the card path, or all slices' routing done first.
- **Effect:** makes 8,192-token chunks fit. Worth ≈+0.6% decode (2,048) to +3% (8,192) if the arena stays permanent.

**I5. An exact tall-M gemv for the card (a large-M path that already exists).**
- **What:** at K = 5120, gate_up's activation row (5,760 floats, padded) fits the 36 KiB LDS buffer only once, so gate_up re-reads its weights for every row (`r4d_ds41_mxfp4.hip:376-381`, `:627-640`).
  - The chunked-K path (chunks a multiple of 1,024 values) is bit-identical by the kernel's own invariant (`:286-288`) and reads the weights once per 8 rows.
  - That is 8× less weight traffic for gate_up and 2.7× for down. Raising the 8-row pass (`kMxXM`) to 16–32 cuts more.
- **Effect:** E2's card MoE cost (≈0.7 s per 1,024-token chunk for residents; ≈3 s per 2,048-token chunk if the card computes all touched experts) drops about 4–6×.
- **TTFT:** no change while drive-bound. It matters for I2 after a drive or RAM upgrade, and at 8,192+ where the gemv approaches drive time (`PREFILL-DESIGN.md:141`).
- **Check:** byte-compare at M from 17 to 1,024 with the P0 device harness.

**I6. A fallback if the P0 gate fails: compute in pieces of the reference height.**
- **What:** QFN computes a span in 2,048-row pieces so its output equals 2,048-token chunks (`GUIDE.md:1140-1143`; `qwen4exp_fp8.cpp:1505-1560`). For ds41:
  - run any op that fails invariance (sparse attention, indexer) layer by layer in 16-row pieces on the base engine's chunk grid, or in pieces of 48 rows or fewer (its exact scalar form);
  - keep the MoE and dense ops at full height.
- **Exactness:** exact by construction against the base engine at 16-token chunks, because causal attention reads the same KV.
- **Cost:** ≈0.4–0.8 s per 1,024-token chunk, under the drive time.
- **Why it matters:** it is cheaper than E1 and keeps today's reference.

**I7. Scheduler settings to reuse (about 0.5 day).**
- Feed the LCP threshold (e.g. 256) to `Scheduler::set_stage_rows` (`engine.cpp:2072-2080`), so `busy_cap` never holds a tall chunk down in the 64–256-token passes where the LRU thrashes (`scheduler.cpp:1185`).
- `--prefill-interleave N` keeps concurrent decoders on cadence (`GUIDE.md:1108-1114`).
- Several prompts can share one tall step and so one expert sweep: two 1K-token prompts in about 13–16.5 s together, against 22–28 s one after the other.
- Exact under E2.

**I8. Prefix cache with a host KV tier (the design's item #2).**
- **QFN precedent:** `--prefix-cache-mib`, exact resume.
- **Size:** ds41's KV is about 55 KB a token across the 4 cards, so 1–2 GiB of host tier holds ≈20–40K tokens. That costs 1–2% of the RAM cache.
- **Effect:** a returning conversation re-prefills only its new turn. A 4K-token history saves ≈140–200 s today, or ≈40–55 s with LCP.
- **Decode:** small (<0.5%, unmeasured).

**I9. How much of each layer to read.**
- QFN stages a layer's whole pooled part, because a Qwen chunk touches nearly every expert.
- ds41 touches 73–83% of non-resident experts at 1,024-token chunks, 82–94% at 2,048, and 97–100% at 8,192 (`PREFILL-DESIGN.md:57`). Whole-layer reads therefore waste 17–27% / 6–18% / under 3% of drive time.
- **Recommendation:** read only routed experts, plus prior-ordered speculation (design P2), below about 4,096 tokens. Read whole layers, with no wait for the router, from 8,192.

**I10. A measurement baseline that already exists (design's W1 arm C).**
- **What:** the base engine above 64 rows reads whole layers into two VRAM buffers, one layer ahead, blocking the issuing thread (`stager.h:223-247`). Above 16 rows it uses the tiled kernel (`r4d_ds41_mxfp4.hip:805`, `:848-849`).
- **Four settings, or it measures the wrong thing:**
  - no B1;
  - no K (otherwise every expert comes from the NVMe);
  - `--checkpoint-interval` at least the chunk size;
  - `RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREFILL_READ_ONLY=1`. It already exists for the card path (`T/core/place/native_file_cache_policy.h:6-19`; `stager.cpp:659-661`; counters at `mover.cpp:1722`), and stops LRU thrash.
- **Model:** each chunk re-reads ≈128 GB from the NVMe in ≈17–18 s. That is ≈115 tok/s at 2,048-token chunks and ≈450 at 8,192: 4–15× today's 30 tok/s.
- **Limit:** replies match the base engine only at the same chunk size.
- **Implication:** E2 plus this path plus I1 and I3 is a low-code exact LCP. It is 17–33% slower than routed reads at 2,048-token chunks and equal at 8,192.

**I11. Pipelining all-reduces on a second lane, as QFN does.**
- About 80 all-reduces of 21 MB per 2,048-token chunk take ≈0.1–0.2 s, under 2% of a drive-bound chunk.
- Revisit only once prefill is compute-bound.

**Lossy options, opt-in only.** None helps prefill while it is drive-bound (≈0–2%):
- `--tp-wire wht6`: measured +37–50% on QFN, where prefill is compute-bound.
- The E4M3-activation MoE kernel `moe_gemm_mxfp4a8` (`T/libr4d/r4d_rows.cpp:8215-8221`).
- `--tp-attn-split rows` (not bit-identical).
- `--attn-prefill-fp8`.

## Survey: offload-engines

# Offload engines: prefill techniques, judged for ds41 on radiance

I did research only. I modified no repository and used no GPU.
- Reading copies: ktransformers @`c987331` (2026-10-28) and ik_llama.cpp @`2484412` (2026-10-29), shallow clones in `/tmp/offres/`. llama.cpp is the local `/home/as/llama.cpp` @`d1a92352c`.
- Radiance references are to `/home/as/radiance-ds41-claude` @`9c8dccb9`, the HEAD that PREFILL-DESIGN.md used.
- The only local computation is arithmetic on the design's constants (`/tmp/offres/scripts/lm_model.py`, single thread, instant). Paper texts extracted for reading are in `/tmp/offres/pdfs/*.txt`.
- Every number marked "model" is mine and unmeasured.

## Bottom line

1. **Above about 1–2K tokens, every engine does the same thing.** It moves each expert to the GPU once per chunk and computes it there with all of the chunk's rows.
   - Examples: ktransformers "layerwise prefill" (≥2,048 tokens for DS-V4-Flash), llama.cpp op offload, MoE-Lightning, MoE-Gen, DuoServe, PowerInfer-2 and vLLM RFC 63666.
   - The design's P1+P3 already does this for experts held in RAM.
   - For us, our limit is NVMe bytes per prompt, not where an expert is computed.
2. **The biggest new lever is layer-major execution.** The design mentions it in one line ("Extra large") but does not develop it. Klotski and MoE-Gen use it; SeqMoE explains why it matters.
   - Each expert is read once per prompt, while the activation arena stays at the chunk size.
   - Model: 4K tokens 15–19 s (P3: 27–33), 8K 17–20 s (P3: 54–66), 16K 18–23 s.
   - It is exact under the same P0 gate as the large-chunk mode.
3. **Its precondition is a cheaper tall MoE on the card, which is also a cheaper win on its own.**
   - At tall heights the gemv re-reads gate_up once per row (`XLM`=1). That costs about 0.02 ms a row, which is VRAM bandwidth, so above about 8K tokens the card becomes the limit.
   - A bit-neutral row-batched or weight-stationary variant cuts this by about 3× (assumed, using the design's E1 factor). It also makes zero-copy compute from pinned RAM feasible, the way vLLM's UVA offloader works.
4. **P3's 50 lent VRAM slots can shrink to a small ring.** DuoServe, ktransformers and the vLLM RFC all stream experts through 2-slot or 2-buffer rings, never a whole lent set.
5. **Split rule.** Use HybriMoE's three-queue simulation instead of a fixed top-S.
   - The card is cheaper than the CPU (computed) above about 10–17 rows per expert for experts in the RAM cache, or 8–13 rows with idea 3's kernel.
   - For drive misses the location does not matter while the drive is the limit.
6. **Already in radiance or in the design, and confirmed by these engines:**
   - next-layer prefetch (P2)
   - computing experts in arrival order
   - dynamic CPU tile claiming
   - 4-row AVX2 blocking
   - a read-only decode cache during prefill
7. **Not applicable:** AMX/AVX-512, NUMA tensor parallelism, expert deferral (decode-only and lossy), CPU attention, KV offload, PowerInfer neuron sparsity, and llama.cpp's 32-token whole-op upload.

## Ideas, ranked

### 1. Layer-major prefill: read each expert once per prompt (exact; extra-large effort)

**What it is.** The engine runs layer L for every chunk of the prompt before starting layer L+1. Chunked prefill today re-reads every touched, uncached expert once per chunk.

**How the other engines do it:**
- **Klotski** (§4–5, https://arxiv.org/html/2502.06888v1) is "designed based on zig-zag block schedule" from FlexGen.
  - A group of batches shares each loaded weight. Expert compute runs across all batches, grouped by expert.
  - Hot experts run first, then the rest in the order their transfers finish.
  - It reports up to 85.12× throughput (Mixtral on an RTX 3090, PCIe 4.0, disk at 1 GB/s).
- **MoE-Gen** (§4.1, Table 1, https://arxiv.org/html/2503.09716v1) accumulates attention micro-batches in host memory, then launches large expert batches.
  - DeepSeek-V2 236B on one A5000 (PCIe 4.0, 512 GB host), 512-token prompts: prefill 841 tok/s, against 77–109 for FlexGen, MoE-Lightning and DeepSpeed.
  - It says hiding expert transfers needs more than 2^11 tokens per expert.
- **SeqMoE** (§8.4, https://arxiv.org/html/2609.12978v1; DeepSeek-V4-Flash is among its models) says chunked prefill is "poorly suited to offloading because it incurs repeated expert transfers across chunks".
- **ktransformers** avoids the problem with whole-prompt chunks and pays for it in VRAM:
  - Kimi-K2 recommends `chunked-prefill-size` 30,000–65,536 and needs "~9GB + incremental cost with prefill length" ([Kimi-K2-Thinking-Native.md:117,132-135](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/doc/en/kt-kernel/Kimi-K2-Thinking-Native.md#L117)).
  - For GLM-5.3-Flash, layerwise mode "normalizes the resident GPU expert count to zero" (GLM-5.3-Flash-Tutorial.md:90).
  - That is the design's P4 with lending.

**Fit for us.** Per layer, the engine would:
- (a) For each chunk of C = 1,024–2,048 rows, run the ops up to the router, with attention against the same layer's KV from earlier chunks.
- (b) Run the routed MoE over all P rows expert by expert. Each expert is fetched once, from VRAM residents, the RAM cache or the drive.
- (c) Run the post ops per chunk.

Memory it needs:
- **Carried state:** 4 hc streams × 5,120 per token (`/home/as/radiance-ds41-claude/arch/deepseek41/deepseek41.cpp:264-267`). That is 80 KB a token at f32: 640 MB at 8K and 2.5 GB at 32K, which can live in pinned host RAM.
- **Drive staging:** about 42 misses per layer-rank, 3.2 GB for all ranks. With one layer of read-ahead it is 6.4 GB, inside the ~14 GB that is free.
- At 8K, the chance that a given expert of layer L+1 is touched is ≥0.97, so P2 becomes "read all of L+1", as the vLLM RFC does.

Exactness:
- The arithmetic is the same as chunked prefill. Layer L's KV, indexer, compressor and hc state depend only on layer L's inputs from earlier chunks, which already exist when layer L runs for the next chunk.
- The all-token MoE needs the same P0 gate as large-chunk prefill: the tall gemv must be batch-invariant and the CPU kernels identical row by row.
- Layer-major with 16-token chunks would avoid P0 but is not attractive. The fixed cost of 76–114 ms per pass remains, and the CPU re-reads each expert every pass (about 14–25 ms a token).

**Estimated effect** (model, `lm_model.py`, drive 2.45–2.70 ms per miss, overlap loss 5% + 0.5 s):

| Prompt tokens | Drive time | Layer-major | P3 (design) | Gain |
|---|---|---|---|---|
| ≤2,048 | — | same as P3 (one chunk) | — | none |
| 4,096 | 13.7–17.5 s | 15–19 s | 27–33 s | ×1.7–1.8 |
| 8,192 | 15.8–18.3 s | 17–20 s (≈P4 without P4's 1.6–3.7 GB/card arena) | 54–66 s | ×3.2–3.3 |
| 16,384 | — | 18–23 s with idea 3; 29–38 s with today's gemv | ~108–132 s (extrapolated) | ×5–6 |

- At ≥16K the card, not the drive, sets the time unless idea 3 is built.
- The growth of attention cost with context is not modelled.
- PCIe: each card uploads at most 2.7 s per prompt.
- **Decode:** no permanent cost. Pair it with idea 5 at the end of the prompt.
- **Risks:** decode for other sequences stalls for the whole prompt unless it is interleaved between layers. The plugin's ops are built per pass (`RAD_RANGE("M",1,rows)`), so layer-major needs per-layer segments that save the hc state.
- **Effort:** my guess is 2–3 weeks after P0 and P1.

### 2. Cheap, exact tall MoE on the card; zero-copy from pinned RAM (exact; 2–4 days, estimate)

**What it is.**
- The decode gemv stages only `XLM` activation rows in LDS. For gate_up that is 1 row of 5,120.
- So an expert "takes them in whole passes of XLM rows, re-reading its weights" (`/home/as/radiance-ds41-claude/libr4d/r4d_ds41_mxfp4.hip:376-381`, `XLM` at `:211`).
- 12.5 MB per row at about 640 GB/s is about 0.02 ms a row. This matches the design's 0.7 s per 1,024 tokens for residents.
- With every MoE row on the card that is about 1.3 s per 1K tokens, so the card becomes the limit above about 8K.

**How the other engines do it:**
- ktransformers' DS-V4 AVX2 MXFP4 kernel: "N-outer loop: each weight row loaded once and shared across M tokens", with 4-token blocking ([mxfp4-moe.hpp:228-229,253](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/kt-kernel/operators/avx2/mxfp4-moe.hpp#L225-L260)).
- vLLM's UVA offloader: "The GPU can then directly access the CPU memory without explicit transfers, at the cost of PCIe bandwidth" (https://docs.vllm.ai/en/latest/api/vllm/model_executor/offloader/uva/).

**Fit for us:**
- Staging activations in LDS as bf16, with exact widening on read, multiplies `XLM` by 2–4. That is the design's item 8, "gemv staging 2+ rows (bit-neutral)".
- A weight-stationary variant goes further: each wave keeps its weight block and loops over rows. Every output keeps its lane order, block order and reduction tree, so it should be byte-identical. It needs a device check like P0's.
- Once weights are read once per expert, zero-copy works:
  - The kernel already takes a per-expert pointer table (`wtab`, `r4d_ds41_mxfp4.hip:201`).
  - RAM-cache slabs are `RAD_MEM_HOST_MAPPED`, with host and device addresses (`/home/as/radiance-ds41-claude/core/place/native_mapped_file_cache.h:196`).
  - Measured zero-copy is 19.4 GB/s, about 0.97 ms an expert.
- Two catches:
  - The packed (Pa) RAM-cache layout is CPU-only.
  - Misses land in malloc'd `HostPlanes` in the file's plane layout (`/home/as/radiance-ds41-claude/core/place/mover.h:608-632`).
  - Each needs either a layout-aware kernel variant (addressing only) or idea 3's relayout writer.

**Estimated effect:**
- Tall MoE on the card about ÷3 (assumed). Layer-major at 16K goes from 29–38 s to 18–23 s; at 32K from 58–74 s to 28–45 s.
- P3 at 2,048: 0–5%, since the card is not the limit there.
- With zero-copy, P3 needs no VRAM staging.
- Decode: none.

### 3. Stream the card's experts through a small ring instead of lending 50 slots (exact; ~2–3 days vs 4–5 for lending)

**How the other engines do it:**
- **DuoServe** (§5.1–5.2, https://arxiv.org/html/2509.07379v1):
  - Prefill fetches every expert of the layer into a 2-slot GPU cache: "one is being used for computation and the other is being fetched".
  - Two streams overlap the fetch with the non-MoE compute.
  - TTFT 1.78–3.8× over its baselines (PCIe 4.0, Mixtral AWQ).
- **ktransformers FP8 layerwise transport:**
  - Two pinned host slots. A rank-0 writer fills expert e+1 while every rank DMAs expert e ([fp8_layerwise_transport.cpp:365-427](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/kt-kernel/fp8_layerwise_transport.cpp#L365-L427), `:562-575`; amx.py:1113-1124).
- **vLLM RFC #63666** (https://github.com/vllm-project/vllm/issues/63666): two per-layer arenas, with layer L+1 assembled while layer L computes.

**Fit for us:**
- P3's 50 slots (0.94 GB a card) exist because the MoE op runs once per layer over all staged experts.
- Instead, launch the op per wave of k experts over a ring of 8–16 slots (0.15–0.3 GB). The expert table already allows a subset.
- This is exact: each expert's rows are independent, and the top-6 combine runs afterwards in its fixed order.
- For misses and packed hits, use ktransformers' pattern: a CPU writer relays into a pinned slot while the DMA moves the previous one.

**Estimated effect:**
- Same TTFT as P3.
- The lend shrinks to the arena alone, and the 0.7–0.95 s refill after a long prompt shrinks with it.
- The static-plan tier flips (P3 #14, the design's largest correctness risk) may not be needed for slots.

### 4. Balance CPU, card and PCIe per layer, and the crossover row count (exact; 1–2 days on `hx_choose`)

**How the other engines do it:**
- **Fiddler** (Alg. 1, §3.3, https://arxiv.org/html/2402.07033v3):
  - Uses the GPU if `cpu_lat(s) > gpu_lat(s) + trans_lat()`. CPU time is linear in tokens; transfer time is constant.
  - Long-prefill TTFT 1.07× vs DeepSpeed-MII and 1.65× vs Mixtral-offloading (§4.2).
- **HybriMoE** (§IV-B, https://arxiv.org/html/2504.05897v1):
  - Three queues: the GPU takes cached experts by load descending; the CPU takes uncached experts ascending, plus cached ones when idle; PCIe moves high-load uncached experts first.
  - A simulated timeline sets the split per layer.
  - Prefill 1.33× over ktransformers on average. In the ablation (Qwen2, 25% cache), scheduling alone gives 1.26× and all techniques 1.31× (Table III).
- **PreScope** also "prioritizes the most heavily loaded experts" (https://arxiv.org/html/2509.23638v1).

**Our crossover** (model):
- CPU cost: 0.20 + c1·(r−1) ms, with c1 between 0.04 and 0.09.
- Card cost from RAM: about 0.8 ms of DMA, plus 0.021 ms a row (today's gemv) or about 0.007 ms (idea 2).
- Break-even r = (0.6 + c1) / (c1 − card cost per row):

| c1 (CPU ms per further row) | Break-even rows, today's gemv | Break-even rows, idea 2 |
|---|---|---|
| 0.04 | 34 | 19 |
| 0.06 | 17 | 12 |
| 0.09 | 10 | 8 |

- Average rows per expert is C/64, so this corresponds to chunks of about 650–2,200 tokens.
- On c1: ktransformers' kernel uses the same 4-row strategy. 24 Zen3 cores give about 2.3 TFLOPS FP32 against 70.8 MFLOP per row-expert, so c1 of about 0.045–0.06 is plausible (estimate).

**Estimated effect:** 0–10% over a fixed top-S at 2,048-token chunks; 0 when the drive is the limit.

### 5. Use the prompt's routing counts to place experts for the decode that follows (exact; 1–2 days)

**How the other engines do it:**
- **ktransformers** `--kt-enable-dynamic-expert-update` "collects actual routing statistics and redistributes GPU experts" during layerwise prefill ([README.md:549](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/kt-kernel/README.md#L543-L551)).
  - Qwen3-Next-80B on 4× 4090, against frequency placement: 10% GPU experts 58.6 → 70.2 tok/s; 40% 72.8 → 81.0; but 80% 100.7 → 92.3 ([experts-sched-Tutorial.md:181-188](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/doc/en/kt-kernel/experts-sched-Tutorial.md#L174-L190)).
- **vLLM RFC:** "Prefill never evicts the decode pool"; prefill's "routing counts seed decode residency".

**Fit for us:**
- Read-only chunks leave the prompt's experts uncached, and a replay measured +17% decode misses from that (design §4.7).
- Admit the top experts by whole-prompt counts, not only those of the last chunk. Choose P3's refill set the same way.

**Estimated effect:** decode after long prompts +2–6% (estimate; check with a `cache_sim2.py` replay). No prefill effect.

### 6. Confirmations: prefetch, arrival order, drive reads

**Next-layer prefetch (P2):**
- HOBBIT: "the prefill stage utilizes all experts of each layer, resulting in 100% prediction accuracy"; prefetch cuts prefill about 10% (https://arxiv.org/html/2411.01433v2).
- DuoServe §5.1: no predictor is needed in prefill.
- vLLM RFC: prefill "streams full layers ... does not predict". The union is ≥98% at 128 tokens and 100% at ≥1K.
- Conclusion: keep P2 as designed (−5 to −10%). At 2,048-token chunks (touch probability 0.82–0.94), order by prior and cap the queue. In layer-major mode, read everything.

**Arrival order:**
- ProMoE prefill ablation (§7.3, https://arxiv.org/html/2410.22134v3): prefetch alone gives "minimal improvement and may even degrade performance"; early preemption 1.27×; reordered inference 2.39×.
- Radiance already computes hits while misses read, and lets demand reads preempt read-ahead. P1 #10's per-unit completion finishes the job.

**Drive reads:**
- PowerInfer-2 reads whole layers sequentially in prefill. On UFS it gets 4 GB/s at 512 KB reads against 450 MB/s at 4 KB (§2.3.2, §7.2.2, https://arxiv.org/html/2406.06282).
- The vLLM RFC stores one file per layer and reads whole records ("a split read was slower than one full read").
- Our reads are already 18.8 MB units at 2.45 ms in bulk (ideal 2.36), with huge pages (`/home/as/radiance-ds41-claude/core/device/aql.cpp:3800`). At most about 5% is left.
- Lossless compression of the drive tier: none of these engines does it. My estimate from the entropy of E2M1 codes is at most 5–10% fewer bytes, for the cost of a CPU decode stage. Not worth it. I did not measure it: `WINDOW_PENDING` was set, so I read no model bytes.

**Already done in radiance:**
- KTransformers' dynamic task scheduling gives up to 1.83× in prefill (§3.2). Radiance's pool already claims tiles dynamically (`/home/as/radiance-ds41-claude/core/place/host_expert_pool.cpp:229-255`).
- ktransformers' 4-row AVX2 blocking is the same plan as the design's `kGemvRows`.

## How each engine splits prefill, and its thresholds

| Engine | How prefill is split | Threshold | Notes |
|---|---|---|---|
| ktransformers, SOSP'25 paper | Routed experts on the CPU (AMX; AVX-512 at ≤4 tokens per expert); attention and shared expert on the GPU | — | 4.62–19.74× prefill over Fiddler/llama.cpp |
| ktransformers kt-kernel, 2026 | Hybrid CPU+GPU below the threshold; above it, "GPU (CPU weights transferred to GPU)", one MoE layer of extra VRAM | ≥2,048 tokens for DS-V4-Flash (default) | [DeepSeek-V4-Flash.md:79-84](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/doc/en/DeepSeek-V4-Flash.md#L79-L84); Native-Precision-Tutorial.md:267-268; Kimi-K2 INT4: 53/184/290 tok/s at 2K/8K/30K on one 4090 (Kimi doc:143-145) |
| llama.cpp | Ops with ≥32 tokens on host weights run on the first GPU that accepts them | 32 tokens (`GGML_OP_OFFLOAD_MIN_BATCH`) | See below |
| ik_llama.cpp | Offload the MoE op if tokens·k ≥ 32·E | 2,048 tokens for ds41 | [ggml-cuda.cu:5340-5369](https://github.com/ikawrakow/ik_llama.cpp/blob/248441295f5f508363a9cfd1bc4a6a3375b458c1/ggml/src/ggml-cuda.cu#L5340-L5369); default 32 at ggml/CMakeLists.txt:131 |
| MoE-Lightning | All prefill on the GPU, KV offloaded to the CPU | — | §4 |
| MoE-Gen | Experts always on the GPU, fed by accumulated batches | — | — |
| PowerInfer-2 | NPU, whole-layer reads from flash one layer ahead | — | — |
| DuoServe | Every expert through a 2-slot GPU ring | — | — |

llama.cpp details:
- Code (local `/home/as/llama.cpp`): `ggml/src/ggml-cuda/ggml-cuda.cu:5524-5543` (threshold check), `:5713` (env read); `ggml/src/ggml-backend.cpp:969-976` (picks the first GPU that accepts the op).
- Since #15346 it copies only the used experts (`ggml-backend.cpp:1701-1780`; 1.16–2.64× on gpt-oss: https://github.com/ggml-org/llama.cpp/pull/15346).
- In PR #13386 (https://github.com/ggml-org/llama.cpp/pull/13386):
  - Llama-4-Maverick, back when every expert was copied: 27 t/s with op offload vs 232 t/s with `--no-op-offload`.
  - DeepSeek-V3 Q2_K_XL on 4 GPUs at ubatch 1024 (a commenter's run): 89.5 t/s with op offload vs 26.3 t/s without.
- In PR #18535 (https://github.com/ggml-org/llama.cpp/pull/18535), Qwen3-Coder-30B with `--cpu-moe` at ubatch 4096: 2,321 t/s with op offload vs 147 t/s without.
- llama.cpp uses one GPU's PCIe link for all uploads; radiance's expert parallelism already uses all four.

## Not applicable to us

- **AMX/AVX-512 kernels and NUMA-aware TP:** this box has AVX2 only and one NUMA node (lscpu).
- **Expert deferral:** KTransformers §4.1 says it "is applied exclusively during the decode phase, not during the prefill phase", because in prefill it "nearly doubles memory access footprints". It is also lossy.
- **CPU attention and KV offload:** the card's compute does not set prefill time, and KV is 13.8 KB a token a card.
- **PowerInfer neuron sparsity:** it needs ReLU-sparse FFNs; ds41 uses SwiGLU.
- **Splitting each expert across the four cards (ktransformers TP):** it changes the down projection's K reduction order, so outputs are no longer byte-identical.
- **llama.cpp's 32-token whole-op upload:** at 32–64-token passes an expert has 1–2 rows, and the CPU is 3–4× cheaper than PCIe there.

## Lossy, opt-in only

- **HOBBIT low-precision copies for misses** (int4 for fp16 models, int2 for int8). It reports ≤1% accuracy loss, and prefill latency −14% to −85% against various baselines.
  - For ds41, a 2-bit drive tier would halve drive bytes: up to about ×1.8 on long prefill while the drive is the limit.
  - In prefill nearly every expert matters to some token, so a per-expert maximum-importance rule keeps most experts at 4-bit. Quality at 2-bit is unknown.
- **Skipping missed experts whose largest gate weight in the chunk is below τ.**
  - It helps short prompts most.
  - KTransformers' expert-skipping baseline lost 13.3% average accuracy when skipping 6 experts per token in decode, so only a tiny τ is plausible.

The other engines in the user's request (deepinfer, qfn radiance, dsv4engine, strata) were outside this source and are not covered here.

Sources: [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf), [LMSYS KTransformers blog](https://lmsys.org/blog/2025-10-22-KTransformers), [ktransformers @c987331](https://github.com/kvcache-ai/ktransformers/tree/c9873312643b615a717832946fba0346e69ab035), [llama.cpp #13386](https://github.com/ggml-org/llama.cpp/pull/13386), [#15346](https://github.com/ggml-org/llama.cpp/pull/15346), [#18535](https://github.com/ggml-org/llama.cpp/pull/18535), [llama.cpp MoE offload guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide), [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), [Fiddler](https://arxiv.org/html/2402.07033v3), [HybriMoE](https://arxiv.org/html/2504.05897v1), [ProMoE](https://arxiv.org/html/2410.22134v3), [HOBBIT](https://arxiv.org/html/2411.01433v2), [MoE-Lightning](https://arxiv.org/html/2411.11217v1), [MoE-Gen](https://arxiv.org/html/2503.09716v1), [Klotski](https://arxiv.org/html/2502.06888v1), [PowerInfer-2](https://arxiv.org/html/2406.06282), [DuoServe-MoE](https://arxiv.org/html/2509.07379v1), [vLLM RFC #63666](https://github.com/vllm-project/vllm/issues/63666), [vLLM UVA offloader](https://docs.vllm.ai/en/latest/api/vllm/model_executor/offloader/uva/), [SeqMoE](https://arxiv.org/html/2609.12978v1), [PreScope](https://arxiv.org/html/2509.23638v1), [MoE-Infinity](https://arxiv.org/html/2401.14361v3) (no prefill-specific mechanism), [Fate](https://arxiv.org/abs/2502.12224v1).

