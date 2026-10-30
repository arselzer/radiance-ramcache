# specdec: research

Speculation can pay on this box, but only after one fix: the drafter itself, not the verify rows, costs about 50 ms a step. With that cost down to about 7 ms, DSpark should give roughly +6–10% at a fixed depth and +12–15% with per-step depth control, on the held-out prompts. Everything below comes from a trace model calibrated on the measurements plus offline analysis; I used no GPU. Nothing has been built or measured on the card yet.

## Why DSpark lost: the verify rows are not the problem

I built a trace model (per layer: card work, then the slower of CPU pool and drive, with one pool and one drive shared by the four ranks). It reproduces four measured sequential configurations within about 1 ms a token: 71.8/71.0, 68.5/67.5, 68.7/68.8 and 65.5/65.6 ms. It also reproduces the DSpark window's own baselines: 77.6/77.8 ms without the bank and 80.1/79.7 ms with it loaded and drafting off.

On the DSpark arms (window ab-20261029T105844Z):

| Verify cap | Rows a pass | Tokens a pass | Modelled verify pass | Measured pass | Left over |
|---|---|---|---|---|---|
| 1 | 2 | 1.83 | 122.8 ms (1.53 steps) | 172.4 ms | 49.6 ms |
| 2 | 3 | 2.32 | 159.2 ms (1.99 steps) | 213.0 ms | 53.8 ms |
| 3 | 4 | 2.65 | 193.0 ms (2.41 steps) | 242.5 ms | 49.5 ms |

- **The working hypothesis is right about the verify pass.** It costs 1.5 / 2.0 / 2.4 single steps for 2 / 3 / 4 rows.
- **It is wrong about the draft.** What is left over is a constant ~50 ms a step, not ~17 ms. The old copy-path engine's sweep is consistent with the same ~50 ms.
- **Where the 50 ms probably goes:**
  - The draft head's experts are read from host memory over PCIe on every query pass: about 340–430 MB a rank (5 rows × top-3 × 3 stages).
  - About 40% of those bytes are wasted. Each stage's last local expert is re-read about 3 times for rows that belong to other ranks, because the live-tail kernel fix is off.
  - At the 19 GB/s a GEMV reaches on host memory (from the q64 kernel receipt), that is 18–23 ms, or more with this GEMV's narrow 4-byte loads. The draft's non-expert work is about 4 ms.
  - About 25 ms is still unattributed, which is why the first measurement below matters.
- **Misses:** drive misses do not amortize for real rows, as you suspected. Rejected rows cost only 0.64–0.74 of a real row's misses (decode misses rose +7 / +21 / +34% at caps 1 / 2 / 3), because a wrong draft routes mostly to the experts the real token at that position needs.
- **Depth choice alone cannot rescue today's drafter.** Picking each prompt's best arm with hindsight gives +0.5% against the base without the bank. If I swap the 50 ms for 7 ms in the measured pass costs, cap 1 gives +9.7%, cap 2 +6.1%, and the per-prompt best +15.5%.

**Exactness:** greedy verify reproduces single-row decode bit for bit, judging from the code.
- The dense matrix-vector kernel picks its split from the matrix shape only, for any row count up to 64.
- The expert kernel keeps the same lane and block order for every row.
- The CPU multi-row jobs are tested bit-identical to one-row jobs.
- The sparse attention keeps its exact kernels up to 48 rows ("so speculative verify reproduces decode").
- All DSpark replies were identical to the base engine's.
- Still to check: per-token logprobs, and the attention compressor's state after rejected rows.

## Ranked list

Gains are against today's 15.5 tok/s, on the 8 held-out prompts with DSpark's measured acceptance, and include the bank's VRAM cost.

1. **Move the draft experts off PCIe: "resident DSpark".**
   - Put the hottest 8 draft experts per stage per rank in VRAM, plus each stage's last local expert. The rest stay where they are. The draft-routing histograms (2 prompts) say the top 8 cover 73–100% of selections.
   - Draft cost about 7 ms, about 0.5 GB of VRAM a rank (59 fewer trunk experts with the bank's own weights).
   - Model: +8.8% at cap 2, +6–10% from the measured-cost swap. Exact, medium effort.
2. **Choose the verify depth per step from cost** (as in EVICT and Cascade): verify the number of drafts that maximizes expected tokens divided by (pass time + draft cost), using each sequence's running acceptance and the drafter's own confidence. Stop drafting while it does not pay.
   - On top of item 1: +12–15%.
   - With 15% lower acceptance (more prose): +2–6%, where a fixed depth loses.
   - Exact, small to medium effort.
3. **Turn on the existing live-tail fix (`RADIANCE_EXPERIMENTAL_DS41_LIVE_TAIL=1`).** No CPU-path run has ever had it on. On the old engine it gave decode 4.97 → 5.36 tok/s and prefill 60.1 → 40.9 s. It also removes the repeated tail work in verify passes, and extending it to the draft head's shape cuts about 45% of today's draft bytes. Free A/B; my rough estimate is +0–4% decode.
4. **Cold draft experts on the CPU pool** instead of VRAM: draft cost about 11 ms, +7.7% (+11% with item 2). Only worth it if VRAM is the binding constraint; it needs per-layer routing counts for the draft head on the CPU path.
5. **Draft head routed only to its resident experts** (the draft becomes approximate, the output stays exact): draft cost about 4 ms, and the bank's 6.7 GB of pinned host RAM is freed.
   - +7–13% if acceptance drops 5%; +4–8% if it drops 10%.
   - Adopt only if measured acceptance loss is 3% or less.
6. **Drop low-confidence draft rows at their first drive read** within the verify pass: +0–2 points over item 2 (noise level in the model). It makes cap 4 safe (−5% becomes +15%). Later.
7. **Reuse the block's leftover proposals** when all verified drafts and the target's extra token match, skipping a re-draft: +0.5–1.5%. Small.
8. **Read ahead the next token's experts from an approximate draft row**, run a step early with misses skipped: +4% to +46% in a what-if, before the draft's own cost, depending on how much of the next token's routing it predicts (0.5–0.9). This is a research bet that needs prediction coverage of about 0.6 or better.
9. **Not on this hardware:**
   - Tree speculation: a sibling branch buys about 0.08 tokens a pass for 5–10 ms.
   - Lookahead/Jacobi decoding: its extra rows each bring their own experts.
   - Layer-skipping self-speculation: every draft token pays the card's ~29 ms of fixed work.
   - Running the draft in the card's idle time: DSpark reads trunk layers 37–39 and the verify pass's last token, so there is only a 2–5 ms window.
   - DSpark's routing as a trunk predictor: its 128 experts a stage are a separate set.
   - Prompt lookup: it already gates itself correctly and verifies nothing on these prompts.

How the value moves with other changes (resident DSpark plus cost-aware depth):
- With a second NVMe: +16–20%.
- With 256 GB of RAM: +20–26%.
- After the FP8 dense weights land: +7–12%. Speculation competes with the card-side items for the same ~29 ms of fixed work a step.

## Most promising design: resident DSpark with cost-aware depth

Expected result is about 16.9 tok/s at a fixed depth and 17.5–17.8 tok/s with depth control, with byte-identical replies. Keep `--expert-plane-mirror-min-rows 7` so verify passes get the decode mirror share.

- **P0 (no code, one window):** measure what the draft costs (see below).
  - Go if the draft passes take 30 ms a step or more on the card, with their expert reads 20 ms or more.
  - Stop if they take 15 ms or less. The excess is then on the host side, so add stream events at pass boundaries before building anything.
- **P1 (about a day):** add `--draft-bank-expert-placement device`, putting all 96 draft experts a rank (1.68 GiB) in VRAM and relaxing `dspark_draft_control_scope`.
  - Window: base without the bank ×2, cap 1 and cap 2 ×2 each, one profiled arm.
  - Go if the draft takes 8 ms or less and decode gains ≥4% (model: +6–8%). Kill if the gain is under 1%.
- **P2 (about a day):** keep only the hot set in VRAM (top 4 or 8 per stage per rank, plus the last local expert), using a draft-phase expert profile. Go if it beats P1 by ≥1.5 points at a similar draft cost.
- **P3 (2–3 days):** cost-aware depth, with per-sequence acceptance tracking, the pass time by width measured online, and a stop-drafting rule.
  - First, log per-step depth, accepted count, pass time and the draft's top-2 logit margin, and replay the policy offline. Build it only if the replay shows ≥4 points over the best fixed depth.
  - Go: ≥10% on the held-out prompts, ≥0% on the 16 chat prompts, all replies identical, and per-token logprobs bit-identical on the 8 prompts.

## What to measure first (Window A, no code, about 25 minutes)

1. **Profiled cap-1 and drafting-off arms** with `--profile-ops` (the `o` part), `AB_REQUESTS=3`, in the DSpark window's configuration (90 GiB cache, 2048 MiB host pool, mirror min rows 7, bank placement `host`). Add up the per-step time of the draft passes (layers 40–42 and the head), including their six expert GEMVs. This decides P0.
2. **Live tail on vs off** on the recommended configuration, two alternations each, plus once with cap 1. `ab_window.sh` has no part for this switch, so it needs a part or environment pass-through. Adopt it if decode gains ≥1% with identical texts.
3. **A draft-routing capture** with `--expert-stats` during a cap-1 arm, on the 8 held-out and 16 chat prompts. This is the hot-set input for P2.

## Files

Everything is in `/home/as/.claude/jobs/635603c1/tmp/research2/spec/`:
- `specsim.py`, `specrun.py`: the model.
- `calib2.py`, `calib3.py`: calibration against the measured configurations.
- `fit1.py`, `fit2.py`: the rejected-row fit and the per-step excess.
- `swapcheck.py`: measured pass costs with the draft cost swapped.
- `final_table.py now|nvme2|ram256|fp8`: the design and scenario gains.
- `designs.py`, `designs2.py`, `now_policies.py`, `combo.py`, `gated.py`, `shadow.py`: the other policies and what-ifs.
- `arms.py`, `acc_per_req.json`: per-arm and per-prompt acceptance.

Sources:
- [Cascade (2506.20675)](https://arxiv.org/abs/2506.20675)
- [EVICT (2605.00342)](https://www.emergentmind.com/papers/2605.00342)
- [ECOSPEC (2607.12696)](https://www.alphaxiv.org/abs/2607.12696.md)
- [MoE-Spec (2602.16052), lossy](https://arxiv.org/abs/2602.16052)
- [DraftExpert (2607.24434)](https://arxiv.org/pdf/2607.24434)
- [SP-MoE (2510.10302)](https://arxiv.org/html/2510.10302v2)
- [MoE-SpeQ (2511.14102)](https://arxiv.org/html/2511.14102v1)
- [PEARL (2408.11850)](https://arxiv.org/abs/2408.11850)
- [SpecExec (2406.02532)](https://arxiv.org/abs/2406.02532)

# specdec: adversarial assessment

## Specdec track assessment (DSpark)

**Verdict:** The researcher has the direction right: the cost is in the drafter, not the verify rows, and most of it can be removed. But the explanation of where the time goes is incomplete in a way that changes the plan, and the gains are about half of what was claimed. Once the pass counts, the draft bank's RAM and the depth controller are corrected:
- Draft-expert placements that pin nothing in host RAM give **+3–7% at a fixed depth** on the 8 held-out prompts (16.0–16.6 tok/s).
- A depth controller with a real per-step signal might reach about +9%.
- With chat-like acceptance the result is between −4% and +1%.
- The researcher claimed +6–10% fixed and +12–15% with depth control (16.9 and 17.5–17.8 tok/s).

Everything below is CPU-only: I re-ran their scripts, read the code, and used the window logs and receipts.

### Findings

**1. The "unattributed 25 ms" comes from a kernel change made overnight (commit 02439b8f, 01:29).**
- Since that commit, the libr4d expert GEMV takes multi-row experts in passes of only as many rows as fit whole in LDS, re-reading the weights for each pass (`r4d_ds41_mxfp4.hip:381`).
  - The LDS budget in `mx_xbuf` (:630) fits one gate_up row at K=5120, so gate_up runs one row per pass. Down fits three rows at K=2304.
- `mx_range` gives the rank's last local expert every row routed to other ranks (:173–180). The live-tail fix that would stop this only accepts the trunk's shape, 96 experts and top-6 (`r4d_ds41_live_tail_policy.h:9`).
- So each rank's draft query pass (15 entries a stage) runs gate_up 15 times a stage, whether the entries are its own or dropped.
  - Over the 37-pass draft-routing histogram (`draftbytes.py`), that is **695 MB a rank a step on average and 715 MB on the slowest rank**.
  - The researcher's 340–430 MB is the pass structure from before the commit: 311 MB average, 378 MB slowest rank.
- At about 17.7 GB/s of zero-copy reads plus about 4.5 ms of non-expert work, this gives the measured ~45 ms. Three independent checks agree:
  - The old engine's head-tail matched pair, run before the commit, saved 13.5 ms a step net, about 19 ms gross. The old structure predicts about 20 ms.
  - The old cap sweep, with realistic concave row scaling, implies a draft of about 10–27 ms. The draft roughly doubled at 01:29, and no window measured DSpark between the two.
  - Rank 0's router waits are 114 ms a pass in the cap-1 arm against 51 ms with drafting off, so the excess is card time, not host time.
- **Consequence:** an exact kernel fix (live tail for the draft's 32-expert/top-3 shape, plus full passes for host-resident weights) takes the slowest rank from 715 MB to 260–310 MB a step. That puts the draft at about 19–23 ms with no VRAM or RAM cost.
  - It cannot reach 7 ms alone, because the expert split is lopsided: 1.7–5.2 local entries a pass per rank.
  - Any design that leaves cold experts in host memory needs this fix.

**2. The pass counts are wrong.**
- `arms.py` and `swapcheck.py` count passes as tokens minus accepted drafts. The trunk's routing-count rows (rows ÷ 40, minus 23 prefill chunks) give 269/213/187/172 verify passes, not 263/207/181/167.
- So each pass costs 168.6/207.0/234.7 ms, tokens per pass are 1.78/2.25/2.57, and the leftover per pass is **46/48/42 ms, not 50/54/50**.
- Re-running the swap with these counts (`swap3.py`), against the 90 GiB no-bank base at a 7 ms draft:

| | Researcher | Corrected |
|---|---|---|
| Cap 1 | +9.7% | +6.3% |
| Cap 2 | +6.1% | +3.7% |
| Per-prompt best (hindsight) | +15.5% | +12.4% |

**3. The bank's pinned RAM is not priced.**
- In the DSpark window the bank's 6.7 GB came out of the cache (90 GiB instead of 96). In that same window, the 90 GiB no-bank arm ran 3.7% slower than the 96 GiB one (77.8 against 74.9 ms a token).
- Keeping 96 GiB plus the bank leaves about 6 GB free, which is under the 8 GiB window guard.
- `final_table.py` keeps 1,424 cache slots for every design. At 1,328 slots, the researcher's own model gives the hot-8 resident design (cold experts pinned):

| | Researcher | With RAM priced |
|---|---|---|
| Best fixed depth | +8.8% | +2.4% to +2.9% |
| Cost-aware depth | +12.7% to +15.4% | +7.9% to +8.4% |

  (`designs_fix.py`, `running_policy.py`)
- This reverses the P1 → P2 order: placements that pin nothing in host RAM win.

**4. The cost-aware depth result assumes perfect information.**
- In `run_spec`, the policy sees the true per-step acceptance probability; the same draw decides acceptance.
- With only a running per-request acceptance estimate, the controller does *worse* than the best fixed depth in the same model. For P1 it gives +2.7%, or +5.0% if it skips the draft, against +6.7% fixed.
- The DSpark confidence head is not converted (`deepseek41.cpp:424`).

**5. The model breaks down near 5 rows.**
- The adaptive arm runs the same full 5-row draft, yet its leftover is only 18 ms (26 ms with the old pass counts), against 42–48 ms at caps 1–3.
- So the model overprices passes of about 5 rows by roughly 25 ms. Its claims for depth 3 and above (items 6 and 2) are unreliable.

**6. `LIVE_TAIL=1` does nothing for the draft.**
- It only accepts the trunk's shape. On the CPU path, the trunk's card GEMV is hidden behind the gather wait in all but about 4% of layers.
- Expect about 0 to +0.5% decode and perhaps 1–2% off TTFT.

**7. Exactness.**
- These claims check out in the code:
  - The dense split choice depends on shape only for up to 16 rows (`r4d_gemm_args.h`).
  - The GEMV keeps each output's order across row counts.
  - Sparse attention stays on the exact kernels up to 48 rows.
  - The multi-row CPU jobs are tested identical to one-row jobs.
  - The attention prefetch also covers verify passes (`deepseek41.cpp:2738`).
- Gaps:
  - Every DSpark identity check had 32–46-token prompts and 64-token replies, so contexts were at most about 110 tokens. That barely exercises compressed-KV and indexer attention, or the compressor and Engram state rollback after rejected rows.
  - Per-token logprobs are unchecked.
  - The FP8 dense kernel merged today (`54ff6dde`) has not run under DSpark.

### Verdicts

Gains are against today's 15.5 tok/s on the 8 held-out prompts.

| # | Proposal | Verdict | Claimed | Corrected |
|---|---|---|---|---|
| 1 | Resident DSpark (hot 8 + tail expert in VRAM, cold pinned) | Plausible, overpriced | +6–10% | +1–3% with the RAM priced. All 96 in VRAM with no pinned copy (P1): +3–7% |
| 2 | Cost-aware depth | Weak as modelled | +12–15% | 0 to +3 points over the best fixed depth with a real per-step signal; negative with running estimates. Keep the ≥4-point replay gate |
| 3 | `LIVE_TAIL=1` | Weak for the trunk; strong as a draft kernel change | +0–4% | Trunk ≈0–0.5%. The draft-shape extension (code) saves 20–25 ms a draft step |
| 4 | Cold draft experts on the CPU pool | Weak | +7.7% | +0–3% with RAM priced; large effort |
| 5 | Draft routed only to resident experts | Plausible, best on paper | +4–13% | +4–7% at top-16 if acceptance loss ≤2%. Top-8 masks about 25% of stage-40 top-1 picks: +1–4%. Needs valid dummy expert pointers or the tail fix |
| 6 | Drop low-confidence rows at their first miss | Weak | +0–2 | ≤+1; relies on the model's pricing at depth 3+ |
| 7 | Reuse leftover proposals | Weak | +0.5–1.5% | ~0 at a 7 ms draft: reused positions accept about 0.65 against 0.81 fresh |
| 8 | Read-ahead from an approximate draft row | Weak | +4–46% | Needs coverage ≥0.6; the t+2 row is right only about 0.5 of the time, and every read-ahead variant measured on this drive was neutral or negative |
| 9 | Not on this hardware | Agree | | |

How the value moves with other changes (best fixed depth, RAM priced):
- Second NVMe: +12–17% (claimed +16–20%).
- 256 GB RAM: +20–23% (claimed +20–26%).
- After FP8 dense weights: +3–8% (claimed +7–12%).

### Options the researcher overlooked
1. **Kernel fix for the draft's shape** ("K1"; exact, no VRAM or RAM, about a day): extend the live tail to the 32-expert/top-3 shape, and use full passes for host-resident weights. The existing 32-expert checker can verify it. It should take the cap-1 arm from 10.76 to about 12.2–12.5 tok/s.
2. **Prune and mask the bank.**
   - In the 2-prompt capture, 49, 76 and 95 of the 128 experts in stages 40, 41 and 42 were never selected.
   - Keep the top 16 per stage per rank (they cover 94–100% of selections): 8 in VRAM, 8 pinned (about 1.8 GB in all), and mask the rest.
   - Expected: draft about 7–8 ms, +4–7%.
3. **Short query block** (rows = cap + 1). This needs `rad_drafter_short_block` declared in the plugin and the `draft-short-from0` requirement lifted. Acceptance with a block shorter than the trained 5 is untested.
4. **Optional lossy acceptance** (what you asked about).
   - Accepting a rejected draft when the target's logit gap to it is under a threshold.
   - In the model, turning 30% of rejections into accepts takes P1 from +6.5% to +10.2%; 50% gives +20.7% (`lossy_accept.py`).
   - It breaks byte identity, and the literature reports distribution distortion.
   - First step: log the target's rank and logit gap for each rejected draft, and replay offline.
5. **Conflict with parallel sequences.** DSpark requires `max_seqs 1` (`dspark_draft_control_scope.h:10`), and batching amortizes the same ~29 ms of card work.

### Recommended order
1. **Window A, no code.**
   - (a) Profiled cap-1 and drafting-off arms. My prediction: draft gate_up GEMVs about 9–13 ms each, downs about 2–3 ms, draft about 40–48 ms a step.
     - Gate_up taking at least 3× the down time confirms finding 1; go to K1.
     - About 2:1 and roughly 20 ms in all means the researcher's structure is right; locate the missing ~25 ms first.
   - (b) **New:** cap 1 on the 16 chat prompts. Kill general-purpose DSpark if first-position acceptance is under 0.70.
   - (c) Draft-routing capture on all 24 prompts, as proposed.
   - (d) Optional: trunk live-tail A/B; expect ±0.5%.
2. **K1.** Go if the cap-1 arm gains ≥10%; stop if under 5%.
3. **One residency window:**
   - Arms: no bank at 96 GiB ×2, P1 at cap 1 ×2, pruned/masked top-16 at cap 1 ×2.
   - Prompts: the 8 held-out plus the 16 chat prompts.
   - Go: ≥+4% held-out and ≥0% chat, texts identical, acceptance loss ≤2%.
   - Kill: under +2%.
4. **Depth control** only if a replay with a real per-step signal shows ≥4 points.
5. **Before adopting anything:** identity checks with replies of 256+ tokens and prompts of 300+ tokens, plus logprobs, with the FP8 kernel included.

DSpark ranks below the FP8 dense weights and the piecewise read-ahead, and rises with a RAM or NVMe upgrade.

### Files
Everything is in `/home/as/.claude/jobs/635603c1/tmp/research2/spec-assess/`:
- `swap3.py`: the measured-cost swap with the corrected pass counts.
- `draftbytes.py`: zero-copy bytes per draft step under each kernel variant.
- `designs_fix.py`, `running_policy.py`, `policy_check.py`: the researcher's model with RAM priced and non-oracle depth policies.
- `lossy_accept.py`: relaxed acceptance in the model.
- `oldsweep.py`, `ttft.py`, `swapcheck2.py`: helpers.

Sources:
- [Cascade 2506.20675](https://arxiv.org/abs/2506.20675)
- [EVICT 2605.00342](https://arxiv.org/pdf/2605.00342)
- [Fuzzy speculative decoding 2502.20704](https://arxiv.org/html/2502.20704v4)
- [Revisiting lossy verification 2607.26627](https://arxiv.org/abs/2607.26627)
- [Relaxed speculative decoding study 2607.08690](https://arxiv.org/pdf/2607.08690)
- [Medusa 2401.10774](https://arxiv.org/abs/2401.10774)
