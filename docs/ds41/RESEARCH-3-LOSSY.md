# lossy: research

I've ranked eight families of lossy options below, with the first three to prototype. Speedups are model estimates for the held-out prompts. The model is a per-request fit, T = 24.7 ms + 2.54 ms × waited misses + 0.133 ms × CPU hits per token (both counts per token, summed over the ranks). The fixed-part fit and a constrained fit (31 / 2.28 / 0.108) give the bounds of each range.
- The fit uses the two measured arms of ab-20261029T124059Z: 80 requests, R² 0.998.
- Fitted on the plain-LRU arm alone, it predicts the packed + second-chance arm's decode time within 0.3–1.5%.
- My cache replay of the 40-prompt trace matches the engine's miss counts within 0.5% (rank 0: 12,054 replayed vs 12,098 measured).
- Current config, held-out prompts, per token over 4 ranks: 11.1 drive misses (4.6% of the 240 selections), 98 CPU hits (41%), 131 on-card selections (55%). That is about 66 ms a token.
- Measured per-request miss rates run 8–20 a token, so anything near zero misses is an extrapolation.

## Ranked list

**1. Miss-aware drop (a).** For a missed expert in decode, write a zero row, keep the other five weights as they are (no renormalization), and read and insert the expert in the background. Misses are spread evenly over layers (0.18–0.35 per layer per token).
- **Speed:** dropping all misses gives 39–43 ms, about 24–26 tok/s (+57–70%). Each dropped miss saves about 2.4 ms.
  - Keeping a missed expert when it is among the token's top 1 / 2 / 3: +43–52% / +32–38% / +22–26%. This assumes misses are spread evenly over ranks; the trace in step 1 below measures the real split.
  - A cap of 4 / 8 drops a token: +14–17% / +29–34%.
  - Background reads put the drive at about 75–80% busy.
- **Quality risk:** moderate and tunable. Literature:
  - Halving active experts while keeping the original top-k normalization cost 0.35 MMLU points on Qwen3.6-35B, against 4.65 with renormalization.
  - Top-1/2 experts are critical; lower ranks tolerate changes.
  - BuddyMoE reports that dropping missed experts hurts accuracy.
- **Effort:** S–M, host side only.
  - `hx_choose` emits zero rows into `hx_rows_` and queues the unit for a background read.
  - The rank needs no new copy: `sorted[i] % top_k` is the router's round order (`router_scored_topk` parks round j's pick on lane j).

**2. Cache-aware routing (g).** Add a bias λ·Δ to the selection score of VRAM- or RAM-held experts; the weights still come from the raw scores.
- **Speed:** about +25–55% if 50–90% of misses go away. It also cuts drive traffic.
- **Quality:** low risk in the literature: on DeepSeek-V2-Lite, misses fell from 28% to 3% with perplexity +0.1–3% and downstream accuracy change under 0.1%.
- **Effort:** S for a static prior toward VRAM residents (one load-time bias edit that every rank applies alike). M for a dynamic RAM-cache mask (a 1.9 KB copy to the card each token).

**3. Late ("deferred") miss outputs (f).** Read the miss asynchronously, compute it when it lands, and add weight × output into the rank's MoE output about 3 layers later, before that layer's all-reduce. This is the useful form of "parallelizing experts".
- **Speed:** about the same as dropping all misses, minus 1–2 ms a token for the late CPU computes.
- **Quality:** the first-order error of a drop is removed; layer L+1's routing and the stored KV still see the gap. I expect it to be clearly better than a drop, but it is untested.
- **Effort:** M. It needs a small add-row op before the all-reduce and one more copy of the routing weights to the host.

**4. Low-bit copies of missed experts on the NVMe (b).** Under 2–3 drop-equivalents a token (a drop counts 1).
- **Speed:** a 2.25-bit copy gives +22%; a 3.25-bit copy gives +10%. Reading the full copy in the background as well brings the drive to saturation, so expect about +15–20% realistically.
- **Blocker:** the 990 PRO has 20 GB free, and the copies need 129–186 GB. A second NVMe with exact split reads gives about +18–20% with no loss.
- **Measured error:** I requantized two real experts (L20, L35; round-to-nearest, per-32 scale searched for least squared error, Gaussian inputs). Expert output error² was 0.05–0.09 at 3.25 bits, 0.22–0.31 at 2.25 bits and 0.46 at 1.83 bits.

**5. Low-bit RAM tiers (also (b)).** Ruled out by the replays: to remove misses you have to make many hits lossy.

| Variant | Speed | Lossy uses a token | Drop-equivalents | Other cost |
|---|---|---|---|---|
| Whole cache at 3.25 bits | +20% | 102 | 5–9 | |
| 600 exact + 1,916 at 1.75 bits | +64% | 45 | about 22 | 23–38 requantizations a token, about 1 GB of extra memory traffic |
| Whole cache at 2.25 bits | +89% | 108 | 24–33 | |

**6. Top-k reduction or weight threshold (c).** Dropping one of six everywhere (original normalization kept) gives only about +9%, for 40 drops a token. Dropping only the 6th-ranked off-card selection gives about +12%. Both are dominated by item 1, because a dropped hit saves 0.13 ms and a dropped miss saves about 2.5 ms.

**7. FP8 (d).** Little to gain beyond the exact FP8-weights work already under way.
- W8A8 for dense GEMVs: 0 to +2%, and only if the exact W8A16 kernel ends up compute-bound. Quality risk is negligible: FP8 activations are DeepSeek's own reference numerics.
- FP8 LM head: about +1% (0.26 ms a token, plus 165 MB a card, about 9 more experts a rank).
- FP8 KV cache: about 0. KV reads are under 0.1 ms a token at these contexts (4 KV source layers, sparse attention).
- 4-bit dense weights: +4–7%, at high risk.

**8. Near zero on this box.**
- (e) CPU FMA or reassociation: at most 1%. The pool already reads at 94 of about 100–103 GB/s; only TTFT might gain.
- (f) Relaxed ordering: 0.
- (f) Splitting an expert's rows between CPU and card: exact, but 0 gain, because the drive or the RAM bandwidth is the limit.
- (f) Replicating hot experts across ranks: 0. Misses per rank are already within ±1.2%, and the drive and the CPU pool are shared.
- (f) Deferring CPU hits as well as misses, as an extreme: +115–165%, but about 109 of 240 selections a token would arrive late, so the risk is high. Try it only after item 3 works.

## Prototype first

**Step 1: a routing-score trace arm.** Exact, S, about one day.
- Record per decode token and layer: the top-6 slots and weights, the 384 router logits (rank 0), and whether each pick was on the card, in RAM or missed. That is about 61 KB a token.
- Offline it gives the rank and weight of misses, and replays drop, threshold, top-k and cache-aware routing.
- Gates: build the protect-top-2 drop if 60% or more of misses sit at slot 2 or below. Build cache-aware routing if λ ≤ 0.5Δ removes at least half the misses while changing at most 3% of all selections.

**Step 2: miss-aware drop, plus a logprob A/B.**
- Arms: exact; drop all; drop slot ≥ 2; drop slot ≥ 4. Each on the 40 validation prompts with `logprobs: 20`, then a teacher-forced re-sync.
- That is about 35–40 minutes of GPU window.
- Go: mean KL ≤ 0.03 (≤ 0.01 is excellent), top-1 agreement ≥ 97%, no degenerate 256-token replies, at +25% or more.
- Kill: drop slot ≥ 4 already exceeds KL 0.03.

**Step 3, picked from steps 1 and 2:**
- Cache-aware routing (static residency prior first) if the trace shows many near-ties at the top-6 boundary.
- Late miss outputs if dropping is fast but its KL is too high.
- Neither, if protect-top-2 already gives KL ≤ 0.01.

## Quality harness

`lossy_ab.py` is ready and was tested offline with a fake server; it does not touch the repo.
- `run` saves token ids plus the top-20 alternatives per token.
- `resync` drives the decode path along the exact engine's reply: at each divergence it appends the reference token and continues, and flags the rows that come from a prefill pass.
- `compare` reports identical replies, first divergence, KL over the top-20 (it matched full-vocabulary KL within 2%), top-1 agreement and the change in logprob of the reference token.
- The ab_window.sh integration (an arm part for the drop switch, plus a re-sync call for lossy arms) still has to be wired in.

## Interactions with the hardware options

The value of item 1 holds after the FP8 residents (still about +65%). It shrinks to about +40% with a second NVMe and to about +5–10% with 256 GB of RAM: it is a software stand-in for those upgrades. With misses dropped, a 2-row MTP verify pass pays off only if the draft costs about 8 ms or less (measured 17 ms); that is for the speculative-decoding track.

## Sources
- [Cache-conditional experts](https://arxiv.org/abs/2412.00099)
- [Training-free halving of active experts](https://arxiv.org/abs/2609.04575)
- [Certain head, uncertain tail](https://arxiv.org/abs/2602.02443)
- [BuddyMoE](https://arxiv.org/abs/2511.10054)
- [HOBBIT](https://arxiv.org/abs/2411.01433)
- [AdapMoE](https://arxiv.org/abs/2408.10284)
- [Lu et al., expert skipping](https://arxiv.org/abs/2402.14800)
- [REAP](https://arxiv.org/abs/2510.13999)
- [DAOP](https://arxiv.org/abs/2501.10375)
- [ScMoE](https://arxiv.org/abs/2404.05019)
- [SwapMoE](https://arxiv.org/abs/2308.15030)
- [MoE offloading with mixed quantization](https://arxiv.org/abs/2312.17238)
- [Accuracy is not all you need](https://arxiv.org/abs/2407.09141)
- [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)
- [llama.cpp KL-divergence tool](https://www.mintlify.com/ggml-org/llama.cpp/api/tools/llama-perplexity)
- [DeepSeek-V4-Flash quantization KL table](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)
- [Unsloth 1.58-bit DeepSeek-R1](https://unsloth.ai/blog/deepseekr1-dynamic)

Files are in /home/as/.claude/jobs/635603c1/tmp/research2/lossy/:
- lossy_ab.py
- test_lossy_ab.py
- rp.py
- validate.py
- feats.py
- fit.py
- fit2.py
- policies.py
- budget.py
- perlayer.py
- firsttok.py
- tiers.py
- alllow.py
- qerr.py

# lossy: adversarial assessment

**Lossy track: adversarial assessment**

**Bottom line**
- **Drop-on-miss is real but over-stated.** It is plausibly worth about +45–70% (central ~+60%), not a firm +57–70%. The fixed part of the time model is an extrapolation, and so far nothing says the quality cost is small.
- **Late miss outputs (f) are about half as good as claimed.** Expect +30–45%, not "about drop-all speed".
- **Two quality-harness flaws and three literature misreadings need fixing before any gate means anything.**
- **FP8 math, low-bit tiers and top-k reduction stay weak.** I agree with the researcher on those.
- **The baseline has moved.** Window ab-20261029T135754Z just measured FP8 dense weights at 16.97 / 16.94 tok/s against 15.11 / 14.98 (+12.6%, replies identical). Rank 0's decode drive layers fell 1278 → 1090. Every lossy gain below is about the same in relative terms on that baseline. Drop-all would give about 24–27.5 tok/s there.

## What I re-checked

**Replay.** `validate.py` reproduces the engine: rank 0 replays 12,054 misses against 12,098 measured, and the others are within 0.3%.

**Time model.** The refit gives the same coefficients (24.74 ms + 2.536 ms a miss + 0.133 ms a hit).
- The quoted R² 0.998 is on per-request totals, so request length inflates it. Per token it is 0.989, which is still good.
- Leave-one-arm-out prediction is off by only +0.33% / −0.11%.
- Splitting hits by whether their layer had a miss gives 0.160 ms (layers without a miss) against 0.053 ms (layers with one). Hits are hidden behind miss reads, as expected. Even so, the drop-all prediction stays at 37.6–38.1 ms across five model forms, plus 1.1–2.2 ms for the background inserts.
- The weak point is the extrapolation. Every one of the 80 requests has at least 8.3 misses a token, so about 21 ms of the predicted saving lies outside the data.
- A bottom-up split of rank 0's decode token (Sc2 arm) points to a floor of roughly 36–47 ms:
  - 42.6 ms waiting at the routers, of which about 17–20 ms is other ranks' misses arriving through the all-reduce;
  - about 9–13 ms waiting at gathers for CPU rows;
  - about 1.6 ms for host decisions.
- Corrected drop-all: 38–45 ms a token, +45–70%.

**Post-stop decode pass (new finding).** Every naturally stopping request runs one extra decode pass, which appears to consume the EOS token. chat-10, which stopped at the length limit, also has one.
- On EOS that pass sends 204–207 of its 240 selections off the card and misses 103–105 experts when the cache is cold (heldout-05 r1–r3, chat-03). That is about 0.26 s.
- That time is included in the measured decode time: the model residuals for those requests are only −1 to −46 ms.
- Without these passes, held-out misses are 10.94 a token (not 11.14), and the worst token has 55 misses (not 105).

**Heavy tail.** About 12% of held-out decode tokens have 20 or more misses (8% or more of their selections), and the worst has 55.
- Out-of-profile content misses much more: SQL, Romeo and Juliet and the prime-number list run 15–16.6 misses a token, against 8–10 for in-profile prose.
- Drop-all therefore concentrates its damage on exactly the content the 40 validation prompts under-represent.

**Literature.**
- **Halving paper ([arXiv 2609.04575](https://arxiv.org/abs/2609.04575), Table 5).** The 0.35 MMLU point needs k2=16, which scales the surviving experts down further. Halving with the original top-k normalisation cost **−3.10** points on Qwen3.6-35B (p = 6×10⁻⁸), and −0.55 (not significant) on the 397B model.
  - The closest analogue to dropping without renormalisation is dropping the two lowest-ranked of eight: −1.10 MMLU, ±0 GSM8K.
  - Those were the least important experts. Cache misses fall at any rank.
  - The same paper finds expert identity matters about six times more than expert weighting.
- **Cache-Prior ([2412.00099](https://arxiv.org/html/2412.00099v2)).** DeepSeek-V2-Lite misses fell 28% → **7%** (not 3%) at λ=0.5, with the **top-2 always kept (J=2)** and perplexity +0.1%. λ=0.8 over-reuses cached experts.
- **BuddyMoE ([2511.10054](https://arxiv.org/html/2511.10054)).** It evaluates no drop baseline; that claim is only in its abstract. Its own buddy substitution cost about 4 ARC points (0.735 → 0.695) at a 75% cache.
- **Calibration point ([AtomicChat table](https://huggingface.co/datasets/AtomicChat/dsv4-eval-artifacts)).** On DeepSeek-V4-Flash, Q4_K_M and MXFP4 quants show mean KLD 0.156 and 87.5% top-1 agreement on wikitext. Absolute thresholds of KL 0.03 / top-1 97% are therefore uncalibrated for greedy chat text.

## Verdict per proposal

**1. Drop-on-miss (a): plausible for speed, quality unproven.**
- Drop all misses: +45–70% (central ~+60%).
- Protecting top-1 / top-2 / top-3 misses:
  - with background reads that yield to demand reads: +42–55 / +29–39 / +18–27%;
  - with plain FIFO reads: +30–39 / +19–27 / +12–19%. A waited miss queues behind the dropped misses' background reads.
- Today `NativeReadWorkers::run` holds `call_mutex_` for a whole read, so background reads must take a separate, yielding path.
- A drop cap of 4 or 8 a token (+14–17% / +29–34%) carries the same caveat.
- The drive is 64–70% busy under drop-all, not 75–80%.
- `sorted[i] % top_k` gives the rank by biased key (score + correction bias), not by weight. A weight threshold needs the six weights copied to the host (24 bytes a layer).
- Effort is M, not S–M: the yielding reader, the determinism rule (risk 1 below) and the stats.

**2. Cache-aware routing (g): plausible, and possibly the best quality per removed miss.**
- Expected gain +10–50%. The removable share of misses is unmeasured.
- Selections only ever change where a miss would have occurred, which is a point in its favour.
- Add the J=2 guarantee.
- The gate "λ ≤ 0.5Δ" is not a near-tie bias. Δ is the full max–min range, so with 384 experts it would make slots 3–6 essentially cache-only. Gate on the substitute's score loss instead.
- The dynamic mask must be one per-token snapshot shared by all ranks. Otherwise the ranks disagree on the top-6 and on its normalisation.
- A static prior applied at load also changes prefill and the KV cache.
- Effort: S for the static prior, M for the dynamic mask.

**3. Late miss outputs (f): weak to plausible.**
- My single-drive timeline model is calibrated: with no deferral it reproduces 65–68 ms against 66.1 measured.
- Deferring 3 layers still stalls 8–9 ms a token (token 47–50 ms, +32–40%). Deferring 6 layers gives 45–48 ms (+38–47%).
- Deferring to the end of the token still stalls 5–6 ms, because reads must land before the LM head.
- The no-stall hybrid (defer, then drop at the deadline) lands only 20–27% of misses in time at 3 layers, and 35–44% at 6.
- Effort: M+.

**4. Low-bit copies of missed experts on the NVMe (b): weak.** It is blocked by the 20 GB free (129–186 GB needed). If the space existed it would give +15–22%.

**5. Low-bit RAM tiers: refuted.** I agree: the re-run replays (`alllow.py`) need 24–33 drop-equivalents a token for +89%.

**6. Top-k reduction or weight threshold on all selections (c): weak.** +9–12%, dominated by item 1.

**7. FP8 math (d): weak, at most +1–3% in total.** Decode GEMVs are bandwidth- or latency-bound, and the exact FP8 storage has already taken the bandwidth half.

**8. "Defer CPU hits as well, +115–165%": refuted as stated.** NVMe time alone is about 27 ms a token at 11 misses, which already exceeds the 24.7 ms floor that +165% assumes. Expect at most about +60–80%, with about 45% of selections arriving late.

**Quality harness `lossy_ab.py`: right idea, biased low.**
- Each re-sync re-prefills the prefix exactly. That resets the KV the drops contaminated and re-warms the cache, so KL is understated.

## Risks the report missed
1. **Replies become timing-dependent.** A unit whose insert is still in flight decides drop or compute, and eviction order depends on when inserts finish. Reserve the slot at the miss decision and treat reserved-but-unpublished units as hits that wait. The replay then matches the engine and replies are reproducible for a given request order.
2. **Damage concentration.** The heavy tail and the out-of-profile content above.
3. **Multi-turn sessions.** Where the KV cache is reused across turns, the next prompt inherits the drop-contaminated KV.
4. **Batching.** With concurrent sequences, misses amortise across rows, so a drop's value shrinks while it degrades several sequences at once.

## Options the report overlooked
- **Weight-threshold drop.** Drop a miss only if its weight is below τ, with a per-token cap.
- **Defer-then-drop hybrid.** As fast as drop-all, with part of the misses delivered late.
- **More background bytes over the mirror.** Read 25–30% of background bytes from the SATA mirror to keep NVMe headroom for protected misses and prefill.
- **Lossy prefill (TTFT).** Drop or route around prefill misses. About 56% of prefill wall time is spent waiting for miss reads. Higher risk, because every later token attends to that KV.
- **Exact side item: skip the post-stop pass.** Skip its misses and inserts when its output and KV are discarded, saving about 0.1–0.26 s per natural stop. First confirm that no session reuses the EOS KV.

## Recommended order, with first experiments
1. **Exact, now.**
   - Merge the FP8 dense weights (measured).
   - Skip the post-stop pass. First a stats count, no GPU; then replies must be identical.
2. **Routing trace arm (the researcher's step 1), extended. Exact, about 1 day plus about 10 minutes of window.**
   - Record the six weights, the keys, rank 0's 384 logits, hit or miss per pick, and the post-stop flag.
   - Offline outputs: dropped weight mass per miss, and the share of misses with a cached alternative within δ of the 6th key.
   - Gates:
     - **weight-threshold drop:** go if 50% or more of misses carry weight below 0.1;
     - **(g):** go if 40% or more of misses have a substitute within δ = 5%, kill below 20%.
3. **Harness fix, no GPU.**
   - Teacher-force through decode passes with a grammar that forces the reference text. The server accepts `grammar`, and logprobs are read from the raw logits (`sampler.cpp:1128`).
   - Fall back to a small forced-continuation flag if tokenisation drifts.
   - Validate on the exact arm: identical tokens and KL = 0.
   - Report KL by prompt family, and add out-of-profile prompts and 256-token replies.
4. **Drop prototype (M), one window: control, drop-all, τ = 0.1, cap 6, each with speed and forced-decode KL.**
   - Kill if drop-all measures below +35%; the extrapolation failed, so re-plan.
   - Go for a variant at +30% or more if its KL is no worse than a "top-5 everywhere" yardstick arm (one-flag router change) and top-1 agreement is 98% or more.
   - Then run a paired downstream check on about 100 problems.
5. **Then (g)** if gate 2 passes: start with a static-prior arm, then the dynamic mask with J=2.
6. **Hybrid (f) only if** drop fails on quality at useful speeds.

My scripts are in `/home/as/.claude/jobs/635603c1/tmp/research2/lossy-assess/`: `feats2.py`, `fit3.py`, `resid.py`, `tail.py`, `eos.py`, `perreq.py`, `defer_sim.py`, `defer_sim2.py`, `protect_sim.py`. All are single-thread and ran after the window had cleared.
