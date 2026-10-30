# Further optimizations: five lenses, each adversarially verified

## Lens: critical-path

### Verified

**Critical-path lens, adversarial verification.** I checked the code at dd0c60de (snapshot: `/home/as/.claude/jobs/635603c1/tmp/moreopt/critpath-verify/stager_dd0c60de.cpp`; `host_expert_pool.cpp` and `readahead.cpp` are unchanged at HEAD). I also checked the counters in the 161035Z, 150322Z, 142506Z, 100728Z, 085239Z and 083552Z windows.

**Verdict:** five candidates survive (two revised, one with a corrected mechanism) and one is dropped. Every survivor is exact. The corrected total is about **+3% central (+2 to +5%)**, not +5–6%. That is about 17.8 tok/s on the finder's 17.25 basis, or about 18.1 from 17.6. Items 1, 3, 4 and 5 all shorten the same ~12–14 ms of pool time a token, so their gains do not fully add up.

### 1. Pool wake (candidate 2): KEEP, revised. +1 to +3% net (central +1.5%); the lock-free wake alone is +0.3 to +1.5%
- **New:** yes. MWAITX and the long spins kept the workers runnable; this does not.
- **Code confirmed:** `wake()` holds `idle_mu_` through `notify_all` (`host_expert_pool.cpp:252-258`), and each woken worker takes it again (`:402-405`). The 23–27 µs submit is that broadcast (2.9 µs under MWAITX).
- **Evidence that awake workers make rows sooner:**
  - MWAITX (100728Z): CPU-row waits fell 46–55 µs per layer with CPU work, and the submit fell 26 µs.
  - s2500 (085239Z): waits fell 28–67 µs, the submit 27 µs.
  - So rows arrived about 54–93 µs sooner per burst. This is an upper bound: both arms delayed the rank threads, which shortens their measured waits.
- **Correction:** sysfs gives C2 a 30 µs exit latency, and the workers idle about 1 ms between bursts, so they sit in C2.
  - A futex wake without the lock removes the lock hold and the reacquire convoy, but not the C2 exit or the wake-up loop.
  - This also explains why C2-off measured neutral: workers that wake fast queue up on the mutex the waker still holds.
  - Lock-free wake alone: 5–25 µs × about 40 critical bursts = 0.2–1.0 ms a token.
  - Adding the router-event pre-wake of about 24 workers: 25–50 µs a burst, 1.0–2.0 ms gross. Its awake cost (24 × ~50 µs a layer, about 2% of MWAITX's extra awake time) is −0.1 to −0.5%.
- **Fix in the test plan:** with the futex wake alone the submit stays about 20 µs, because the kernel's wake loop remains. Watch the time from submit to the first tile, not the submit.
- **Drop part 3**, the wake when the mirror leg lands. The mirror lands only 0.18–0.25 ms before the NVMe leg, about the workers' whole 200 µs spin. The spinning workers would hold CPUs exactly when the 7 kept readers need to wake on I/O completion, which is the MWAITX hazard.
- **Drop the s400/s600 probes.** They repeat the s2500 and MWAITX experiments.
- **Test:** B, B-Wf, B-Wf-Wp24, three interleaved rounds, with counters for submit to first tile, submit to the 28th worker running, and pre-woken workers that slept again. Keep it if CPU-row waits fall by 20 µs or more per layer with CPU work and router waits stay flat.
- **Optional, needs your decision (system-wide setting):** B-Wf-Qc (C2 off). This is new evidence against the +0.6% power verdict, because the convoy would have hidden C2-off's gain.

### 2. Urgent read-ahead finish (candidate 1): KEEP with a corrected mechanism. +0.5 to +1.4% (central +0.9%)
- **The premise is wrong:** queue depth is not the limit. piecebench measured 4 × 512 KiB pieces in flight at 6.77 GB/s on a quiet box, faster than a demand read (~6.2).
- **The gap is still real:**
  - In the engine, urgent pieces read the remaining 12.45 MB in 2.53 ms (4.9 GB/s).
  - One-unit demand reads read 16.36 MB in 2.75 ms (5.95 GB/s).
  - Both face similar contention: the chance that another rank reads the drive in the same layer is about 0.17 for each, and demand reads also meet pieces already in flight in 2,562 of 6,856 cases.
  - So the piece path loses about 28% of its quiet-box speed and the demand path about 4%.
- **Likely cause: per-piece CPU round trips.**
  - Each unit needs about 20 reader wake-ups.
  - Each wake-up pays the C2 exit or lands on CPUs held by the burst or the 200 µs spin.
  - After every piece the waiting rank thread is woken by `done_cv_.notify_all` under `mu_`.
- **Saving:** reading the rest at the demand rate takes 12.45 / 6.0 + 0.07 = 2.15 ms against 2.53 ms. That is 0.38 ms × 2.43 units × 0.85 ≈ 0.8 ms a token, or about half of that if the variant falls short.
- **Preferred form:** one read per extent tail on the kept readers, since the 4-pieces-a-read form still needs about 6 round trips.
- **Test:** first the no-code diagnostics B-Pq8 and B-Pb2048, judging only the "finished while the layer waited … each" counter (expect it to fall toward ~2.1 ms). Then B against B-Pu, three rounds, after the verify arm.

### 3. Sliced inserts (candidate 4): KEEP. +0.2 to +1.0% (central +0.5%)
- **Status of the idea:** the ranked plan (#7) already specified slices with a cap. Commit e8560766 built a single wait before the whole insert (`stager.cpp:2613`) instead. Iy3000 measured about +0.2% on its clean rounds, which fits removing only the first burst an insert overlaps.
- **Calibration from the direct-insert change (100728Z):**
  - 75.43 against 76.24 ms a token, for 36–54 MB less traffic per insert at about 13.5 inserts a token.
  - That is at most 1.1–1.65 ms per GB. It is an upper bound because it includes the writer-lock memcpy.
- **Today:** 9.63 inserts × 37–55 MB = 356–528 MB a token, worth at most 0.4–0.87 ms. Slicing removes about 60–80% of that.
- **Also add** the plan's non-temporal stores, which save about 18 MB of read-for-ownership traffic per insert.
- **Pitfall the finder missed:** `tile()` calls `drained()` before `t->left.fetch_sub` (`host_expert_pool.cpp:346-352`). Once inserts wait on `quiet_cv_`, every burst's last job pays a lock plus `notify_all` before its ticket completes, on the critical path. This also affected Iy3000. Decrement the ticket first.
- **Test:** B against B-Is1024, three rounds. Misses a token should stay the same and CPU-row waits should fall. Add a counter for insert time that overlaps queued pool work.

### 4. Predictor off the burst (candidate 3): REVISE down. +0.2 to +0.6% only with parallel or pool-run scoring; moving the call alone ≈ 0 (−0.3 to +0.3%)
- **Facts check:** rank 0's choose is 8.0–8.9 µs against 4.0–4.5 µs on the other ranks, and its gather-wait swing is real.
- **But it is FIFO position, not critical path.** The post (an allocation plus `notify_one`, about 4.4 µs) puts rank 0's jobs behind about 250 µs of other ranks' work.
- **Bound on the removable contention:** in 142506Z, the predictor and the read-ahead together raised CPU-row waits by only about 0.2 ms a token per rank.
- **Lead time:** posting in `hx_finish` delays the prediction by the burst (about 0.4 ms, and about 3 ms in rank-0 miss layers).
  - Pr4 restores the arrival time, but its 4 threads land in the workers' spin window.
  - **Better variant:** score L+2 as lowest-priority pool jobs right after the burst. That is 3.93 MB at about 90 GB/s, roughly 45 µs on workers that are already awake.
- **Test:** B against B-Pr4 (or the pool-scored variant), three rounds. Check µs per prediction, rank 0's choose at about 4.3 µs, and that read-ahead MB per used unit does not fall.

### 5. Interleaved row streams (candidate 5): KEEP, gated on a bench. +0.3 to +0.9% if the bench shows 3% or more, otherwise 0 (expected about +0.3%)
- **Support:** tiles held in L3 run at about 130 GB/s with 40 workers (IMPL-L3 review), so the kernel has compute headroom left.
- **Caveat:** down tiles are about 74 KB, so a 4-way interleave gives streams of about 18 KB, short for the L2 prefetcher. Try 2 streams on gate_up tiles first. The finder's 8% top end equals the synthetic benchmark's ceiling; 3–5% is realistic.
- **Test:** kernel identity tests and the pool-shaped bench under the build lock, then B against B-Hs2, three rounds, only if the bench shows 3% or more.

### Dropped
- **Gate the handoff (candidate 6): at most +0.2%, for M–L effort.**
  - After the ticket completes, the host's exposed gap is about 2–5 µs: lease releases (about 0.1 µs a weight under the per-rank cache mutex) plus the first copy's doorbell. Later packets arrive while the card runs the copies.
  - The card takes about 1–3 µs to see a gate signal in host memory.
  - That leaves at most about 3 µs × 40 layers ≈ 0.12 ms a token.
  - Fl, the same kind of host-issue saving, was neutral.
  - The proposed build criterion measures the wrong gap: it looks before the counts copy, not between the ticket and the first CPU-row copy.
  - Holding leases until the next submit risks the leased-tail insert refusals (bypasses) seen at 01:32.

**Free probe (no code, exact):** one B-s0 arm (pool spin 0). It was neutral before read-ahead in pieces, but the 200 µs spin after each burst now overlaps every urgent finish and miss read.

### Finder's report

The best decode arm's token splits into four parts. About 43% is drive misses. About 31% is the cards' attention-and-router chain, which mostly belongs to the cards lens. About 21% is the CPU pool's compute in layers without a miss, and the per-token tail is about 3%. I found six exact levers on this path that are not on the excluded list; together they are worth about +5–6%, or roughly 18.2–18.4 tok/s. None needs a user decision, and none is lossy.

Data: rounds 2–3 of the best arm (`…-F8-Pp4-Ei-Pm56-Pe`) in `ab-20261029T161035Z`, 960 decode tokens, 57.97 ms a token (17.25 tok/s). The code references are at main `dd0c60de`. The tree moved during this review, but the paths cited here did not change.

## Decomposition

**Measured, per rank (ms a decode token):**

| Part | r0 | r1 | r2 | r3 |
|---|---|---|---|---|
| Router wait, one-token passes (µs a layer) | 32.80 (818) | 34.06 (850) | 33.83 (844) | 34.76 (867) |
| Decision, total (choose / submit, µs a layer) | 1.42 (8.7 / 26.6) | 1.17 (4.3 / 24.6) | 1.19 (4.3 / 25.2) | 1.11 (4.2 / 23.4) |
| New miss reads on the issuing thread | 4.95 | 4.98 | 5.04 | 5.09 |
| Read-ahead units still being read when their layer asks (units a token × ms each) | 1.51 (0.58 × 2.59) | 1.46 (0.61 × 2.41) | 1.57 (0.64 × 2.45) | 1.62 (0.60 × 2.69) |
| Waits for CPU rows, all passes (decode share, estimated*) | 13.52 (~8.5) | 12.23 (~7.1) | 12.48 (~7.3) | 10.69 (~5.5) |
| What is left: host issue and per-token tail | ~8.8 | ~9.2 | ~9.0 | ~9.9 |

\*Prefill's share is estimated from the long-prompt arm: 47.4 s over 15,760 prefill layers is ~3 ms a layer, so ~2.5 s of the 920 prefill layers here.

- **One missed expert:** the mirror leg takes 2.54–2.59 ms and the NVMe leg 2.72–2.77 ms, for about 16.4 MB from the NVMe at ~6.1 GB/s. The issuing thread is back after 2.77–2.81 ms.
- **Where the 26.6 µs submit goes:** almost all of it is waking ~40 sleeping workers. `wake()` notifies them while holding `idle_mu_` (`host_expert_pool.cpp:252-258`), and each worker takes that same mutex again when it wakes (`:402-405`).

**Critical path a token, worked out from the counters above:**
- **Miss layers, ~43%.** There are about 8.5 distinct layers a token with a drive read: 9.3 rank-layers, about 8% of them coinciding. Each costs about 3.0 ms: 2.73 ms of read, plus about 0.25 ms to wake the pool and compute the missed expert, plus the handoff. That is about 25 ms. Summed over ranks the read waits are 26.2 ms a token.
- **Pool compute in the 31.5 layers without a miss, ~21%.** Each costs about 0.38 ms: the decision, plus a burst of ~1.8 cached experts × 18.0 MB at ~94 GB/s (0.34 ms), plus the handoff. That is about 12 ms.
- **Card chain, ~31%, by difference.** About 40 × 0.45 ms, or 18 ms: gather, MoE all-reduce, attention, attention all-reduce, router, topk/scatter, four device-to-host copies, interrupt.
- **Per-token tail:** about 1.5–2 ms.
- **Skew between ranks:** in layers without a miss, r0 finishes last in the pool's first-in-first-out order. Its predictor post comes before its choice, so r0's choose takes 8.7 µs against about 4.3 µs on the other ranks. r3 waits 49 µs a layer longer at the router than r0. In miss layers, each waiting rank absorbs about 2.5 ms in its next router wait.

## Candidates, most valuable first (all exact)

**1. Finish a partly read read-ahead unit at full drive depth: about +1.7% (range +1 to +2%).**
- **Mechanism:** when a layer takes a unit the read-ahead only partly read, stop new pieces and let the ones in flight land. Then read each extent's unread tail as one large read on the rank's kept plane readers, in the same call as any new misses of that layer. A simpler variant stays inside PieceReader: an urgent unit takes four consecutive pieces a read, with up to 8 readers, while non-urgent pieces stay at 4 or fewer in flight. The same bytes land at the same offsets.
- **Arithmetic:**
  - There are 2.43 such units a token, each waiting 2.41–2.69 ms (mean 2.54).
  - 6.3–6.7 MB of each unit is already in when its layer asks, so about 12.4 MB is left. About 10.1 MB of that is read as urgent pieces: 23,009 urgent pieces ÷ 1,165 units = 19.8 pieces of 512 KiB each, one read call a piece, 0.392 ms a piece.
  - That is about 4.9 GB/s, against 6.1 GB/s for a demand read. At 6.1 GB/s the remainder takes about 2.05 ms, saving about 0.47 ms a unit.
  - 2.43 × 0.47 × 0.85 (layers where no other rank reads the drive) ≈ 1.0 ms a token.
- **Effort:** S for the PieceReader variant, S–M for the kept readers. **Risk:** low.
- **Test:**
  - Arms: B and B-Pu, three interleaved rounds. Pu is a new part, for example `..._PREFETCH_URGENT=deep`; also try B-Pu-Pn2 (non-urgent pieces at 2 or fewer in flight). Run a verify arm `verify-f-M1-H-R-Rd13-Pp4-Pu` first.
  - Probe with no code: one B-Pq8 arm. It raises the depth for urgent and non-urgent pieces together, and 2,560 of 6,856 demand reads already meet pieces in flight.
  - Expect "finished while the layer waited … each" to fall from about 2.5 to about 2.05 ms.
- **Why it is not covered:** Pp4 was adopted as a whole, but its finishing path was never tuned. No Pq or Pb arm has ever run, and Pf0 re-reads the whole unit, which is slower.

**2. Take the pool's wake-up off the critical path: about +1.5% (range +1 to +2.5%).**
- **Mechanism:** three parts.
  - Workers sleep on a futex word, and `wake()` issues the futex wake without holding any lock.
  - One rank records an extra interrupt event right before each decode layer's router GEMV (the op reading `blk.L.glm_router.weight`, found at init). At the gather, its thread waits for that event, wakes about 24 workers, then waits for the counts.
  - In a miss layer, the issuing thread wakes the pool when its own read (the mirror tail, which lands first in about 78% of calls) returns while NVMe reads are still in flight. For a read-ahead unit being finished, it wakes the pool when the last read is issued.
- **Arithmetic:**
  - Today the submit is 23–27 µs of the 28–36 µs decision.
  - In the MWAITX arm (`ab-20261029T100728Z`, workers kept awake), the submit fell from 29 to 3 µs and the waits for CPU rows fell by 0.9–1.3 ms a token a rank. So the CPU rows were ready about 50 µs a layer sooner.
  - Capturing 25–40 µs on each of about 40 bursts a token (31.5 layers without a miss plus 8.5 missed-expert jobs) gives 1.0–1.6 ms. That is +1.7–2.8% before the cost of extra awake time.
  - That cost: MWAITX lost 7% for about 1,500 µs × 40 workers of extra awake time a layer. Here it is about 50–100 µs × 24 workers, roughly −0.2 to −0.5% if the cost scales linearly.
- **Effort:** S–M. **Risk:** medium. Rank threads woken onto CPUs held by spinning workers can wait about 1 ms; waking only a subset of about 24 workers limits this.
- **Test:**
  - Arms: B, B-Wf (futex wake only) and B-Wp24 (pre-wake 24), three rounds.
  - Probes with no code: B-s400 and B-s600. A spin that long covers the ~450 µs gap between bursts in layers without a miss, and shows the wake's value against the awake-time cost.
  - Expect the submit to fall to 5 µs or less and the CPU-row waits to fall. Router waits must not rise. Add counters for submit-to-first-tile time and for pre-wakes whose workers went back to sleep.
- **Why it is not covered:** MWAITX and long spins kept all 40 workers runnable between bursts. This keeps them asleep and wakes some 50–100 µs ahead. The counts spin targeted the rank threads' wait, not the pool's wake.

**3. Run rank 0's predictor outside the pool's burst: about +0.6% (range +0.3 to +1.5%).**
- **Mechanism:**
  - In the early-issue path the predictor post sits before `hx_choose` (`stager.cpp:1953-1956` before `:1984`). In the non-early-issue pieces path it sits after `hx_submit` (`:872-876`), which that path's comment justifies as keeping the predictor off the pool's wake-up.
  - Move it to rank 0's `hx_finish`, after its ticket completes. At that point the burst is over and memory is idle during the card chain. Optionally score on 2–4 threads.
- **Evidence:**
  - In every early-issue arm, r0's choose is 8.0–8.9 µs against 4.0–4.5 µs on the other ranks; in non-early-issue arms it is 4.4–4.6 µs.
  - r0's CPU-row waits are 1.1–1.4 ms a token above r1's with early issue, but 0.3–0.4 ms below without it.
  - The predictor streams 3.93 MB a layer at 583–607 µs, mostly during the ~0.34 ms burst.
- **Arithmetic:**
  - Removing that memory contention from the burst is worth 0–20 µs × 40 layers, or 0–0.8 ms.
  - With 4 threads the prediction arrives about 0.15 ms sooner. That is about 0.4 MB more read ahead a unit, so about −0.07 ms × 2.43 × 0.85 ≈ 0.15 ms a token.
- **Effort:** XS (one call moved) to S (threads). **Risk:** very low; the predictions are hints only.
- **Test:** B, B-Pr and B-Pr4, three rounds. Expect the predictor to fall from 597 µs to about 250 or about 100 µs a prediction, r0's choose to about 4.3 µs, and the read-ahead's bytes for units used to rise.

**4. Slice background inserts and stream them only while the pool is idle: about +0.7% (range +0.5 to +1.1%).**
- **Mechanism:** relay each missed unit into its reserved slot in about 1 MB slices. Before each slice, if the pool has work queued or reserved, sleep on the pool's drained notification (`wait_idle` already exists). The slot is published only when complete.
- **Arithmetic:**
  - Each of the 9.6 misses a token triggers an insert of about 37 MB: an 18.8 MB read plus an 18.0 MB write. That is 355 MB a token, about 23% on top of the bursts' traffic.
  - One insert thread needs 3–4 ms for it, longer than the ~1.1 ms between bursts.
  - Bursts are about 13 ms a token; an insert is active about half the time and takes up to ~10% of memory bandwidth. 13 × 0.5 × ≤10% ≈ 0.3–0.65 ms.
- **Effort:** S–M. **Risk:** low; a unit enters the cache a few ms later.
- **Test:** B and B-Is1024, three rounds. Misses a token should stay unchanged and CPU-row waits should fall.
- **Why it is not covered:** "insert yield (inconclusive)" is on the excluded list, and I'm raising it again deliberately. The tested yield (`stager.cpp:2624`) waited once for an idle pool and then streamed all 37 MB. An insert lasts longer than the gap between bursts, so that design could not keep inserts out of the next bursts. Its window's second round also dipped about 8% in two arms at once. Slicing, with a check before each slice, is a different mechanism.

**5. Interleave 2–4 row streams per worker in the CPU GEMV: about +0.75% (range +0.4 to +1.5%).**
- **Mechanism:** the one-row kernel walks a tile's rows strictly in sequence (`ds41_host_experts.cpp:460-470`, `:578-590`). Each worker would instead walk 2–4 row groups of its tile interleaved. Each row's arithmetic is unchanged, so the bits are identical.
- **Arithmetic:** the GEMV runs at 94 GB/s. The memory benchmark (`membw_pf`) reached 101.7 GB/s at 40 threads with four interleaved streams a thread. Bursts plus missed-expert jobs are about 11.2 ms a token; a 2–8% gain on that is 0.22–0.9 ms.
- **Effort:** S–M, with the existing kernel identity tests plus a pool-shaped bench under the build lock. **Risk:** low.
- **Test:** the bench first (keep it only if it is 3% or more faster), then B and B-Hs4, three rounds.
- **Why it is not covered:** the 07:58 measurement found this headroom but nothing was built.

**6. Gate the handoff: queue the CPU rows' copy, the gather and the rest of the layer behind a gate the pool's last tile opens: about +0.6% (range +0.4 to +0.8%, more if the card chain waits on the host).**
- **Mechanism:**
  - Today, after the last tile, the rank thread releases leases, issues 1–3 blit copies (system-scope acquire), an interrupt event and the gather, then the all-reduce (`stager.cpp:2787-2875`).
  - Instead, queue all of that behind `rad_stream_gate` (`device.h:138-158`; the engine does not use it today), opened by whichever of the last worker and `hx_finish` comes second. Release the leases at the next layer's submit.
- **Arithmetic:** a host handoff of about 8–15 µs becomes about 2–3 µs of the card polling the gate. 6–12 µs × 40 layers ≈ 0.24–0.48 ms a token.
- **Effort:** M–L. **Risk:** medium. The Engram layers' host ops must never meet a closed gate, and the gate's queue room has to be accounted for.
- **Test:** probes first: one B-Ct16 arm (card stamps, run through `chain_report.py`) and one arm with `RADIANCE_DEBUG_ISSUE=1` (host cost per launch, rank 0). Build only if the card idles 25 µs or more a layer before the counts copy. Then B and B-Gt, three rounds, plus a 256-token chat soak.
- **Why it is not covered:** live-chain overlap (Ov) changes card-side concurrency and the fetch list (Fl) changes the copy count. Neither moves the host's post-pool handoff.

## Checked and dropped
- **Order in which the pool serves the four ranks:** not a lever. Without a miss, the rows are ready when the pool's total work is done; reordering only changes whose handoff comes last. In a miss layer the one critical job, the miss, already goes first.
- **Starting miss reads sooner (io_uring or similar):** posting misses at the choice (Mp) moved the first NVMe read from 51–64 µs to 21–24 µs after dispatch, but the NVMe leg stayed at 2.68–2.77 ms. The reads are drive-bound.
- **Counts round trip:** the counts spin and the fetch list were both neutral. Fusing router, topk and scatter into one kernel is card work for the cards lens.
- **CPU rows read directly by the gather (no copy):** under 5 µs a layer, below 0.4%.
- **Finer tiles for the missed-expert job:** it is bandwidth-bound, so finer tiles don't shorten it.
- **A card-side predictor two layers ahead:** about +0.6 ms more read-ahead lead, roughly +0.5–1% on top of items 1 and 3, at M–L effort. A follow-up only.

All six items fit one window after implementation: B, Pu, Wp24, Pr4, Is1024 and all four together, three rounds, with the verify arm first. The no-code probes (Pq8, s400/s600, Ct16, `RADIANCE_DEBUG_ISSUE=1`) can run in the next window now.

The parsing scripts are in `/home/as/.claude/jobs/635603c1/tmp/moreopt/critpath/`:
- `parse_arms.py`
- `decomp.py`

## Lens: cpu-dram

### Verified

**Verified ranking for the cpu-dram lens.** Four candidates survive, all with lower estimates than the finder gave. Together #1–#3 are worth about +1.2–3.5% (central ~2%), not +3.5–4.5%, because all three save time in the same CPU bursts.

**Corrections that apply to every item**
- **Burst size.** The 8 benchmark prompts decode with about 2.2 CPU hits per layer-token over all ranks, not 2.8 (2.8 comes from the long-prompt trace).
  - Source: the r2 log's 14,704 hit units, of which about 3,950 are prefill hits (912 prefill layer-passes at ~4.3 each). That leaves about 10,750 decode hits.
  - Cross-check: hx_submit ran on 9,943 layers. Removing about 902 prefill layers (98% of prefill layer-passes have CPU work) leaves 47% of one-token rank-layers with CPU work. Scaling the trace's distribution to that gives about 2.4 CPU units a layer-token, so about 2.2 hits.
  - So a hit burst is about 40 MB (~430–480 µs), not 50 MB.
  - About 6% of layers have no hit (from the long trace), so critical hit bursts are about 29 a token, plus 8.5 urgent miss bursts (37.5 critical bursts in all).
- **Burst headroom.** I discount in-burst DRAM traffic by 0.6–0.8, because the pool's 99 GB/s still sits below the 101.7–103 GB/s multi-stream ceiling.

**1. #3 Pool wake through a raw futex, no `idle_mu_` (Fw). Keep, with a different mechanism and different expected counters. Exact, S.**
- **Code-verified mechanism:** `wake()` (host_expert_pool.cpp:252) holds `idle_mu_` through the whole `notify_all`.
  - glibc's broadcast is one `futex_wake(INT_MAX)`. The kernel costs the waker about 1.3–1.7 µs per sleeper (wakebench2), so ~40–60 µs for 40 workers.
  - No woken worker can leave `cond_wait` until that ends. They then re-take the mutex one by one, and ranks 2–4 queue on it in `submit()`.
  - The finder's poolsim fit is calibrated on burst5, which calls `notify_all` outside its mutex. So the fit models the wrong mechanism, and the engine's wake is worse than burst5's.
- **Wrong counter prediction:** submit will not fall to ~5 µs. The first waker still pays the kernel's wake loop.
  - Expected "EP decision submit": ~25 → ~12–18 µs a one-token layer.
  - Expected per-submitting-layer time: ~55 → ~20–30 µs.
- **Gain:** about 10–35 µs saved per burst × 37.5 critical bursts = 0.4–1.3 ms a token.
  - Ceiling: keeping the workers awake (MWAITX, an always-awake bound) cut gather waits by 0.5–0.6 s a rank, about 1.0–1.2 ms a token. That figure is confounded: the same arm raised router waits by 2.5 s a rank.
  - The C2-off test gave only +0.6%, so C-state exit, which a futex cannot remove, is a small share.
  - **Estimate: +0.5–1.6%, central ~1.0%.**
- **Test:** run B against B-Fw over 3 interleaved rounds.
  - Log the first-claim time of the 1st/10th/20th/40th worker after a submit, the EP submit time and the gather waits.
  - Add a lost-wakeup stress test to `host_expert_pool_test`.
  - Optional: a 2-line arm that only moves `notify_all` after the lock is released (`{lock;}` then notify), to isolate the lock-hold effect.

**2. #2 Insert copies in slices with non-temporal stores (Is, plus In = NT only). Revise. Exact once an `sfence` precedes publish. S–M.**
- **Not new.** RESEARCH-2 #7 already proposed holding inserts in slices plus NT stores, at +0.4–0.8%. Only the wait-before-start part was built and measured (Iy3000, neutral).
- **New evidence, which raises the case:**
  - glibc 2.43 (verified): `x86_non_temporal_threshold` and `x86_rep_movsb_threshold` are both 0x1800000 (24 MiB). So `pk_pack_rounds`' two `memcpy` calls (11.8 + 5.9 MB) use cached stores, which read each destination line first (if Zen 3 elides nothing): about 53 MB of traffic per insert (35 MB with NT stores).
  - The finder cited memexp's 2.68 ms, but that is the old pack-into-slot form. Today's insert is a plain `memcpy`, and memexp's memcpy variant took 5.70 ms per copy and cut the concurrent GEMV rate by 12% (31.5 → 27.6 GB/s).
- **Gain:** at 9.6 inserts a token × 3–5.7 ms each, an insert is running during 50–95% of bursts.
  - Per overlapped burst, 4–8 GB/s × 0.45 ms = 1.8–3.6 MB, i.e. 18–36 µs zero-sum.
  - × 37.5 critical bursts × overlap × 0.6–0.8 = 0.2–1.0 ms a token.
  - **Estimate: +0.4–1.5%, central ~0.8%. NT stores alone: +0.15–0.6%.**
- **Test:** run B, B-In and B-Is.
  - Use 256 KB slices, not 1 MB: a 1 MB slice takes 50–100 µs alone.
  - Wait through `wait_idle()`, not by polling, cap the wait at 3–5 ms, and apply it to decode only.
  - Log bytes copied while backlog > 0, insert duration and gather waits.

**3. #1 Router predictor scoring in int4 with bf16 rescoring of the top 12 (Pi4). Keep, revised down. Exact (it only steers read-ahead). S–M.**
- **Verified figures:** 3.93 MB per prediction, 18,278 predictions at 597 µs (6.6 GB/s).
- **Why less in-burst traffic than claimed:** a ~450 µs benchmark burst is shorter than the predictor, which then finishes alone at ~12–16 GB/s.
  - Only about 1.6–2.2 MB lands inside the burst, i.e. 16–22 µs per burst.
  - × 29 bursts × 0.6–0.8 = 0.28–0.51 ms a token.
- **Estimate:**
  - Removing the predictor entirely (on the card): +0.5–0.9%.
  - Int4 + rescoring: +0.3–0.65%.
  - Not priced: one fewer busy CPU thread, and the prediction landing ~0.4 ms sooner for read-ahead.
- **Free extra fix:** with Ei, `hx_early_decide` (stager.cpp:1964) posts the predictor before `hx_choose`.
  - Rank 0's "choose" is 8.1–8.9 µs against 3.8–4.4 on ranks 1–3 in every Ei arm. Before Ei it was 4.3–4.6 µs, with the post (4.3–5.9 µs) counted separately after submit.
  - This undoes the earlier "post after submit" fix. Moving the post back is worth ≤0.1–0.2%.
- **Test:** run B against B-Pi4 together with the post move.
  - Predictor time should fall from ~597 to ~150–200 µs, and rank 0's choose should drop to ~4.3 µs.
  - Watch read-ahead used units and their wait per unit.
  - A debug-only check against bf16 scoring should show the top 4 agreeing in ≥99% of layers.

**4. #4 Per-tile overheads. Revise down; drop the helper threads. Exact, S.**
- `x_lanes` running on every tile and `run_one` scanning all 128 slots per claim are both confirmed in the code.
- During the DRAM-bound part of a burst, these costs only cost bandwidth indirectly, by lowering the number of loads in flight. That is about 2–6 µs per burst.
- **Estimate: +0.2–0.5%**, from reusing a worker's `x_lanes` when its last tile was in the same job and phase, and from an active-slot bitmap for claims.
- **Helpers dropped:** an issuing thread caught in the middle of a tile (30–60 µs under contention) when its own ticket completes delays its gather by about as much as the model's 6–14 µs gain.
- **Test:** bundle it with Fw, or arm `Ph`. Adopt on claim and skip counters with a non-negative mean.

**5. #5 BIOS settings. User decision.**
- **APBDIS, fixed SoC P-state, DF C-states:** 0–1%. In the 8-thread `burst` bench, the slowdown after a 1–2 ms gap appears only when workers sleep. Spinning or MWAITX-waiting workers show none, even though DRAM is idle during the gap. So the penalty follows core sleep (CC6), not fabric idleness, and C2-off measured only +0.6%.
  - Pre-check: rerun `burst 8 12` and `burst 8 48` after any BIOS change.
- **`dmidecode -t 17`:** a free check. The facts are verified: NPS1, 16 EDAC ranks, THP madvise, acpi-cpufreq/schedutil with a 1 ms rate limit, C2 exit 30 µs.
- **NPS4: drop.** CCD confinement measured about 0 in burst4/5. It needs NUMA-striped slabs (L effort) and a reboot that affects QFN. It cannot be pre-measured on an NPS1 box.

**Window plan and adoption.** The finder's 15-arm plan (B, B-Pi4, B-Is, B-Fw, B-all × 3 rounds) is fine. Single items worth 0.3–1% sit inside the ±1% run noise, so a "+0.7% with no losing pair" rule will often reject real effects. Decide each item on its counters plus a non-negative mean, and let the combined arm confirm the total.

My scripts are in `/home/as/.claude/jobs/635603c1/tmp/moreopt/cpu-dram/verify/`: `rankfrac.py`, `prefill_frac.py` and `wakemodel.py` (uses the finder's poolsim2).

### Finder's report

**CPU expert path and DRAM: candidates, most valuable first**

The pool already streams at the DRAM ceiling once its bursts get going. In burst4/burst5 (pool-shaped, 40 workers), each extra expert adds 190 µs, about 99 GB/s. The multi-stream read ceiling is 100–103 GB/s, which is 73–75% of 136.5 GB/s. Only hardware or BIOS settings can raise that ceiling.

The software gap is in what happens around each burst, so the levers below target that:
- **Wake-up overhead per burst:** 3 packed experts take 622 µs; at 99 GB/s they would take 545 µs.
- **Other threads' DRAM traffic inside the bursts:** the router predictor and the miss inserts.

The numbers used throughout come from window ab-20261029T161035Z, best arm, r2 logs:
- 57 ms a token.
- 2.8 hits a layer-token over all ranks, 18.0 MB each (long-prompt traces).
- 9.6 decode misses a token: 4,622 one-token misses over 481 tokens.
- About 8.5 critical miss layers a token, so about 31.5 hit bursts a token sit on the critical path.

A note on the arithmetic: "zero-sum" means every byte another thread moves during a DRAM-bound burst delays the burst by bytes ÷ 99 GB/s. I discount that by 0.5–0.85 because not all of a burst is DRAM-bound.

**1. Router predictor scans 3.93 MB of bf16 every decode layer, alongside the burst (exact: it only steers read-ahead)**
- **Mechanism:**
  - With `Pp4` on, rank 0 posts the predictor at the counts sync (`hx_early_decide`), so it runs in parallel with the hit burst.
  - Each prediction reads 384×5120 bf16 = 3.93 MB. The 157 MB of routers never stay in L3.
  - Logged: 18,278 predictions at 597.2 µs each, which is 6.6 GB/s for the whole burst.
- **Gain:**
  - 3.93 MB at 99 GB/s ≈ 40 µs per burst. Over 31.5 bursts that is 1.26 ms a token, 2.2% zero-sum, so +1.1–1.9% after the discount.
  - **On the card:** removes it all. The card already holds the routers and idles waiting for CPU rows.
  - **int4 + bf16 rescoring of the top 12:** 1.1 MB, −72% → +0.8–1.4%.
  - **int8:** +0.55–0.95%.
  - **Side benefit:** the prediction lands about 0.4 ms sooner. Read-ahead units used today still wait 2.42–2.69 ms at their layer. The read-ahead lens should price that; I have not counted it.
- **Effort and risk:** S–M on the CPU, M on the card. The risk is hint quality.
- **Test:** part `Pi4` (`RADIANCE_EXPERIMENTAL_DS41_PREFETCH_SCORE=int4`). Gate it with a check counter that compares the top-4 against bf16 scoring (expect under 1% of layers to differ). Watch the predictor's scoring time (597 → about 150–200 µs) and the read-ahead used and waited figures.
- **Not covered:** the read-ahead itself was adopted with this cost inside its +3.6%; nobody has measured the predictor's DRAM cost.

**2. Miss inserts: write-allocate copies that overlap the next bursts (exact)**
- **Mechanism:**
  - `pk_pack_rounds` copies the codes with `std::memcpy`: 11.8 MB and 5.9 MB.
  - glibc 2.43's `x86_non_temporal_threshold` is 24 MiB (`ld.so --list-tunables`), so both copies use cached stores and read every destination line first. That makes about 53 MB of traffic per miss, roughly 510 MB a token.
  - `hx_submit_misses_now` queues the insert right behind the urgent jobs. An insert lasts about 2.7 ms (memexp contend: 2.68 ms), so about 46% of it overlaps the urgent burst and roughly two hit bursts.
- **Gain:**
  - At a contended 6–10 GB/s, that is 8.6–14 MB per insert inside bursts, 83–135 MB a token, 0.84–1.36 ms.
  - Result: +0.7–2.0%, central about +1.3%.
  - Non-temporal stores alone remove about a third of the traffic.
  - **Calibration:** Iy3000, which only held each insert's start, cut rank 0's gather waits from 5,970 / 5,859 / 5,983 to 5,837 / 5,587 ms, about −0.47 ms a token. Its tok/s change stayed within noise.
- **Effort and risk:** S–M. Copy with streaming stores in 1 MB slices, only while `pool->backlog()==0`. Bound the wait, and apply it to decode only so prefill inserts do not pile up. Risk is low.
- **Test:** part `Is` (slices + non-temporal stores), optionally `In` (non-temporal stores only). Add counters for bytes copied while the pool was busy and for insert duration.
- **Not covered:** Iy3000 waited only before starting, and a 2.7 ms insert then spans about two bursts. The 24 MiB threshold is a new finding.

**3. Pool sleep and wake through a futex, without `idle_mu_` (exact)**
- **Mechanism:**
  - `wake()` holds `idle_mu_` across `notify_all` (about 25 µs for 40 sleepers).
  - Every woken worker must lock it again (glibc `cond_wait`, contended lock) and unlock it with a `futex_wake`. Those hand-offs happen one at a time.
  - While `sleepers_ > 0`, other ranks' `submit()` calls and the `wake()` in `tile()` after each gate_up also queue on `idle_mu_`.
- **Evidence:**
  - Submit takes 23–25.6 µs per one-token layer, about 54 µs for each layer that actually submits (47% of layers). MWAITX had it at 3 µs.
  - burst5 measures 622 µs for 3 packed experts. My calibrated model (`poolsim.py`) gives 559 µs with instantly awake workers and 586 µs for a flat futex wake (30±10 µs C2 exit, 0.5 µs per kernel wake). Neither covers the last 36 µs per burst.
  - wakebench: the last of 7 waiters runs at 24–88 µs (p50), against 9–20 µs for the first.
- **Gain:**
  - Saving 15–40 µs per hit burst (31.5 a token) plus 25–45 µs per urgent burst (8.5 a token) gives 0.7–1.6 ms a token.
  - The C2-off test measured only +0.6%, so the C-state share is small. Range +0.5–2.5%, central +1.3%.
- **Effort and risk:** S, about 40 lines. Risk: lost wake-ups; add a pool stress test.
- **Test:** part `Fw` (`..._HOST_EXPERTS_FUTEX=1`). Add a counter for the first claim of the 1st / 10th / 20th / 40th woken worker. Expect submit to fall to about 5 µs per layer.
- **Not covered:** the rejected items all kept workers runnable. This adds no spinning.

**4. Per-tile overheads in the pool (exact, nearly free)**
- **Mechanism:**
  - `x_lanes` runs on every tile: about 1.3 µs at K=5120 and 0.8 µs at K=2304, about 12 µs per worker per 3-expert burst, and ×rows in multi-row jobs.
  - `run_one` scans all 128 slot headers per claim.
  - The issuing threads spin in `HostExpertPool::wait` instead of computing their own rank's tiles. The model puts the value of that help at 6 µs per burst after #3, 14 µs before it.
- **Gain:** about +0.5–0.9% decode. Prefill bursts and N=4 batches gain 4–7% of burst CPU time.
- **Effort:** S. **Test:** part `Ph`.

**5. BIOS and hardware (user decision)**
- **Readable facts:**
  - NPS1, one node.
  - 8 channels, each with one dual-rank x4 RDIMM (EDAC ranks 0–15).
  - THP set to madvise.
  - acpi-cpufreq with 2.85 / 2.1 / 1.5 GHz plus boost; schedutil rate limit 1 ms.
  - C2 exit latency 30 µs.
  - The DMI tables are root-only.
- **Checks:**
  - `sudo dmidecode -t 17`: every DIMM should show a configured speed of 2133.
  - Try APBDIS=1, a fixed SoC P-state P0 and DF C-states off. Untested; expect 0–1%.
  - NPS4 only together with NUMA-aware tile claims. That is L effort, also affects QFN, and might give 0–3%. Before any engine work, test it with `membw_pf` (local against interleaved memory).

**One window for 1–3:** arms B (`all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe`), B-Pi4, B-Is, B-Fw and B-Pi4-Is-Fw, three interleaved rounds (15 arms, about 30 min). Adopt an item if its mean is at least +0.7% with no losing pair, its counter moves as predicted, and every reply is identical. Together these should be about +3.5–4.5%.

**Checked and dropped:**
- **Prefetch, multi-stream and memory-level parallelism:** the bursts' 99 GB/s is already within 3% of the 101.7–103 GB/s ceiling.
- **Small tail tiles or a down projection split into K-phases:** the model saves only 2–10 µs per burst.
- **Faster kernel compute:** only the ramp is compute-bound.
- **CCD confinement:** burst4/burst5 show −0.2 to −0.3%.
- **Entropy coding:** the codes carry 3.894–3.897 bits of 4.
- **1 GB pages:** the slabs are already madvised for THP.
- **Thread count:** already measured.
- **Zero-copy miss codes:** reading the codes with O_DIRECT straight into the reserved slot is M–L effort for +0.2–0.6% beyond #2.
- **Free check:** nothing logs the slabs' huge-page coverage after pinning. `pin_pieces` applies `madvise` before the KFD lock, but coverage is never confirmed.

My scripts, single-threaded and read-only, are in `/home/as/.claude/jobs/635603c1/tmp/moreopt/cpu-dram/`: `trace_decode.py`, `trace_crit.py`, `poolsim.py`, `poolsim2.py`, `poolsim3.py`.

## Lens: drive-cache

### Verified

**Verdict for lens drive-cache.** The finder's +2–4% comes down to about **+1.5–2.5% exact** if every candidate that survives is built. The biggest correction is to candidate 1: raising the mirror share is worth about 0, and its only real value is re-adopting Ms3+Mp, which was already measured. Candidate 2 holds up with somewhat lower numbers. Candidate 3 must be measured before anything is built. Candidate 4 holds. Most of item 5 is dropped.

## Per-candidate checks

**1. Mirror share above 0.13 with Ms3+Mp — revise (the share step is worth ~0)**
- **Not new:** shares above 0.13 were never tested after the kept readers, so that part is genuinely untested.
- **Cherry-picked baseline:** the finder's "mirror 2.38 / NVMe 2.70 / back 2.74" comes from rounds 2–3 only.
  - Round 1 of Ms3+Mp gave mirror 2.48 / NVMe 2.70 / back 2.82.
  - Across all 3 rounds, caller back is 2.766 ms for Ms3+Mp against 2.799 for B.
  - So Ms3+Mp alone saves 0.033 ms × 6.8 one-unit reads × 0.9 critical = 0.20 ms a token, about **+0.36%**.
  - About half of the finder's 0.12 ms saving is that Ms3+Mp effect, measured on rounds 2–3 only.
- **Ignored spread:** with Ms3+Mp the mirror lands last in only 4.7–11.4% of calls, but when it does it is 0.23–1.18 ms late (round 1: 0.72–1.18 ms). That tail is heavy. Equal mean legs mean the mirror lands last about half the time.
- **Variance fit** (`v3_mirror_var.py`): body N(0, σ) plus a rare Exp(τ) excess, fitted per arm family.
  - Ms3 family: σ 0.14, q 0.095, τ 0.60 (good fit).
  - Moving 0.13 → 0.14 changes a one-unit read by −23 to +10 µs; 0.13 → 0.15 by −39 to +69 µs.
- **Timeline model with spread and a mirror shared by ranks** (`v4_timeline_var.py`):
  - 0.14: +0.09 to +0.16%.
  - 0.15: −0.10 to −0.15%.
  - The finder's deterministic model gave +0.4% and +0.3% for these steps.
- **The extension** (larger share when another rank reads): +0.2–0.3% even as an upper bound where all ranks adapt. Realistically +0.05–0.15%, because the first rank to dispatch cannot know. The finder said +0.3–0.5%.
- **Corrected:** Ms3+Mp +0.3–0.4% by counters, plus Rd14 +0–0.2%. Drop Rd15 and the extension.

**2. A1 (finish the partly read unit with a mirror part) + B1 (int8 predictor routers) — keep, revised**
- **New:** yes. `hx_ra_take` today only re-reads a barely started unit (under 2.44 MB) afresh. It never finishes the rest with the mirror, and the scoring speed was never changed.
- **Counters check out:** scoring 583–597 µs (715–719 µs in the 256-token chat runs). Used units have 6.2–7.0 MB read ahead and the layer waits 2.32–2.80 ms for the rest.
- **The model overstates the waits:**
  - It has 10.71 misses a token against 9.61 measured; scale its percentages by 0.9.
  - Its finish wait is 2.72–2.74 ms against about 2.51 measured.
- **A1:**
  - All of its value is the mirror part. Without it A1 is −0.09%.
  - With spread and a shared mirror: +0.41–0.52%, against the finder's +0.67%.
  - A safer variant, "A1-lite", keeps the urgent pieces for the container part and adds only the mirror part beside them: +0.48–0.59%.
  - Why A1-lite is safer: per the read-ahead plan's piecebench, 4 × 512 KiB pieces read at 6.77 GB/s, while demand preads get about 6.1–6.2 GB/s. Moving the rest onto big preads could cost up to about 0.17 ms a unit.
  - Hazard: the posted miss read (Mp) must not include a unit whose planes `hx_ra_settle` may replace. That is the IMPL-MISS review's finding 1a/1b (use after release, self-deadlock).
- **B1:**
  - +0.64% in the model, unchanged by the spread fix.
  - Scoring is bound by memory latency, not compute: about 0.15 ms of FMA work against 0.59 ms measured. So int8 should take about 0.3 ms.
  - Possible extra: the predictor reads 3.9 MB a layer during the pool's DRAM-bound burst, and B1 halves that. This could add up to +0.35% (the earlier cost estimate was about 11 µs a layer on rank 0).
  - It frees 78 MB of RAM.
- **Corrected:** B1 +0.4–0.8%, A1 or A1-lite +0.3–0.5%, both together +0.6–1.1% (the finder said +1.2%).

**3. One-ahead tier — revise to "measure first"**
- **Weak prior:** one-ahead read-ahead is listed "still dead" (RESEARCH-2-RANKED-PLAN.md:330). In the night log it measured +1.9% within noise at 02:48 and −0.5% at 09:03. The finder's exclusion note misses this.
- **Shadow data** (ab-20261029T085239Z, Ps10 arm): top-4 recall is 36.0–37.0% one-ahead against 30.6–32.8% two-ahead. The finder's "at top-2 both catch the same misses" is really the same *count*; the overlap was never measured.
- **Incremental recall:** somewhere between 4.7% and 36.5% of misses. My guess is 11–17%, which is model P1 ≈ 0.15–0.25, the low end of the finder's 0.25–0.40.
- **Not modelled:** a second scoring a layer (CPU occupancy, cf. the MWAITX −7% lesson), and that one predictor thread cannot deliver both predictions 0.35 ms after the post (the model assumes it can).
- **Corrected:** +0.3–0.8%, only on top of A1+B1. Do not build until a counter shows ≥10% of misses caught by one-ahead top-4 and not by two-ahead top-4.

**4. Side store for the 7 never-packable experts — keep**
- **New:** yes.
- **Counters:** refused inserts were 26/11/17/16 a rank (6 distinct units) in the held-out arm, and 314 over 2,941 decode tokens in the 256-token chat runs (0.107 a token). The replay gives 0.094.
- **Gain:** 0.1 × 2.7 ms × 0.9 ≈ 0.24 ms a token, about **+0.4%**, plus a small TTFT gain.
- **Cost:** about 140 MB of planar planes in total, at most 40 MB a rank. That is small, but count it in the memory budget.
- **Must also:** make `hx_ra_plan` treat side-stored units as cached, and run the epoch check as for whole read-ahead units.
- A planner alternative (prefer these units for VRAM) is worth only about +0.27%, since their access rate (0.013 a token) is below the marginal VRAM resident's (0.022).
- **Corrected:** +0.3–0.5%.

**5. Lower value**
- **Read-ahead pieces from the mirror: drop.** It contradicts the read-ahead design note ("read-aheads must read every byte from the NVMe; the RAID mirror is too slow for their tails"). The model also ignores that a SATA piece in flight delays a demand mirror leg, which has only about 0.3 ms of slack. Realistically −0.5 to +0.2%.
- **Scoring on the card: not new.** It is PLAN.md P0.4's original design (xprefetch-shaped). Worth about +0.2–0.4% beyond B1, effort M–L.
- **Pb256 / Pq2 / Pp6:** agree, no window of their own.
- **"Checked and dropped" list:** spot-checked against `m1_stats` (88.3% single-rank miss layers, 9.75 misses a token on held-out round 1, flat per-layer misses). Consistent.

## Surviving candidates, ranked (all exact; none needs a hardware decision)

1. **B1, int8 predictor routers** (per-expert scale, 4 accumulators): +0.4–0.8%, effort S, frees 78 MB.
   - Test: B vs B+B1 (Pm56-Pe base), ≥4 interleaved rounds. Expect "us a prediction" about 590 → 300 and MB read ahead per used unit about 6.6 → 7.5, with planned/used ratios unchanged.
2. **A1-lite** (mirror part beside the urgent pieces; or A1): +0.3–0.5%, effort S–M.
   - Test: a verify arm `verify-f-M1-H-R-Rd13-Pp4-<new part>` must show 0 differ. Then B+B1 vs B+B1+A1, ≥4 rounds. Expect "finished while the layer waited" about 2.5 → 2.3 ms.
3. **Never-cached side store:** +0.3–0.5%, effort S, about 140 MB.
   - Test: B vs B+side store, ≥4 rounds. Expect refused inserts down to 1 per distinct unit and decode misses about −0.1 a token; judge by counters, since 0.4% is below one round's noise.
4. **Re-test Ms3+Mp with Rd14:** +0.3–0.5% (Ms3+Mp by counters; the share step is about 0). No code.
   - Test: filler arms in the next decode window: B, B-Ms3-Mp, B-Ms3-Mp with Rd13→Rd14, ≥4 rounds. Judge by the "one unit … caller back" line (expect 2.80 → 2.77 → 2.75–2.77), and check the read-ahead wait does not rise. Skip Rd15.
5. **One-ahead tier:** +0.3–0.8%, only after B1 and A1.
   - Test: first add an overlap counter to the shadow mode (`Ps`) for misses caught by one-ahead top-4 but not by two-ahead top-4. Build only if it is ≥10% of misses.

Everything is in `/home/as/.claude/jobs/635603c1/tmp/moreopt/drive-cache/`:
- `v1_counters.py`, `v2_compact.py`: per-arm counter extraction from the window logs.
- `v3_mirror_var.py` (output `v3_mirror_var.out.txt`): the variance fit and share pricing.
- `v4_timeline_var.py` (outputs `v4_timeline_var.out.txt`, `v4_ext.out.txt`): the finder's timeline model with mirror spread and a shared mirror.
- `m7_main.out.txt`, `m7cd8.out.txt`, `m1_stats.out.txt`: the finder's scripts re-run.

### Finder's report

The misses lens has little exact headroom left: the five candidates below add up to about **+2–4%**. Stalls on misses are about 24 ms of the 56.8 ms token. The drive is already reading at full speed, the read-ahead is limited by how far ahead it can predict, and the cache policy is at a local optimum.

Three numbers set the limits:
- **A no-miss layer is about 0.83 ms.** That is (56.8 − 9.14 miss-layers × ~2.6 ms) / 40. A two-ahead lead is therefore about 1.65 ms, and today the prediction takes 0.65 ms of it.
- **A lone miss is bandwidth-bound.** The NVMe leg reads 16.4 MB at about 6.0–6.1 GB/s.
- **In the timeline model, even a perfect two-ahead predictor gives only +7.6%** (+13% with items 2 and 3 added).

All estimates come from replaying the 40-prompt trace (ab-20261029T115523Z), re-targeted to today's placement (712/720 VRAM experts, 1,424-unit SC-LRU, warm-up). The replay reproduces the engine: 9.75 decode misses a token against 9.62 measured. I also built a timeline model calibrated to window ab-20261029T161035Z: 6.5 MB read ahead per used unit against 6.3 measured, and a 2.74 ms wait for the rest against 2.39–2.72 measured.

## Ranked candidates

**1. Raise the decode mirror share now that the mirror tail is split and reads are posted (Ms3 + Mp + Rd14 or Rd15)**
- **Kind:** exact; no code.
- **Mechanism:** Ms3 and Mp were only ever measured at share 0.13 and called neutral. Their counters show the mirror leg now finishes early and could take more bytes.
- **Arithmetic** (one-unit miss reads, mean over ranks):
  - B: mirror leg 2.57 ms, NVMe leg 2.75 ms, the caller back at 2.80 ms.
  - B-Ms3-Mp: mirror 2.38 ms, NVMe 2.70 ms, back at 2.74 ms. The mirror has about 0.32 ms of slack.
  - Moving x MB from the NVMe (6.1 MB/ms) to the mirror (about 0.67 ms per extra MB): 2.70 − x/6.1 = 2.38 + 0.67x gives x ≈ 0.39 MB. That is a share of about 0.15, and both legs land at about 2.64 ms (back at about 2.68 ms).
  - That saves about 0.12 ms on each of 6.8 one-unit miss reads a token: about 0.8 ms, or +1.4%. 88% of miss layers have a single rank missing, so most of this is critical.
  - The model, which ignores variance, gives +0.6% at 0.13, +1.0% at 0.14 and +0.85% at 0.15. **Expect +0.6–1.2%.**
- **Effort / risk:** none / low. The mirror leg may land last more often.
- **Window** (×3 rounds, interleaved; `MIRROR=` set):
  - B = `all-t40-f-M16-H3-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe`
  - B with Rd13 replaced by Rd14, plus `-Ms3-Mp`
  - B with Rd13 replaced by Rd15, plus `-Ms3-Mp`
  - Watch the "caller back" figure on the "one unit" line.
- **Not covered by the exclusions:** they rejected Ms3 and Mp at a fixed 0.13. No share above 0.13 was tested since the kept readers.
- **Optional extension:** a larger share when another rank's demand read is already in flight. This applies to 10.8% of miss layers and is worth +0.3–0.5%.

**2. Finish a partly read read-ahead unit inside the rank's demand read (A1), together with an int8 copy of the predictor's routers (B1)**
- **Kind:** exact. Predictions are only hints.
- **Mechanism, A1:** at take time, cancel the unit in the PieceReader and read the rest, [started × piece, end), through `read_unit_planes` with an offset. Use large preads and a mirror part sized so both legs land together (about 1.4 MB). Then wait for the pieces still in flight.
- **Today's path:** urgent 512 KiB pieces, NVMe only. Used units have 6.3–6.7 MB (34%) read before their layer, and the layer waits 2.39–2.72 ms for the rest.
- **Mechanism, B1:** scoring takes 592.9 µs a prediction. That is one thread reading 3.9 MB of bf16 routers per layer from DRAM while the pool is streaming, and it eats about 40% of the two-layer lead. An int8 copy (per-expert scale; 79 MB instead of 157 MB) should take about 0.3 ms.
- **Model:**
  - A1 alone: +0.67%. The result is the same however urgent pieces share the drive (weights 2, 4 or 8); without the mirror part A1 is −0.2%, so its value is the second device.
  - B1 alone: +0.65%, with 6.5 → 7.5 MB read ahead per used unit.
  - Both: **+1.2%**.
- **Effort / risk:** A1 S–M (about a day), B1 S. Low risk: the ranges don't overlap, and the existing failed-piece and epoch checks still apply.
- **Window:**
  - B vs B with the new parts, for example `Pr` (A1) and `Pi8` (B1), ×3 rounds.
  - Plus a verify arm `verify-f-M1-H-R-Rd13-Pp4-Pr`, which must show 0 differ.
  - Counters: "ms scoring", and MB read ahead per used unit.
- **Not covered by the exclusions:** Pp4 is in the exclusion list as measured, but neither its completion path nor its scoring speed was changed. `_PREFETCH_FINISH=0` discards the bytes already read.

**3. A one-ahead tier on top of the two-ahead plan (needs item 2)**
- **Kind:** exact.
- **Mechanism:** score layer L+1 from layer L's router input as well. Add uncached top-K units the two-ahead plan missed, and cancel two-ahead units the one-ahead prediction rejects.
- **Data:** one-ahead uncached precision at positions 1–3 is 78/64/52%, against 67/47/30% two-ahead. At top-2 both catch the same misses (261 vs 255 of about 1,550).
- **Model:** with guessed incremental recall of 0.25–0.40, waste of 0.02–0.03 units a layer a rank, and 40–70% of two-ahead waste cancelled, the tier adds **+1.1–1.9% on top of item 2**. It is worth about 0 without A1. A realistic incremental recall is about 0.18, so **expect +0.5–1.5%.**
- **Effort / risk:** S–M. It needs B1's time budget, about 0.6 ms of predictor time a layer, possibly on two threads.
- **Window:** measure first. Add a shadow counter for misses caught by the one-ahead top-K but not by the two-ahead plan (`Ps` style), then run the arms.
- **Not covered by the exclusions:** pieces mode never computes the one-ahead prediction.

**4. Keep the 7 experts the packed cache can never hold in a small side store**
- **Kind:** exact. They are computed from their planar planes, the same path a miss already uses.
- **Data:** 223 of 25,395 replayed decode misses (0.88%, 0.094 a token) fall on those experts. The engine logged 11–26 refused inserts a rank in 8 prompts.
- **Gain:** 0.094 × 2.65 ms ≈ 0.25 ms, about **+0.4%**. It is below the threshold but nearly free: at most 2 units a rank, about 38 MB.
- **Effort:** S. On a refused insert, keep the planes and hand them to later accesses as a "whole" ahead unit.
- **Window:** B vs B plus a new part (for example `Nv`); the shutdown line's refused-insert count should fall to about 1 per distinct unit.

**5. Lower value**
- Read-ahead pieces also from the SATA mirror (w3 codes, yielding to demand mirror legs): +0.6% in the model, M effort.
- Scoring on the card on lane 1: +0.4% beyond int8, M–L effort.
- Free arms Pb256, Pq2 and Pp6: ±0.2% in the model, and Pp6 was measured equal to Pp4. Not worth a window on their own.

## Checked and dropped (exact, with numbers)

- **Cache policies (replayed decode misses vs today):**
  - ARC: +13.8%.
  - Protecting units used in ≥2 or ≥3 requests: +2.8% to +9.8%.
  - Probation for experts the profile never selected (P = 8–64): −2.1% to +12.6%, not monotonic in P, with prefill misses up as much as 18%.
  - Graded second chances: −0.4% to +5.6%.
  - Demoting prefill-only units at the end of prefill: +3.4% decode, +44% prefill.
  - Refilling profile-hot units between requests: at best −0.9%, with +6% prefill misses and 3.8 GB of reads a request.
  - Belady's optimum is −52%, but that gap is knowledge of the future. The only cheap feature that separates victims is profile share > 0 (71% vs 44% come back), and the second chances already use it.
- **Recency within the request and seeding from the prompt:**
  - Of decode misses, 0.67% are units the same request already touched, and 79% were last used by an earlier request.
  - 13.5% were never seen before, and 6.7% are warm-up units evicted before first use.
  - Even an oracle that admits only the prompt units decode will use saves 9.5% of decode misses but adds 27% to prefill misses.
- **Read start latency, io_uring, queue depth, read sizes:** Mp already starts every read within 21–46 µs instead of 55–270 µs, yet the NVMe leg ends only 0.03 ms sooner (2.717 vs 2.748 ms). The drive is the limit.
- **Per-layer K:** decode misses are flat across layers (19–33 per 100 tokens).
- **Read-ahead that does not yield to demand reads:** −2.4% in the model.
- **Three-ahead prediction (guessed precision):** no gain in the model.
- **Dynamic VRAM residency:** the marginal VRAM experts get 0.022–0.025 accesses a slot a token, while an extra cache slot saves only 0.0027–0.003 misses a slot a token.

Every replay is single-threaded and stdlib-only. The scripts are in `/home/as/.claude/jobs/635603c1/tmp/moreopt/misses/`:
- `mlib.py`: shared replay helpers.
- `m1_stats.py`: miss structure.
- `m2_policies.py`, `m4_variants.py`, `m4b_probation.py`, `m6_prefill_admit.py`, `m6b_prefill_demote.py`, `m9_refresh.py`: cache policies.
- `m3_victims.py`: features of eviction victims.
- `m5_vram_marginal.py`: value of the marginal VRAM residents.
- `m7_timeline.py`, plus `m7b_recall.py`, `m7c_tier.py`, `m7d_mirror.py`, `m7e_sens.py`: the timeline model and its variants.
- `m8_never.py`: misses on the never-cached experts.

## Lens: card-vram

### Verified

**Verdict: all four survive, but the numbers change.** Candidate 1's replay is wrong in both directions, its gain range holds only at d=32 rather than d=64, and it costs TTFT. Candidates 3 and 4 are slightly smaller than claimed.

All re-checks were single-thread replays and log reads, each under 4 s; nothing was built and no GPU was touched. My replay matches the best arm's own counters (ab-20261029T161035Z, Pm56-Pe-r2):
- **File-tier units per rank and run:** replay 17,121; engine 16,897–17,410 (mean 17,077).
- **Decode misses:** replay 9.53 a token; engine 9.61 (4,622 over 481 tokens).
- **Prefill misses per run:** replay 4,759; engine 4,752.

**How I convert counts into time:**
- **One CPU unit** is worth c = 0.13–0.18 ms of critical path.
  - The chat-256 arm runs 63.3 ms a step against 57.6 ms on held-out, with about 41 more CPU units a token and the same miss rate. That gives 0.14 ms a unit.
  - A per-layer model fitted to rank 0's 7.9 ms gather wait gives 0.15–0.18 ms.
- **One extra miss** costs m = 2.4–2.8 ms. A lone one-unit miss takes 2.80 ms from its dispatch, less the pool burst it hides.

## Ranked survivors

### 1. Request-adaptive VRAM region — revise. Exact, large effort, gated on vpack's device check
**Novelty:** genuine.
- The heat engine is off under `--deterministic`.
- xprefetch covers only the pinned pool and excludes native DeepSeek (REVIEW F5).
- The engine survey deferred "background promotion into VRAM".
- It shares machinery with the excluded prefill card share (P3-lite: slots, SDMA copies, `hx_choose` handing units to the card). Build that machinery once.

**Errors in the finder's replay:**
- **The stream is too small.** Promoting each rank's in-sample top 141 units leaves 80.5 file-tier units a token (held-out). The real FP8 residents, rebuilt from the sealed profile, leave 98.7. The rebuilt resident masses match the log exactly (rank 0: 6.035599 at 598 residents, 6.544383 at 712). So the region saves about twice the CPU units the finder claimed.
- **The static comparison is the wrong one.** Comparing against "the next d units kept static" is not the trade. The real trade demotes d residents into the file tier, where they take space in the RAM cache (they are profile-hot and get second chances). The finder did not count the extra misses or the TTFT cost this causes.
- **The lag claim is wrong.** A two-token lag costs 13–20% of the hits, not half (d=32: 22.9 → 18.4; d=64: 32.9 → 28.6).
- **The file buffers are smaller than stated.** "two file buffers 35.86 MiB" is the total for both, about 2 units. Null entries would free 2 slots, not 4.
- **Kh does not belong on timing arms.** The README says steps are not issued ahead while it is on.
- **0.38 ms a unit is too high.** That figure is the packed pool bench's per-expert time; the measured critical-path slope is 0.13–0.18 ms.

**Corrected replay** (vpack residents 743/752, held-out, per decode token over the 4 ranks; the status quo is 95.8 CPU units and 9.16 misses):

| Variant | CPU units | Misses | Fills | Prefill misses | Time saved (before fill costs) |
|---|---|---|---|---|---|
| d=16 from residents | −12.25 | +0.19 | 18.4 | — | 1.07–1.76 ms |
| **d=32 from residents (best)** | −19.85 | +0.39 | 21.8 | +4.4% | 1.5–2.6 ms; per-layer model 1.8–2.2 ms |
| d=64 from residents | −26.88 | +0.84 | 24.8 | +7.9% | 1.2–2.8 ms (extra misses cancel extra hits) |
| d=64, region copy removed from RAM | −26.88 | +0.50 | 24.8 | — | needs write-back from the card on eviction |
| 9 slots from headroom, nothing demoted | −7.50 | 0 | — | 0 | 0.98–1.35 ms |
| Same 9 slots kept static | −0.82 | −0.05 | — | — | 0.23–0.29 ms |

A recency slot is worth about 8–9× a marginal static slot.

**Fill traffic costs:** 0.2–0.8 ms a token, from DRAM contention during pool bursts, link-gate holds and host bookkeeping.

**Net gain:**
- **d=32, held-out:** 0.7–2.4 ms of about 57 ms, **+1.2–4.2% (central +2.5%)**.
- **d=32, chat:** 37.08c − 0.39m − (0.3–1.0) = 2.7–5.4 ms of 63 ms, **+4–8.5%**.
- **TTFT:** about +3–4% longer, from the extra prefill misses.
- **No-demotion variant** (headroom slots plus the 2 file-buffer slots): **+0.7–2.4%**, with no TTFT cost.

**Implementation gaps the finder did not mention:**
- The vpack mark is static. `issue.cpp:258–262` sets it from `weights[w].vram_bytes > 0`, "once, at the first refresh". Region entries need a mark that changes with the slot.
- `hx_early_decide` fails with RAD_E_STATE if any selected unit is left off the CPU.
- The B1 cap assumes the CPU takes every routed unit.
- The DMA lane is not active in the best config ("copy lists not used").
- Pm56 prefetches 56 MB a layer into the 64 MB on-die cache, and fills could evict it.
- Publish a slot only after the host sees its copy land, with no ready event. Otherwise the compute stream waits on the copy.

**Test:** run the vpack device check; then `B-Vp` against `B-Vp-Vd32` and `B-Vp-Vd16`, three interleaved rounds plus Uchat256, with no Kh, decode-only fills and the link gate held from counts to gather. Watch hits, fills, late fills, misses and TTFT.

### 2. Ct16 trace, then the IMPL-CHAIN fusions — keep, measurement-gated. Exact, medium effort
**Checks:**
- No chain trace has ever run; there are no `chain-*` files in any window.
- The Ct part is in main's `ab_window.sh` (lines 774–777).
- The plain-layer chain checks out: 20 launches, about 289 µs of profiled op time plus about 72 µs of gaps.

**Correction:** the Ov counters show 686 launches a decode pass (17 a layer) actually went unordered, with no gain (16.85/16.65 against 16.36/17.07 tok/s). So the barrier wait itself is worth about zero. A fusion only saves each launch's fixed dispatch and ramp cost, about 2–5 µs, not 6–7 µs. Six launches give 12–30 µs a layer, or 0.5–1.2 ms a token.

**Gain:** **+0.5–2%** (finder: +1–2.5%). That the attention prefetch helped (+2.8%, then +1–1.4% more) shows the chain is on the critical path.

**Test:** one `B-Ct16` arm. Build only the fusions whose removed launch shows at least 3 µs of card idle before it, and at least 20 µs a plain layer in total. Then `B` against `B-Fu`, three rounds, no Kh.

### 3. Read-ahead predictor on the card — revise. Exact (hints only), medium effort
**Corrections:**
- In the early-issue path the post happens before `hx_choose`, not after `hx_submit`. That is why rank 0's choose time is 8.5 µs against 4.3 µs.
- The predictor already scores bf16 router copies (157 MB of host RAM).
- Used read-ahead units wait 2.42–2.69 ms out of a 2.8 ms miss, because the read-ahead is limited by drive time: 11.6 s of reader time was held back by demand reads, and 4 readers are shared by all ranks.

**Gain:** about 0.55 ms of extra lead (597 µs of scoring less about 50 µs on the card) buys about 0.2–0.35 ms for each used unit. At 2.42 used units a token that is 0.5–0.85 ms, **+0.5–1.5%** (finder: +1–2%).

**Design changes:**
- Run the GEMV after the scatter, not alongside the router, which is on the critical path.
- Read the f32 router already in VRAM, so card 0 needs no extra copy.

**Test:** first a cheap CPU probe that splits the scoring over 3–4 threads. Then `B` against `B-Cp`, three rounds. Watch the "finished while the layer waited" line.

### 4. `--gpu-headroom-mib` 256 → 96 — keep. Exact, trivial
- 96 MiB is the engine's own default (`config.cpp:81`, "default 96"). Every recent card settles at 301.8 MiB free against a 291.38 MiB target.
- The 160 MiB freed is 9.3 packed slots a rank.

**Gain:**
- As static residents: 0.82c + 0.05m = 0.23–0.29 ms; the embedding experiment implies up to about 0.09% a resident. **+0.4–0.8% decode, TTFT −1.5%** (prefill misses −1.7%).
- As region slots after candidate 1: **+0.7–2.0%**.
- **Your decision:** the smaller safety margin.

**Test:** add a Gh part to `ab_window.sh` (the 256 is now on line 220), then `B` against `B-Gh96`, three rounds, plus one soak watching the "settled at" line. `EXTRA_ARGS` works too, but it applies to every arm in the window.

**Dropped items worth revisiting:** "Splitting replicated weights" (q_a plus kv frees 16 slots a rank) and CWSR queue memory (106 MiB a card) were judged as static residents. As region slots, the split could net about +0.6–1.6% against the all-gather's −1.4%. Revisit both only after candidate 1 exists.

Everything is in `/home/as/.claude/jobs/635603c1/tmp/moreopt/card-vram/`:
- `region_replay.py`, `region_replay.out`
- `layer_model.py`, `layer_model.out`
- `mass_check.py`, `mass_check.out`
- `profile_check.py`, `profile_check.out`
- `segs.py`, `trace_peek.py`

### Finder's report

**Card lens: four candidates, ranked. The only big one is a request-adaptive VRAM region (about +2.5–4% on the held-out set, +5–8% on chat-style prompts).** I also measured the code-compression idea on the real weights, and it is dead.

Baseline numbers (best arm, ab-20261029T161035Z, Pm56-Pe-r2, rank 0, 481 decode tokens):
- **Token time:** 57.2 ms.
- **Waiting for the counts:** 32.2 ms a token. This includes the attention phase and other ranks' misses.
- **Waiting for CPU rows:** about 7.9 ms a token (199 µs a layer). I got this by taking the 6552.7 ms of gather waits and removing prefill's share: 920 prefill layers × 2.97 ms each, a cost taken from the Ulong arm. On the chat prompts (Uchat256 arm) the same estimate is 400 µs a layer, about 16 ms a token.
- **Synchronous miss reads:** about 4.9 ms a token. Read-ahead units finishing while their layer waits add about 1.5 ms.
- **Attention chain:** in the profiled arm (ab-20261029T154954Z Ds0-o, plain layer 4), 20 dependent launches run from the ffn all-reduce to the scatter. They take 306 µs of op time plus 20 gaps of about 3.6 µs, so roughly 0.3–0.38 ms a layer.

## Ranked candidates

### 1. Request-adaptive VRAM region: recently used experts kept on the card (exact; large effort)
- **Mechanism:**
  - Take d of each rank's static residents (d = 32–64) and use them as a recency-managed region.
  - When the CPU computes a cached unit for the second time within a ghost list, copy its packed RAM-cache slot into a free region slot. The copy goes through the existing SDMA lane (`aql.cpp`: 28.7 GB/s "at no cost to the kernels"; the link gate yields every 1 MiB).
  - Once the copy has landed, point the unit's gate_up and down table entries at the slot, in stream order, with vpack's bit-0 mark (vpack is merged in 5ad108c6; its slot format is the RAM cache's own packed form). `hx_early_decide` then leaves the unit out of the CPU's share.
  - Evict by LRU: unpublish the unit, then let the copy into that slot wait on an event behind the last kernel that reads it.
  - Keep touching the RAM-cache LRU on region hits. Otherwise a unit leaving the region can turn into a drive miss.
  - The `xprefetch` code (card-side `xp_logits`/`xp_plan`/`xp_step`, compare-and-swap table redirects; off under `--deterministic`) shows the redirect pattern to copy.
  - Under the same switch, publish a null entry for CPU-owned units so the card stops computing their stray rows. That also frees the two 35.86 MiB file buffers.
- **Gain** (`vram_dyn2.py` replay of trace ab-20261029T115523Z, 40 validation prompts; FP8 + vpack emulated by moving each rank's 141 most-used units to VRAM):

  | Prompts, d a rank | Region hits / token (4 ranks) | Same slots kept static | Net CPU units saved | Fills / token |
  |---|---|---|---|---|
  | Held-out, d=64 | 18.9 | 6.9 | 12.0 (of ~80) | 21.8 (409 MB) |
  | Held-out, d=32 | 12.3 | 3.6 | 8.7 | 18.6 |
  | Chat, d=64 | 36.1 | 10.9 | 25.2 (of ~110) | 27.6 |

  - A copy has a whole token to land. A one-token copy lag is needed: a two-token lag halves the hits.
  - The fills come to about 1.8 GB/s a card, 11% of cards 2/3's 16 GB/s, and about 7 GB/s of DRAM.
  - Arithmetic, held-out: 12.0 units × 0.20 ms (mrbench, 94 GB/s) to 0.38 ms (packed pool bench) = 2.4–4.6 ms less pool work a token. That fits inside the measured 7.9 ms wait: 57.2 → about 54.8 ms, +4.4% at most. Discounted for the card's longer MoE chain and copy interference: **+2.5–4%**.
  - Chat: 25.2 units × 0.2 ms = 5 ms of about 63 ms, against 16 ms of waits: **+5–8%**.
- **Risks:**
  - Stream-order hazards around table flips under early issue.
  - The card's routed chain may become the long pole (profiled at 173 + 109 µs for gate_up and down).
  - Creating the SDMA queue may cost something (the docs record 13 ms a step for a fourth *compute* queue); measure it.
  - Copies landing during pool bursts compete for DRAM; hold the link gate around the blit, gather and all-reduce.
- **Test:**
  - Before building, run one `x` arm on the current best config and rerun the replay. Build only if the net gain at d=64 is at least 8 units a token.
  - Window arms: `B-Vp` against `B-Vp-Vd64` and `B-Vp-Vd32`, three interleaved rounds, plus the Uchat256 arm for both, with `Kh` on every arm. Here B = `all-t40-f-M16-H3-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe`.
  - Add a verify mode in which the CPU also computes region units and compares rows.
  - New counters: hits, fills, late fills, evictions, link-hold time.
- **Not in the excluded list:** "packed VRAM residents" is static capacity. The engine survey mentioned "background promotion into VRAM" only as low-confidence and deferred. The new evidence is the replay: recency-managed slots catch 2.7× (held-out) to 3.3× (chat) more accesses than the same slots in the static plan.

### 2. Measure the attention chain with `Ct16`, then fuse its serial launches (exact; medium effort)
- **Step 0 needs no code.** A `B-Ct16` arm (the part is already in main) records card-0 start and end stamps. No `Ct` arm has ever run. `chain_report.py` gives, per layer type, the chain's wall time, busy time, summed gaps and critical path. It also gives the card's routed chain against the CPU-row blit, which item 1 depends on.
- **Fusions** (designed in IMPL-CHAIN, none built; each must byte-compare on the card):
  - router + topk + scatter
  - q_rope into q_b's epilogue
  - o_f into the attention epilogue
  - q_an into q_a through a last-block finisher
  - moe_gather + ffn all-reduce, a ds41 form of libr4d's `ar_gather_hc_write_4rank`
- **Gain:**
  - Six fewer launches a layer at about 6–7 µs each unprofiled (3.6 µs gap plus about 3 µs launch floor) = 36–42 µs a layer, 1.4–1.7 ms a token, +2.5–3% at most.
  - Discounted to **+1–2.5%**, because live-chain overlap freed about 6 attention-phase launches a layer and measured +0.2% ± 1%.
- **Test:** build only if `Ct16` shows at least 60 µs a plain layer of gaps plus small kernels. Then `B` against `B-Fu`, three rounds, with `Kh`.
- **Why not excluded:** only the overlap is excluded; fusion removes launches instead of reordering them.

### 3. Run the read-ahead's router predictor on the card (exact, hints only; medium effort)
- **Mechanism:**
  - Today the CPU predictor scores each prediction in 597.2 µs (10.9 s a run on one thread). It is posted after `hx_submit`, so the read-ahead for layer L+2 starts about 0.6 ms after layer L's counts.
  - On the card, an `xp_logits`-style GEMV of L+2's router on lane 1 (no fourth queue), plus top-4 into a pinned mailbox, would be ready about 50 µs before L's counts.
- **Gain:**
  - About +0.6 ms of lead on an effective lead of about 2.3 ms.
  - The 280–306 used read-ahead units a rank currently finish while their layer waits 2.42–2.69 ms each. At 0.3–0.6 ms saved each: about 1150 × 0.45 ms / 481 = 1.1 ms a token, **+1–2%**.
  - It also frees about 25% of a core and 1.7 GB/s of DRAM. Rank 0's choose time (8.5 vs 4.3 µs on the other ranks) is this post.
- **Test:** `B` against `B-Cp`, three rounds. Watch the "finished while the layer waited X ms each" line and the router-predictor scoring time.
- **Why not excluded:** read-ahead in pieces is adopted. This changes only where its predictions come from.

### 4. Lower `--gpu-headroom-mib` from 256 to 96 (exact; your decision on the safety margin; nearly free)
- **Basis:** every card settles at 301.83 MiB free against a 291.38 MiB target. 160 MiB is about 9 packed residents a rank.
- **Gain:** **+0.35–0.7%**. The low end is the static-extension replay (6.9 units a token per 64 slots); the high end is FP8's measured 0.077% per resident.
- **Risk:** running out of VRAM if scratch or code memory grows later.
- **Test:** add part `Gh<N>` (ab_window.sh hard-codes 256 at line 154). Run `B` against `B-Gh96` three times, plus one soak watching the "settled at" line.

## Checked and dropped
- **Lossless compression of resident experts' codes:** measured on 24 weights across layers 1–33 (`fp4_entropy.out`). H = 3.895 bits per code, 3.882 with byte context, so 2.9% is the theoretical ceiling. A block-decodable static code averages 125.7 of 128 bits a block, which is about zero.
- **FP8 or BF16 forms of the LM head and the mHC fn weights:** the release head is genuine BF16 (low mantissa bits uniform), and the 0 of 491,520 F32 `hc_*_fn` values have a zero low half. No exact narrower form exists.
- **Splitting replicated weights:** wq_a, wkv, indexer q_b, routers and hc fn total about 840 MiB a card. Splitting them buys about 16 residents (+1.1%) but costs an all-gather a layer (about 20 µs × 40 = −1.4%), so roughly zero net.
- **Zero-copy or reactive card compute of RAM hits:** DRAM is the shared ceiling (about 100 GB/s, and the pool already streams 94).
- **One-ahead predictive staging alone:** cards 2/3 move about 4.8 MB in the roughly 300 µs attention window, below the half unit a layer it would need.
- **Not exact or not a lever:** expert-to-rank re-partitioning changes the sum grouping, so it is not exact. Rank 0's shorter counts wait is the shared pool's FIFO order, not a lever.
- **System settings:** CWSR queue memory (106 MiB a card) is a system-wide setting and about +0.4%. The 304 W vs 317 W power caps don't matter while decode is latency-bound.

All scripts and outputs are in `/home/as/.claude/jobs/635603c1/tmp/moreopt/cards/`:
- `vram_dyn.py`, `vram_dyn2.py` (region replays)
- `fp4_entropy.py`, `fp4_entropy.out` (code entropy)
- `dtype_probe.py` (head and mHC weight forms)
- `optab.py`, `optab_detail.py`, `layer_chain.py` (decode-only per-op tables from the profiled arm)

## Lens: serving

### Verified

**Serving lens: adversarial verification of the 6 candidates**

Five of the six candidates survive. Two of the survivors need corrections: #1 has a conflict the finder missed, and #2's mirror-share half was already proposed but never run. #3 and #4 are not new. #5 should be dropped, because it brings back a wasted pass that a standing queue currently suppresses.

How I checked:
- I reran the finder's scripts s1, s2, s2b and s3; every number reproduces.
- I re-derived the counters from windows ab-20261029T083552Z, T153750Z, T124059Z and T164814Z.
- I read the code at HEAD d9fc54fd. Another agent is editing scheduler.cpp and consumer_skip.h in the main tree right now, so line numbers drift. The finder's engine.cpp:791 is now :825, scheduler.cpp:2249 is about :2423, and ab_window.sh:42 is :58.
- I ran three single-thread checks (`v2_mirror.py`, `v3_lanes.py`, `v6_calib.py`). There was no GPU use, no build, and nothing written outside `/home/as/.claude/jobs/635603c1/tmp/moreopt/serving/verify/`.

## Verdicts

**#1 Prompt lookup with issue-ahead kept: REVISE (exact). The cost attribution is new; prompt lookup itself is not.**
- **Mechanism confirmed.**
  - `--spec-ngram 3` sets n_spec=3 (config.cpp:1122). `target_pipelining()` then returns false (draft_experiment.h:36), and `issue_ahead` returns at once (engine.cpp:754).
  - In T083552Z the one-token router waits are identical in both arms: 25.09 s over 19,240 layers against 24.99 s over 19,160, i.e. 1.304 ms a layer either way. So the extra time sits between steps.
  - The spec arms' larger KV pool (772 against 680 tokens) left VRAM placement unchanged, so it is not a confound.
- **Correction: the loss is larger than the finder's 0.7–1.3 ms a step.**
  - Raw difference: 85.16 against 83.91 ms a step.
  - The one verify was a 2-row pass (draft_positions reached 1, accepted 0), worth about 0.09 ms a step, not 0.3.
  - The base arm also carries heldout-05's post-EOS pass (481 against 480 steps). That pass costs 0.06–0.29 s, and heldout-05 ran +11% in both spec arms for that reason.
  - Lost issue-ahead is therefore 1.25 + (0.12 to 0.60) − 0.09 ≈ **1.3–1.8 ms a step**. On a per-request basis, the other seven prompts lost 1.9–2.2%. The finder's 0.7–1.3 ms is low.
- **Missed conflict.**
  - scheduler.cpp:324 refuses the exact consumer-suffix skip (Cs) whenever `n_spec != 0`.
  - Turning on `--spec-ngram` therefore turns off Cs, which models at −10% / −25% / −29% TTFT at 933 / 1,890 / 3,086 tokens.
  - The fix needs a lookup-only exemption. It is plausible: there is no drafter context, and verify steps come after the prompt. The consumer-skip proof has to be extended to cover it.
- **Wrong figure.** The "≈2× on a code edit" in lookup.h comes from a regime where a verify row costs 0.06 of a step. Here a row costs about 0.5: 2 / 3 / 4-row verifies cost 1.53 / 1.99 / 2.41 single steps (RESEARCH-3-SPECDEC).
- **Corrected gain on copy spans.**
  - Formula: (1 + a + … + a^k) ÷ verify cost, then −2 to −3% for running unpipelined during spans.
  - k=2 gives 1.19 / 1.32 / 1.40× and k=3 gives 1.19 / 1.39 / 1.50× at acceptance a = 0.8 / 0.9 / 0.95.
  - On a whole reply with 60–80% copied tokens: **+10% to +35%**.
  - On every measured receipt the gain is **0**: one proposal reached in 480 steps.
- **Design note.** The lookup needs the committed token, so a proposal can only start one step late under issue-ahead. An alternative is to build on the Ss peek (the post-stop switch), which reads the token before the next issue.

**#2 Multi-row decode passes get the decode settings: KEEP. Part (a) is not new; part (b) is new.**
- **Code confirmed.**
  - The predictor post and the read-ahead require `pass_tok_ == 1` (stager.cpp:809, 873, 881, 1603, 1680).
  - The mirror tail is chosen by `rows >= min_rows` (mover.cpp:989).
- **Counters confirmed** (N=4 arms in T153750Z):
  - "0 pieces read" and "0 inputs posted".
  - Rank 0 decode miss layers ≈ 1,093 of 5,040 (21.7%), from 1,646 drive-reading layers minus about 553 in prefill.
  - NVMe busy 50% / 52%.
- **(a) is the RESEARCH-3 concurrency assessment's item 3.** It proposed the in-flight rule at +1.5–3.5% and an Rm5 arm with a "≥2%" gate. That arm was never run.
- **(a) recomputed** with the assessment's model on the FP8 world (`v2_mirror.py`):
  - A lone-miss penalty of 0.3–0.5 ms reproduces the measured batched arms: the model gives 18.1 / 17.7 at N=2 and 20.6 / 20.3 at N=4, against 17.6 and 20.0–20.2 measured.
  - Removing that penalty (the in-flight rule) gives +3.4–5.7% at N=2 and +2.2–3.6% at N=4.
  - Full exposure is the upper bound: the 0.18 share costs 3.6 ms a lone miss against 2.8 ms at 0.13, which gives +9.2% / +5.8%.
  - Rm5 gives slightly less than the in-flight rule, because it also puts layers with two or more misses at 0.13 (about one third of miss layers at N=4).
  - **Corrected: Rm5 +1.5% to +5%; in-flight rule +2% to +6%.**
- **(b) is uncertain.**
  - In its favour: at N=4 the two-layer lead is 9.5 ms, against about 2.8 ms at N=1, so a used unit could land whole.
  - Against it: four rows mean about four times the planned units, i.e. roughly 2.7 GB/s of wasted reads on top of about 4 GB/s of demand. Pieces also never start while a demand read is in flight, and 62% of N=4 layers have one.
  - **Corrected: 0 to +3%.** Gate it on a shadow count before building.

**#3 Two-lane decode pipelining: KEEP as the long-term item (exact, L effort). Not new.** It is RESEARCH-3's item 4, and the assessment's recommendation 3.
- **Model rerun** (`v3_lanes.py`, FP8 world):
  - Batched N=1 / 2 / 4 with the assessment's extra cost terms (called B′ there): 15.6 / 18.1 / 20.6. Measured: 15.8–16.7 / 17.6 / 20.0–20.2. That is within 3%, so the assessment's ±10% gate for going ahead is met.
  - The lane simulation gives 2×1 lanes 24.4 and 2×2 lanes 25.9. These omit the B′ terms; with them, 2×1 is about 23.2–23.7.
  - At 60–80% realisation: **≈20–23.5 tok/s aggregate and 10–11.8 per stream**. That is +14% to +34% over batched N=2 and 0 to +17% over batched N=4.
- **New feasibility caveat.**
  - The engine already runs three queues a card: compute, lane 1 and the mover's (mover.h:407). A fourth queue costs 13 ms a step.
  - Lane B would therefore share lane 1 with the on-die attention prefetch (deepseek41.cpp:2732–2749, Pm56/Pe, about +4% at N=1), or the prefetch must move to the mover's stream.

**#4 Lossy decoder replay: KEEP (opt-in lossy). Not new.** It is PREFILL-ENGINE-SURVEY idea 3, with the same figures: "−30% at 1K, −24% at 3K on top of #1".
- **The replay reproduces exactly:**

  | Prompt tokens | Exact skip | Lossy K=9 | Change vs exact skip |
  |---|---|---|---|
  | 272 | 10.7 s | 8.8 s | −18% |
  | 933 | 27.7 s | 19.0 s | −31% |
  | 1,890 | 51.2 s | 35.7 s | −30% |
  | 3,086 | 70.9 s | 54.8 s | −23% |

- **Effort is M, not S.**
  - consumer_skip.h requires that no row ever reads a window key the request did not write.
  - A constant K-suffix makes the first ~127 suffix rows of layers 21–39 read unwritten slots. It therefore needs a per-layer window floor in the attention op, which is DeepInfer's masking.
  - Decode needs no mask, since K·16 ≥ 144.
- **Schedule fix.** Take the union with the exact triangle. As written, K=9 loses the exact skip's −2.7% at 82 tokens, and K=17 loses its −3.3% at 272 tokens.
- **Prerequisite.** Cs is merged but not yet measured.

**#5 Keep issuing ahead while a queue stands: DROP.**
- **Missed cost.**
  - Today a standing queue also suppresses the post-EOS pass, because nothing is issued ahead while a request waits. The queued request is admitted at the EOS commit.
  - Re-enabling issue-ahead brings that pass back (57 ms warm, up to 0.29 s with cold EOS experts) in front of the queued request.
- **Net effect.**
  - N=1: 63 × 1.3–1.8 ms gained against 57–290 ms lost per natural stop. That nets −6% to +3% per 64-token request and 0 to +3% for 256-token requests.
  - N=4 with more clients than slots: +0.2–0.8%.
- Below the bar. Revisit only alongside Ss.

**#6 Expert profile from served traffic: KEEP, weak (exact; needs a user decision).**
- **Reproduced.**
  - Chat: 141.9 file-tier selections a token (card share 40.9%), 15.78 misses a token.
  - Held-out: 109.0 (card share 54.6%), 13.68 misses.
  - Two-fold split, weight 1: the held-out chat half gains +7.60 / +4.67 selections a token; the in-family prompts lose 2.92 / 5.02.
  - The learning curve is noisy: means of +1.6 to +6.9, single splits from −5.2 to +21.8.
- **Unsupported claim.** "~100 RAM-cache slots freed, −7% misses" comes from no script; s2 counts card selections only. Drop it.
- **Calibration** (`v6_calib.py`): FP8 moved the same window (10.55 held-out selections a token) onto the cards for about 5 ms a token, i.e. about 0.45 ms per selection a token. The chat-versus-held-out gap implies about 0.23 ms.
- **Corrected: +0.5% to +5% on out-of-family traffic, −1% to −5% in-family.**
- **Confound.** Chat prompts are 14–28 tokens against 35–46, so fewer of their experts are warmed by prefill. The held-out repeats ran equal to round 1, so there is no warm-repeat bias.

**Configuration notes.** These check out: the 8K-context arm placed 710 against 712 VRAM units and an arena of 11.31 against 7.09 MiB, consistent with about 3.6 KiB a token. One addition: the indexer's width is max_ctx/ratio + 1 (deepseek41.cpp:1575), so a very large `--max-model-len` also adds a small amount of indexer work every step.

## Surviving candidates, ranked

| # | Candidate | Label | Corrected gain | One-line test plan |
|---|---|---|---|---|
| 1 | #2 multi-row decode passes get the decode settings | Exact; (a) no code, (b) 1–2 days | (a) +1.5–5% (Rm5) or +2–6% (in-flight rule), N=2–4; (b) 0 to +3% | One window: B-Pf-N4-C4, -Rm5, -Rm5-A5, B-Pf-N2-C2, -Rm5, ×2 interleaved, logprobs equal to the serial arm, gate Rm5 ≥ +2%. Build (b) only after a shadow arm (multi-row predictions posted, read-ahead off) shows ≥25% of misses predicted and readers not blocked. |
| 2 | #1 prompt lookup, issue-ahead kept while no long match is live | Exact; S–M, plus a Cs exemption (S) | 0 on held-out and chat; +10% to +35% on copy-heavy replies (+14% to +51% on spans) | Patched build: B and B-Sg3-Rm5 ×2 on held-out (within ±0.5% of B) and on a new Ucopy receipt at 256 tokens (≥ +10%), replies and logprobs identical, Cs off or exempted in all arms. |
| 3 | #3 two decode lanes | Exact; L effort; only if multi-client use is real | ≈20–23.5 aggregate, 10–11.8 per stream (+14–34% over batched N=2, 0–17% over N=4) | Design spike that settles lane 1 versus the prefetch, then a HOST-backend two-lane identity test, then 2×1 against B-Pf-N2-C2 and B-Pf-N4-C4, logprobs equal, gate ≥ +15% over N=2. |
| 4 | #4 lossy decoder replay, K=9, unioned with the exact triangle | Opt-in lossy; M effort | TTFT −18 / −31 / −30 / −23% against the exact skip at 272 / 933 / 1,890 / 3,086 tokens | After a Cs arm validates the exact skip: K=9 and K=17 arms on the long receipt (TTFT against Cs) plus Lq teacher-forced KL on prompts of 1K tokens or more. |
| 5 | #6 re-profile from served traffic | Exact; needs a user decision | +0.5–5% out-of-family, −1 to −5% in-family | Rt trace of real traffic (≥2K decode tokens), profile = sealed + traffic, make PROFILE overridable at ab_window.sh:58, A/B on a mixed receipt, gate ≥ +2% on the mix. |

Dropped: #5.

## Cross-cutting findings for the parent
1. **Check before adopting Ss (skip the post-EOS pass, in A/B).** The Sg3 counters put the cost of turning issue-ahead off at 1.3–1.8 ms a step: 1.5–2.2% at 84 ms, and 2–3% at today's 57 ms if it stays constant. Ss reads every one-sequence step back before issuing the next (`outcome_decides_next`), and its design assumed this costs "tens of µs". Its real cost lies between those two bounds, so read its shutdown turnaround counter before adopting it. Ss is worth +0.1–0.9% on held-out.
2. **Any speculation turns the consumer skip off.** Prompt lookup and DSpark both trip `n_spec != 0` (scheduler.cpp:324). Any speculation arm should be compared with Cs off, or the refusal exempted for lookup-only speculation.

Files are in `/home/as/.claude/jobs/635603c1/tmp/moreopt/serving/verify/`:
- v_s1.py
- v_s1_out.json
- v_s2_reprofile.py
- v_s2b_curve.py
- v_s3_lossy_replay.py
- v3_lanes.py
- v2_mirror.py
- v6_calib.py

### Finder's report

**Serving and workload lens: six ranked candidates for ds41**

Only this lens is covered. I used no GPU, ran no build and changed no file outside my scratch directory. Gains are against the current best configuration (B = `all-t40-f-M16-H-p256-m98304-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe`). Each estimate is labelled measured, replayed or modelled.

## Ranked candidates

### 1. Prompt lookup on copy-heavy replies, keeping the next step issued ahead (exact)

- **Mechanism.**
  - Passing `--spec-ngram K` sets `n_spec=K`. `DraftExperiment::target_pipelining()` (`core/sched/draft_experiment.h:36`, used at `core/engine.cpp:791`) then turns the next-step issue-ahead off for the whole run, because prompt lookup is neither serial nor block drafting.
  - So prompt lookup costs time even on prose, where its gate verifies nothing.
  - Fix: keep the issue-ahead on when lookup is the only drafter. Skip it only while a sequence holds a live long-match proposal.
  - Then enable `--spec-ngram 3` with `Rm5`, so that verify passes of 4 rows or fewer use the decode mirror share.
- **Measured cost today** (ab-20261029T083552Z, `step_ns_decode / steps_decode`):
  - Base: 83.7 and 84.1 ms a decode step. Sg3: 85.3 ms. Sga3: 85.0 ms. That is −1.2% to −1.9% (decode 12.09 vs 11.92 and 11.95 tok/s).
  - In each spec arm, 1 proposal was verified and 0 accepted. That verify accounts for about 0.3 ms a step, so about 0.7–1.3 ms a step is the lost issue-ahead.
- **Gain on copy spans** (modelled):
  - Verify passes cost 1.56, 2.03 and 2.46 single steps for 2, 3 and 4 rows. Source: the DSpark window, with the spec assessment's corrected pass counts and the draft time removed.
  - Tokens per pass are Σ aʲ for j = 0..k. At per-position acceptance a = 0.8, 0.9 and 0.95:
    - k=2: 1.16× / 1.29× / 1.36×
    - k=3: 1.18× / 1.38× / 1.48×
  - Copied tokens route to experts the prompt's prefill just put in the RAM cache, so the upper end is plausible.
  - Elsewhere the gain becomes about 0 instead of −1.5%. `lookup.h` reports a replay on 24 real replies: 0.999–1.014× on prose, and about 2× on a code edit.
  - Which share of real traffic is copying (tool-call JSON, code edits, quoted text) is your call.
- **Effort and risk.** Effort is S–M: a scheduler predicate plus the lookup/issue-ahead hand-off. Risk is that this touches upstream scheduler logic; output is identical by construction, because lookup only chooses what gets verified.
- **One window.**
  - Arms `B`, `B-Sg3-Rm5`, `B-r2`, `B-Sg3-Rm5-r2` (patched binary). Run them on the held-out receipt and on a new `Ucopy` receipt (8 prompts that return edited code, JSON or text of 150–300 tokens) with `AB_MAX_TOKENS=256`.
  - Read the stats JSON `draft_positions` and step times.
  - Gates: held-out within ±0.5% of B; `Ucopy` at least +15%; replies identical.
- **Why it is new.** DSpark is on the excluded list; prompt lookup is not. RESEARCH-2 dropped it only on held-out prompts, where 0 drafts were accepted. Nobody traced the cost to the pipelining switch.

### 2. Batched decode passes get the decode-only settings (exact)

- **Mechanism.**
  - These features are keyed on `pass_tok_ == 1`: read-ahead in pieces (`stager.cpp:1274`) and the predictor post (`:795`, `:851`, `:1626`).
  - The mirror tail is chosen by rows ≥ `min_rows` (`mover.cpp:989`).
  - So a 2–4-row decode pass gets no predictor, no read-ahead, and the prefill mirror share of 0.18. Both batched sequences under `Pf` and speculative verify passes are affected.
  - Measured in the N=4 arms (ab-20261029T153750Z): "0 pieces read", "router predictor: 0 inputs posted", and miss reads of 2.38–2.61 ms a batch weight against 2.23–2.54 at N=1.
  - Fix (a), no code: `Rm5` (optionally with `A5`) for N≤4.
  - Fix (b), code: post all rows to the predictor, plan the union of their predictions, gate the read-ahead on "pure decode pass of 4 rows or fewer", and pick the mirror share by misses in flight (2 or more → 0.18) instead of by rows.
- **Gain** (modelled, about +3% to +6% aggregate at N=2–4):
  - (a) +1.5% to +3.5%.
    - At N=1, a 0.18 share cost about 4–8% on lone misses: −4% against no mirror (02:28), against +3.7% for 0.13.
    - Arithmetic per lone miss: 3.23 MiB on SATA ≈ 3.6 ms, against max(2.59, 2.75) = 2.8 ms at 0.13.
    - At N=4: 1,087 of 5,040 decode layer-passes miss on rank 0 (21.6%), so about 62% of layers stall on some rank, 67% of them on one rank alone.
  - (b) +2% to +3%. Pp4 measured +3.6% at N=1. The NVMe is only 50–52% busy at N=4 without read-ahead.
- **Effort and risk.** (a) none; (b) 1–2 days. Risk is low (read hints only). The predictor costs 597 µs a prediction on its own thread; even with four rows it stays well under N=4's 4.75 ms layer interval.
- **One window.** Arms `B-Pf-N4-C4`, `B-Pf-N4-C4-Rm5-A5`, `B-Pf-N2-C2`, `B-Pf-N2-C2-Rm3-A3`, ×2 interleaved, then (b)'s build. The log must show pieces > 0 and predictions > 0. Replies must be identical.
- **Why it is new.** Only the SCHED_PREFILL_ALONE switch is on the excluded list, and it was measured with these features silently off. Mirror split and miss post are N=1 tail changes.

### 3. Two-lane decode pipelining for two or more clients (exact, L)

- **Mechanism.**
  - Two sequences run ordinary 1-row (or ≤16-row) passes on two issue lanes, offset by about half a layer.
  - One lane's CPU rows and drive misses then overlap the other lane's card work.
  - Collectives are issued in a fixed lane order on every rank, and no fourth hardware queue is added (a fourth costs about 13 ms a step).
- **Data.**
  - Measured: batched N=2 is 17.6 tok/s aggregate and N=4 is 20.0–20.2 (4.4 a stream), against 15.8–16.7 at N=1.
  - The concurrency assessment's simulation (tbo_sim, FP8 calibration) puts 2×1 lanes at 24.4 aggregate (12.2 a stream) and 2×2 at 25.9. The same model predicted batched N=4 at 20.6 (measured 20.0–20.2).
- **Gain.** Using the assessor's 60–80% realisation of the simulated gain: 20.9–22.6 aggregate and 10.5–11.3 a stream.
  - Against batched N=2: +19% to +28% aggregate.
  - Against batched N=4: +4% to +13% aggregate, at 2.4–2.6× the per-stream rate.
- **Effort and risk.** Effort is L (2–3 weeks). Risk: collective ordering and deadlock; the CPU pool and drive are shared.
- **One window.** First gate on a HOST-backend identity test. Then run `B-Pf-N2-C2` against the two-lane arm, and `B-Pf-N4-C4` against 2×2, with logprobs equal to the serial arm.
- **Why it is new.** RESEARCH-3 ranked it "strongest long-term" but nobody designed or built it.

### 4. Opt-in lossy decoder replay on top of the exact consumer-suffix skip (TTFT)

- **Mechanism.**
  - Layers 0–20 and layer 20's compressed and index keys still run on every pass, so global attention stays exact.
  - Layers 21–39 run only on the prompt's last K passes; K=9 is DeepInfer's KEEP_TOKENS=140, and Strata's DS4_YOCO=2 is the same idea.
  - The deep layers' 128-row windows start empty at the kept suffix. That is the approximation.
- **Replayed TTFT** (`s3_lossy_replay.py`, prefill-replay's validated cache model and law L2). The exact-skip column reproduces PREFILL-REPLAY's −2.7, −3.3, −10.4, −24.8 and −28.6%.

  | Prompt (tokens) | Today | Exact skip | Lossy K=9 | Change vs exact skip |
  |---|---|---|---|---|
  | 272 | 11.0 s | 10.7 s | 8.8 s | −18% |
  | 933 | 30.9 s | 27.7 s | 19.0 s | −31% |
  | 1,890 | 68.1 s | 51.2 s | 35.7 s | −30% |
  | 3,086 | 99.2 s | 70.9 s | 54.8 s | −23% |

  With K=17, against today: −32%, −44% and −42.5% at 933, 1,890 and 3,086 tokens.
- **Quality risk: moderate.** DeepInfer calls it approximate; Strata says it "shifts the answers a little".
  - Gate it with the lossy KL harness on prompts of 1K tokens or more, plus long-context checks.
  - Apply it only above a length threshold, and never write its KV into the prefix cache.
- **Effort.** S, since the exact skip has been reported done (task #55; I did not check the code).
- **One window.** The long receipt with exact skip, K=9 and K=17, plus forced-decode KL against the exact replies.
- **Why it is new.** Only the exact skip is on the excluded list. The lossy truncation appeared only in the survey's lossy table.

### 5. Keep issuing ahead while requests wait but none can be admitted (exact, S)

- **Mechanism.**
  - `step_next_locked` returns nothing whenever any request waits (`scheduler.cpp:2249`).
  - When every `--max-num-seqs` slot is busy, or KV is full, the full step cannot admit anyway, so it can issue ahead.
- **Gain.** The lost issue-ahead is 0.7–1.3 ms a step (from item 1's data).
  - At N=1 with a queue (57 ms steps): +1.2% to +2.3% while the queue stands.
  - At N=4 (190 ms steps): +0.4% to +0.7%.
  - Cost: a waiting request can be admitted up to one step later.
- **One window.** `B-N1-C2` and `B-Pf-N4-C8`, base against patched; compare step time and identity.
- **Why it is new.** The concurrency assessment noted the 1–3 ms a step loss but proposed no fix.

### 6. Expert profile from the traffic actually served (exact, your decision, weak)

- **Why it matters.** The sealed profile is 721 decode tokens from 16 prompts in 8 domains, the same families as the held-out set.
  - 40-prompt trace, same placement (`s1_tiers_by_prompt.py`): chat decode has 40.9% of selections on the card and 15.8 misses a token; held-out has 54.6% and 13.7.
  - 12:41 validation: chat ran at 13.56 tok/s against 15.35 for held-out (−11.7%).
- **Replay** (`s2_reprofile.py`, `s2b_curve.py`). Re-ranking today's last 105–114 card slots a rank with one half of the chat prompts' counts:
  - The other half of the chat prompts gains +3.0 to +6.1 selections a token on the card (out of sample).
  - The in-family prompts lose 3.5 to 5.0 selections a token.
  - About 100 RAM-cache slots a rank are also freed (about −7% misses).
  - Net: roughly +2% to +5% on out-of-family traffic, and about the same loss in-family.
- **Effort.** S: an `Rt` arm (now merged), `rtrace.py`, then a new profile. `ab_window.sh` hard-codes `PROFILE` (line 42), so making it `${PROFILE:-…}` is a one-line change.

## Configuration notes (not optimizations)

- **KV for N sequences.** The pool backs one `--max-model-len` sequence, so overflow at N>1 means preemption by recompute (about 35 ms a re-prefilled token).
  - Cost of sizing up: about 3.6 KiB a context token a card (Lm8192 against Lm512: KV +23.4 MiB, arena +4.2 MiB).
  - 4 × 8K tokens ≈ −0.35% decode; 128K ≈ −1.3%; the model's 1M maximum ≈ −10%.
  - The container's own default is 512.
- **Interleave and N.** Use `--prefill-interleave 1` with `Pf` so a long prompt cannot stall all decoders for its whole prefill. N=3 gives 84% of N=4's aggregate gain at 6.0–6.6 tok/s a stream (assessment).
- **Benchmark representativeness.** The held-out prompts are in the profile's own families, so they overstate general traffic by about 12%. Back serving claims with a mixed receipt.

## Dropped (below the bar, or no data to support them)

- **Sampling:** device-side, about 10 µs a step per a comment in `sampler.cpp`; nothing to gain.
- **Cold Engram rows on new text:** 0.4–0.9% (RESEARCH-2), and they are already prefetched just before each lookup.
- **FP8 LM head (lossy):** `logits_gemm` takes 525 µs a call, so it would save about 0.45%.
- **Mixed steps:** throughput-neutral by arithmetic (12 prompt tokens + 4 decode rows ≈ 12/16 of a pass + one N=4 step).
- **Online VRAM re-placement:** item 6's replay bounds its value, and runtime tier changes under a static plan are L effort.
- **Two TP2 replicas:** half the VRAM and RAM each puts far more misses on the one drive.
- **Lazy or parallel warm-up:** would save about 15 s of the 46–50 s startup. It only matters if the GPUs are swapped between QFN and ds41.

Scripts and output are in `/home/as/.claude/jobs/635603c1/tmp/moreopt/serving/`: `s1_tiers_by_prompt.py`, `s1_out.json`, `s2_reprofile.py`, `s2b_curve.py` and `s3_lossy_replay.py`. All are single-thread; `s3` reads the prefill-replay modules with `-B`, so nothing was written there.

