# Prefill and speculative decoding: where the room is, and the plan (2026-11-04)

Synthesis of research workflow wof0gb00v (two investigators per topic, then skeptics), re-checked against the window logs
and the source. Read-only: nothing was built, served or run on a card. "Measured" means read from a log or JSON;
"est." means arithmetic, shown. Where a skeptic corrected an investigator, the corrected figure is used. Refuted
items are listed under "Dropped".

**Paths and names**
- `src` = `/home/as/radiance-ds41-claude` at `f84b23db` (read with `git show f84b23db:<path>`).
- `B` = `src/tools/ds41-host-experts/build`; `R` = this directory.
- Logs:
  - `CR2` = `B/ab-20261104T003545Z/all-…-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs-Kh-Lm131072-Ulong48k.server.log`;
  - `CR4` = its `M4096…Lc4096…Pci4096` twin;
  - `CR100` = the three `…Ulong100k` logs of the same window (2K with hashes, 2K without, 4K with);
  - `CP8` = `B/ab-20261103T221104Z/…-Rb-Vp-Cs-Kh-Lm8192-Ulong8k-r2.server.log`;
  - `W2K` = `B/ab-20261031T090154Z/…-Lc2048-P1-Qe64-Cs-Mb1-Kh-Lm32768-Uharness.server.log` (a warmed server);
  - `W4K` = `B/ab-20261031T162629Z/…-M4096-…-Lc4096-P1-Qe64-Mb1-Pci4096-Cs-Kh-Lm32768-Uharness.server.log`;
  - `F3q` = `B/ab-20261104T081812Z/…-Cs-Qa0-Pc-Pce-Pcg16-Kh-Lm32768-Uharness-r2.server.log` and its `.json`;
  - `QA` = `/home/as/ds41-claude-build/work/autonomous-20261101/stable_e184_Qa_B/window/`.
- Configurations and parts:
  - C2 = Lc2048 + P1 + Qe64 + Mb1 + Rb + Vp + Cs (the recommended config); C4 = the same at 4,096-token chunks (Pci4096).
  - `Ada` = `RADIANCE_EXPERIMENTAL_DS41_LCP_ADMIT=all`; `LiN` = `..._LCP_INSERTQ=N`; `Cf` = `..._CACHE_FILL=1`.
  - Dpp = DSpark with the frozen prefix policy `v1:0.938:0.975`. Ax = the R9700 expert worker.

## 1. Summary

**Prefill: there is a lot of room, and the first part costs no code.**
- **Long prompts mostly re-read the same experts.** Phase 1 (P1) reads the same experts from the drive in every chunk,
  while about 500 RAM-cache slots per rank sit empty.
  - At 48K / 2K chunks, rank 0 made 8,177 drive reads. At most 862 distinct units were ever needed: 3,216 planned
    file-tier units (CR2:74) minus 2,354 warmed (CR2:173), with 0 evictions (CR2:543).
  - 317 slots were still idle at the end (CR2:546).
  - Cause: P1 borrows the cache read-only and admits only the units the prompt's last 128 rows select
    (`src core/sched/scheduler.cpp:151-176`, `core/place/stager.cpp:5126-5131`, `core/place/card_expert_stager.cpp:1416-1418`).
- **The fix is an existing switch that has never run under P1.** `Ada` (ab_window.sh:2651) admits every unit P1 reads.
  - Est. at 48K: 115 → ~62-68 s at 2K chunks, 76.6 → ~58-63 s at 4K.
  - Est. 100K: 222 → ~130-140 s; 6,817 tokens: 17.1 → ~12-13 s.
  - It changes residency only, so replies stay byte-identical.
- **After that, prefill sits on a compute floor.** Warmed servers already show it:
  - 2,048-row × 20-layer chunks take 2.27-2.40 s (55-59 µs a row-layer);
  - 4,096 × 20 take 4.26-4.30 s (52 µs) (W2K:651-744, W4K:599-641).
  - About two thirds of the floor is the busiest rank's MoE. Its exact kernel runs at about 20% of the card's peak
    (§2.2), so most of the remaining room is there: a per-unit CPU/card split, the packed row at block speed, a
    row-batched exact kernel, and cross-rank balancing.
- **Layer-major (Phase 2) does not beat that floor with today's kernels.** At 192 GiB, `Ada` already delivers most
  of its drive saving. Defer it until the MoE kernel is about 2× faster (then 48K would take about 35-42 s).

**Speculative decoding: one real lever, refused by today's config.**
- **It helps one stream only.** A verify row costs about half a decode step, because every extra row brings about
  2.3-2.9 more CPU-path units a layer (F1 fit with the worker: T(R) = 15.5 + 14.5·R ms).
  - Even perfect drafts cap the gain near 2× at one stream, and near +13% at 8 streams.
- **The best measured gain is on the shelf but refused in C2.** In the decode-only config, Dpp gave code +20.7% and
  chat +8.3% (ab-20261031T165338Z against ab-20261030T205440Z's Rb+Vp plain arms).
  - Five separate checks refuse speculation under C2 (§3.3).
  - In C2 with the worker it is est. code +14..+22%, chat +2..+9% at one stream.
- **Prompt lookup measured −1.5%** (NIGHT-LOG.md:117). Most of that loss is that `--spec-ngram` turns issue-ahead off
  for the whole run (`src core/sched/draft_experiment.h:79-81`, `core/engine.cpp:1081`).
  - It pays only on copy-heavy agent output (est. +8..+13% decode there) and only after that fix.
- **Neither moves concurrent throughput**, the user's first priority. The online worker set (window F4) is the lever
  for that.

| # | Item | Est. gain | Effort | Exact by | First step |
|---|---|---|---|---|---|
| P-1 | `Ada` (+ `Li64`): P1 admits every unit it reads | Fresh server: 48K −41..−46% (2K) / −18..−24% (4K); 100K −37..−41% / −24..−27%; 6,817 −24..−29%; harness −4..−10%. Warm server −3..−8% | XS (config) | residency only | window F5 |
| P-2 | `Cf`: fill the free RAM-cache slots at startup (merge claude/ds41-cache-fill 8822ae0c) | Fresh server, with Ada: 6,817 → 10.2-10.8 s (−37..−40%); 3,086 → 5.5-6 s; harness −15..−25%. Steady state ~0 | S (merge) | residency only | F6 |
| P-3 | Per-unit split: P1's few-row units run on the idle CPU pool | Harness −5..−10%; 320-row passes −10..−20%; 2K chunks −3..−6% | M | same bytes on either side | build, then F7 |
| P-4 | Packed P1 row at block speed | −6..−8% at the floor | S-M | p1 device check | card checks, then F7 |
| P-5 | Exact row-batched P1 MoE kernel (Phase 3) | ×1.5: −12..−22% at the floor; ×2: −25..−33% | L | same per-output order | model-free card checks |
| S-1 | Dpp + worker in the decode-only profile (zero code) | Code +17..+24%, chat +3..+10% at one stream; single user only | XS (arms) | both already exact; add Axv with DSpark | F6 |
| S-2 | Dpp admitted in C2 | One stream: code +14..+22% (+18..+25% with in-pass snapshots), chat +2..+9%, held-out +9..+15%; harness wall −1..−3.5%; N4/N8 −2..−4.5% while the bank is loaded | M-L (5-8 days) | verify ≤ 3 rows in the decode band | after S-1's go/no-go |
| S-3 | Prompt lookup with issue-ahead kept, plus a v2 gate | ds41 harness 0..+2%; edit-heavy agent proxy +8..+13%; chat and held-out 0. No VRAM tax | M | greedy verification | only if the Uedit replay ≥ +8% |

## 2. Prefill

### 2.1 Measured anatomy (fresh servers, P1, prefill hashes on)

| Prompt / chunk | TTFT s | chunk walls s | tok/s | rank-0 drive units (incl. read-ahead) | rank-0 drive busy | ring-full waves | source |
|---|---|---|---|---|---|---|---|
| 6,817 / 2K | 17.13 | 17.02 | 398 | 1,470 (1,137 demand) | 89.0% | — | CP8:498-503, :536; `…r2.json` |
| 6,817 / 4K | 12.69 | 12.56 | 537 | — | — | — | ab-20261031T162629Z, without Rb/Vp (README.md:166) |
| 32,768 fill / 2K (a 6,817-token text repeated) | 64.19 | 63.03 | 511 | 5,239 (4,231 demand): the same text's units are re-read in every chunk | 85.9% | — | `…Ufill32k.server.log:557`; README.md:251 |
| 47,917 / 2K | 115.10 | 111.75 | 416 | 8,177 (6,958 demand, 1,219 dropped read-ahead) | 78.2% | 9,500 / 15,829 (60%) | CR2:496-521, :554 |
| 47,917 / 4K | 76.58 | 70.91 | 626 | 4,481 (3,968 demand) | 68.6% | 5,524 / 8,992 (61%) | CR4:496-508, :542 |
| 99,987 / 2K | 221.74 (224.48 hashes off) | 216.47 (223.87) | 451 | — | — | — | CR100 chunk lines (my parse); `…Ulong100k.json` |
| 99,987 / 4K | 165.10 | 159.00 | 606 | — | — | — | CR100 (4K) |

(— = not re-read for this note; the investigator's parser gives them in the prefill-anatomy report.)

**Chunks.**
- **Producer chunks (20 layers).**
  - At 2K they take 4.25-4.72 s with 318-348 rank-0 drive units each (CR2:497-520).
  - At 4K they take 5.22-7.64 s with 343-361 units (CR4:497-506).
- **The final chunk** runs 37 layers on the cards plus 2-3 cut layers on the CPU path, and reads 584-637 units:
  - 9.85 s at 48K/2K (CR2:521);
  - 9.03 s at 48K/4K (CR4:508);
  - 6.71 s at 6,817 tokens (CP8:503).
- **A simple time model reproduces the chunk walls:** wall ≈ max(compute floor, 4 × units × 18.8 MB ÷ drive rate).
  - The drive delivers 5.4-7.1 GB/s across the four ranks (the per-rank GB/s on the chunk lines, × 4).
  - Checks:

    | Case | Model | Measured |
    |---|---|---|
    | 48K/2K, 330 units, 5.6 GB/s | 4.4 s | 4.25-4.72 s |
    | 6,817 chunk 2, 296 units, 6.2 GB/s | 3.6 s | 3.57 s (CP8:499) |
    | Warm 6,817 chunk 2, 200 units | 2.5 s (≈ the floor) | 2.46 s (`QA/quiet_Qa8_first.server.log:522`) |

  - So the drive hides under compute up to about 170-190 units per 2K chunk. This model replaces the investigators'
    "F + 0.011 s a unit". That one implies a floor of F ≈ 0.8 s, which the floor chunks below contradict, so it
    overstated every large gain.
- **Read-ahead precision is about 68% at 48K/2K:** 3,792 posted, 2,573 used, 1,219 dropped (CR2:554).
- **Where the waits go.** At 48K/2K rank 0's issuing thread waited 37.9 s for drive units (CR2:554). At 4K the wait
  moves to the ready slots: 18.0 s (CR4:542).

**Re-reads.**
- Rank 0's 8,177 drive reads at 48K/2K cover at most 862 distinct units, so each unit was read ≥ 9.5×.
  At 100K it is ≥ 19×.
- RAM-to-card copies repeat the same way: 24,179 ring copies of RAM-cache units against ≤ 2,531 cached units, so
  ≥ 9.6× (CR2:546, :554).
- Bytes per token stay flat as the prompt grows: about 13 MB a token at 2K and about 7 MB at 4K (investigator fit:
  (N/C)·R(C) + K, with R ≈ 24.8 / 26.5 GB per producer chunk and K ≈ 21 GB).

**Context cost.** At equal drive units (≈325-331), the 100K/2K producer chunks grow:
- with hashes on, from 3.99 s (chunks 2-9) to 4.57 s (the last nine);
- with hashes off, from 4.27 to 4.64 s;
- at 4K, from 6.02 to 6.40 s.

That is +6..+14% across the prompt (my parse of CR100). At the floor it would be +0.4-0.6 s per late 2K chunk.

**First-chunk anomaly.** The first P1 chunk after a server start sometimes runs slowly:
- 9.56 s against 2.94 s for the same 688 rows and 279 units (`…Ulong8k-r1.server.log:498` vs CP8:498);
- 25-28 s on the carry build, where it came from planes-pool growth (`…Qk96-Qt-…-r1/r2.server.log:515/516`).

### 2.2 The floor and what it is made of

**Floor chunks.** On a warmed server, with ≤ 124 drive units:
- 2,048 × 20 chunks take 2.27-2.40 s (W2K:651, 665, 680, 695-696, 711-712, 727-728, 743-744), and 2.397 s on
  today's Vp build (`QA/quiet_Qa0.server.log:522`);
- 4,096 × 20 chunks take 4.26-4.30 s (W4K:599, 613, 627, 641).
- The unit count does not move them: 85 units gave 2.283 s and 124 units 2.300 s.
- So a producer layer costs about 115 ms at 2K and 214 ms at 4K.

**Decomposition (est.).** 2K / 4K per layer:

| Part | 2K | 4K | Basis |
|---|---|---|---|
| Dense half, 11.7 µs a row | 24 ms | 48 ms | investigator, from the P0 m3-dense / m3-sparse logs (all-reduce and indexer included) |
| Busiest rank's MoE | 75 ms | 142 ms | 1.95 / 1.84 local slots a row (skeptic replay of `R/aux-followups-20261104/traces.pkl`; the rank mean is 1.50) × 18.9 µs a packed expert-row |
| The rest | 14-25 ms | 14-25 ms | gather, all-reduce waits, the tail fill of about 32 waves a layer (15,829 waves over 498 layers, CR2:554), barriers |

- Per-rank load swings layer to layer. In the device check, rank 0 holds 1.13 slots a row at L12 and 1.44 at L36,
  while rank 2 holds 0.65 at L25 (the "local" counts at M 2,048, `B/ab-20261104T003545Z/devcheck-p1.log:132-142`).

**The MoE kernel.**
- It costs 16.4-16.6 µs per expert-row in block form and 18.7-19.1 packed, at M 1,024-2,048 (devcheck-p1.log:120-142).
- An expert-row is 35.4M MACs: 4,608 × 5,120 + 5,120 × 2,304 (`src tools/ds41-host-experts/ds41_p1.h:125`).
  - That is 1.9-2.2 TMAC/s, about 20% of the est. 9.8 TMAC/s mul-then-add peak (4,096 lanes × 2 × 2.4 GHz ÷ 2).
  - The Phase 3 design target was 3-5 TMAC/s (PREFILL-200-DESIGN.md:144). Its "5.7 TMAC/s" tiled figure counts padded
    tiles; the live rate in that log was 0.7 (PREFILL-200-DESIGN.md:529).
- In the served config every P1 unit is packed: the drive packer reports "6958 packed, 0 block" (CR2:554), the RAM
  cache is packed, and so are the Vp residents (CR2:540). So the slower packed rate applies to all of them.

**Short passes pay per unit, not per row.**
- F3's 320-row follow-up turn took 1.73 s over 37 layers: 47 ms a layer, or 146 µs a row-layer (F3q:551). That is
  2.6× the 2K floor per row.
- At 320 rows a rank-layer touches about 49 experts. 54% of them have ≤ 4 rows, yet they carry 11% of the rows
  (skeptic replay). Each still costs an 18 MB copy and a wave slot.

### 2.3 Cold against warm servers

- **Every window arm starts a fresh server.** The same 6,817 tokens took 11.83-11.85 s after 82 + 3,086 tokens on one
  server, against 17.13-17.22 s alone (`src tools/ds41-host-experts/README.md:136-142`).
- **Its chunks read far less when warm:** 178/200/208/406 drive units against 279/296/311/584
  (`QA/quiet_Qa8_first.server.log:521-524`, CP8:498-503).
- **A warmed server's 6,817 is already near the floor:** 1.19 + 2.40 + 2.66 + 4.45 s (`QA/quiet_Qa0.server.log:521-524`).
- **So the README's 6.8K/48K/100K times are cold worst cases.** Quote every gain against both a cold and a warm baseline.

### 2.4 Harness turns (F3, Qa0, without the worker)

Summed TTFT was 61.3 s. Per turn: 15.72, 7.27, 3.17, 1.75, 3.18, 4.16, 4.14, 6.75, 2.61, 7.40, 4.34, 0.49, 0.31 s
(F3q `.json`).

- **Turns 1-2 start cold and take 23.0 s.**
  - Turn 1 runs a 2,048 × 37 chunk in 8.23 s (511 units) and a 973 × 40 chunk in 7.40 s (500 units) (F3q:513-514).
  - Turn 2 runs 4.92 s (267 units) and 2.32 s (148 units) (F3q:526-527).
- **The follow-up turns are bound on the card side.** 11,260 of 15,917 waves found the ring full, while the drive was
  busy only 47.8% (F3q:694).
- **Turn 6 is a 312-row pass on the CPU path:** 4.16 s (F3q `.json`; AUX-FOLLOWUPS-20261104.md:50).
- **Decode share of all time:**
  - 0.44 here (48.8 of 110.1 s);
  - 0.36-0.37 with the worker (41.8-42.0 of 114.0-114.6 s; `B/ab-20261104T064211Z/…Uharness-Ax-r1/r2.json`).
- **Replies are short:** at most 128 tokens (F3q `.json`), and chat-framed without thinking
  (`src tools/ds41-host-experts/ab_requests.py:185-190`).

### 2.5 Ranked levers

Gains are against C2 on a fresh server (6,817: 17.1 s; 48K: 115.1 s; 100K: 221.7 s; harness: 61.3-63.5 s with Qa0)
unless noted. From P-3 on, the gains apply on top of P-1/P-2.

| # | Lever | Mechanism | Est. gain (corrected) | Effort | Exactness | Status |
|---|---|---|---|---|---|---|
| P-1 | `Ada` (+ `Li64`) | Every unit P1 reads goes to the insert threads (4 a rank, stager.cpp:2221). `all` sets admit row 0 (scheduler.cpp:157, batch.h:206-210), so `p1_admits` is always true (stager.cpp:5130). Idle slots fill first, then LRU with second chances. `Li64` raises the queue cap from 16 (stager.cpp:1735, 4853-4856) as insurance | See §2.7. 48K/2K 115 → 62-68 s; 48K/4K 76.6 → 58-63 s; 100K 222 → 130-140 s / 165 → 120-125 s; 6,817 → 12-13 s; fill32k 64 → 41-46 s; harness −4..−10%; warm servers −3..−8% | XS | Residency only: the same packed bytes reach the same P1 kernel. Ad128 already admits under P1 with every hash identical | Never run (only Ad128, ab-20261101T014215Z) |
| P-2 | `Cf` startup fill | Fills each rank's 538 free slots with zero-decode-share units: 297 by prefill count, 241 in layer/expert order, in 2.3 s (`B/ab-20261103T143939Z/…-Cs-Cf-L1024-Pf-N4-C4-Pmr4.server.log:179-180`). Drive-only units in layers 0-19 fall from 422 to 49 (replay) | Fresh servers only. With Ada: 6,817 → 10.2-10.8 s; 3,086 → 5.5-6 s; 48K −46..−50% at 2K; harness −15..−25%. Ada adds 0..−5% for one prompt once Cf is in. Steady state ~0. Startup +1.6-2.7 s a rank | S: merge ~1k lines (not in f84b23db) | Residency only, through the warm's insert path | CH measured decode only (NIGHT-LOG.md:626) |
| P-3 | Per-unit split inside P1 passes | After the counts, units with ≤ r* rows that sit in the RAM cache run on the CPU pool in place. Their rows are copied over the card's before the gather, as cut layers already do (CR2:518; `src core/place/host_expert_pool.h:5-7`). The cards stream the rest. This replaces the 320-row cliff (Qm) with a per-unit choice | 320-row passes −10..−20%; ~1K passes −5..−10%; 2K chunks −3..−6%; harness −5..−10% (turn 6 alone 4.2 → ~2 s). The skeptic proposed −15..−25% for 320 rows; I lowered it because one pool serves all four ranks (host_expert_pool.h:9-10): at 320 rows about 4 × 26 units × 0.35 ms ≈ 36 ms of pool time a layer, against a 47 ms layer (pool cost 0.30 ms a unit + 0.05 ms a row, AUX-FOLLOWUPS-20261104.md:110-112). So r* must adapt per pass | M (3-5 days) | P1 rows equal 16-token passes (devcheck-p1.log:176), whose non-resident experts the CPU path computes. Extend the p1 check to split cases | New (skeptic); sketched only for link-bound cases (PREFILL-200-DESIGN.md:281) |
| P-4 | Packed P1 row at block speed | Either a faster packed decode in `moe_gemm_ggml_mxfp4_gemv_tall_p1`, or a card-side packed → block expansion into the block-sized ring slot (`kSlotBytes` = block size, ds41_p1.h:130). Residents need the first | 18.9 → ~16.5 µs per expert-row is −13% of the MoE, so −6..−8% of a chunk at the floor (2K: −10 of 115 ms; 4K: −19 of 214) | S-M (2-4 days) | p1 device check on packed tables | New (skeptic) |
| P-5 | Exact row-batched P1 kernel (Phase 3) | Decode each weight block once per 32-row tile. Keep each output's lane = block mod 32 order, ascending blocks, mul then add, and the same butterfly | At the floor: ×1.5 gives −12..−20% (2K) / −18..−22% (4K); ×2 gives −25..−33%. Nothing while drive-bound. 2× is plausible but unproven under the order rule | L (1-2 weeks) | Emulator plus the p1 device check to M 8,192, 0 rows differing | Design only (PREFILL-200 Phase 3) |
| P-6 | Exact cross-rank balancing of streamed units | The busiest rank (1.84-1.95 slots a row against a mean of 1.50) sets each layer. A less loaded card computes a streamed unit and sends its rows peer-to-peer to the owner's slots; the gather and the rank-ordered all-reduce stay as they are. About 82% of touched units are streamed (31,185 ring copies of 37,980, CR2:554) | −13..−20% of MoE time, so −6..−10% at the 2K floor and −8..−13% at 4K. ~0 while drive-bound | M-L | Same kernel and input bytes; outputs are copied, not re-reduced | Designed for link-bound cases (PREFILL-200-DESIGN.md:276-280) |
| P-7 | 4K chunks + A10 lending | At the floor 4K beats 2K by 6-11% a producer row (52 vs 55-59 µs). Today it costs decode −3.0% / −4.6% at N4/N8 (window CW, NIGHT-LOG.md:622); A10 lends the larger arena back (lend cap above kMaxLend 64, ds41_p1.h:165; cdbad1a7 incomplete) | After Ada: 48K/100K −7..−11%, 6,817 −10..−15%, harness 0..−3%. Decode −0.5..−2% with A10 (est., unmeasured) | M (2-4 days) | Kh/Kld at 4K are measured; row-slice caps need gates at the 4K slice boundaries | 4K is opt-in today |
| P-8 | Re-tune Qw4 / Qt (tail-once) / Qm / Qa after Ada | Per-wave costs matter at the floor: about 32 waves a layer at 0.3-0.4 ms of tail fill each (A7). Qt sits on claude/ds41-p1-carry | Long prompts −2..−6%; harness −1..−3% (Qm). After Ada read-ahead targets only uncached units (stager.cpp:5114), so the Qa setting stops mattering | XS (+ S to split Qt off the carry branch) | Scheduling only; tail-once waves pass the p1 check (devcheck-p1.log:176) | Qw4 measured −3.4..−5.1% at 6,817 (PREFILL-QW-20261103.md) |
| P-9 | Aux card computes the P1 rows of its resident units | The worker's 1,513 units (about 9.5 a layer a rank) are copied to the rank cards in P1 today | After Ada: long chunks −6..−12%, harness follow-ups −6..−10% | M-L | Same ISA and bytes; needs an aux p1 byte check | Needs the R9700 free |
| P-10 | Exact faster indexer at long context | index_score grows from 11-13 to 329 µs a call at 100K (README.md:252-257) | After Ada: 100K −3..−6%, 48K −1..−2%, 6.8K and harness ~0 | M | lsparse check, 0 diffs to width 524,289 | — |
| P-11 | Cache every eligible unit (~212.6 GiB) | 3,172 × 17.16 MiB = 53.2 GiB a rank | After Cf + Ada: 6,817 −5..−10%, 48K/100K −1..−2%, harness −4..−8%. MemAvailable would fall from 35-41 GB to ~13-19 GB | XS + RAM risk | Residency only | Only with ≥ 15 GB of headroom proven |
| P-12 | A second NVMe | Doubles drive supply | Today: −25..−45%. After Ada/Cf: −3..−10% (first touches only) | Hardware + S | I/O only | Not needed after P-1/P-2 |
| P-13 | Prefill-aware eviction (keep layers 0-19) | Evict upper-layer units first | Warm 6,817 −2..−5%, 48K/100K ~0..−2%, harness −1..−3%. LRU has no heat ties, so it can raise decode misses | S-M | Residency only | Low priority |
| P-14 | Prefetch the consumer layers' units during the producer sweep | — | ~0 today, because the drive is 69-80% busy. After Ada: −1..−3% at 48K/100K, −5..−12% at 6.8K. Needs somewhere to put the units | S-M | I/O only | Low priority |
| P-15 | Layer-major (Phase 2) / 8K chunks | §2.6 | With today's kernels ≈ the 4K floor: 48K ~52-58 s, 100K ~112-125 s. With a 2× kernel: 48K ~35-42 s | L / M | By construction (§2.6) | Defer |

**Harness perspective (est.).** Today's harness is about 104 s: TTFT about 62 s with Qa0, plus decode about 42 s with
the worker.
- P-1 + P-2 take about 9-15 s off TTFT on a fresh server.
- P-3 + P-8 take another 3-6 s.
- Speculative decoding (S-2) takes 1-3.5 s.

### 2.6 Layer-major prefill (Phase 2): verdict

**What it would save.**
- The drive bytes per prompt (est., investigator) at 48K:
  - one segment: about 52 GB (hard cap 65 GB);
  - S = 16K-row segments: about 108 GB;
  - S = 8K: about 186 GB;
  - today: 627 GB (2K) / 345 GB (4K).
- At 100K: about 203 GB (S = 16K) against 1,271 / 699 GB today.
- PCIe copies per card at 48K: from 561 / 319 GB to about 100 GB (S = 16K).

**What it costs.**
- Kept activations of 172.3 KiB a row (`90648e2c:docs/DS41-PHASE2-AFTER4K.md:81`), plus a candidate-keep term that
  grows with context (:83-86). That is 1.35 GiB a card at 8K rows and 2.69 GiB at 16K, plus 40 KiB a row of window KV.
- These must be lent to decode residents between prompts.
- A seam after the scatter and before gate_up in `MoeFP8::pass_head` (`src arch/common/rad_block_moe_fp8.h:1735`),
  per-piece launch captures, KV and window reservation for whole segments, Engram history kept in order, and atomic
  checkpoint publication.
- 2-3 weeks of work plus two windows.

**Why it should wait.**
- **Drive savings.** At 192 GiB the cache can hold every distinct unit of a prompt (≤ 862 a rank at 48K). With `Ada`
  each unit then crosses the drive once, which is layer-major's main drive saving, for no code.
- **Speed.** With today's kernels both layer-major and large chunks land on the same compute floor (about 48-52 µs a
  row-layer at large M): 48K ~52-58 s against ~58-63 s for Ada at 4K. That is a ~10% gain for weeks of work.
- **Where its value is.** It amortizes PCIe copies and per-chunk costs once the MoE kernel is about 2× faster (48K
  ~35-42 s). It also keeps a 96 GiB box from re-reading, because there admit-all thrashed (PREFILL-REPLAY.md:58-62).
- **Revisit when** P-5 reaches ≤ 12.6 µs per expert-row, or when a ≤ 128 GiB budget becomes a target again.

**Exact by construction, if:**
- dense, norm, attention, indexer and compressor ops run at the original 2K/4K piece heights, in position order;
- each expert replays each piece's original stable sort;
- the six-slot gather and the rank-ordered all-reduce run per piece.

Its gates would be Kld, Kh at 6.8K-100K, a host verify arm, a 2-slot ring stress arm, and the prefix / EORS / cancel
cases.

### 2.7 Expected speeds (est.)

All on fresh servers with prefill hashes on; walls from the max(floor, drive) model. Each Ada estimate is built from:
- the first chunk unchanged (it is cold);
- the producer chunks at the measured floor plus the context term (§2.1);
- the final chunk at about 300 first-touch consumer units;
- the measured time outside the chunks (3.35 s at 48K/2K, 5.67 s at 4K, 5.27 s at 100K/2K, Kh included).

| Prompt | Today 2K / 4K (measured) | Ada 2K / 4K | Cf + Ada | + P-3, P-4, P-6, P-8 at 4K | + ×2 kernel, 4K or S = 16K |
|---|---|---|---|---|---|
| 6,817 | 17.1 / 12.7 s | 12-13 / ~11-12 s | 10.2-10.8 (2K) / 8.5-9.5 (4K) | ~8.5-9.5 s | ~7-8 s |
| 47,917 | 115.1 / 76.6 s | 62-68 / 58-63 s | 58-66 (2K) | ~48-55 s | ~35-42 s |
| 99,987 | 221.7 / 165.1 s | 130-140 / 120-125 s | ~125-137 (2K) | ~95-110 s | ~80-95 s |
| 32,768 fill | 64.2 s (2K) | 41-46 s | — | — | — |
| Harness, 13 turns (Qa0) | 61.3-63.5 s | −4..−10% | −15..−25% | a further −5..−10% | — |

Arithmetic for 48K/2K with Ada:
- 3.50 s for the first chunk (CR2:496);
- 21 producer chunks × (2.30-2.40 s floor + 0.1 s average context) = 50.4-52.5 s;
- 2.55-2.7 s for the 21-layer chunk;
- 4.3-5.0 s for the final chunk (2,032 × 37 chunks with 244-289 units took 4.24-4.38 s, W2K:653-746);
- 3.35 s outside the chunks.

That sums to 64-67 s, or about 62-65 s without Kh.

Arithmetic for 48K/4K:
- 4.77 s for the first chunk (CR4:496);
- 10 × (4.26-4.30 + 0.1-0.2) = 44-45 s;
- 6.4-7 s for the final chunk (4,080 × 37 chunks took 6.3-6.7 s, W4K:600-642);
- 5.67 s outside the chunks.

That sums to 61-62 s.

After Ada, 4K beats 2K by only 5-10%: both chunk sizes sit on the floor.

### 2.8 Dropped or corrected (prefill)

- **REFUTED: Qa0 for cold long prompts.**
  - F3's fresh first turn took 14.60 / 14.92 s with Qa8 and 15.72 / 16.36 s with Qa0, +7..+12% (F3 JSONs;
    README.md:89-90).
  - Keep Qa8 for cold long prompts and Qa0 for harness turns. After Ada the setting barely matters.
- **Subsumed by `Ada`:**
  - carry with preallocated buffers: −0.45..−0.6 s a chunk when it did not stall (CP `…Qk96-Qt…-r1.server.log:516-520`);
  - overlapping the next chunk's reads.
- **Dropped:**
  - compression of unit reads: packed is 4.4% smaller (CR2:539), and MXFP4 codes do not compress;
  - "8K chunks before 4K + A10".
- **Corrected: drive-bound gain models.** Every gain built on "F + 0.011 s a unit", or on a 42-47 µs floor, was
  re-anchored on the measured floor chunks (55-59 µs at 2K, 52 µs at 4K).

## 3. Speculative decoding

### 3.1 What was measured (one stream)

| Window | Setup | Result |
|---|---|---|
| ab-20261029T083552Z | prompt lookup `--spec-ngram 3` | 11.72 / 11.76 against 11.92 / 11.86 tok/s; 1 proposal verified, 0 accepted (NIGHT-LOG.md:117). The loss is issue-ahead switched off, 1.3-1.8 ms a step (MORE-OPTIMIZATIONS.md:783-791) |
| ab-20261030T205440Z | DSpark depth 1, decode-only (M16 + Rb + Vp), Dv24 | code 29.79 vs 26.49 (+12.5%), held-out 34.15 vs 31.24 (+9.3%), chat 26.92 vs 26.60 (+1.2%) (README.md:275-282) |
| ab-20261030T213623Z | depth 2 / 3 | code depth 2 30.73 (2.60 tokens a step, 84.6 ms); depth 3 30.08; chat depth 2 −11.7% (NIGHT-LOG.md:184) |
| ab-20261030T234957Z | DSpark beside concurrency | 1 stream code +12.3%; 4 streams 36.9 vs 38.8 (−4.5%): nothing drafts, and the bank's VRAM costs about 100 residents a card (NIGHT-LOG.md:190) |
| **ab-20261031T165338Z** | **frozen prefix policy Dpp** (k = 0 if p1 < 0.938, 1 if p2 < 0.975, else 2; draft_policy.h:45-50) | **code 31.97, chat 28.85 / 28.77 (r2)**; fixed depth 1 29.73 / 26.22; fixed depth 2 30.53 / 23.22 (summed predicted_n / predicted_ms of the arm JSONs). Against the Rb+Vp plain arms of ab-20261030T205440Z (26.49 / 26.60): **+20.7% / +8.3%**. 128/128 replies identical (NIGHT-LOG.md:260) |

- The window's own plain arms (25.87 / 25.88) lack Rb/Vp, so they are not the reference.
- Acceptance in the capture:
  - P(first draft) is 0.865 on code and 0.614 on chat;
  - p1 ≥ 0.99 occurs on 70% of code steps and is accepted 0.97 of the time (chat: 54% of steps, 0.86).
- No DSpark arm has ever run on harness text, which C2 refuses. No identity check covers contexts beyond a few
  hundred tokens (RESEARCH-3-SPECDEC.md:184-186).

### 3.2 Cost model

**Old engine** (decode-only, no worker): steady cycles (DECODE-40-PATH.md:46-66).

| Steady cycle | Code | Chat |
|---|---|---|
| Plain step (1 ÷ the Rb+Vp plain tok/s of ab-20261030T205440Z, another window) | 37.75 ms | 37.6 ms |
| Verify 0 drafts, query still runs (C0Q) | 41.5 ms | 43.0 ms |
| Verify 1 draft | 64.2 ms | 66.8 ms |
| Verify 2 drafts | 85.7 ms | 89.5 ms |

- Each verified row costs 21.5-23.8 ms, 0.57-0.62 of a step. The query, observer and wait cost about 3.5-5 ms on
  every step, draft-free steps included.
- Dpp's code arm ran 1,753 steps for 3,995 tokens: 2.28 tokens a step at 71.3 ms. That means about 1.35 verified
  rows a step: (71.3 − 41.5) / 22.1.

**Today, F1** (`B/ab-20261104T051616Z`, step_ns_decode / steps_decode):

| | T(1) | N4 (3.79 rows) | N8 (7.59 rows) | Fit |
|---|---|---|---|---|
| C2 | 36.98 ms | 88.75 ms | 157.5 ms | T = 18.7 + 18.3·R |
| C2 + worker | 29.96 ms | 73.40 ms | 125.3 ms | T = 15.5 + 14.5·R |

**Why a verify row is expensive.**
- Consecutive tokens share few CPU-path units: only 24% (held-out) and 34.5% (chat) of a token's CPU picks were the
  previous token's (Oct 29 route traces, specdec-anatomy).
- So each extra row adds about 2.3-2.9 CPU units a layer, and a verify row costs 0.48 (F1 multi-sequence slope) to
  0.57 (DSpark's own rows) of a step.
- Speculation saves almost no memory traffic. It spreads the fixed card part, about 13-16 ms, over more tokens.

**Ceilings.**
- One stream: about 2× even with perfect drafts.
- At 8 streams a token already costs 16.4 ms. A draft must then be accepted ≥ 0.88 of the time, and the ceiling is
  +13% at 100% acceptance.
- At 4 streams (19.4 ms a token) the break-even acceptance is 0.75.
- The cheaper the CPU path becomes relative to the card part, the more speculation pays. The worker and its online
  set (F4) push in that direction.

### 3.3 What blocks it in C2 (verified at f84b23db)

Each part of C2 refuses speculation on its own:
- **Large-chunk prefill:** `core/sched/scheduler.cpp:107-108` and `core/place/stager.cpp:1592-1594`.
- **P1:** `stager.cpp:4762`. Line 4766 also refuses the DSpark head tail, but the Dv24 arms never configure it
  (the Dpp code log has no "fixed head tail" line). So P1 and the worker (`core/engine_aux.cpp:108-112`) need no
  head-tail work for Dv24.
- **End-of-request snapshots (EORS):** `scheduler.cpp:396-398`. A snapshot is reserved only on rows that carry no
  drafts (`scheduler.cpp:3069-3075`).
- **Consumer skip:** `scheduler.cpp:780-782`.
- **The Dpp profile:** it requires `max_seqs == 1`, depth 5 and a 2-row short block (`core/sched/draft_policy.h:10-12`).

Lookup needs only part of that admission work:
- it has no drafter context, so it needs no consumer-skip cone and no bank;
- but it turns pipelining off whenever `n_spec > 0` without a serial or block head (`draft_experiment.h:79-81`,
  `engine.cpp:1081`), and the post-stop skip goes with it (`engine.cpp:1095`).

**Exactness of verification.**
- Each draft is compared with the target's own draw at the RNG coordinate p.pos = n_out + k
  (`core/sample/sampler.cpp:818-821`), so any drafter is exact at any temperature if the logits are bit-identical.
- Bit-identical logits hold while a sequence's verify pass stays within the decode bands: the MoE gemv to 16 rows,
  norms to 32, sparse attention and the indexer to 48 (README.md:319).
- The aux worker takes verify rows, which are trunk passes, and declines draft passes
  (`stager.cpp:765`, `:2323`).

### 3.4 Ranked methods

| # | Method | One stream (est., corrected) | 4 / 8 streams | Effort | Exact | Verdict |
|---|---|---|---|---|---|---|
| S-1 | Dpp + worker in the decode-only profile (no code) | Code +17..+24%, chat +3..+10% over the same profile with the worker. Prefill is slow there (3,086 tokens 28.1 s, README.md:273) and `max_seqs` is 1 | n/a | XS: arms, plus ab_window support for Ax beside Ds parts | Each feature is exact; add an Axv arm with DSpark loaded | **Now**: it calibrates S-2 and gives a single-user mode today |
| S-2 | Dpp admitted in C2 | Code +14..+22%, chat +2..+9%, held-out +9..+15% after the draft-free grid steps. With snapshots taken inside the verify pass: code about +18..+25%. Harness wall −1..−3.5% (decode is 0.37 of it) | −2..−4.5% while the bank is loaded (measured without the worker; may be smaller with it, because displaced units enter the worker's set, `stager_aux.cpp:41-71`). A cost gate at N4 est. −1..+6% on code-like text, chat negative | M-L (5-8 days) | Gates in §4 | **Second**, if S-1's harness go/no-go passes |
| S-3 | Prompt lookup: issue-ahead kept on, plus a v2 gate | ds41 code-review harness 0..+2%, code256 0..+1.7%, chat and held-out 0. Edit-heavy proxy (846K tokens of Claude Code transcripts, 93% tool payloads) +8..+13%. Without the issue-ahead fix: −2..−4% | 0..+2% | M (3-5 days; less after S-2) | Exact (greedy verify) | **Conditional** on ds41's own agent output |
| S-4 | Query skip / device-side depth | DSpark pays its ~4 ms query on the 22% (code) and 36% (chat) of steps where it verifies nothing (C0Q vs plain; `src core/engine_dprob.cpp:98-105`). Skipping the query after low-p runs: est. +2..+5% on chat-like and harness text | — | S-M | Exact | With S-2 |
| S-5 | Hybrid: Dpp per step plus lookup proposals | Edit-heavy +11..+19% (about lookup alone −5..+2 points); code and chat equal to Dpp | as Dpp | S-M after S-2 and S-3 | Exact | Later |
| S-6 | Depth 3 when p3 ≥ 0.99 | Code +1..+3%, only if a depth-3 capture shows ≥ 0.9 acceptance there. Fixed depth 3 lost to depth 2 (NIGHT-LOG.md:184) | 0 | S + a capture arm | Exact | Later |

**Dropped (spec decode)**
- REFUTED: the bank on the R9700. 288 draft experts (about 5 GiB) would cost the worker about 19% of its units:
  −2.4% at N4 and −3.5% at N8. That about equals the XT bank tax it removes.
- The whole drafter on the R9700: 8.6-9.8 GiB displaces about a third of the worker. Net −3..+1% at one stream and
  −1..−3.5% at N4/N8 against the bank on the XTs (LONGCTX-FIFTHCARD-20261103.md:490-503).
- Trees: a sibling row brings 0.15-0.25 tokens for 0.48-0.57 of a step.
- Layer-skip self-speculation: a full-depth draft costs 0.35-0.45 of a step, against the DSpark query's ~0.11.
- Lookahead / Jacobi: about 20 extra rows a step.
- Expert-aware drafting: ≤ 1%, because routing is learned at every layer.
- Cross-session suffix tree: +0.1 point.
- Re-fitting the policy to today's costs: ~0 (±0.5 point), in-sample.
- Bulk verify of long copies at P1 height: ≤ 1%. Park it.
- Lossy relaxed acceptance: excluded by the exactness rule (HANDOFF.md:8-9). It is opt-in research only and would
  need a capture of the target's logit gap.

**Dpp arithmetic** for S-1 and S-2 (est.):
- With the worker:
  - the plain step falls 17-19% (F1: 36.98 → 29.96 ms);
  - a verify row falls 15-25% (the worker serves 28-31% of CPU-path rows, F1);
  - the ~3.75 ms query (C0Q minus plain, code) does not shrink.
- C2 + worker: 30.0 + 3.75 + 1.35 × (16.1-18.8) = 55.5-59.1 ms for 2.28 tokens. That is 38.6-41.1 tok/s against
  33.4, +15..+23%.
- Draft-free grid steps cost 10-20% of that gain (the skeptic's range is +14..+22%).
- Chat runs lower, because its acceptance is 0.82 against 0.95.
- Harness: decode is about 42 s of about 104 s. A guessed ×1.10-1.20 on harness text saves 3.8-7.0 s, less the grid
  loss and +0.5-1.5 s of TTFT for the bank's VRAM.

## 4. Ordered implementation plan

The windows assume four XTs unless noted, a fresh server per arm, one throwaway first arm (F3's first arm ran 89.1 s
against 72.0 s), and the research locks held across the series. **F4 (the aux-dyn worker window) runs first, as the
user asked.**

**Step 1 — window F5: P1 residency (config only, stable build, no code). About 7 arms, about 35-40 min.**
- **Arms:**
  1. C2-Kh-Lm131072-Ulong8k: same-window cold reference, hash 8c583a99. CR's own Lm131072 Ulong8k took 22.98 s,
     so do not reuse it.
  2. C2-`Ada`-`Li64`-Kh-Lm131072-Ulong8k.
  3. C2-Ada-Li64-Kh-Lm131072-Ulong48k: hash 87d48520; against CR2's 115.10 s TTFT and 111.75 s of chunk walls.
  4. C4-Ada-Li64-Kh-Lm131072-Ulong48k: against 76.58 / 70.91 s.
  5. C2-Ada-Li64-Lm131072-Ulong100k, hashes off: against CR100's 224.48 s.
  6. C2-Ada-Li64-Qa0-Pc-Pce-Pcg16-Kh-Lm32768-Uharness: hashes 13/13; against F3's Qa0 arms (61.3-63.5 s; decode
     25.4-25.6 tok/s with Kh).
  7. Decode guard: Ulong48k followed by the held-out N8 wave on the same server, with and without Ada.
     - This needs a small client option to send a wave after the timed prompt (XS; not in ab_window.sh today).
     - The same option gives the warm baseline (held-out wave, then Ulong8k), worth running as two extra arms if time
       allows.
- **Read:**
  - each chunk line's drive units and wall;
  - the admission line: admitted, relaid, and "dropped at the insert queue's cap" (CR2:562);
  - idle slots (CR2:546) and MemAvailable;
  - drive busy and ring-full waves (stats line).
- **Exactness:** every hash and reply equal to base1031 / the receipts. Any difference stops the window.
- **Adopt `Ada` at ≥ 192 GiB if:**
  - producer chunks after the third read ≤ 150 rank-0 units (expect ≤ 60);
  - 48K/2K TTFT ≤ 75 s;
  - 0 drops at the insert cap (otherwise rerun with Li256);
  - N8 decode after the long prompt within 2%.
- **Fallback:** if decode loses more, add an "admit into idle slots only" mode (S). It still covers the fresh-server
  case, since the layer-0-19 drive-only set (~422 a rank) fits the 494-541 idle slots.
- Keep Ad128 at 96-128 GiB.
- **Effort:** 0.5 day for the client option and the README; no engine build.

**Step 2 — merge Cf, and window F6 (needs the R9700 free for the spec arms). About 9 arms, about 50 min.**
- **Build:**
  - merge claude/ds41-cache-fill 8822ae0c (branch head 6341f3ce) into decode-impl: 0.5-1 day, host tests;
  - add ab_window support for Ax beside the Ds/Dpp parts;
  - re-run DEVICE_CHECKS=aux on the build's libr4d.
- **Prefill arms:**
  1. C2-Cf-Ada-Li64-Kh-Lm131072-Ulong8k: expect 10.2-10.8 s.
  2. C2-Cf-Ada-Li64-Qa0-Pc-Pce-Pcg16-Kh-Lm32768-Uharness.
- **Spec arms (S-1)**, in the decode-only profile (M16 + Rb + Vp + Ax, the Dpp arm's parts from ab-20261031T165338Z):
  3. plain Ucode256;
  4. Dpp-Dpc Ucode256;
  5. plain Uchat;
  6. Dpp-Dpc Uchat;
  7. plain Uharness (prefix cache on, no EORS);
  8. Dpp-Dpc Uharness;
  9. Axv with Dpp: 0 rows may differ.
- **Optional arms:**
  - a C2-Ax N4 pair with the DSpark bank loaded and without, which measures the bank tax with the worker (the user's
    first priority);
  - a Ct4 trace in one Dpp arm, to settle the drafter-cost dispute (2.3 / 4.7 / 8-9 ms, LONGCTX-FIFTHCARD-20261103.md:518-526).
- **Go/no-go for S-2:**
  - replies identical in every arm;
  - harness decode ≥ +6..8% (RESEARCH-20261103.md:373-374);
  - code ≥ +15%.
- **For S-3, without a GPU:**
  - build the Uedit receipt: read_file outputs plus "reply with a DSML str_replace or rewrite";
  - run it once as a plain C2 arm in F5 or F6;
  - replay ds41's own replies with `/tmp/specdec-research/ds_bucket.py`. Copy that directory into `R/` before a reboot.
  - Go if the v2 gate reaches ≥ +8%.

**Step 3 — build the prefill compute items, then window F7.**
- **P-3, the per-unit split (3-5 days).**
  - Mechanism:
    - after the counts, cost each local unit for the card (copy plus rows) and for the CPU (0.30 ms + 0.05 ms a row);
    - fill the shared pool up to the card's per-layer time;
    - write CPU rows over the card's slot rows before the gather, using the cut-layer path.
  - Lower Qm once this lands.
  - Gates:
    - p1 device check extended with split cases at M 320-4,096 (rows equal);
    - Kh at 8K and 48K, harness 13/13;
    - Kld with Cs off (3,397 / 3,397 positions).
- **P-4, the packed row at block speed (2-4 days).** Gates:
  - the p1 device check, packed tables included;
  - its timing lines must show ≤ 16.6 µs per packed expert-row at M 1,024-2,048.
- **P-8:**
  - split tail-once (A7) off claude/ds41-p1-carry;
  - pre-warm the planes pool and P1 staging at startup (task 90). That removes the first-chunk anomaly, which costs
    3-7 s and pollutes A/B pairs.
- **F7 arms (about 6):**
  - Uharness Qa0 + Ada, with and without the split, A-B-B-A;
  - Qm192 against Qm320 with the split;
  - Ulong8k and Ulong48k with P-4 + P-8;
  - Ulong8k r1 / r2: the first chunk must be ≤ 1.2× the second.

**Step 4 — S-2, DSpark-Dpp in C2 (5-8 days; can be built in parallel with Step 3), then window F8.**
- **Pieces:**
  - verify passes of ≤ 3 rows admitted under the large-chunk pins and P1 (scheduler.cpp:107-108, stager.cpp:1592-1594, :4762);
  - the deferred drafter context pass (Dx), so the drafter never runs at chunk height; drafter-only ops are exempt
    from the pins;
  - a consumer-skip cone at the tap layers 37-39: widen the cut to ≥ 129 rows, or tap layer 39 from layer 38's rows.
    The cut leaves about 272 / 144 / 16 rows there (AUX-FOLLOWUPS-20261104.md:36-37);
  - EORS taken inside the verify pass when every draft up to the grid row was accepted (`reserve_ckpt` takes a row
    offset, scheduler.cpp:1517), and otherwise a draft-free grid step as the fallback;
  - the Dpp profile generalized from `max_seqs == 1` to "one sequence decoding" (draft_policy.h:10-12);
  - S-4's query skip.
- **Gates:**
  - harness prefill hashes 13/13 with drafting on;
  - replies equal to base on code, chat and held-out at T = 0, plus one seeded T > 0 pair;
  - Kld 3,397 with DSpark off;
  - long-context rollback: Ulong8k and Ufill32k prompts followed by drafted replies, replies equal to base (never
    tested beyond a few hundred tokens);
  - EORS resume equality;
  - Axv with DSpark loaded;
  - N4/N8 with the bank loaded, for the tax.
- **F8 arms (about 8):** the gates above, plus code / chat / held-out N1 plain vs Dpp (A-B-B-A on code) and the N4 pair.

**Step 5 — S-3 lookup, only if Step 2's Uedit replay reaches ≥ +8% (3-5 days, less after Step 4).**
- Pieces:
  - keep issue-ahead on in lookup-only mode: treat lookup like a block drafter in `target_pipelining`, and drain only
    while a long-match proposal is live (MORE-OPTIMIZATIONS.md:904-906);
  - the v2 gate in `lookup.h`/`.cpp`: estimates kept across requests, keyed by match length (cap 64) plus an in-run
    flag, with depth k ≤ 15 chosen to maximize (1 + E[accepted]) / T(1 + k);
  - a cap of 16 verify rows a step until a byte gate passes (gemv_vp serves ≤ 16, stager.cpp:1597-1599).
- Same gates as Step 4. K = 15 at 8 sequences costs about 62 MiB of logits a rank (deepseek41.cpp:1122).

**Step 6 — P-5, the exact row-batched P1 kernel (1-2 weeks), with a model-free card-check window.**
- First an emulator, then the p1 device check to M 8,192 with 0 differing rows.
- Target ≤ 12.6 µs (×1.5), ideally ≤ 9.5 µs (×2), per expert-row at M ≥ 256.
- Then P-6 (balancing; size it first from F4's route trace taken without P1), P-7 (4K + A10), and P-15 only after P-5.

## 5. Measurement rules for these windows

- **Never time with Kh arms' TTFT alone.** Kh copies large pools synchronously at the last prefill step and turns
  issue-ahead off (LONGCTX-FIFTHCARD-20261103.md:165).
  - The replies after Kh prompts decode at about 13 tok/s, against 20.9 tok/s with hashes off (CR `…Ulong*.json`).
  - Use chunk walls in Kh arms, and pair one hashes-off arm when TTFT itself matters.
- **Report cold and warm baselines.** A fresh server is the worst case (§2.3).
- **Discard or repeat first-chunk anomalies** (§2.1) and the first arm of a window.
- **Single A/B pairs carry about ±5% of order effect** (HANDOFF.md:203). Use A-B-B-A for anything to be adopted.
- **Watch NVMe thermals.** Over 100-220 s of sustained reads, part of the 100K context slope may be drive slowdown.
  Log the 990 PRO's temperature in the Ulong100k arm.

## 6. Decisions for the user

1. **`Ada` changes RAM-cache residency.** P1 hits become LRU-touched, and a long prompt's units displace part of the
   unprotected warm set.
   - Adopt it at ≥ 192 GiB if N8 decode after a long prompt stays within 2%? Keep Ad128 for 96-128 GiB?
2. **`Cf` adds 1.6-2.7 s a rank to startup** and helps only a fresh server's first prompts. Adopt it if F6 confirms?
3. **DSpark in C2 conflicts with the first priority.** Its bank costs about 85-100 residents a card, −2..−4.5% at
   N4/N8 while loaded. The options:
   - (a) keep it as a single-user profile only;
   - (b) build S-2 into C2 and accept the tax, or unload the bank under load (that mechanism is unmeasured, and bank
     lending lost 4.9-6.5% before, NIGHT-LOG.md:265);
   - (c) skip it and put the effort into the online worker set.
   - Recommended: S-1 now, S-2 only if F6's harness numbers pass.
4. **Lookup depends on the user's real agent traffic.**
   - Does the user's agent emit file edits and rewrites (copy-heavy), or reviewer prose like the ds41 harness receipt?
   - Is thinking on or off, and at what temperature? Every byte gate so far ran at T = 0.
   - A sample of real agent logs, tokenized with the model's tokenizer, would settle S-3 without any GPU time.
5. **Phase 2 layer-major: defer** (2-3 weeks of work for about 10% with today's kernels), unless 100K-prompt TTFT
   matters more than everything else.
6. **The R9700 is shared.** The S-1 arms, P-9 and the worker-based calibrations need it free of training and BoltzGen.
7. **More RAM pressure.** Caching every eligible unit (P-11) would leave about 13-19 GB available.
8. **Lossy relaxed acceptance stays out** unless the user explicitly wants an opt-in research arm.

## 7. Sources

**Notes in `R/`:**
- HANDOFF.md:8-9, :147-151, :203;
- NIGHT-LOG.md:117, :126, :159, :184, :189, :190, :260, :265, :622, :626;
- DECODE-40-PATH.md:44-66;
- MORE-OPTIMIZATIONS.md:780-791, :900-906;
- RESEARCH-20261103.md:346-378;
- RESEARCH-3-SPECDEC.md:182-187, :214-221;
- PREFILL-200-DESIGN.md:81, :138-146, :272-281, :529;
- PREFILL-REPLAY.md:58-62;
- PREFILL-QA0-20261103.md, PREFILL-QW-20261103.md;
- LONGCTX-FIFTHCARD-20261103.md:165, :488-526;
- AUX-FOLLOWUPS-20261104.md:27-60, :100-195.

**Code at `f84b23db`:**
- `core/sched/scheduler.cpp:107-108, 151-176, 396-398, 592-597, 780-782, 1515-1520, 3064-3075`;
- `core/sched/batch.h:203-210`;
- `core/place/stager.cpp:765, 1588-1599, 1730-1737, 2216-2224, 2318-2326, 4758-4769, 4845-4860, 5110-5131`;
- `core/place/card_expert_stager.cpp:1410-1419`;
- `core/place/host_expert_pool.h:5-12`;
- `core/place/stager_aux.cpp:40-71`;
- `arch/common/rad_block_moe_fp8.h:1735`;
- `core/sched/draft_experiment.h:74-81`;
- `core/engine.cpp:1076-1098`;
- `core/engine_dprob.cpp:40-55, 92-108`;
- `core/engine_aux.cpp:40-62, 100-116`;
- `core/engine_bringup.cpp:1621-1628`;
- `core/sched/draft_policy.h:1-56`;
- `core/sched/lookup.h:40-92`;
- `core/sample/sampler.cpp:816-822`;
- `core/place/dspark_draft_control_scope.h:14-31`;
- `arch/deepseek41/deepseek41.cpp:1118-1124, 1306-1322`;
- `tools/ds41-host-experts/ds41_p1.h:84-90, 124-132, 160-166`;
- `tools/ds41-host-experts/ab_window.sh:2650-2653`;
- `tools/ds41-host-experts/ab_requests.py:183-191`;
- `tools/ds41-host-experts/README.md:84-90, 136-142, 166, 247-257, 273-282, 319`.

**Windows** (under `B/`):
- ab-20261104T003545Z (CR, with devcheck-p1.log:118-177);
- ab-20261103T221104Z (CP); ab-20261031T090154Z; ab-20261031T162629Z;
- ab-20261104T081812Z (F3); ab-20261104T064211Z (F2); ab-20261104T051616Z (F1);
- ab-20261103T143939Z (CH);
- ab-20261031T165338Z (Dpp); ab-20261030T205440Z.

Also `QA/quiet_Qa0.server.log` and `QA/quiet_Qa8_first.server.log`.

**Investigator and skeptic reports.** Their replays are in `/tmp/specdec-research/` and
`R/aux-followups-20261104/traces.pkl`. Two of their figures are used here without re-running: the busiest-rank load
and the per-unit row skew.
