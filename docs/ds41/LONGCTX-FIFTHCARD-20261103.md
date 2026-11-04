# Long context beyond 32K and the fifth card: one plan (2026-11-03)

This synthesizes five investigator reports (growing KV, fixed KV, fifth-card expert worker, drafter on the fifth card,
other fifth-card roles) and three skeptic passes. It adds one new read: window CQ (`ab-20261103T230607Z`, finished
23:44 UTC), which ran the first C2 servers above 32K.

Read-only: nothing was built or run, and no file was changed except this one.
- **Measured** means taken from logs, receipts or sysfs.
- **Estimate** means arithmetic, a model, or a replay of recorded route traces.
- Code citations read `path:line @commit`.

Commits named here:
- `0fb1f93e`: decode-impl, the recommended C2 (Lc2048 + P1 + lent arena + Rb + Vp).
- `43a1d1d2`: next-batch, which holds the growth stack (worktree wt-sync).
- `f4fe2ff1`: batch-admit, window CQ's build.
- `fa9d0411` and `4ed9cf6c`: the private fa9 line and the TARGET bank.

## 1. Summary

**Long context: contexts above 32K already work with a fixed KV reservation.**
- Nothing refuses `--max-model-len` up to 1,048,576 for one sequence.
- Under `--deterministic` the KV pool is a fixed startup reservation, paid for with VRAM expert residents
  (`core/engine_bringup.cpp:1894`, `core/mem/vram_budget.cpp:118-139, 190-195` @0fb1f93e).
- Window CQ started C2 at Lm 65,536 / 131,072 / 262,144 (measured):
  - startup budgets matched the fixed-KV model within one unit;
  - all 8 held-out replies were identical to the base;
  - one stream on short prompts decoded at 29.55 / 29.38 / 28.73 tok/s, against 29.04 at Lm512 (one arm each).
- Not yet shown:
  - exact prefill of prompts longer than 32K under C2 (no C2 prompt above ~9.8K tokens has ever run);
  - decode speed at a long *actual* context;
  - 512K under C2.
- One window (LC1, ~65 min) covers all three. If it passes, recommend Lm131072 for harness use.
- Measured caution: after a 32,768-token prompt, decode ran 4.4-5.6 ms a token slower than after an 82-token prompt
  (about 10%). If the cost grows with context, decode at 100K could be 25-35% slower. LC1 measures it.

**Growing KV: not the route to >32K.**
- It refuses C2: P1, the lent arena, more than one sequence, the prefix tier and lending.
- Its first cohort funds the full maximum at startup, so it frees no expert residents compared with a fixed pool.
- The bug is real and lies below the KV policy:
  - A component reproduces it without the model, using synchronous copies.
  - The data are not lost but displaced by exactly one re-committed granule.
  - Most likely cause (H1): a GPU access uses a stale address translation after unmap → handle release → new handle
    → re-map at the same address.
  - A ROCr/thunk binding error (H2) would leave the same signature.
- Next is a model-free standalone repro plus the existing failure-only mapping capture (one short window), not an
  engine fix.
- Until it is resolved, nothing should unmap and re-map VRAM at runtime. That covers the TARGET bank, KV↔expert
  lending and any fifth-card role.

**Fifth card: the R9700 is gfx1201 with ~27.85 GiB free.**
1. **Resident expert worker: build it.** It is the only role that attacks the measured priority-1 bottleneck.
   - Estimate: +15-25% at 4 and 8 streams on held-out prompts, +10-17% on chat and long prompts, −2 to +8% at one
     stream.
   - A 16 GB XT in its place gives about half that.
   - Exact by construction.
   - About 1-2 weeks of work plus one 1.5-2 h window.
2. **Phase 1 "cold unit" layout for prefill: unproven.** Revisit after #1.
3. **Drafter on the fifth card (your question): possible and exact, but small.**
   - Gain: about −1 to +3% at one stream over DSpark on the XTs, and nothing for concurrency.
   - Blocked today, because C2 refuses DSpark; making it admissible takes 4-7 days.
- **Rejected:** a shared VRAM cache read over P2P, an extra P1 streaming card, a VRAM tier for KV or prefix
  snapshots, index planes, a separate service, and TP5.

**First actions:**
1. LC1 preparation (one agent, CPU only, ~1 day).
2. Window LC1.
3. The aux expert worker, built in parallel.
4. Window F1 to qualify it.
5. A short model-free window on the KV bug, chained to a stop that is happening anyway.

Section 5 has the gates. Section 6 lists the decisions only you can make.

## 2. New measurements from window CQ

Sources:
- `tools/ds41-host-experts/build/ab-20261103T230607Z/<arm>.server.log`, lines 36, 73-74 and 195-196;
- the reply table in `window.log`;
- the NIGHT-LOG CQ row.

Setup: build f4fe2ff1, C2 configuration, four XTs, one stream, 8 held-out prompts of 32-46 tokens, 485 decode tokens
an arm, Kh off.

| `--max-model-len` | KV a card | Static residents r0 / r1-3 | Lent slots | Effective r0 | Rope/fixed region | Decode, 1 stream |
|---|---|---|---|---|---|---|
| 512 (first arm) | 21.61 MiB | 667 / 676 | 41 | 708 | 20.27 MiB | 29.04 tok/s |
| 65,536 | 286.98 MiB | 635 / 644 | 51 | 686 | 52.02 MiB | 29.55 |
| 131,072 | 486.98 MiB | 621 / 630 | 48 | 669 | 84.02 MiB | 29.38 |
| 262,144 | 886.98 MiB | 594 / 603 | 40 | 634 | 148.02 MiB | 28.73 |

- **Exactness:** replies were 112/112 identical to the merged base (`base1031.json` = `ab-20261031T104524Z/base.json`).
- **The model held:** the fixed-KV model predicted 635/643, 622/630 and 594/602 static residents with 51/48/41 lent
  slots, all within one unit of the measurement.
- **Cost at one stream:**
  - The Lm512 references are 29.04 (this window's first arm), 29.29 (window CG) and 29.78 (window CM).
  - Against them, 128K costs about 0 to −1% and 256K about −2 to −3.5%.
  - These are single arms, and arm order alone moves results by ~5%.
- **Not measured:** 4 and 8 streams at a large Lm. The estimate is −1 to −4% at 128K; the model's one-stream figure
  of −2% came out at about 0.
- **Host RAM:** MemAvailable stayed above ~36 GB (`*-Lm262144.mem`).
- **Batched admission (Ba):** the same window found it exact and +8-13% at 4 and 8 streams, and it is to be adopted.
  The windows below should use it as their base once it is merged.

## 3. Long context beyond 32K

### 3.1 The fixed-KV route (recommended)

**How the reservation is sized** (source @0fb1f93e):
- Without lending, the VRAM split gives KV one full-length sequence and the experts the rest
  (`core/mem/vram_budget.cpp:190-195`). Lending is always off under `--deterministic` (`core/engine_bringup.cpp:1894`).
  `--max-model-len` is therefore the pool that N sequences share.
- Only two things are refused:
  - fixed costs larger than the card (`vram_budget.cpp:118-139`);
  - a prompt of max_ctx tokens or more, which gets a 400 `context_length_exceeded` (`core/server/oai.cpp:1677-1696`).
- Two things only produce warnings (`core/engine_bringup.cpp:4144-4191, 986-999`):
  - a pool smaller than one sequence;
  - positions beyond YaRN's reach of 16 × 65,536 = 1,048,576.
- Per-card costs that grow with Lm:
  - KV: 3,200 B a token plus an ~86 MiB window allowance;
  - two rope tables: 512 B a position;
  - the index-score plane: rows = min(max_tok, max(256, 64Mi/(Lm+1))) (`arch/deepseek41/ds41_index_memory.h:8-22`).
    That is about 256 MiB up to 256K, 512 MiB at 512K and 1 GiB at 1M.
- Prefix caching keeps a quarter of the pool free (`core/sched/scheduler.cpp:1294`). Idle harness conversations must
  fit in 75% of Lm or move to the host tier.

**Cost of the reservation:**

| Lm | Effective r0 residents | Decode, 1 stream, short context | Decode, 4/8 streams, short context |
|---|---|---|---|
| 128K | 669 (measured) | ≈0 to −1% (measured) | −1 to −4% (estimate) |
| 256K | 634 (measured) | −2 to −3.5% (measured) | −4 to −7% (estimate) |
| 512K | ~565 (estimate) | −8 to −15% (estimate) | not estimated |

**Unexplained 512K/1M result** (research build 77, M16, Kh on):
- A 16-token decode after a 6,817-token prompt took 1,918 ms at 512K and 2,135 ms at 1M, against 771 ms at 32K
  (`work/autonomous-20261101/growing-kv-{raw32,512k,1m}-77-run1/window/inner/cold.requests.json`).
- That is about 1.1-1.4 s extra per request, not necessarily a slower rate. The cause is unknown.
- CQ at 256K (Kh off, 485 tokens) shows nothing like it.
- Measure 512K once before recommending anything above 256K (LC1 arm 9).

**Decode at a long actual context** (separate from Lm, measured, single requests):
- Stable build e184, M16, Kh on (`work/autonomous-20261101/filled32k-common0118-base-window2/inner/approved_e184.{0,1}.raw-response.json`):
  - 47.1 ms a token over 64 tokens after a 32,768-token prompt;
  - 42.7 ms a token over 8 tokens after an 82-token prompt.
- fa9 fixed reservation (`filled32k-fa9-fixed-certificate-root1/certificate.json`): 54.7 ms against 49.1 ms.
- So a 32K context costs +4.4 to +5.6 ms a token.
- The investigator's estimate of 0.3-1.5 ms per 100K tokens of context was about 10× too low.
- If the cost is linear in context, C2 would decode at ~20 tok/s at 100K (estimate). LC1 arm 8 measures it.

**Prefill time on long prompts** (estimates for C2):
- Model: 2.52 ms a token (linear) plus the O(N²) indexer.
- 48K: ≈2.2-2.4 min. 100K: ≈4.8-5.9 min, i.e. 280-370 tok/s, below the 500 tok/s target.
- C4 (4,096-token chunks): about 0.74× those times.
- Plain M16 at 35.4 tok/s: ≈23 min at 48K.

**Host RAM is not a constraint:**
- Snapshot store: 8-11 slots × 22.59 MiB at 64K-512K.
- Kh pins ~1 KiB × Lm a rank.
- MemAvailable stayed at or above ~34 GB in every C2 arm.

**Harness limits:**
- The per-request timeout is a hard-coded 600 s (`tools/ds41-host-experts/ab_requests.py:199` @0fb1f93e). That is
  enough for C2 up to ~100-150K, too short for M16 above ~20K and for LC2.
- The default `--max-model-len` is 512 (`ab_window.sh:604`); the parts `Lm<N>` and `L<N>` set it (`:2109`, `:2128`).
- No prompt receipt above ~16K tokens exists in `work/extra-prompts/`.
- Kh copies large pools synchronously at the last prefill step and disables issue-ahead. Never use Kh arms for timing.

### 3.2 Exactness for prompts longer than 32K

**Anchors that already exist** (plain M16, stable build):
- Ulong8k (6,817 tokens): all-ranks hash `8c583a99…`. It was identical at 32K, 512K and 1M, and across builds.
- fill32k (32,768 tokens + 64): all-ranks hash `71eaab9a74338840` with a fixed 333-character reply
  (`filled32k-common0118-base-window2/inner/approved_e184.*`).

**Above 32K there is no base anchor:**
- An M16 run costs ~23 min at 48K and ~47 min at 100K.
- Comparing C2 with C4 is a real cross-check for chunking errors, since they differ in chunk size, index-run split and
  P1 waves.
- But both use the same LCP-pinned index path with the same plane rows, so a width-dependent defect in index_score or
  top-k would pass both.
- The index kernels have run on cards only up to width 32,769 (the 2-run and 3-run cases; Lc4096 in
  `ab-20261031T162629Z`).

**Close the gap at kernel level before LC1:**
- The card check's sparse fixture places its chunk at position 333 (`tools/ds41-host-experts/ds41_mxfp4_device_check.hip:1092-1135`
  @0fb1f93e).
- It compares M-row launches with single rows and with launches of 16 (`devcheck-lcp-lcp.log`).
- Its top-k selections are made on the host, so the top-k kernel itself is not exercised.
- Extend it to positions 65,537 / 131,073 / 262,145 / 524,289, and cover index_score, the top-k kernel and
  attn_sparse_sink.
- Zero differing values at every width make C2 equal to M16 for these operations at those widths.
- Source reading finds only INT_MAX guards and per-1,024-key emit counters, so no limit at 65,536
  (`libr4d/r4d_ds41_sparse.hip:1565-1568, 1687-1688, 1708-1722`; source-derived, not run).

### 3.3 Why growing KV is not the route

**Scope** (source @43a1d1d2):
- The switch requires TP4, max-num-seqs 1, deterministic mode, no speculation, no host prefix tier, no KV streaming
  and no LoRA. It refuses P1, P1_LEND_ARENA and TBO (`core/engine_kv_growth.cpp:16-30`).
- Its runtime scope also refuses lending, the stager and P1 (`:57-70`).
- So C2 and concurrency stay out of scope even after a fix.

**Funding:**
- MAX_MIB must equal the full granule-rounded reservation, and the card must fund it at startup
  (`core/mem/kv_live_budget.h:78-93`).
- Expert residency is fixed at startup ("Deferred backing is funded KV capacity, not unused expert-pool slack",
  `core/engine_bringup.cpp:4901`).
- So growth gives the experts nothing compared with a fixed pool of the same maximum.
- A real gain needs lending (the TARGET bank 4ed9cf6c), which brings back runtime unmapping.

**On decode-impl 0fb1f93e:** growth cannot be reached. `kv_growth.h` is used only by tests, and `kv_quiescent_plan` is
defined but never called. The claim that the commit has "no growth code" is literally false.

**Its value even if fixed:**
- At 128K the fixed pool costs rank 0 39 effective residents (~5.5%), measured at about 0-1% at one stream.
- Sharing would matter only at 512K and above.
- The fifth-card worker adds ~1,500 units, far more.

### 3.4 The growing-KV bug: evidence, hypotheses, tests

**Where the code lives:**
- The stack is next-batch 43a1d1d2 (wt-sync).
- Its growth files are byte-identical on fa9d0411 and on its diagnostic children 4c6199b8 and c45c07bd.
- The TARGET bank (4ed9cf6c) uses the same commit/decommit pattern (`core/place/kv_expert_bank_native.cpp:74-80`
  @4ed9cf6c).

**Mechanics** (source @43a1d1d2):
- Commit (`a_vmem_commit`) creates a handle, maps it and sets access, once per 2 MiB granule
  (`core/device/aql.cpp:3816-3845`).
- Decommit (`a_vmem_decommit`) unmaps and then **releases the handle immediately** (`:3851-3867`).
- VMem maps and unmaps one granule at a time (`core/device/dispatch.cpp:285-345`). The calls come through
  Pool::commit_backing_granule and release_backing_granule (`core/mem/pools.cpp:150-164`).
- Every changed plan releases and re-commits granules (`core/mem/kv_growth.h:110-136`,
  `core/mem/kv_backing_transaction.cpp:132-152`).
- So each regrow gets new physical pages at the same addresses.
- The engine's bookkeeping is consistent: one handle per granule. This is legal API use, but it maximizes reuse of
  physical pages.

**Evidence (measured):**
- **Reproduced without the model:**
  - The native component `fa9-kv-remap-component-window320-*` runs the real KVManager, live controller, Pool/VMem and
    HSA backend, with synchronous HSA copies, full drains and no model.
  - It failed 3 of 3 uninstrumented runs, at frames 6, 12 and 12.
  - The compressed and index ID tables were identical in every frame.
  - Three instrumented 24-frame reruns passed: `fa9-vmem-timeline-window1`, `-window2` and `fa9-kv-kernel-mapping-window1`.
- **Decisive capture** (postfail2, frame 12, rank 0, recomputed from `inner/raw/frame12-rank0.postfail-KV-granule*.bin`):
  - G5 held its own bytes.
  - G7 was all zero.
  - G9 held G7's bytes, G17 held G9's, G25 held G17's, G33 held G25's and G37 held G33's.
  - G37's own bytes appear nowhere.
  - The 266,240 B of metadata were exact.
  - The earlier capture postfail1 (frame 6) had G5 all zero.
- **Correction to the notes:** HANDOFF.md:118 and NEXT-STEPS.md:43 say "granule 33 zero". It was G7.
- **No faults:** no GPU page fault was logged for any failure on Nov 2-3. The only amdgpu faults are at
  2026-11-01T01:30:58, from the ID divergence that 77f22d8e fixed (`/var/log/kern.log`).
- **Only re-mapped addresses fail:**
  - The first grow passes, and so does the fixed pool.
  - HYST108, which never changes its mappings after the first request, passes too, but that evidence is thin (2 replies).
- **Failures fall on first-use frames:**
  - Frame 6 is the first native entry_store dispatch (profile 0, producer 1).
  - Frame 12 is the first profile-1 reader.
  - Profile-0 data are zeros, so errors in frames 0-11 could stay hidden.

**Hypotheses, ranked:**

| # | Hypothesis | For | Against or unknown |
|---|---|---|---|
| H1 | Stale GPU translation: after unmap, release, new handle and re-map at the same address, some accesses still use the old translation, while the driver has already given the old pages to the next granule's handle | Fits the shift by one granule, the zero granule, valid-but-wrong pages with no faults, failure only at re-mapped addresses, intermittency, masking by instrumentation, and first-use frames (a later allocation or dispatch could flush the stale entries) | The direction is unknown: stale writes and stale reads leave the same signature. The driver mechanism is unverified (flush skipped or too light; CPU page-table updates racing the invalidation; a flush by PASID while no VMID is held) |
| H2 | ROCr/thunk binding error: a re-created handle imports the wrong buffer (a recycled dmabuf fd or GEM handle) | Same signature | A persistent misbinding should fail every time, but the instrumented reruns passed |
| H3 | A drain gap on the model path | Not excluded for the model | Not needed: the component fails with synchronous drains |
| H4 | Index rows are written through the compressed table and read through the index table (`arch/deepseek41/deepseek41.cpp:3792, 3800, 3907-3908, 3974-3992` @fa9d0411) | It caused the real faults on Nov 1 | Inactive now: the min-ID allocator keeps the tables identical (`core/mem/kv_backing_allocator.cpp:93-99, 159-200`) |

**Discriminating tests, in order:**
- **T-b, standalone HSA repro.** Model-free; half a day of code and ~20 min of window.
  - A small program reserves an address range and cycles commit → write → read → unmap → release over ~7 granules.
  - Variants:
    - as the engine does it, with a new handle each time;
    - the same handle re-mapped at each address;
    - release quarantined for N cycles;
    - a fresh allocation, or a first kernel dispatch, between write and read;
    - a perturbed free-page order;
    - a compute writer against an SDMA writer;
    - one card, and four cards in one process.
  - A tight one-card loop alone may not reproduce, which is why the allocation, first-dispatch and four-card variants
    are included.
  - Gate: the engine variant reproduces within 200 cycles, and the same-handle variant shows 0 mismatches in at least
    10^4 cycles.
  - The quarantine variant tells stale writes from stale reads: zeros against old poison.
  - A repro here is a ROCr or driver fault that can be reported upstream.
- **T-a, failure-only mapping capture.** The design exists: `work/autonomous-20261101/fa9-kv-failure-hsa-retain-mapping-source`
  (manifest 93f8c1cb…; HANDOFF "Next3").
  - After a failure, it asks the kernel which buffer each address maps to.
  - The right buffer means a transient translation (H1); a wrong buffer means a misbinding (H2).
  - It adds nothing before the failure, so it should not mask the bug.
- **T-c, trace.** Only if T-b does not reproduce.
  - Rerun the unchanged component under sudo trace-cmd or bpftrace, recording page-table updates per granule and TLB
    flushes.
  - The binary runs 24 frames and stops at the first failure (`fa9-kv-remap-component-source-retry7/native_remap.cpp:219-224`),
    so budget at least three runs. Tracing may hide the bug.
  - Accept either direction (the new page of G(k+1) is the old page of G(k), or the new page of G(k) is the old page
    of G(k+1)), and record which one occurred.
- **T-d, CPU-only fixture.** No window, about 2 hours.
  - A random KVManager walk asserting that the compressed and index tables stay identical and that the required set
    covers every held fragment.
  - This guards against H4 coming back.

**Fix directions** (only after the mechanism is known):
- The proposed engine mitigations (grow-only, or a per-address handle cache) buy nothing over a fixed pool.
- They also cannot pass the component as written:
  - The component asserts the startup bitmap after every release (`native_remap.cpp:214`), so grow-only fails it by
    construction.
  - A handle cache never returns VRAM.
- If KV↔expert sharing is wanted later, reusable pages must never be unmapped at runtime. That needs address-stable
  lending.
  - The current layer-major KV layout has ~9 MiB per layer stripe at 128K, smaller than one 18 MB expert unit.
  - So this means real layout work (large).
  - Rebase the TARGET bank (4ed9cf6c) and the `KV-EXPERT-COUPLING-20261101.md` design the same way.
- A driver A/B such as `amdgpu.vm_update_mode=0` needs a module reload, which stops QFN, whisper-server and BoltzGen.
  Only with your approval.
  - Today the value is −1 (`/sys/module/amdgpu/parameters/vm_update_mode`).
  - With full-size BARs, compute VMs probably already use CPU page-table updates.

## 4. The fifth card

### 4.1 Hardware and current use (measured, sysfs read 23:5x UTC Nov 3)

**The R9700:**
- KFD node 1, PCI `c3:00.0`, device 0x7551.
- `gfx_target_version 120001` (gfx1201, the same target as the XTs, device 0x7550).
- 128 SIMDs, 2350 MHz maximum clock, 256-bit memory at the XTs' clock.
- 34,208,743,424 B of VRAM, of which 4,301,783,040 B are in use.

**Who uses it:**
- whisper-server (pid 1145533, running since 18:00:05) holds 4,240,601,088 B on it
  (`/sys/class/kfd/kfd/proc/1145533/vram_11920`).
- Production radiance holds nothing on it, and ~16.8 GB on each XT.
- 27.85 GiB is free.

**Link:**
- The host link is Gen4 x16 through root port `c0:01.1`; the card's 32 GT/s is only its internal switch.
- Rank 3's XT (`c6`) sits on `c0:03.1`: the same root complex, a different port.
- KFD lists P2P links to all four XTs, but no P2P bandwidth or latency involving c3 has ever been measured.
- The Oct 30 shared-root copy result (17.1 GB/s) predates the slot move: it was measured on XTs, and no BDFs were
  recorded.

**Card order and rank binding:**
- KFD order: node 1 c3 (R9700), node 2 c6, node 3 83, node 4 48, node 5 03.
- Ranks bind to the first `--tp` visible devices of the first card's architecture
  (`core/engine_bringup.cpp:189-229, 4261-4267` @0fb1f93e).
- Without a mask the R9700 becomes rank 0. That already happened in production (NIGHT-LOG.md:510).
- Windows pin the four XTs by UUID (`work/chain47.sh:13`).

**No fifth rank:**
- TP5 is refused, because 384 experts do not split over 5 ranks (`arch/deepseek41/deepseek41.cpp:1104-1108`).
- The VRAM split is sized from the smallest card (`core/mem/vram_budget.cpp:42-63`).
- So a fifth card helps only in an auxiliary role.

### 4.2 Roles, ranked

| # | Role | Priority served | Estimated gain | Effort | Exact? | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Resident expert worker**: the hottest CPU-path units stay resident on the extra card; input rows go in, complete bf16 rows come out before the unchanged gather and rank fold | 1, concurrent decode | 4/8 streams: +15-25% held-out, +10-17% chat and long; 1 stream: −2 to +8%; with a 16 GB XT: +6-13% | 1-2 weeks + 1 window | yes, by construction | **Build** |
| 2 | **P1 cold-unit layout**: drive-only units resident on the extra card and computed there for large chunks | 2, prefill | unknown | M-L, after #1 | yes | Revisit after #1 |
| 3 | **Drafter**: the whole DSpark drafter, or only its draft-bank experts, on the extra card | 1-stream decode only | −1 to +3% over XT DSpark at 1 stream; 0 against plain at 4/8 streams | L, plus 4-7 days of prerequisite work | replies yes | Only if 1-stream speed matters |
| – | Shared VRAM expert cache read over P2P | – | negative: one ~26 GB/s link against ~120 GB/s of RAM | M | yes | Reject |
| – | Extra P1 streaming participant | 2 | 0-10%: the links are ~29% used, and rank 3 shares root c0 | L | yes | Reject |
| – | VRAM tier for KV or prefix snapshots | 3 | ≈0: the host tier exists, and a host restore uses four links | M | yes | Reject |
| – | Index-score planes or long-context attention state | – | ≈0 or negative: per-pass scratch, replicated caches read every token | XL | – | Reject |
| – | Separate service (today's whisper and BoltzGen use) | – | 0 for ds41 | – | – | Status quo |
| – | TP5, or the R9700 as a rank | – | refused by the source | – | – | Reject |

### 4.3 Role 1: the resident expert worker

**Why it works:**
- Concurrent decode is limited by the CPU reading expert weights from DRAM.
  - At C2 with 4/8 streams, rank 0 waits ~8 s an arm at the gathers, about 57-61% of a step (window CM,
    `ab-20261103T211924Z`).
  - A pool job's first tile starts 682 / 1,231 µs after it is submitted, so work queues.
  - Going from R to C2 cut CPU experts 3.4% and step time 1.9-2.2%: an elasticity of 0.56-0.65.
- The CPU path already reproduces the cards' decode chain bit for bit
  (`tools/ds41-host-experts/ds41_host_experts.h:1-26`).
  - 14,708 and 31,669 rows matched with 0 differing (NIGHT-LOG.md:13, :30).
  - CQ's chain check: 48 rows, 0 differing.
  - gemv_vp equals the plain gemv: 0 of 6,087,680 values differ (README.md:278).
  - So an extra card running the same kernels returns the same rows.

**Capacity and absorption** (estimates):
- 26.85 GiB after ~1 GiB of workspace holds ~1,600 packed units (18,001,920 B) or ~1,533 native units (18,800,640 B).
- The replay used the Oct 29 route traces against the rebuilt C2 plan. It reproduces the measured CPU work: 98.7 /
  367.9 / 722 CPU experts a step at 1 / 4 / 8 streams.
- Ranked by the sealed profile, ~1,600 units absorb:
  - 31% of CPU-path selections on the 8 held-out prompts;
  - 22.5-23% on the 16 chat prompts and on the long prompts.
- The "oracle" figure of 52-86% is in-sample hindsight. Orderings trained on the other traces never beat the sealed
  profile by more than 0.8 points.
- A 16 GB XT (~800-900 units) absorbs 17% / 11.5% / 12%.
- The held-out prompts share task families with the calibration profile, so they are the optimistic case.

**Gain** (estimate):
- Method: absorption × elasticity, with elasticity 0.5-0.65. The C2→C4 slope (~0.8-1.26) is confounded by its
  4,096-token chunks.
- 4 and 8 streams: +15-25% on held-out prompts, +10-17% on chat and long prompts.
- 1 stream: −2 to +8%.
  - A layer's CPU wait is mostly fixed cost there (~0.2 ms of pool wake).
  - Misses, about 43% of the one-stream critical path (MORE-OPTIMIZATIONS.md:104), stay on the CPU.
  - The worker's round trip (~0.1-0.35 ms a layer, estimated) has never been measured.
- PCIe traffic stays under ~0.4 MB a layer at 8 streams, so sharing root c0 with rank 3 should not matter for decode.

**Design** (on decode-impl plus Ba, default off):
1. **Switches:** `RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=<MiB cap>`, an optional device selector by BDF or UUID, and
   `..._AUX_POLICY=verify`.
2. **Discovery:**
   - Candidates are visible devices beyond `--tp`, discrete, with the same gfx target as rank 0.
   - They need fresh free VRAM of at least the cap plus headroom.
   - Log the PCI root and any rank that shares it.
   - Warn and stay off if the TP set mixes card types while a same-architecture card is left over.
   - No candidate means off, and the four-card path is unchanged.
3. **Unit set:**
   - The global top N CPU-path units in planner order (`core/place/planner.cpp:476-497`, shares from
     `core/place/xstats.cpp:517-536`).
   - Skip VRAM, pinned and lent units.
   - The set is static for the server's life, with no runtime vmem release (section 3.4).
4. **Loading:** after the RAM-cache warm-up, copy the units from their packed slots, read them back once and compare.
5. **Hooks** (`core/place/stager.cpp` @0fb1f93e):
   - `hx_choose` marks worker units before the borrow (`:2327`).
   - After the CPU submit (`:2903`), on each rank's own stream on the extra card:
     - copy the rank's rows from `hx_x_` (`:2113`) into the card's VRAM (kernels should not read pinned host memory
       directly);
     - relaunch gate_up, swiglu_clamped and down;
     - copy the output rows to a private pinned staging buffer;
     - signal completion.
   - `hx_finish` (`:4258-4272`) waits for the pool ticket and the worker's signal, then places the worker's rows in
     `hx_out_` at their sorted positions, before the existing copy (`:4346`).
   - `hx_drain` (`:4358`) waits for the worker too.
6. **Kernel geometry:**
   - The decode gemv reads every expert listed in the offsets (`libr4d/r4d_ds41_mxfp4.hip:643-645, 1357-1360`).
   - So pass compacted sorted/off arrays for the worker's units, keeping the live-tail range convention (`:546, :643`).
     The alternative is the P1 tall form's skip mark (`:1268-1270`). Either needs its own byte check.
   - Use the base's kernel family for that pass size: gemv or gemv_vp, up to 16 rows.
   - At first, decline larger passes and the drafter's layers.
   - Template: P1's relaunch with a 96-entry weight table (`core/place/card_expert_stager.cpp:709-750`), rebound to
     the extra card's buffers and streams.
7. **Deadline:** if a worker job is late (another process loading the card), the CPU computes those rows, which is
   exact either way. A late worker output lands only in its private staging buffer.
8. **Counters at shutdown:** units held, jobs, rows, time waited, deadline misses, and differing rows in verify mode.
9. **Harness:** new arm parts `Ax`, `Axv` and `Ax0` (five cards visible, worker off). Only those arms append the extra
   card's UUID, found from sysfs by capability.

**Do not port the earlier prototype's code.** The f520 source is source-only, unbuilt and NOT RUNNABLE. It targets fa9,
which is not an ancestor of 0fb1f93e, and it is audit-only: a two-entry queue, one thread, full intermediate readback
and no quiet mode. Reuse only its discovery and capacity rules
(`work/autonomous-20261101/aux-resident-expert-source/README.md`).

**Risks:**
- **Sharing the card.**
  - whisper-server's 4.2 GB is permanent.
  - BoltzGen jobs select the R9700 by UUID (HANDOFF.md:141), and their VRAM use is not recorded.
  - Holding the research locks did not stop BoltzGen before (NEXT-STEPS.md:65).
  - `ab_window.sh` has no guard for other processes using the cards; whisper ran during windows CM, CP and CQ.
  - So the Ax arms must check the extra card's clients at start. A sharing policy is your decision (section 6).
- **Cost of a fifth visible card, even with the worker off.**
  - Every pinned range is mapped to every visible device (`core/device/aql.cpp:4049-4052` @0fb1f93e). That includes
    the RAM-cache slabs, up to ~192 GiB.
  - This means longer pinning and page tables in the extra card's VRAM.
  - The machine key changes from `-x4-` to `-x5-` (`core/format/machine_key.cpp`).
  - Measure this in the Ax0 arm. If it matters, map only the worker's own buffers to the extra card.
- **Clocks.**
  - The XTs sat at 41 MHz sclk in ~35% of decode samples (NIGHT-LOG.md:626), and a lightly loaded extra card will
    also ramp down between jobs.
  - Measure the round trip at the real per-layer cadence.
  - Forcing the 'high' power level on a shared card would also affect whisper and BoltzGen.
- **Static unit set.** Absorption depends on the traffic. A set adapted online, refilled from RAM, is a later idea.
- **Uncounted upside.** The worker's units no longer need RAM-cache slots for decode (~28 GB), which could reduce
  misses.

**Long-context synergy:** the worker more than offsets a fixed 128K or 256K pool. Those cost 39 or 74 effective
residents a rank; the worker adds ~1,500 units.

### 4.4 Role 3: the drafter on the fifth card (your question)

**Today (measured):**
- DSpark at depth 1 gains at one stream: code +12.5%, held-out +9.3%, chat +1.2% (README.md:206-210 @0fb1f93e).
- Its bank costs 96-97 trunk residents a rank at 192 GiB (`ab-20261030T205440Z`).
- Loaded beside concurrent requests it costs −4.5% at four streams (README.md:215-218).
- C2 refuses DSpark under:
  - large-chunk prefill (`core/place/stager.cpp:1593-1595`, `core/sched/scheduler.cpp:107-108`);
  - Phase 1 (`stager.cpp:4686`);
  - end-of-request snapshots (`scheduler.cpp:320-322`);
  - consumer skip (`scheduler.cpp:704-706`).

**The whole drafter on the extra card (estimate):**
- It needs ~8.6-9.8 GiB:
  - all 384 draft experts, 7.22 GB;
  - the full bf16 vocab head, 1.32 GB;
  - FP8 dense weights, ~0.53 GiB;
  - Markov tables, 126 MiB;
  - optionally the embedding, 1.23 GiB.
- A 2-row query on one card reads ~2.0-2.15 GB: ~3.5-4.3 ms, slower than the XTs' sharded pass.
- Net at one stream against XT DSpark: about −1 to +2% a cycle. Reaching +2.5% needs the head to stay on the XTs.

**Only the draft-bank experts on the extra card, through the worker:** frees ~1.26 GiB a XT (~70 residents), at the
cost of three mid-pass round trips.

**At 4 and 8 streams:** 0 against plain. It only removes the bank's VRAM tax, which the recommended configuration does
not pay.

**Exactness:**
- Acceptance compares each draft with the target's own token, at a per-position random coordinate
  (`core/sample/accept.h:4-12`, `core/sched/draft_gate.h:26-27`).
- A one-card drafter may propose different tokens, which changes acceptance only, not replies.

**Prerequisites:**
- DSpark inside C2 (RESEARCH-20261103.md:346-378, 4-7 days). Its cheap go/no-go comes first: a plain-vs-DSpark pair on
  harness turns before any build (:373-374).
- An auxiliary-device concept: today only TP ranks get devices, streams and peer setup
  (`core/engine_bringup.cpp:4261-4290`).
- A one-rank drafter program running beside the four-rank trunk, which does not exist.
- `--draft-model` is refused (`core/config.cpp:238, 900-908`).
- The DSpark scopes hard-code TP4 (`core/place/dspark_draft_control_scope.h:15-20`, `core/engine_dbank.cpp:53-54`).

**Unresolved:** four drafter-cost figures disagree:
- 2.3 ms of host wait;
- ~4.7 ms (NIGHT-LOG.md:184);
- 8-9 ms "drafter plus tail" (RESEARCH-20261103.md:394-395);
- 31-43 ms from the 96 GiB gate split, probably an attribution artifact.

Device timestamps in one DSpark arm would settle this.

**Verdict:** third. Run the go/no-go pair only if single-stream speed matters to you.

### 4.5 Role 2: the P1 cold-unit layout

**P1 at 6,817 tokens is limited by the drive and the ring, not the link** (window CW,
`…-Kh-Lm8192-Ulong.server.log:569-677`):
- Rank 0 moved 129 GB in 16.6 s: 7.8 GB/s against ~26.7 GB/s of link.
- The drive was busy 69-81% of the time, with 1.6-3.9 s of drive waits.
- 69% of waves found the ring full.
- ~43% of drive units were read ahead and then dropped.

**Why the worker's units would not help:** they are mostly RAM-cache hits, so computing them on the extra card saves
DMA, which is not the limit.

**What would help, and why it is doubtful:**
- Only drive-only units would help (~1,281 units, ≈22 GiB), and those compete with decode capacity.
- A 208 GiB RAM cache gave no prefill gain (NIGHT-LOG.md:189). That was measured at 1,024-token chunks, where the
  drive was not P1's limit; at today's 2,048-token chunks the drive is busy 69-81%, so the question is not fully closed.

**Next:** cut the wasted read-ahead first, and revisit after #1.

## 5. Ordered plan with gates

**Arm prefixes:**
- C2 = `all-t40-f-M2048-H-p256-m196608-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc2048-P1-Qe64-Mb1-Rb-Vp-Cs`,
  plus `Ba` once it is merged.
- C4 = `all-t40-f-M4096-H-p256-m196608-G-R-B6-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Ss-Pu-Is-Pi8-Ns7-Fw-Mr32-Lc4096-P1-Qe64-Mb1-Pci4096-Rb-Vp-Cs`.

Unmarked arms run one stream on the held-out prompts.

### Step 1: LC1 preparation (no window; one agent, CPU only, ~1 day)

**Work:**
- Prompt receipts in `work/extra-prompts`, with names without `-`:
  - `long48k`: ~185K characters of `/usr/share/perl/5.40.1/pod/perldiag.pod`;
  - `long100k`: perldiag plus licence texts, `max_tokens` 128;
  - `fill32k`: the frozen 32,768-token body from `approved_e184.role.json`.
- Make the request timeout in `ab_requests.py` configurable from the environment (default 600).
- Extend the sparse fixture to widths 65,537-524,289 for index_score, top-k and attn_sparse_sink.
- Pin the four XT UUIDs in the AR/AG row checks (`ab_window.sh:1558/1625/1714`). They unset the device mask, so today
  they test c3, c6, 83 and 48, and never 03.
- Correct "granule 33" to G7 in HANDOFF.md and NEXT-STEPS.md.

**Gate:**
- The fill32k body is byte-identical to the reference request.
- Existing arms are unchanged at the default timeout.
- The extended fixture builds.
- Real token counts are read from LC1's first arm.

### Step 2: window LC1 (~65 min, ~80 with the optional arms)

**Arms (9):**
1. C2 (reference)
2. C2-Kh-Lm131072-Ulong8k
3. C2-Kh-Lm131072-Ufill32k
4. C2-Kh-Lm131072-Ulong48k
5. C4-Kh-Lm131072-Ulong48k
6. C2-Kh-Lm131072-Ulong100k
7. C4-Kh-Lm131072-Ulong100k
8. C2-Lm131072-Ulong100k (timing: Kh off, 128 tokens)
9. C2-Lm524288

Optional arms 10-13: 8-stream A-B-B-A, L1024 against Lm131072.

Card checks include the extended fixture.

**Gate:**
- Arm 2's hash is 8c583a99.
- Arm 3's hash is 71eaab9a74338840, with an identical 64-token reply.
- C2 and C4 give equal hashes and text at 48K and at 100K.
- The fixture shows 0 differing values at every width.
- Replies match base1031.
- At 512K, startup is near ~524 static residents and ~41 lent slots, and decode is within 15% of arm 1.

### Step 3: README update (no window; small)

**Work:**
- Recommend Lm131072 (C2 + Pc/Pce/Pcg16) for long-context harness use, with the measured time to first token and
  decode at 100K.
- L1024 stays for throughput benchmarks.
- If arm 8 is more than 25% slower than short-context decode, profile one decode step at 100K.

**Gate:** all LC1 gates passed.

### Step 4: build the aux expert worker (no window; CPU builds; 1-2 weeks, one agent)

**Work:** as designed in section 4.3.

**Gate (host tests):**
- No candidate, a wrong ISA, too little VRAM, or a mixed TP set each turn the worker off cleanly.
- Compaction and run-copy tests pass.
- With the switch off, the code path is unchanged.
- The Ax, Axv and Ax0 grammar passes a dry run.

### Step 5: window F1 (~1.5-2 h, BoltzGen idle)

**Card checks on the extra card:**
- SiLU recapture equal to `silu_table_gfx1201.bin`;
- `vpack check 48`;
- gemv and gemv_vp bytes at M1/M4/M8/M16 equal to an XT's;
- the per-layer round trip with whisper resident;
- the card's sclk.

**Arms (12):**
1. C2
2. C2-Axv at 4 streams
3. C2 at 8 streams
4. C2-Ax at 8 streams
5. C2-Ax at 8 streams
6. C2 at 8 streams (arms 3-6 form an A-B-B-A)
7. C2-Ax at 4 streams
8. C2 at 4 streams
9. C2-Ax at 1 stream
10. C2-Ax0 at 4 streams
11. C2-Ax-Kh at 8 streams
12. C2-Ax Kld

**Gate:**
- 0 differing rows between the worker and the CPU.
- Replies match base1031.
- Kld matches at 3,397/3,397 positions.
- Kh matches CQ's 8-stream Kh arm, 8/8.
- 8 and 4 streams each gain at least 10%.
- 1 stream loses no more than 2%.
- Ax0 stays within noise of C2, with the extra card's VRAM growth and pin time logged.

### Step 6: root cause of the growing-KV bug (~20-30 min of window, model-free)

**Work:**
- T-b, the standalone repro (half a day of code).
- T-a, if it is built.
- T-d, on the CPU only.
- Chain the window to another planned stop.

**Gate:** as in section 3.4. No engine fix before the mechanism is known.

### Step 7: if F1 adopts the worker (later windows; M)

**Work:**
- Update the README.
- Run an 8-stream pair on the chat prompts.
- Later: use the worker for CPU-path prefill passes (17-319 rows, which needs the tall kernel and its own byte gate), and
  try an adaptive unit set.

**Gate:**
- The same byte gates as F1.
- Before any window, replay must show absorption of at least 30% on the chat and long prompts.

### Step 8: only on your decision (LC2 takes 2.5-4 h of downtime)

**Candidates:**
- LC2: 200K and 500K prompts, C2 against C4 plus the compact index floor, with a 1-hour request timeout.
- The drafter path: the go/no-go pair (D0), then DSpark inside C2, then a drafter on the extra card.
- A redesign for KV↔expert sharing.
- A driver A/B.

**Gate:** as in sections 3 and 4.

## 6. Open questions for you

1. **Sharing the R9700.**
   - May ds41 hold ~26-28 GiB on it while whisper-server keeps its 4.2 GB?
   - When BoltzGen needs the card, which should apply?
     - ds41 restarts without the worker;
     - a fixed cap that leaves room for BoltzGen (how much?);
     - a runtime yield, which is more work.
2. **Will the R9700 stay, or be replaced by a 16 GB XT?** A 16 GB card makes the worker worth about half.
3. **Long-context target.** Is 128K enough? Qualifying 256K-1M (LC2) costs 2.5-4 h of production downtime and needs a
   1-hour request timeout.
4. **Default `--max-model-len`.**
   - One server at Lm131072 for both harness use and concurrency? Its one-stream cost measured ≈0; its 4/8-stream cost
     still needs measuring.
   - Or keep L1024 for throughput work?
5. **Single-stream speed.** Should DSpark be made admissible inside C2 (4-7 days, after a cheap go/no-go), and later
   run as a drafter on the extra card? It adds nothing to concurrent throughput.
6. **Growing-KV root cause.**
   - Spend one short model-free window on it now, or after the worker?
   - A driver A/B needs a module reload that stops QFN, whisper and BoltzGen.
7. **Testing on the R9700 outside production windows.** May model-free GPU tests (for example T-b) run on the R9700
   alone outside the QFN stop windows, given that it shares the card with whisper-server? Default: no.

## 7. Claim ledger: corrections to the investigator reports

**Refuted or corrected:**
- "Granule 33 was zero" (HANDOFF.md:118, NEXT-STEPS.md:43) → G7 was zero.
- "0fb1f93e has no KV-growth code" → it has growth code that cannot be reached.
- "Only the 2-run index case ran on cards" → the 3-run case ran too (`ab-20261031T162629Z`). Widths above 32,769 are
  still untested.
- The cited window `ab-20261030T090558Z` does not exist → it is `ab-20261103T090558Z`.
- "No window above Lm32768" → stale. CQ ran 64K, 128K and 256K with short prompts.
- "C2 needs 4.5-5 min at 100K" → 4.8-5.9 min (estimate).
- "LC2 takes ~2 h" → 2.5-4 h, with a request timeout of at least 1 h.
- "Long context costs 0.3-1.5 ms a token per 100K" → measured +4.4 to +5.6 ms a token at 32K.
- "Research build 77 decoded at 7.8 / 7.0 tok/s" → it was ~1.1-1.4 s extra in a 16-token decode, cause unknown.
- "The bank displaces ~92 residents" → 96-97 a rank at 192 GiB.
- "An oracle reaches 52-86% absorption" → in-sample only; out-of-sample orderings reach at most ~24%. The profile
  follow-up and its gate of 30% on chat and long are unlikely to pay.
- "Drafter on the extra card: +0-5%, central +2.5%" → about −1 to +2% for the whole drafter.
- "Drafting at 4/8 streams would not pay" → true for chat and at 8 streams; about break-even for code at 4 streams.
- The growth bug was attributed to "the index writer's page against the IDs" → that was the Nov 1 divergence, fixed by
  77f22d8e. The current failures have identical IDs.
- "The worker needs compaction" → or the P1 skip mark, or dropped-slot rows. Each needs its own byte check.
- T1's gate encoded only one direction → accept either. The component needs at least 3 runs, and tracing may hide the
  bug.
- "The mitigation gate (component, 100+ frames)" → cannot pass as written.
- "Run a vm_update_mode=0 A/B" → requires a module reload and your approval.
- "Both research locks plus BoltzGen idle is a guard" → it is not.
- "A 212,992 MiB RAM-cache arm" → deprioritized. It gave no gain before, at 1,024-token chunks (NIGHT-LOG.md:189,
  HANDOFF.md:54), and the extra 16 GiB caches decode-hot units rather than P1's cold ones, so it is a weak proxy.
- "Rank selection by the largest group of identical cards" → it must be bounded by `--tp`, stay opt-in, and create
  nothing on the extra card when the worker is off.

**Unverifiable estimates:**
- Every fifth-card gain.
- The other-roles replay, which was not reproduced; a cross-check suggests its upper end is optimistic.
- The 4/8-stream decode-loss model at large Lm.
- The driver mechanism.
- P2P to and from c3.
- Contention between c3 and c6.
- BoltzGen's VRAM use.
- The worker's per-layer round trip.
