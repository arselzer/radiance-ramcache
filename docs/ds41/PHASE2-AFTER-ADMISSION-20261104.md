# Phase2 after current admit-all: implementation status and next evidence

2026-11-04, source/selected-existing-log review only. No build, test, import, simulation, trace processing, GPU work, engine edit or model change. Current four-XT/192GiB expert-cache configuration remains the comparison; all effects of a future Phase2 executor are unmeasured. Original IQ4-Engram hybrid reply/state baseline stays binding.

## Current evidence changes the reason to pursue it

F5 qualified quiet47,917-token TTFT100.254→61.789s (−38.37%) with Ada/Li64; subsequent canonical N8 server rate46.672→47.162tok/s showed no measured regression. See `F5-ADMISSION-20261104.md:25–38`.100K and prefix admission are still unmeasured. The old several-times/drive-reread forecast is inapplicable after this change.

A bounded read of the two **matched48K Kh** logs in `tools/ds41-host-experts/build/ab-20261104T121517Z/` supplies more useful current traffic evidence. Identify the original Ad128 and Ada/Li64 `...-N8-C1-Pf-Pmr8-Ba-Kh-Lm131072-Ulong48k-r1.server.log` through `work/f5/B-summary.json`; rank0 footer lines571/572, other ranks608–686. These are one original long request plus its retained16-token tail, not quiet timing arms.

| P1 metric | Ad128 | Ada/Li64 | Meaning |
|---|---:|---:|---|
| Rank0 actual drive reads, including read-ahead |8,177|1,900|Storage pressure fell; Ada is not a literal once-only-read guarantee: dropped read-ahead still exists |
| Rank0 demanded drive units |6,958|609|The source shifts mainly to RAM leases |
| Rank0 drive busy / P1 chunk span |80.2% /106.87s|29.7% /62.26s|Read-worker activity fraction, **not GPU utilization or critical-path attribution** |
| Rank0 issuing-thread wait for drive units |38.103s|3.168s|A large old wait has already disappeared; overlaps other phases |
| P1 ring copies, ranks0/1/2/3 |31,210 /31,760 /31,657 /31,341|identical|Actual successful P1 weight-copy enqueues |
| P1 DMA GB, ranks0/1/2/3 |561.84 /571.74 /569.89 /564.22|identical|2,267.69GB total, log-rounded; repeated **aggregate** traffic remains |

These bytes are **GU+DN weight copies into P1's prefill supply ring**, counted after `Device::copy` succeeds by `core/place/card_expert_stager.cpp:292–310`. They exclude the arena/ring lender's later decode-resident reloads (`:1118–1253`, separate `lend_` counters at `:1455`). The same logs report48/47/47/47 residents reloaded once; those189 reloads must not be subtracted from, or renamed as, P1's125,968 ring copies. The logs do not identify repeated Plan unit IDs across pieces, representation equality, independently completed DMA duration, or the fraction on the critical path. Unchanged aggregate volume is a reason to measure reuse, not a speedup estimate. Ring-full waves and ready waits describe backpressure and can be effects of slower GPU consumers.

## Existing checkpoints: real source, no active executor

| Checkpoint / worktree | Implemented source | Qualification / missing part |
|---|---|---|
|90648e2c, `wt-phase2-plan`|Unused `core/place/ds41_phase2_retention.h`: aligned conservative retained-plane accounting and variable device/host budget checks; HOST fixture source; `docs/DS41-PHASE2-AFTER4K.md`|No allocator, runtime flag, architecture/scheduler/P1 caller or executor. Documentation says source-only/unexecuted. |
|285d2e62, `wt-phase2-exec`|Unused `core/place/ds41_phase2_routes.h`: owned original-piece route frames and local expert-union metadata; HOST fixture source; `docs/DS41-PHASE2-EXECUTOR-PLAN.md`|Despite branch name, **no executor**. Max4 original pieces/8,192 rows. Metadata does not grant placement, launch or lifetime authority. |
|6dfbada2, `wt-phase2-trace`|Defaultoff actual route/copy audit, `ds41_phase2_audit{.h,.cpp,_stager.cpp}`, real producer witness and P1 copy callback, bounded readback/output; fixture source and route-audit document|Instrument source exists; no completed build/HOST/GPU receipt found in the selected HANDOFF/NIGHT records. It is absent from current stable runtime. Its source admission explicitly requires static EP4, **C4096 and early issue OFF**, and refuses TBO (`ds41_phase2_audit_stager.cpp:12–16`). Current C2048/Ei arms cannot use it unchanged. |

The helpers/tests were prepared, not established as passed by these branch descriptions. Test targets are automatically globbed by `tests/CMakeLists.txt`; absence of an explicit named CMake entry is not proof of missing registration. No Phase2 execution/model speed has been qualified. Historical2K→4K copy reduction in the old reports is a real earlier result on a different configuration, not current Ada/P4 qualification.

## Smallest next measurement and cheaper alternative

1. **Finish P4's model decision first.** Select its eligible current baseline; preserve Ada/Li64, W/G, Rb/Vp, consumer cuts and all original receipts. Do not port the whole obsolete trace cohort or weaken its C4/Ei refusals.
2. If another source observation is justified, reuse the **already written successful-copy callback** in `wt-phase2-trace/.../card_expert_stager.cpp:311` and its header observer seam. The smallest new observer can record actual sequence/original piece/absolute range/rank/layer/Plan epoch plus `(unit, packed, GU bytes, DN bytes, source kind)` per successful P1 copy. It does **not** need a GPU route-ID/sort readback just to count repeated copied weights. Bound it to four original pieces/8K rows and fixed per-rank storage/output caps, with complete terminal coverage and unchanged normal end-pass drain. Source/ABI/Plan/library/prompt authority must be pinned externally; source/format/epoch changes cannot be merged as one reusable unit. This lighter copy ledger is a proposal, not another implementation already present.
3. Reconcile its sums exactly to P1 counters over the same covered passes. Report actual repeat copies of the **same immutable Plan unit and representation**, separately from CPU cut pieces/residents and separately from lender reloads. A successful enqueue is not an independent per-copy completion timestamp; normal model completion/byte gates remain required. First off/on observation window is correctness/coverage only, not a performance claim. Existing full route-audit source is useful later for original sorted-range/frame proof, after a narrow current-profile port and HOST/native gates.
4. Before an executor, the cheapest actual H2D-reduction experiment is the **existing exact4K chunk path with Ada/Li64**, versus current2K, using fresh current-cohort native/Kh gates, quiet6.8K/48K brackets and same-process postlong N8. F5 did not qualify that combination. Larger activation reservations reduce decode residents; actual arena lending/return and N8 must pay for them. Existing4K/lending source is not automatic adoption of the unqualified larger lender cap or P4 behavior at a new height.

An idle-plane preparer could be redundant because W already warms it. Actual repeated H2D is therefore the more concrete next **measurement** lead; neither this priority nor saved byte volume establishes an executor's benefit. Retrying the old12-slot ring/grace/cache-policy experiments without new ownership/capacity mechanics does not hold experts across40layers.

## Smallest missing active slice

If the current copy ledger and4K comparison justify structural work, start with **one fresh sequence, no prefix/resume/EORS, no concurrent decode/draft/aux/media/LoRA, fixed KV, at most8K unsampled rows**, retaining its original C2048 piece boundaries. The existing6,817 prompt has688/2,048/2,048/2,032 card-piece rows (6,816 total); do not create a single8K numerical call or infer boundaries by ceil(prompt/C). The final normal prompt/sampling path stays ordinary.

The minimum functional design is producer-only layer-major: run layers0–19 across these original pieces, then finish layers20–39 in original piece order from owned L20 inputs. It still needs:

- **Owned batch/KV span transaction before writes:** projected original `plan_chunk` descriptors, immutable token/position/cu/slot/state-index/mask arrays, all-rank funding and full-span block leases. Current `KVManager::ensure` recycles from the ordinary piece's first query (`core/mem/kv.cpp:1324–1380`); later pieces must not recycle/reassign blocks containing earlier pieces' still-live other-layer stripes. No ordinary step commit to obtain the next descriptor. The current architecture entry takes one mutable RadBatch; a new explicit bounded span-entry contract is required, rather than hiding an opaque span inside an existing field or repeatedly calling normal step while suppressing publication. Abort/drain must agree on all ranks before any collective participant exits.
- **A real post-scatter/pre-GU seam:** current `MoeFP8::pass_head` performs router/scatter **and** GU/SwiGLU/DN/shared work (`arch/common/rad_block_moe_fp8.h:1735–1850`); the architecture's Seg2 calls it or even the whole pass (`deepseek41.cpp:3734–3762`). Seg1/Seg2/Seg3 alone cannot expose a union. Own each piece's FFN/routing/shared/mHC/down-output buffers and immutable launch tensor/parameter arrays; bind declared retained storage so ordinary arena liveness cannot reuse it. Retain the fused next-layer `xn` output (`:3774–3785`). P1 currently accepts only one capture triple per layer/pass (`card_expert_stager.cpp:770`).
- **One unit supply, every original piece:** fetch the same packed-RAM/pinned-pool/file representation once per layer/span, then replay each selected piece at its old height/stable sorted ranges. Keep its GU/SwiGLU scratch sequentially valid and its down output retained; release the ring slot only after the last piece consumes it. Preserve original six-slot gather, shared contribution, ascending-rank F32 fold and BF16 store. Ada/Li64 admission/recency order and later decode residency need an explicit gated policy; do not import the old last128-only plan by accident.
- **Attention/state semantics:** dense/norm/rope/index/compressor launches keep original heights, absolute positions and per-row causal/full/key cuts. Later pieces' written compressed/index pages must remain masked from earlier queries. Preserve compressor entry/work/tail stores exactly once in original token order, raw F32 tails/padding, and per-layer Engram history. Index selection/keep and piece-local compressor work survive until their real consumers. No new sort, enlarged reduction, deduped Engram gather or approximate arithmetic.
- **Atomic completion:** publish no prefix/checkpoint/n_computed/sampler state midspan. Even prefixoff validation requires owned snapshots of original logical piece state before later stripes overwrite it; final-span state cannot stand in for earlier Kh checkpoints. Cancel/failure drains compute/DMA/CPU/readers/leases/tables then discards the unpublished fresh sequence; no implicit rollback proof. Prefix/EORS/COW reuse requires a later, separately funded per-checkpoint all-layer transaction and stays refused initially.

The first independent implementation prerequisite is an immutable, funded **original-piece/span owner plus non-recycling KV/activation lifetime**, with allocation refusal before mutation and ordinary-C2 fallback. It is not another unused byte-budget helper. The active post-scatter/replay integration only follows that proof; no implementation is authorized or supplied here.

## Bounded memory and host-transfer costs

The conservative inventory is176,446B/row at max_ctx8K. Its variable candidate mask is `4*ceil(ceil((max_ctx+1)/8)/32)` B/row for this helper's ratio1/block8. At the current max_ctx131,072, `keep_words=513`, so the inventory becomes **178,366B/row (174.186KiB)**. Source: retention header:61–90; current architecture's keep formula `deepseek41.cpp:1134,1180`. Do not use172.3KiB unchanged at128K or1M.

| Retention scope | Per rank, GiB | All4 rank copies if placed on host, GiB |
|---|---:|---:|
|Bounded8,192-row span|1.361|5.443|
|Bounded16,384-row span|2.722|10.887|
|Whole47,917-row prompt, illustrative only|≈8.0|≈31.8|
|Whole99,987-row prompt, illustrative only|≈16.6|≈66.4|

These are arithmetic storage models, **not actual allocation measurements**; add alignment, original-piece route/capture metadata, host/pinned allocation padding, final-hidden/sampler rows, checkpoint copies and dense/ring/KV scratch. The helper permits16K inventory, but existing route-frame/audit scope stops at8K. Whole-prompt retention is unnecessary and cannot be assumed to fit today's headroom while preserving192GiB cache. A bounded host alternative preserves variable cache budgets only with aggregate admission against actual host/cgroup room, future unbacked expert cache and system headroom; refuse/fall back rather than silently shrink the cache.

Device-resident retention avoids host activation shuttling but needs actual placement. Conservatively add full retained storage; no subtraction of an old arena subset is certified. Plain BF16 WINDOW backing alone grows about**240MiB/card for C2→8K** or560MiB/card for C2→16K (40layers×1,024B×extra rows, before block/table alignment). Thus additional retained+window storage is about1.60/3.27GiB/card, before the other costs, unless a smaller certified rolling lifetime replaces it. Existing Qe64 does not automatically lend these new allocations back to decode residents. All-card funding, safe loan/reclaim and N4/N8 impact remain open.

Host spilling has a substantial additional traffic term. Illustratively, shuttling the **entire**178,366B retained inventory once out and back per producer layer would move about342GB/card at48K and713GB/card at100K over20producer layers, or1.37/2.85TB across4ranks. That is a **conditional roundtrip model**, not a necessary lower bound, actual copy plan or latency forecast; selective placement can move less and capture/replay phases can add trips. The next design must list exactly which fields remain on device and which cross at which phase. Measure sustained all4-card H2D+D2H/DRAM overlap and synchronization cost on that real schedule; single-direction peak bandwidth cannot price it. Required cost is actual added activation bytes/achieved mixed-direction throughput plus nonoverlapped waits, compared with the measured weight-DMA critical path saved.

## Priority gate

Require current same-unit copy overlap and per-phase critical-path evidence; actual bounded arena/host/KV funding with resident restoration; native original-piece row/sort/gather/logit and every raw state/window byte gate; fixed-context6.8K/48K and100K when filled-context scope is qualified; smaller-budget actual misses; cancel/abort/recovery; and quiet whole-TTFT/postlong N8 brackets. Preserve every original raw oracle, including padding/unused declared rows; no geometry or common-view waiver. Prefix/EORS follows its separate transaction proof. A source helper, a large traffic number or favorable old drive model does not clear any gate.

Current decision: **measure and test existing chunk reuse first; keep Phase2 executor deferred** until those results justify its state/placement cost. No several-times forecast survives current admission evidence.
