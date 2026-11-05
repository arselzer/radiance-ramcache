# Aux placement and scheduling: source-only follow-up

2026-11-05. Read existing source/reports and bounded log tails only; no imports, helper runs,
replays, simulation, tests, builds, APIs or GPU work. No implementation is proposed as qualified.

**Answer:** a worker holding experts outside RAM can be more valuable per hit, because a useful
hit may remove a drive/packing stall as well as CPU work. A stronger first candidate is to keep
hot CPU experts on the worker but remove their mandatory RAM duplicates, letting RAM hold the
colder experts. Actual current capacity is close enough to the complete routed-expert set to
make this a high-priority structural experiment, not merely a small hit-rate adjustment.
The useful target is **frequency × marginal critical-path time saved
per VRAM byte, less migration/contention cost**, with a mixed set when both kinds have value.
Current serving cannot realize a worker-only cold tier merely by changing its ranking.

## What exists, and what actually passed

| Work | Status and evidence |
|---|---|
| Static hot CPU-path worker | Adopted optional/current32d. F1 ab-20261104T051616Z:104/104 replies,8/8 Kh,3397/3397 Kld, Axv0 differing rows; total decode +24.1% N8,+18.1% N4,+19.9% N1. F2 ab-20261104T064211Z: chat N8 +19.4%, harness decode +14.7%, all-turn wall −5..−7%, prefill TTFT substantially unchanged. |
| Dynamic sharing + online set | Implemented on private wt-aux-dynamic/e9aafb1a; coherent build and32 HOST entries plus harness checks recorded. NOT merged/adopted/current32d and NOT GPU/model qualified. F4A ab-20261104T113441Z failed/refused all9 arms: figml occupied11,748MiB; no aux/auxswap/Axv measurement. Do not report old replay forecasts as results. |
| Online policy already implemented | Decayed live hit counts + sealed prior; per-rank replacement quotas,16 spare slots,2 swaps/token (credit capped16), every step, half-life512, prior256, margin100%=strict improvement. Boundary publication, retired-slot fences, bounded waits,2MiB copy pieces, readbacks, sharing release/reacquire and dry-copy/no-flip arm already exist in the private implementation. This is not new work to reinvent. |
| Missing trace / parked gates | F3 harness trace was refused with P1; F4A's supposed no-P1 trace still had Vp and failed startup. No valid harness route trace from those attempts. Dynamic cross-queue auxswap, actual swaps/Axv, release/reacquire/tenant lifetimes and dry-copy cost still need real gates. Existing DSpark+worker S1 remains deferred; current finite D2 qualification does not cover aux/prefix/concurrency. |
| Other already identified levers | Finish-time decline at low concurrency,17–319-row prefill extension, global inter-rank quota changes, prompt seeding, elastic partial-arena release and mapping RAM slabs to aux were explicitly deferred in AUX-FOLLOWUPS/aux-dyn SPEC. Cap27000 and backoff8 were suggested switches, not new adopted gains. Qa0 was separately measured/adopted for the harness; its gain must not be counted again as a worker improvement. |

Current card discovery is already capability/topology based: same supported architecture, suitable
free capacity, preferably an unshared root complex, then free VRAM and deterministic tie breaks.
It is not a job finish-time scheduler. Current lanes accept when ready, otherwise decline/busy/
backoff; no measured CPU-versus-aux completion predictor chooses each job. The old generic
CPU/copy-link greedy split also failed to beat all-CPU policy, so static bandwidth intuition is
insufficient. Earlier copy/overlap studies showed DRAM/PCIe contention can consume gross gains.

## Correct ownership and sizes

`arch/deepseek41/deepseek41.cpp:1097` declares routed experts **expert-parallel**, not TP width
shards:384 whole routed experts/layer,96 owned by each of the four ranks. A worker unit is one
owned `(rank, layer, local expert)` gate_up+down pair, corresponding to one whole global routed
layer-expert. The shared expert is separately width sharded and is not one of these candidates.
Each pair is full gate_up4608×5120 and down5120×2304 (FF2304). It is not one quarter of an expert.

`ds41_aux.h:396–449` fixes current worker passes to **at most16 token rows**, at most96 sorted
top-k slots. At current raw-round budgets its packed arena stride is **18,001,920 bytes/unit**;
the actual capacity formula subtracts per-rank lane/table workspace before dividing the chosen
VRAM cap. F1 actually held **1,513 unique whole units across all four owner quotas** in26000MiB
(25.37GiB unit residency), not1513×4 replicas. F1's quoted39% is calibration mass of eligible
CPU-path candidates, not39% of all model weights or general workload coverage.

Candidates exclude primary-card slab/ring/lent residents, unsupported/malformed packed units
and nontrunk/nonrouted layers. The static set orders eligible units by sealed profile share.
It therefore complements primary-card residents, but in F1 all1513 loaded from packed RAM.

## The missing cold-tier lifetime/fallback seam

There is already a cold startup loader: `stager_aux.cpp:252–265` reads original HostPlanes and
uses the existing exact packed representation when a unit is absent from RAM. But the live
serve path `stager.cpp:2350–2381` only assigns an aux unit after successful `borrow_host_unit`.
An absent RAM lease increments `aux_stats_.uncached` and follows the normal CPU miss path.
`aux_fallback` at stager_aux.cpp:593 consumes the kept RAM leases. Private online filling still
uses RAM-only `aux_source` at wt-aux-dynamic/stager_aux.cpp:521. Thus cold preload/ranking alone
does not remove a demand read or free duplicate RAM capacity.

The minimal meaningful extension is:

1. Acquire a worker residency/read lease carrying worker/slot generation, unit identity and
   owner rank before removing a unit from CPU miss submission. Keep it until its actual GPU
   read completes; an abandoned job is not a completed reader. Reuse the private boundary/
   retirement machinery rather than inventing another slot pool.
2. Let resident-only units enter the existing compacted aux job without a RAM lease. Preserve
   sorted row ownership, output placement, original packed bytes/kernels and original rank fold.
   Keep the file/mirror epoch authenticated. Existing RAM-hit admission/touch semantics can
   remain unchanged initially; avoiding a cold read must actually avoid its early submission.
3. On decline/failure/deadline/release, materialize the original unit through the ordinary
   cache/file HostPlanes miss path, own its buffers through exact CPU ticket completion, then
   commit through the same existing row-placement path. A late GPU result stays abandoned and
   cannot overwrite the CPU result. Reserve bounded fallback staging before admission so that
   worker-only does not assume free RAM; include it in variable-budget accounting.
4. Release/replacement first withdraws admission, then drains actual readers before recycling
   a slot/arena. Cancel must join/release both CPU and aux ownership. Axv must compute the
   original uncached CPU path too; require actual cold offloads and forced cold fallbacks,
   not merely a passing arm whose cold path remained unused.

No value conversion, expert pruning, routing change, reordered reduction or approximate
fallback is needed. The following is a capacity bound, not an implemented all-fit claim.

## Actual union-capacity bound: potentially enough for all routed experts

The complete trunk set is40×384=15,360 whole expert units. At the standard aux packed stride,
15,360×18,001,920B is **257.51953125GiB**. The seven over-budget scale-tail units need their
existing exact exception storage, rather than pretending every unit fits that standard slot.

Accepted F5B `ab-20261104T121517Z/...-Ada-Li64-N8-C8-Pf-Pmr8-Ba-Lm131072-F5post-r1.server.log`
records these actual counts, read from bounded startup/shutdown text:

| Resource | Actual count / accounting | Unique-coverage treatment |
|---|---|---|
| Primary static VRAM |618/627/627/627 whole experts; **2,499 total** | Disjoint from SSD cache; includes one raw-tail expert with a block-form down weight |
| Native packed RAM cache |**2,848 GU slots AND2,848 DN slots per rank**,11,392 pair-slot capacity | Upper bound on complete unique pairs; actual identities/partial pairs must be checked |
| Net RAM accounting |47.75GiB logical payload/rank,47.99GiB conservative backing +2.33MiB metadata under48GiB cap |192GiB advertised cap is not192GiB expert payload; exact class payload totals190.986862GiB |
| Separate HostPinned routed units |3/rank, **12 total**, from the actual3840-unit/rank plan | Independent fixed host-pool units, not additional mapped-cache slots |
| Raw-tail side store |actual2/2/1/1, **six units** held,108MiB total | Other six of the seven uncacheable scale-tail units; the seventh is already primary resident |
| Aux capacity, separate F1/F2 measurement |**1,513 units** at26000MiB; same pair layout/capacity formula | Currently duplicates RAM by design; hypothetical disjoint capacity requires the new lifetime path |
| P1 lent activation VRAM |48/47/47/47, **189 currently published** after48K | DO NOT add as189 guaranteed unique units: these are profile-hot SSD units whose RAM copies are not excluded; they are withdrawn for larger prefill steps |

The actual RAM cache was full in that F5 arm at the end (all2848 slots resident in each class),
but aggregate class counts do not prove the same2848 unit identities in both classes or the
union's coverage. Cache warm-up also loads only positive-profile units; F1 left203–219 pair
slots idle/rank even after the short heldout requests. A full-coverage policy needs explicit
identity/coverage accounting and loading of the otherwise-zero-profile tail, not only capacity.

If disjoint and complete, the arithmetic is **2,499+11,392+12+6+1,513=15,422 units**, versus15,360:
only **62 units of global slack**, roughly1GiB. Even excluding pinned/side-store extras leaves
44 slots of aggregate slack. This makes near/full routed-expert coverage plausible, but does
not prove it. Missing pair matches, required staging/spare capacity, duplication, scale-tail
exceptions, context reservations and per-rank imbalance matter at this small margin.

With the stated fixed stores, the extra disjoint units needed per owner rank are
**369/360/361/361 (1,451 total)**. Historical F1/F2 N8 aux quotas were396/348/377/392: blindly
reusing those would leave rank1 **12 units short** despite the global surplus. Actual131K aux
quotas have not been measured. The private online policy preserves rank quotas; global quota
rebalancing was deferred. An explicit coverage floor or bounded cross-rank budget reassignment
is needed before an all-fit claim. Sixteen online spare slots would also reduce the1513 active
residency to1497, leaving only46 aggregate spare units under the same assumptions.

For comparison, actual F1 heldoutN8 and F2 chatN8 atL1024 had665/674/674/674 static units
(2,687 total),41 lent units/rank and the same2848 cache-pair capacities. F2's prefix harness
atLm32768 had642/650/650/650 (2,592),53 lent/rank and worker quotas393/353/375/392. Those extra
primary residents cannot be transferred to the accepted131K/N8 configuration. Its fixed KV
reservation is555,718,080B (~530MiB/card), with the log explicitly saying no KV is lent; empty
logical tokens do not make another cache-sized VRAM budget available. P1's lendable top of the
activation arena varies with step height and is taken back during large prefill. Growing KV
and elastic reuse have separate unresolved gates and are not assumed here.

The proper next census is small metadata: union the actual complete GU+DN unit IDs in RAM,
static/pinned/P1/side-store/aux IDs at matching boundaries, retaining rank ownership and source
epochs. Existing `Mover::units_cached`/present-many infrastructure can avoid payload reads;
no such new census or simulation was run here. Count overlap explicitly. An all-fit result
would eliminate warm **routed-expert** NVMe misses, not Engram/other checkpoint I/O, startup
reads or misses after the optional card is released.

## Preferred first variant: hot worker plus RAM deduplication

Keep the compute-frequent set on aux, but make its CPU RAM lease optional. While a validated
aux residency generation is live, use no-touch RAM checks on worker hits and deprioritize its
RAM entries for eviction/admission instead of refreshing their normal LRU/protection. Let the
freed RAM store colder experts. This preserves the demonstrated high worker compute utilization
while expanding unique coverage; deliberately cold GPU placement might have less compute value.
Compare both after accounting for critical-path wait and readback costs rather than assuming
one wins. The exact fallback/residency lifetime seam above is common to both variants.

Do not impose a blanket RAM ban. **Prefill passes above16 rows cannot currently run on aux.**
They still need its experts on the CPU/P1 path. Options, in increasing scope, are temporary
prefill-specific RAM admission, ordinary original-SSD fallback, or bounded exact aux-VRAM→host
packed readback under the resident generation lease, then existing CPU/P1 consumption. A D2H
readback must finish before source-slot recycling and its host buffer must survive every CPU
ticket/H2D reader. It copies original packed bytes; no conversion is permitted. Phase-end
admission should restore useful disjoint coverage rather than preserve unnecessary duplicates.
Zero expert NVMe misses would not mean zero weight-transfer cost: a frequently selected hot
aux bank could require repeated D2H readbacks on successive P1 chunks, contending with primary
H2D traffic, especially on the shared root. The62-unit slack cannot retain1513 additional RAM
duplicates without evicting some other unique experts. This is a reason to compare a cold/mixed
GPU bank for prefill, or later bounded phase-boundary swaps (hot GPU for decode, hotter RAM for
prefill), pricing all copies and retirement. Do not assume one fixed placement wins both phases.
Future tall aux kernels are a separate byte-gated optimization, not a requirement to explore
deduplication. Release/cancel/failure must revoke the RAM discount and preserve ordinary four-
card operation; do not delay a tenant's release by trying to copy the entire26GiB bank first.

## Ranked next design

1. **Build the hot-worker/RAM-dedup seam and coverage census as the structural candidate.** The
   current aggregate bound is enough to justify prioritizing it, with exact fallback/lifetimes,
   per-rank coverage floors and explicit prefill behavior. No all-fit/TPS result exists yet.
2. **Qualify the already-built bounded online set and sharing in parallel.** It attacks demonstrated
   workload mismatch without the new file-fallback lifetime seam. Existing engine-rule replay
   q5.out predicted CPU-row absorption30.1→40.1% heldoutN8 and21.6→41.9% chatN8; those are replay
   outputs over1481 old decode tokens, not TPS or current-profile results. Older +5..8%/+15..21%
   speed forecasts extrapolated measured elasticity and priced swap traffic; still unmeasured.
3. **Add completion-aware decline/dispatch.** Estimate CPU completion from submitted work,
   remaining critical misses and calibrated per-job costs; estimate aux finish from all four
   lanes' queue/service times, copy bytes and shared-root pressure. Offload only when the
   resulting layer completion improves by a margin. Do not treat miss reservations as already
   executing CPU work, and do not sum waits across concurrent ranks into a latency gain.
   Use per-device/per-root observations, with bounded smoothing and conservative fallback.
4. **Compare cost-weighted mixed/cold placement against hot-worker/RAM-dedup.** Rank each eligible unit by observed rate ×
   marginal saved layer time/slot byte, subtracting expected copy/readback/DRAM/PCIe interference
   and amortized replacement cost. A cold hit counts only the read/pack/insert stall not hidden
   by read-ahead or other computation. Keep a minimum residency interval, score-improvement
   hysteresis and both byte/time swap budgets; worker releases must leave the four-card path
   exact. Do not allocate everything to rare cold units merely to improve unique-cache hit rate.
5. **After evidence:** prompt seeding for N1 or17–319-row support. Coverage quota changes are
   required sooner if the actual131K ownership census shows a rank shortfall.
   The last requires tall-kernel byte gates and bigger workspace; existing16-row worker cannot
   accelerate most large P1 chunks. Old harness estimates were only1.1–1.5s of~72s summed TTFT
   and predate Qa0/admit-all; do not add them to today's measured prefill gains.

Useful actual bounds: F1/F2 static absorption was29.4% heldoutN8,20.7% chatN8,25.5% harness and
17.3% long Kld. F2 harness extra wait was715–958ms/rank over~43s decode, versus34–39ms/rank in
F1 N8; meanN1 roundtrip~270us versus134us idle auxrt. N8 logged roundtrips1.3–2.9ms are collection-
time upper bounds, not isolated service times. Outliers8–33ms and missed deadlines/backoff
justify conservative finish prediction. F1 N8 Ax-r1's retained4KB tail logs721 demand reads,
10,456MB read-ahead traffic and1 deadline miss; these prove real I/O exists, not its per-unit
critical-path value or today's cache profile. Worker choices must be measured against current
2K/admit-all/fixed-context settings, not transferred from old L1024/F1 configurations.

## Small future comparison, after separate component gates

Do not launch during the current GPU hold or displace the user's auxiliary-card tenant. First
qualify existing dynamic jobs/slot swaps and the cold-residency/fallback component separately.
Then one~9–11-arm canonical window can compare one new policy at a time, all at the SAME
four-XT/current192GiB/fixed131K/2K settings and SAME aux VRAM budget:

- Startup original heldout control, timing ineligible.
- Hot-worker/dedup Axv/Kh byte gate, requiring actual VRAM-only units and CPU fallback coverage.
- SAME hot set with normal duplicates versus hot set with RAM dedup at original N8:
  OFF/ON/ON/OFF, all eight original
  requests/caps each time, quiet diagnostic-free timings.
- Dedup forced-late/fallback gate, requiring actual cold fallback coverage and exact replies.
- Dedup controlled release/reacquire/cancel-recovery gate with original requests and full ownership.
- Optional complete original13-turn Qa0/Pc/Pce/Pcg16 normal/dedup pair; no Ba beside prefix cache.

Keep cost-weighted cold/mixed placement for a separate equal-budget comparison after that first
causal result; do not stack new dispatch and replacement policies into the same throughput A/B.

Use fresh native byte gates, original independent reply/count/full-Kh authorities, unchanged
weights/routing/arithmetic/output ordering, full accounting and every failed record. Record
critical CPU miss wait, aux extra wait, canonical N8 rate/wave wall, actual cold rows, NVMe/pack/
insert bytes, migration/readback bytes, queue depth, deadline/fallback counts and residency
turnover. Increased absorption without better canonical throughput/wall is not a win. Keep
variable RAM/VRAM budgets, arbitrary supported fifth cards and exact four-card fallback; defer
smaller-cache generalization until directly tested. No aux gain or new qualification exists here.

Sources: AUX-FOLLOWUPS-20261104.md; work/aux/{SPEC.md,README-WINDOW.md};
work/aux-dyn/{SPEC.md §R,README-WINDOW.md,replay/q5.out}; HANDOFF/NIGHT F1/F2/F3/F4A/S1 entries;
current32d core/{engine_aux.cpp,place/stager.cpp,place/stager_aux.cpp,place/aux_expert_worker.cpp};
tools/ds41-host-experts/ds41_aux.h; arch/deepseek41/deepseek41.cpp; private online source as named above.
