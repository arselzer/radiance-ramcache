# Bounded live physical KV growth

Status: complete opt-in source cohort, not built or executed. This connects the backing policy,
all-rank transaction, backing-aware allocator and actual Engine pre-admission hook. It is not a
measured optimization or a token-exactness result. The ordinary serving path stays the default.

The first cohort requires native DeepSeek4.1 Flash, TP4, max-num-seqs1, deterministic fixed expert
placement and an explicit speculative depth0. It refuses P1/arena lending, TBO/second-lane
declarations, a generic prefill stager, actual drafting, media/embedding/generator/LoRA, host-prefix
tiers, streamed KV, general KV loans and backends without real elastic VMem. Retained VRAM prefix
references and raw-state/window checkpoint snapshots remain supported. Plain BF16 and packed
288/68-byte FULL pages use the same physical-growth machinery. No expert resident is added when
KV backing is released: the expert pool remains fixed. Joint expert/KV funding is separate work.

## Explicit physical budgets

All six environment settings are required to opt in. Unset/0 enables the existing fixed-backing
path; supplying growth tunables while the main switch is off is a refusal.

```
RADIANCE_EXPERIMENTAL_DS41_KV_GROW=1
RADIANCE_EXPERIMENTAL_DS41_KV_GROW_MIN_MIB=0
RADIANCE_EXPERIMENTAL_DS41_KV_GROW_MAX_MIB=<actual rounded KV VA reservation>
RADIANCE_EXPERIMENTAL_DS41_KV_GROW_WARM_MIB=0
RADIANCE_EXPERIMENTAL_DS41_KV_GROW_HYSTERESIS_MIB=0
RADIANCE_EXPERIMENTAL_DS41_KV_GROW_HEADROOM_MIB=<explicit runtime free-memory floor>
```

Values are nonnegative decimal MiB; MAX is positive and MIN<=MAX. MAX must equal the entire
granule-rounded stable KV reservation on every rank. Startup names the required value on a
mismatch. For example,187MiB logical with a2MiB backend granule requires188MiB physical MAX.
The first cohort deliberately funds every advertised logical page; a lower physical cap requires
a separate admission/can-ever-fit proof and is refused here. No context or expert-RAM budget is
hardcoded. MIN is a total minimum, not an extra allocation beside mandatory holders. The actual
backend granule rounds MIN up, MAX down, and charges sparse mappings and the tail honestly.

The full maximum is checked against fresh free memory after eager context/weight allocations,
plus late step/queue allocations and the larger of configured startup GPU headroom and explicit
runtime headroom. Only the minimum and all raw-state/window pool floors are physically mapped at
startup. Thus free memory exceeds the eager-backing case, but is reserved funding for later KV,
not unused expert-pool capacity. At runtime, each positive mapping peak must leave the explicit
runtime floor plus remaining lazy-allocation allowance free. A pure release may heal pressure.
External allocation races still return a fatal, safely unpublished transaction failure.

## Admission, publication and lifetimes

The Engine disables pipelining/ahead and calls `Scheduler::step_quiescent` for every ordinary
plan. It holds the scheduler ownership mutex across the drain, shrink and complete planning
phase, including prefix lookup/adoption, COW/page allocation and BatchBuilder upload. Its paired
rank barrier must be in an even parked generation before and under this mutex; the sole step
thread has not released the next generation, so CPU rank issuers cannot restart during mapping.
Parking is distinct from GPU completion. The callback drains both Ctx compute lanes, host
row-prefetch, FileStager CPU/miss work and RAM insert helpers, mover mapped-source leases and
transfer queue, and the checked tier queue on every rank before changing backing.

None of those drain helpers calls Scheduler: FileStager end_pass/hx_drain finished CPU expert
work before the rank join, insert/read-ahead workers use only mover/RAM-cache locks, and already
submitted transfer packets need no scheduler callback. `prepare_release`, hash staging, snapshot
restore and tier_tick remain outside this callback. Cleanup ends the controller lease under the
same scheduler lock on success, failure or exception, avoiding recursive locking or an unowned
allocation interval.

The allocator publishes no unbacked ID. Before grab/COW/specific reservation/adoption/restoration
it requests exact candidate IDs. The controller translates all bound layer fragments, including
cross-granule pages, and performs a real all-rank transaction before any requested ID is handed
out. Leader sequence and retained-prefix refs, pending COW source/destination refs and transient
multi-group allocation intents pin their physical pages. Copies already removed from the pending
inventory are covered by the preceding stream drains. Whole raw-state/window pools are fixed
floors. Native sparse padding reads a separate zero key rather than a speculative cache block0,
so it adds no unowned page0 floor.

All ranks stage replacement backed-ID masks/heaps/counts before withdrawal. New backing becomes
allocatable only after every rank succeeds. Commit failure removes every newly mapped extent,
including a failed-access mapping recorded as physically present, then restores old backing
where possible. Any failed cleanup/restoration poisons publication; even a successful rollback
is fatal to this Engine cohort and withdraws every rank. Scheduler's separate allocation-error
latch prevents eviction/retry or a page-table upload after a fatal mapping result. Stable VA,
maximum layer strides and captured addresses never move.

At the next drained pre-admission boundary, released sequence/prefix/COW holders permit physical
shrink to the mandatory union plus configured warm/minimum room and hysteresis. Retained prefixes
can keep fragmented granules pinned. There is no background relocation or live stride resize.
Kh hashing copies only the referenced physical FULL-page runs, preserving the old logical BF16
hash order/length while avoiding D2H reads through unmapped max-VA holes.

## Receipts and pending validation

Every real transaction logs one `DS41_KV_GROW rank=... epoch=... cause=startup|grow|shrink` row per
rank, with logical/reserved/granule/old/actual/min/max bytes. A successful startup also logs
`DS41_KV_GROW_READY` and per-rank `DS41_KV_GROW_BUDGET` free/actual/deferred/maximum bytes. No-op
plans add no transaction marker. A model gate must require actual grow and shrink markers for
all four ranks, stable VA/strides, and Kh/token identity, rather than treating READY as activity.

Prepared new targets: `kv_live_budget_test` (three parser/tail/headroom/funding cases),
`kv_live_controller_test` (three actual host Pool/VMem/all-rank-manager composition cases) and
`kv_hash_runs_test` (two sparse plain/packed physical-fetch cases). Controller cases cover real
minimum backing, multi-group growth, prefix-pinned shrink, final release, stable addresses/strides,
all-rank fatal funding refusal before ID allocation and rejection outside the ownership lease.
The allocator/transaction prerequisites add fault-injection, COW/cross-granule and scheduler fatal
cleanup cases. CMake globs add these files automatically after regeneration. No builds, syntax
checks, tests, emulators, simulations, GPU work or services were run in this source lane.

Root must run serialized host gates, a coherent full-plugin rebuild (packed ABI requires it),
then a locked QFN-stop/restore card window. Compare fixed versus growing plain storage first,
then packed only after its own codec/producer/reader gates pass. Include prefix reuse/cancel,
long/decode Kh, a small-budget/96GiB expert-cache path beside current192GiB, and forced growth.
No recommended configuration changes until those gates pass. The earlier
`DS41-KV-GROWTH.md` records the policy-only milestone; this file records the connected caller.
