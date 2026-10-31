# Backing-aware KV allocation

Source only. This lane supplies the real KVManager allocator/publication adapter for the
existing backing transaction. No environment switch, engine startup enablement, physical
budget controller or runtime saving is introduced here. Those belong to the separate engine
integration. No source test has been run in this lane.

Call `enable_backing_mode()` after group planning and before carve. The carve reserves the
existing maximum addresses and strides, skips its eager physical commit, and initially exposes
no allocatable pages or state slots. The unchanged legacy carve remains the default.

`KVManagerBackingPublisher(KVManager*)` implements the transaction's stage/withdraw/publish/
rollback/poison interface. Stage derives an explicit committed page mask from every layer
fragment of each physical page, including packed pages crossing mapping granules. It builds
both old and target filtered min-heaps/counts before any withdrawal, validates every held page
and pending COW source/destination, and requires the target's complete window/raw-state floor.
An empty old bootstrap map is valid only without holders. Withdrawal hides all allocatable
IDs/counts; final publication uses allocation-free swaps. Poison is idempotent and hides every
ID. This mask is separate from the contiguous expert-loan mark; loans and streaming are not
supported in backing mode.

The leader's synchronous growth callback receives exact `KVBackingBlock {group,block}` IDs.
`backing_block_ranges` translates them through each rank's own fixed physical layout. The
integration must append those ranges to `backing_required_ranges(leader)`, execute the all-rank
transaction under its already-held pre-plan quiescence lease, and publish every rank together.
No callback may acquire Scheduler ownership recursively. Followers must have common eligibility
before following the leader or reserving the same restore ID.

Allocation preflight pins intended writers before they receive refcounts. `ensure` and window
seeding may grow several groups before grabbing anything: a later group's transaction must not
release an earlier group's newly mapped candidates. The manager includes these temporary IDs
in its required inventory and validates them in the target publication until the complete
attempt returns. Actual grabbed/COW/retained-prefix IDs then carry the ordinary references.
Free blocks count only currently published mapped IDs; admission availability includes future
unbacked free capacity. The initial engine cohort will fund the whole rounded logical maximum;
an optional pure capacity callback exists for future tighter physical ceilings.

`KVBackingGrowResult {status,fatal}` distinguishes ordinary budget pressure (`FULL`, not fatal)
from a failed map or invalid lease. Fatal errors latch in the manager, prevent all further
allocation, and are exposed through `IKVManager::allocation_error`. Scheduler checks the latch
before eviction/preemption retry and before BatchBuilder uploads; `step_quiescent` returns the
original failure. Its optional allocation-free/noexcept `after_plan` callback clears the engine
lease while the same ownership lock is still held, including failure and exception exits.
A rolled-back map failure is still fatal for inference: the controller must poison every rank,
even though the transaction restored the old heaps successfully.

Prepared host sources cover empty bootstrap/fixed floors, mask-filtered heaps, intended writer
lifetimes across groups, prefix retention/adoption, COW, exact/specific restore, cross-granule
pages, ordinary budget refusal versus fatal mapping, follower mask agreement, and actual manager
publishers in a later-rank failed-map rollback. Mapping backends in those cases are metadata
fault fixtures; the existing Pool/VMem transaction case separately exercises real host mapping.
Scheduler sources cover admission/running failures with zero eviction retry or batch publication
and lease cleanup on success, failure, throw and fatal mapping. They do not establish device
mapping, generated-token correctness, expert-pool growth or long-context serving.
