# Physical KV transaction prerequisite

Source only; no build, test or runtime/model gate has run. This is an executable transaction
component with a real `Pool`/`VMem` adapter, a readonly `KVManager` holder inventory and prepared
fault-injection test source. There is **no engine caller, KVManager backing-mask publication
adapter, runtime switch or active dynamic KV allocation yet**. Default serving behavior remains
startup backing of the whole carved KV pool. The pure plan is documented in
[DS41-KV-GROWTH.md](DS41-KV-GROWTH.md); physical packed pages in
[DS41-KV-PAGES.md](DS41-KV-PAGES.md) retain their fixed maximum virtual geometry.

`kv_backing_transact` enters a mandatory quiescence lease, captures each rank's holder inventory
under that lease, reads actual physical maps/backend granules, plans all ranks and stages both
old and target publications. Funding reserves both the forward peak and rollback-to-old cost
before the first mapping change. Every publisher then withdraws all allocatable IDs. Each map or
unmap is one physical granule, followed by an actual-ledger read, including after errors. Only
when all ranks match their target maps does the transaction publish their preallocated target
masks and leave quiescence. Publication callbacks are infallible, allocation-free swaps.

Failure never assumes a negative backend call left mappings unchanged. Rollback first removes
every newly mapped extent on every rank, including physical-but-inaccessible extents left by a
failed access grant. Only after every new extent is gone does it restore any early-released old
backing and then restore old publications. Early releases contain no held data, so their stale
free-page bytes do not need restoration. If rollback unmap/read/commit fails, every publisher and
funding lease is poisoned and inference remains stopped. A failed restore is fatal even when its
physical map happens to equal the old map: physical presence does not prove successful access.
The separate VMem error-ledger fix `29319f85` preserves physical records when access-error undo
cannot unmap a new granule; this component depends on that truthful bookkeeping.

The real adapter binds the owning device for each backend operation. `Pool` exposes its actual
rounded VA reservation, actual backend granule, physical presence queries and explicit whole
physical-granule transaction methods. Ordinary logical `Pool::commit/decommit` ranges are not
extended past `Pool::bytes()`. A tail granule may extend beyond the logical pool budget, but its
whole size is charged physically:187MiB of logical KV with a2MiB backend granule needs188MiB if
every logical byte is backed. A187MiB hard physical maximum cannot authorize that mapping. The
caller must explicitly fund the rounded physical cap; the component never silently rounds the
hard maximum upward. Nonelastic pools refuse dynamic operations instead of claiming a saving.

`KVManager::backing_required_ranges(owners, out)` translates the scheduler leader's FULL-page
refcounts through each rank's own physical group offsets, layer stripes and page pitch. Prefix
owners remain counted after their sequence ends. Pending COW sources and destinations are pinned
even if cancellation removed their final logical owner. Every window/state allocation is a
conservative fixed floor, retaining the FP8-roundtrip window and raw F32 compressor state.
Followers use the leader's ownership, not their own empty refcount arrays. Active KV expert loans
and streamed KV are refused. The caller must drain copies already removed from the pending list,
and must append any external holder and future write ranges. The method changes no allocation,
free list, backing or prefix state.

## Exact live-engine seam still required

`Engine::step_loop` calls `sched_.step()` (or promotes an already issued step) before
`prepare_release`. Scheduler admission and growth have already adopted prefix IDs, allocated
pages and queued COW by then. `prepare_release` parks rank threads, follows the leader's loan
marks and composes streamed views, but an ahead pass or draft/tier/mover copy may still be active
on the device. Appending a decommit to `loan_tick` is unsafe for the same reason.

The first live mode should disable step pipelining and tree/block/serial speculation, acquire
the scheduler ownership lock before planning/admission/tier restore, and drain **all** rank
compute/second-lane/draft, mover transfer, FileStager/P1 and tier-copy work. The quiescence lease
must stop HTTP cancel/prefix release/publication while inventories and publications are staged.
Idle/deferred-cancel paths need the same lease. Device selection must be returned to rank0 when
the transaction leaves. Encoder work is another protected reader/writer and must also be drained
or refused in the initial mode.

A real publisher must make every `grab_block`, `reserve_blocks`, `reserve_specific_block`,
`adopt`, follower/tier restore and COW allocation consult the published backed-ID masks, counting
a page eligible only when **all** its physical layer fragments are backed. Rebuild/preallocate
free heaps/counts before withdrawal; swap without allocation after all-rank success. Existing
`live_` is a contiguous expert-loan frontier, not a backing bitmap. Merely lowering a live mark or
decommitting a granule while free-list IDs remain usable is not acceptable. Admission must be
able to request/fund a future backed reserve before taking those IDs, without preempt/recompute
livelock. Maximum logical IDs, tensor shapes, group bases and layer strides stay fixed for
captures; no logical resize accompanies a physical map change.

The initial integration can preserve all existing expert residents and use the configured KV
maximum as its physical budget. That can free physical VRAM while context is short, but it does
not place more experts in the fixed startup weights pool. Increasing expert residents requires
a real joint KV/expert physical ledger, explicit elastic expert capacity, drain/eviction before
KV growth, and rollback funding that retains the old budget until commit. Prefix-held high IDs,
page/layer granule sharing and fragmentation can limit shrink. Min/max/warm/hysteresis must be
explicit, and all accounting must work for arbitrary host/cache budgets and the128GB spill path.

## Prepared gates

`kv_backing_transaction_test` covers all-rank withdrawal before mapping, late-rank failure,
early-release budget peaks, inaccessible maps after failed commits, rollback-unmap poison,
failed old-access restore despite equal physical maps, failed ledger reads, quiescence/stage/
funding/inventory refusals, actual granule mismatch, logical-tail physical charging and failed
tail-map rollback. Its host-only real-Pool case maps/releases actual host VMem without changing
addresses; its manager fixture retains prefix refs and cancelled pending COW on leader/follower
layouts while pinning raw state/window floors. These sources have not run. They do not replace
the eventual engine publication adapter, GPU VM behavior, cache-budget regression, long logical
KV/logit/token gates, prefix/cancel/tier lifetimes or measured expert/memory results.
