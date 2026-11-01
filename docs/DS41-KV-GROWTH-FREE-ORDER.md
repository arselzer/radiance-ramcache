# Growing KV: preserve legacy physical ID order

Source-only fix off `53fe48b0758765a91589e7d6d30ad0d4b7e86870`. No build, test,
simulation, device or model execution was performed in this lane. The original failed run is
preserved at `work/autonomous-20261101/growing-kv-512k-run1` in the build workspace.

## Failure and existing producer contract

The raw 512K cold configuration reserved 1,686,110,208 bytes per rank with 2,097,152-byte
granules. Group 6 (`kv_index_r1`) begins at 1,013,919,744, contains 524,313 physical IDs,
and uses 256 bytes per layer/block. ID 4304 starts exactly at global granule 484. The failed
entry-store destination was this group's base; the first missing write was ID 4304.

The first compressed-r1 page also maps global granule 547, which covers the end of the index-r1
group starting at index ID 520400. The old backing allocator preferred an already backed free
ID to an unbacked lower ID. After low index IDs 0..4303 were consumed it could therefore return
520400, while the compressed group returned 4304. Existing DS41 producer paths deliberately
share the compressed group's slot mapping for the latent and index stores; the index reader
uses the index group's table. A one-granule mapping patch would not repair that ID mismatch.

The fixed allocator returns the global minimum logically free ID. Paired groups with equal
allocation, retention, COW and release histories consequently receive equal IDs. This change
restores that existing contract in backing mode; it does not introduce independent index
producer mapping, change a kernel, or broaden the supported engine scope.

## Private candidates and public backing

`KVManager::backing_stage` builds the same existing private min-heap from every logically free
ID below the live mark, regardless of its current backing. The separate committed mask and
`free_usable_` count continue to describe only completely backed IDs across every layer
fragment. Withdrawal, all-rank publication, pending-COW checks, preflight intent pins, rollback
and fatal poison are unchanged. No unbacked candidate is an allocatable public ID.

`backing_need(g,N)` extracts the lowest N distinct free candidates from that heap, discarding
only stale held/duplicate entries. It restores candidates before invoking any callback, then
requires backing for those exact IDs. A callback may swap the heap publication; no reference
to the old heap is used afterward. Ordinary budget refusal cannot fall through to a higher
already mapped ID. A missing/invalid candidate or successful callback without complete backing
fails closed. The existing final `grab_block` backing check remains in place.

The N-entry workspace is the existing candidate vector, reserved before any heap removal.
Restoration cannot exceed the heap's pre-extraction size/capacity. There is no second block-sized
heap or bitmap and no per-allocation scan of the refcount array. Single-ID allocation outside
preflight uses the mapped minimum directly; other peeks cost O(N log B) plus amortized removal
of stale entries, with B private heap entries. Actual transaction staging still visits every
ID and stores both old/target publications, as it did before this fix. Populating the existing
reserved heaps now touches all free candidate entries; no latency or memory-cost claim follows
from source inspection.

## Allocation-path audit

- `ensure` (including partial-block COW) and `seed_window` preflight each group's exact candidates
  through `grow_for`/`backing_need`; intent IDs remain pinned across later group transactions.
  Their second pass and direct/bulk reserve calls reach `grab_block`, which checks the exact
  current minimum again before changing its refcount.
- Specific-ID restore uses `backing_need_ids` before reservation. Its stale heap entry remains
  harmless: held entries are skipped and duplicate free entries are deduplicated during a peek.
- Prefix adoption requires existing held IDs and verifies their backing before adding refs.
  Fork/retain do not create an unbacked reference. Pending copy source/destination ranges remain
  in the required inventory even after their final logical reference is released.
- Release, window recycle, cancel-like sequence cleanup and snapshot seeding keep their original
  ID/refcount behavior. Public free counts remain backed-only; logical admission availability
  can include future backed capacity. Follower mask agreement, quiescent callback ownership,
  all-rank transaction rollback and fatal poison are not relaxed.

Only backing-enabled implementations change. The fixed allocator, geometry, budgets, flags,
ABI, producer/reader arithmetic and expert residency remain unchanged.

## Prepared host cases and next gates

New auto-globbed target: `kv_backing_free_order_test`, four host-only metadata cases:

1. Exact native 1607-MiB/512K geometry: G483/G484 boundary, G547 high-tail diversion, publication
   rollback, ordinary refusal of low ID 4304 despite mapped high IDs, then exact legacy ID 4304.
2. Fixed versus backing paired-group traces through native M16 chunks and boundary heights
   2144/2151/2152/2153/4303/4304/4305; all table IDs, refs and state slots must agree.
3. Prefix retain/fork/COW/adopt, pending-copy inventory, release/reuse/cancel-like cleanup and
   nonempty window snapshot seeding, comparing original fixed and backing managers.
4. Bulk and specific reservation, stale duplicate heap entries, distinct ordered candidates,
   and invalid-lease/fatal callback refusal with every subsequent allocation withdrawn.

The fixture touches manager metadata only. Its legacy host-VMem reference reserves/maps native
anonymous address space but never reads or writes cache payloads. It requires the host backend
and 2-MiB VMem granules; a skip is not evidence that these cases passed.

Root must build/run this target plus existing `kv_backing_allocator_test`,
`kv_backing_transaction_test`, `kv_window_test`, `kv_live_controller_test`, `mem_test` and
`sched_test` regressions under the normal serialized host rules, then rebuild
the entire coherent engine/plugin cohort. Fresh guarded real-card transaction/publication
proof and the failed 512K model case must pass before any growing-KV qualification. Preserve
the full rounded physical MAX, variable cache budgets, physical ledger/COPY/Kh/reply checks
and normal production-container restoration. This source fix alone proves none of those gates.
