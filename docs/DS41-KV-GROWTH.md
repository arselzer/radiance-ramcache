# Incremental physical KV backing

Status: policy prerequisite only, default behavior unchanged. `kv_growth.h` has no runtime
caller and cannot itself save VRAM. Its host test source has not yet been built or run.

Current `KVManager::carve_into` reserves fixed layer stripes and backs all carved bytes at
startup. Logical page allocation grows inside that backing. The existing loan recalls expert
copies from unused FULL stripes but does not map or unmap physical memory. General lending is
disabled under deterministic placement. `Pool` and `VMem` already support fixed virtual
reservations with granule commit/decommit; unsupported backends retain full physical backing.

The new pure policy accepts the actual mapping, explicit minimum/maximum physical budgets,
spare backing and shrink hysteresis, and every byte range held by a sequence, retained prefix,
COW operation, pending copy, fixed window/state or next step's writes. Any live byte pins its
whole granule. The planner keeps existing spare mappings before allocating alternatives,
coalesces adjacent changes and orders enough unheld releases before additions to fit the
physical maximum. Reducing a budget can release old backing before growth; required data is
never clipped. Invalid ranges and impossible budgets leave the output unchanged.

Runtime integration still needs all of the following:

- A fixed maximum VA layout and page-ID capacity; packed physical strides must not change after
  startup. Window/state pages remain mandatory. A retained prefix is a holder even after its
  original request ends.
- A complete holder-to-granule inventory, including pages crossing granule boundaries in each
  layer stripe, copy/event lifetimes and the next chunk. The policy cannot infer missing holders.
- A quiescent transaction across all TP ranks: retire readers/copies, recall expert loans, fund
  the joint ledger, commit mappings, then publish new allocatable block IDs and block tables.
  Failure must keep the actual mapping ledger truthful and must never expose unbacked blocks.
- Explicit expert-pool physical reclaim and expansion. Unmapping KV while keeping the expert
  pool's startup budget fixed does not produce more expert residents. Changing residency must
  separately pass the base-byte gate under native DS41 CPU/card/P1 paths.
- Cached graph addresses and maximum strides remain fixed; no relocation of live KV is needed.
  Empty holes can prevent full shrink, so reclaimed space depends on actual granule occupancy.
- Startup and runtime reports distinguish logical capacity, reserved VA, actual committed bytes,
  retained prefix bytes and the joint expert/KV budget. Lower budgets must preserve spill paths.

The prepared host cases cover tightly packed rows crossing granules, fixed/prefix holders,
hysteresis, budget reduction, mandatory-capacity refusal, integer bounds and randomized
sequence/prefix inventories. They model mapping order and budget peaks. They do not establish
GPU mapping, graph safety, multi-rank rollback or generated-token correctness.
