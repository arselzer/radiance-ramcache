# Exact DS41 physical KV pages and later physical growth

Source preparation off `3fe07391956274d0529b95d553a234f5e2032cf7`. No build, host/device test,
model gate, measured saving, or performance claim accompanies this document. The production
configuration remains plain BF16. All runtime checks belong to the integration owner's serialized
build queue and two-lock QFN stop/restore window.

## Fixed packed-page contract

The compressed latent and index keys already undergo FP4 roundtrip before being stored as BF16.
Packing captures those original codes/scales and actual BF16 results; it must never quantize the
stored BF16 values again. The supported active domain is the existing producer output, initialized
zero pages and byte copies. An arbitrary BF16 writer is not supported. The independent codec/card
oracle must establish producer closure, including signed zeros, BF16 subnormals, Inf/NaN payloads,
the index producer's exponent255 and its F32 half-product FTZ case, before runtime enablement.

The public storage IDs are zero/plain, one/FP4 E4M3 group16 and two/FP4 power-of-two group32. The
power-of-two descriptor is not UE8M0. A row is planar: original FP4 nibble payload followed by one
scale byte per group. Stored scale0 contains two literal BF16 patterns and a bit selector inside
that group's existing payload, so no raw side allocation, relocation pointer or full BF16 shadow
is required. A zeroed row decodes to positive zero. Invalid stored metadata is an error; it must
not silently decode as a plausible value. The codec owns the exact descriptor semantics.

| Logical producer row | Payload | Scale plane | Physical page/layer |
|---|---:|---:|---:|
| 512 BF16 latent elements, groups16 | 256 B | 32 B | 288 B |
| 128 BF16 index elements, groups32 | 64 B | 4 B | 68 B |

There is no per-row16-byte alignment requirement: consumers decode with byte loads, page copies
support a multiple of4 bytes, and pool group bases retain the existing allocation alignment. The
68-byte index stride is four-byte aligned even when a later layer stripe is not16-byte aligned.
Padding it to80 bytes would cost30 extra bytes per logical token in the current cache geometry.

`RadKVGroupDecl` retains its logical BF16 geometry and appends default-zero `storage_format` and
`storage_block_bytes`. The latter is one layer's whole physical page. Plain requires both zero.
The first packed contract permits only FULL, one-head, logical BF16 groups whose entire logical
page is one original512/128-element row. The core checks the format, physical pitch and logical
geometry during declaration and manager planning. It binds a packed layer as honest U8
`[blocks, physical_page_bytes]`; an ordinary rank4 BF16 reader cannot mistake it for its layout.
Every actual RAD_KV operand is guarded before launch against the selected band/domain: matching
declared/resolved format, the exact dedicated packed op/kernel and its cache operand role, plus
the physical dtype, shape and pitch. Plain fallback readers/writers, a later mismatching band,
packed attention's ordinary BF16 window slot and sliced packed views are refused. Declaration
alone cannot prove this: the ABI records buffer liveness, but actual KV handles arrive at issue.
Packed streaming is refused. Plain BF16, the raw F32 compressor tail and the window remain
available with their existing layout and configurable cache/expert budgets.

The current retained source groups are three ratio2 groups and one ratio1 group, each with latent
and index pages. Full-history cost per logical token per card changes from3200 bytes to
`3*(288+68)/2 + (288+68) = 890` bytes, a72.1875% reduction in that component. At32768/524288/1048576
tokens, full-history pages cost27.8125/445/890 MiB rather than100/1600/3200 MiB. The fixed window,
F32 tails, RoPE, index scores, other activations and allocation alignment are additional and
unchanged. These are layout arithmetic, not measured process VRAM. A decoder transient, if any,
must be charged separately and may not be presented as permanent-cache savings without reporting
its peak.

Every physical page is self-contained. The manager's layer fragment is the physical pitch;
allocation, admission, expert loan strips, COW, prefix entry accounting and idle-tier transfer
already use that fragment. Refcounts continue to count logical blocks, including cached prefix
owners. A snapshot still carries raw state and the window only; FULL pages remain referenced by
the prefix cache. Replicated tier slots now also require equal storage format, block size and
bound-layer count, since equal byte widths alone do not imply equal content. No persistent tier
is shared between engines with different declarations.

Prefill `Kh` hashes logical BF16 bits. Packed debug hashing decodes one page into bounded scratch
and preserves every old per-position hash-run boundary, including the two halves of a ratio2
page. Hashing complete decoded pages instead would alter the length-mixing receipt despite equal
values. Unwritten physical pages remain unreadable until the producer publishes their completed
logical block; the producer must write the complete physical row before readers or prefix publish.
Zero decoding is defined for explicitly initialized zero pages, not a promise that malloc/device
allocations are automatically cleared.

## Current physical backing versus logical growth

The existing allocator does not physically grow with a sequence. `KVManager::carve_into` commits
the whole carved range at startup and states that it stays backed for process lifetime
(`core/mem/kv.cpp`, backing section). `ensure` adds blocks from that already backed free list.
`grow_for` can recall expert loans but does not map more VRAM. `lendable` exposes unused FULL
group stripes; `Engine::loan_tick` and the mover temporarily place reproducible expert copies
there. This saves expert movement while retaining all KV backing; it is not physical KV shrink.
The recommended deterministic mode disables the general loan path.

`Pool::init_elastic`, `commit` and `decommit` in `core/mem/pools.cpp` already reserve stable virtual
addresses and map/unmap physical granules. Commit rounds outward; decommit rounds inward. Without
VM support, the documented fallback allocates the maximum and these calls are no-ops. The KV pool
is created through this seam, while `Pools::weights()` is presently initialized as a fixed
allocation. Therefore merely decommitting idle KV does not let the fixed expert pool exceed its
existing size: physical KV growth must coordinate a real expert-capacity policy as well.

## Safe opt-in growth implementation boundary

Reserve an explicit maximum virtual address range, logical block IDs, each group base and every
layer stride once at startup. Keep those addresses stable for block tables, captured passes,
prefix refs and tier transfers. Configure a minimum and maximum physical KV budget separately
from this addressability limit, plus fixed state/window floors, a chunk reserve and hysteresis.
Compute every physical charge from the actual group fragment, including288/68-byte packed rows;
do not assume the192GiB host cache, full RAM residency or a particular prompt/context size.

At a quiescent step boundary, before new IDs/table entries become visible:

1. Determine which granules the requested blocks touch on every rank. Include all bound layer
   stripes, window/state floors, pending COW destinations, tier-copy destinations and a step's
   reserve. A granule can contain bytes from adjacent pages/groups; ownership is physical, not
   merely a count of logical tokens.
2. Recall lent strips and drain all readers/copy events. Remove reproducible expert copies and
   reduce their allowed physical allocation before committing KV. The admission calculation must
   account for both pools and headroom; releasing a loan descriptor alone releases no physical
   memory.
3. Commit the required KV granules on all ranks. Stage the new backed-ID/free-list frontier until
   every rank succeeds. If any rank fails, roll back newly mapped granules and staged expert/KV
   capacity changes while IDs/tables remain unpublished. Only a successful all-rank transaction
   can admit the request. Treat an irrecoverable rollback failure as an engine error, never a
   partly backed block table.
4. After sequence/prefix release, consider shrinking whole unowned granules. A prefix reference,
   fork, snapshot/tier-copy lifetime or pending COW still owns its pages even when the originating
   request ended. Retain minimum/chunk hysteresis, drain readers, unpublish backed IDs first,
   decommit, then make the returned physical budget available for expert copies.

An initial implementation can shrink only suffix granules of each fixed layer stripe. The
existing ascending free-block heap and held-block histogram make that tractable, but a high
prefix-held ID can pin a large suffix and holes cannot be reclaimed by counting free blocks.
General granule ownership/fragmentation tracking may later reclaim holes without moving virtual
addresses. Physical unmapping does not change kernel captures; changing a base, stride or bound
tensor shape would require invalidating/rebuilding them. Non-VM backends must report fixed-max
fallback instead of claiming dynamic VRAM savings.

The root lane's separate [growth policy design](DS41-KV-GROWTH.md), header
`core/mem/kv_growth.h` and `tests/kv_growth_test.cpp` were prepared at `c5498719` (integration
`2167bd20`). They plan actual mapped/holder granules, min/max/warm budgets and hysteresis, including
early release when needed to cap peak. That policy is preparation, not wired runtime growth.
Runtime work still needs backed-ID admission changes in `KVManager`, shared expert/KV
physical budget enforcement, quiescent all-rank transaction integration, drain/rollback handling
and source/card lifecycle gates before any active deployment.

## Source gates supplied by this lane

`tests/ds41_kv_page_test.cpp` covers physical format refusals,1/2MiB budgets, multiple nonadjacent
bound layers, physical capacity and rank2 U8 binding, plain rank4 BF16 fallback, shuffled ratio1/2
logical debug runs with literal signed-zero/NaN bits, COW/cancellation, prefix retention/adoption/
eviction, packed streaming refusal and equal-width/different-format replicated tier refusal.
Its host-backend snapshot case keeps raw F32 tails and BF16 windows unchanged. ABI size/offset
expectations are updated with the declaration. The codec lane supplies numerical/producer closure
oracles separately; the runtime lane supplies the default-off producers/readers and dispatch
markers. All are required before a model byte/memory window; no source test result substitutes
for the card gate or long KLD, logical `Kh`, prefix/cancel and concurrent reply checks.
`kv_storage_guard_test` and the packed-KV case in `runtime_test` cover every allowed cache role
and its format/op/band/physical-view refusals. LCP's index-run report counts the packed alias;
producer aliases retain the existing one-kernel-per-op admission rule and require their original
norm pins in the architecture, not a new permissive multi-kernel family.
