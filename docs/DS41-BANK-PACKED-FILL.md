# Exact packed CPU-cache source for DSpark bank fill

Source only, not built or executed. This follow-up is independent of the first
bank-lending measurement and changes no kernel, public ABI, cache budget, bank
geometry, mask, weight address or publication protocol. It depends on the pure
inverse in daa19f6f. The root's independent bank callback return-type fix
2a9469fa must also be present in the integrated build; it is not repeated here.

`RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_PACKED_FILL=1` explicitly enables
this source adapter and requires `...DSPARK_BANK_LEND=1` with its existing valid
backup cap and all bank admission restrictions, including Dx. Unset or exactly
`0` preserves the existing raw-RAM/original-file branches and their log strings.
Other values, or enabling it without lending, refuse startup. The option never
increases the expert cache budget. Current192GiB, reduced96GiB and future128GiB
cache configurations use the same contract.

The published CPU packed form already preserves the original nibble bytes and
every original e8m0 byte. `packed_scales_equal` checks all scales before cache
publication. P1's direct packed transport is already implemented; RingLender and
the first bank version instead use original-file extents when a CPU lease is
packed. This adapter supplies the missing packed-to-native inverse at bank fill
only. It does not repeat the earlier lossy weight/dtype proposals, introduce a
different weight representation, or claim a measured I/O or throughput benefit.
The archive's packed-cache/refused-expert and original-file fallback results
remain relevant: a partial cache hit cannot authorize a partial native pair.

## Admission and byte contract

`draft_bank_packed_fill::prepare` consumes the actual Program/Plan, unit ID and
two active `Mover::HostUnit` leases. It checks:

- A file-tier trunk unit, layer0..39/expert0..95, native GU then DN, with no
  slab, movable/host-forced/draft placement, packed VRAM layout or duplicated
  weight. Logical shapes must be4608x5120 and5120x2304, with conditional unsharded
  native `ggml_mxfp4` weights. Stored U8 row shapes must be4608x2720 and5120x1224.
- Raw stored sizes12533760 and6266880, offsets0 and12533760, and both the unit
  size and physical destination slot exactly18800640. These geometry-derived
  sizes have no inter-weight or end padding; the independent native BLOCK
  oracle compares every byte. Router masks are separate512-byte planes.
- Both active leases actually say packed; the HostUnit unit/count/packed fields,
  host/device pointers, raw sizes and64-byte-multiple physical slot lengths agree
  with their owning immutable leases. Mover's borrow already checks each slot
  against the configured cache class size. Neither input, workspace nor relay
  may alias another. Packed header shapes, extents, exception ordering and bounds
  are then checked by the independent pure inverse for each complete matrix.
- Both weights use the trunk source owner and same source generation. The real
  Mover's source epoch check runs before and after inverse preparation, and again
  before and after the synchronous card copy. All ranks are rechecked before
  cohort publication. A stale source aborts the fill and enters the existing
  full-cohort restore path.

The inverse reconstructs original scale bytes and uses the existing integer
native relayout. No float operation, requantization or memory-kind change occurs.
Both complete inverses must succeed in the backup's existing one-pair relay
before the engine issues its single whole-pair H2D copy. No source bytes point
directly at a card-native reader. On validation/inverse refusal, with source
epochs still valid, the engine clears the entire relay and uses the unchanged
original-file reader/relayout. A second-matrix refusal may leave a completed
first matrix in the relay, which is never copied by itself.

Both immutable cache leases stay held through synchronous copy completion,
including file fallback and a failing copy. `copy_sync` drains its stream even
when enqueue fails. Only after every selected pair on every rank has landed
does the existing controller publish any borrowed residency. Its full-cohort
rollback/strict restore, parked aliases, mask checks and query gate are unchanged.

## Bounded memory and counters

There is no new payload buffer, host worker, vector, heap allocation or source
read allocation during a transition. One `Workspace` and fixed counters are
retained inside `DraftBankLoan`, so its complete `sizeof` remains included in the
explicit backup cap and host-room preflight. The enabled path additionally
charges a4096-byte call-metadata allowance. The source call chain holds only two
small parsed views/headers, a maximum32-byte scale expansion, two plane pointers,
integer/SSE relayout locals and scalar temporaries; it has no recursion or
variable/model-sized stack storage. The4096-byte allowance is conservative source
accounting, not a measured compiler stack watermark. Existing source extents are
still preallocated/charged for fallback. All future unbacked cache backing and
configured system headroom remain reserved.

The original `DSBANKLEND` startup/totals strings remain unchanged. Enabled-only
`DSBANKPACK` markers report configuration, each rank's fill and final totals:

- `packed_hits`: both inverses completed; a subsequent copy/rollback failure is
  still separately visible. `packed_refusals` counts attempted packed pairs
  refused before copy. `packed_misses` counts a failed paired cache borrow.
- `packed_source_bytes` counts physical packed bytes read on complete inverse
  hits; `packed_raw_bytes` is their exact native BLOCK output size.
- `raw_ram_bytes` and `source_file_bytes` distinguish the original two sources;
  `inverse_ns` includes validation plus both inverse calls, including refusals.
  `whole_pair_fill_bytes` counts successful complete copies. It is not an SSD
  bandwidth, physical disk-read or end-to-end performance claim.

## Prepared validation, all unexecuted

Exact host target `draft_bank_packed_fill_test` has eight cases: strict opt-in;
actual paired cache leases/full native GU-DN bytes and guards; fifteen malformed
plan/weight/source variants; inconsistent or inactive leases; failure of the
second inverse after the first completed; malformed/truncated second packed
record; source epoch failure before/after conversion; and source/relay/workspace
aliases. Keep the existing `ds41_packed_inverse_test`, `draft_bank_lend_test`,
`draft_bank_backup_test` and `native_mapped_copy_mover_test` gates too. Root runs
exact host ctest names under build.lock with `-LE gpu`; none has run in this lane.

Separate target `ds41-bank-packed-fill-check` is never registered as a host test.
Inside an ordinary two-lock QFN stop/restore window only, with all four cards
visible, its CLI is:

```text
ds41-bank-packed-fill-check --inside-guarded-window --backup-mib 512
```

It refuses a host backend or non-gfx1201/TP4 inventory. Before host payload
allocation it reserves512MiB of conservative oracle room plus2GiB headroom.
Its test-only original BLOCK reference/planes/cache and initializer are included
in the explicit cap. Through the production preparer, actual packed-cache leases
and production synchronous copy adapter, three profiles exercise all256 code and
original scale bytes, low/high packed bases, raw exceptions, native GU full rounds
and DN partial tails. Each full pair is copied/read back on every card and all
18800640 bytes are compared to the independent libquant original relayout, with
complement-seeded readback. Leases remain active until copies complete. It also
checks stale-source/malformed-second-record refusal, partial-target rollback and
a real successful-but-short restore that must poison the whole image. Masks are
checked during each strict cohort restore. Required literal success starts:

```text
PASS: DSBANK_PACKED_FILL_CARD ranks=4 profiles=3 pair_bytes=18800640 compared_bytes=225607680 leases=through_sync_copy partial_fill_rollback=1 short_restore_fatal=1 masks=1
```

This tool does not prove actual Engine publication, every source-file epoch or
model replies. Require fresh receipts bound to the entire checker binary/core
candidate and full libr4d SHA, alongside the existing bank backup/draft/Rb/Vp
gates; do not reuse a receipt from the first fallback-only window.

The later model window must keep the first measurement intact and compare
packed-fill0/1 with identical bank+Dx configuration, hot set, mixed workload,
grace/cooldown and explicit cap. Use matched192GiB and96GiB roles plus a plain
base reply reference;128GiB is an optional additional functional budget, not a
hardcoded admission value. Require startup enabled proof, nonzero actual packed
hits on every rank, exact replies/counts/hashes, no partial publication, observed
4-running→anchor-solo-resume, all-rank restore/mask/alias proof and successful
production restore. The reduced-cache role must actually observe paired cache
misses and positive file bytes; a96GiB label alone is not coverage. If this frozen
mixed workload has no miss, mark the miss gate incomplete and schedule a separate
predeclared miss workload rather than claim coverage. Price startup, inverse,
fill, restore/readback, idle and teardown in end-to-end results. No adoption or
speedup claim is authorized by source review or the raw-copy oracle alone.
