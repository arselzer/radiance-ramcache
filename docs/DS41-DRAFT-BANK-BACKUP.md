# Native DSpark hot-bank byte backup

Source only. No build, host test, card oracle or model result was run by this
feature lane. The lender/controller integration and all validation belong to
the integration owner. This helper adds no environment enable or engine caller.

`DraftBankBackup` captures the unchanged native EP4 DSpark bank after loading,
parked-alias publication and the existing router-mask write/readback. Each of
four ranks must have N hot experts in each of three stages, for N=1..32. A hot
slot is the full native 18,800,640 bytes (GU 12,533,760 plus DN 6,266,880), at its
existing `Mover::unit_base` address and native offsets. Nothing is reencoded or
requantized. Packed, movable, host-forced, differently laid-out or overlapping
slots are refused. Every parked expert must alias the first hot weight of its
own stage and slot, in the same Plan order as `draft_vram_aliases`.

The immutable snapshot includes every hot-slot byte, including padding, and
the three separate 512-byte router correction planes per rank. All hot and
parked Program and Residency pointers, fixed hot-slot bases, and mask pointers
are recorded. Mask planes are invariants, not lendable storage: a changed mask
causes permanent failure rather than silent repair. Startup capture requires
the normal mask operation to have succeeded; the snapshot cannot establish
that operation's history.

## Budget and storage

An explicit positive `DraftBankBackupBudget.backup_mib` is required. No amount
is derived from installed RAM, and this code does not resize the expert cache.
One pinned allocation per rank contains its immutable snapshot followed by one
native-pair scratch buffer. Capture readback, restore readback and target-file
relay use that scratch sequentially. Relay and readback are reported as roles
of the same allocation, with the shared peak charged once.

Per-rank pinned request is
`(3*N + 1)*18,800,640 + 3*512` bytes. The real adapter conservatively charges
`round_up(request, 2 MiB) + 2 MiB` for each rank, covering AQL's no-SVM
`pin_whole` host-pool allocation as well as its smaller SVM physical rounding.
This is a ceiling for the backing allocator's request, not a claim that all
drivers physically consume the ceiling. The cap also includes the complete
fixed helper object, its bounded startup descriptor, and caller-supplied
`controller_metadata_bytes`. The caller must include any retained lender/IO
metadata outside the helper. No unbounded heap vector stores backup metadata.

Before any pinned allocation, the entire cohort's charged size is admitted by
the existing `lockable_room` MemAvailable/every-cgroup check, reserving
`max(0, expert_cache_budget_bytes - expert_cache_backed_bytes)` plus
`other_future_bytes` for future allocations and keeping
`system_headroom_bytes` free. Actual backed expert-cache bytes are supplied by
the engine. Fixed metadata is charged conservatively even if its storage
already exists. The controller must admit any additional allocation it makes
before constructing/allocating that storage; it must not allocate uncharged
relay, snapshot or readback buffers after this check. These rules apply equally
to 96, 128 and 192 GiB cache configurations.

`DraftBankBackupDeviceIO` uses the same `RAD_MEM_HOST_PINNED` allocation kind,
the rank's existing stream, synchronous raw copies and a return to rank0's
selected device. It drains the stream even after enqueue failure. Injected IO
must outlive the helper. Null IO selects the helper's owned real adapter.
`device_io().copy_sync(...)` and `scratch(rank)` let the controller use this
same bounded relay for file-to-bank target fills.

## State and failure contract

Capture is one-shot. It allocates all four rank buffers before copying any
bank bytes, then reads each span twice. Before each verification read, every
scratch byte is set to the complement of its saved reference. A truncated
read cannot pass merely because untouched bytes were already zero. Strict
`memcmp`, rather than a hash, decides byte equality. A skipped first-capture
zero byte that is already zero remains an exact snapshot; the second read
still proves the resulting snapshot bytes.

`mark_borrowed()` clears readiness before the controller overwrites any hot
slot. Restore checks invariant planes, writes every raw byte back to the
original addresses, strictly reads back every slot, and checks planes again.
The wrapper checks every Program/Residency alias before and after the cohort
restore. Any copy, allocation, pointer or byte failure permanently latches
failure; a late rank failure cannot publish a partially restored cohort.
Completed restore count advances only after the complete cohort passes.
The FNV image hash is diagnostic and names the immutable cohort image; it is
not an independent per-rank proof or a replacement for the byte comparison.

`ready()` means a valid captured image/current restore, and is a cheap Boolean
query. It does not grant scheduling permission. The controller must require
its own Restored phase, no active context/query, proper all-rank drains and
at least one new successful cohort restore after each Lent phase. Expensive
`validate_invariants()` is for transition boundaries, never every query.
Readback/restore failures are fatal; the controller must not fall back to
serving a potentially borrowed or partially restored bank.

## Required validation

The auto-discovered host target `draft_bank_backup_test` contains native
metadata admission for all N=1..32 and four ranks, refusal/alias mutations,
variable cache budgets, cap/room/allocation failures, truncated capture and
readback, repeated exact restores with relay reuse, later-rank partial writes,
mask corruption, overlap/overflow and one-shot lifetime cases. Tiny spans run
the actual image copy mechanism; native shape cases allocate no model payload.

The controller integration still needs a two-lock four-card window exercising
real startup capture, target fill, pointer/mask stability, exact restore and
fatal partial-copy handling. Model gates must include solo→concurrent→solo,
cancellation and repeated transitions, compare reply/token bytes with the
merged base and fixed DSpark controls, and retain reduced-cache/miss coverage.
Charge fill, strict restore/readback and hysteresis in end-to-end timings;
there is no throughput or adoption claim from these sources.
