# Optional bank backup subphase timing

Source only; no build, host test, syntax check, device oracle or model run was
performed in this lane. This is diagnostics, not a runtime optimization. It
does not change FNV, source/restore bytes, memcmp, copy order, synchronization,
alias/mask invariants, bank ownership or query readiness.

`RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_TIMINGS=1` is an explicit strict
opt-in and requires `...DSPARK_BANK_LEND=1` with its existing valid budget and
admission. It is independent of packed fill. Unset/exact0 leave timing off;
other spellings, or1 without lending, refuse startup. Existing callers of the
private Config parser retain their previous defaults. When off, no new clock
or timing log call occurs and every timing counter remains zero. Existing
`DSBANKLEND`/`DSBANKPACK` log strings remain unchanged. Retained counter storage
still exists and is honestly included in normal `sizeof` budgeting.

The clock is the existing `rad_mono_ns()` steady-clock source. Each bucket stores
cumulative `ns/attempts/ok/requested/okbytes` as checked uint64 values. These are
elapsed host wall nanoseconds, including preemption. Requested/call-success
bytes do not claim measured physical DMA traffic or pure GPU time. A successful
copy API can still have a short-copy fault; the unchanged strict readback and
memcmp decide whether the bytes are exact. A failed memcmp can exit early, so
only requested bytes, not a fabricated full memory-read count, are recorded.

| Bucket | Exclusive operation being timed |
|---|---|
| `alloc` | Each existing pinned allocation call, including its own allocator/pinning work |
| `zero` | Each existing memset of the saved reference span before first D2H |
| `cap_d2h` | First synchronized D2H reference copy |
| `cap_fnv` | The original dependent FNV byte loop, same order and arithmetic |
| `cap_complement` | Existing complement seed for the second capture read |
| `cap_verify_d2h` | Second synchronized D2H capture read |
| `cap_memcmp` | Existing exact capture memcmp |
| `restore_h2d` | Existing synchronized full raw-span H2D restoration |
| `restore_complement` | Complement seed for raw restored-span readback |
| `restore_d2h` | Synchronized raw restored-span D2H readback |
| `restore_memcmp` | Existing exact restored-span memcmp |
| `invariant_complement` | Complement seed for immutable router mask readback |
| `invariant_d2h` | Existing synchronized mask readback |
| `invariant_memcmp` | Existing exact mask memcmp |

Each full copy count/requested byte total is retained even when its API reports
failure; `ok`/`okbytes` distinguish reported completion. Allocation and memcmp use
their actual pointer/result as success. Zero/hash/complement count completed
loops. No copy is added to measure a phase. Each copy still selects its rank,
enqueues, synchronizes even after enqueue failure and returns to rank0.

After it enters image capture, the native wrapper emits ONE bounded `DSBANKTIME` line after capture (including
post-capture invariant validation) and after each complete or failed restore
attempt. Lines identify operation/status/ready/failed/overflow, successful
restore generation, unchanged hash, actual retained timing `stats_bytes`, extra
workspace charge, clock and column schema. They contain all14 cumulative
buckets. Per-transition differences come from successive cumulative snapshots;
do not add repeated snapshots or label them exclusive interval totals. A
pre-lend invariant check accumulates until the next restore line. Generic tiny
Image tests expose stats directly and do not introduce a runtime log bypass.

The buckets are exclusive, but their sum is not the entire Engine startup or
transition timer: source selection, metadata/pointer checks, host-room admission,
drains, source reads/relayout, publication and log formatting are outside them.
The existing fill/restore/drain/startup timers remain nested in actual process
and request wall times. Never add these timers again to full lifecycle cost.
Packed `inverse_ns` remains a separate existing fill diagnostic.

Counter updates are transactional: if any attempts/success/time/byte addition
would overflow, every field of that bucket remains unchanged, overflow latches,
and the Image is poisoned before it can become query-ready. Clock regression or
an unrepresentable clock value also poisons the image. A partial operation that
could not fit a counter is not silently wrapped/saturated into a valid sample.
No clear/reset API exists. Lend, restore, release, poison and failed/repeated
capture retain counters and the overflow latch. A newly constructed Image starts
all-zero. Successful retry of a failed/overflowed image remains forbidden.

Memory accounting grows by the actual retained `sizeof(DraftBankTimings)` in
Stats/Image/Loan, automatically included in their existing cap/host preflights.
Enabled timing additionally charges8192B fixed call workspace. The formatted
line bound is2188B: fixed header plus14 names≤32B and five20-digit columns each.
The existing logger simultaneously uses a4096B stack buffer. Both plus1024B
scalar/varargs reserve fit8192B. The line is shorter than that logger buffer,
so its long-line heap allocation path is not used. There is no new payload
buffer, worker or allocation during capture/restore. No cap expands automatically;
an explicitly insufficient backup cap still refuses before pinned allocation.

New SOURCE cases in exact host target `draft_bank_backup_test` cover timing
off/on same hash/bytes/exact copy order; zero off counters; transactional overflow
and lifetime latch; exact tiny exclusive counts/requested bytes and workspace
charge; failed allocation/copy and failed recapture; successful-but-short restore
with failed memcmp; and injected image-counter overflow remaining poisoned.
`draft_bank_lend_test` adds the strict independent configuration boundary.
Tests intentionally assert no positive duration threshold: zero nanoseconds at
clock resolution is not proof of missing work. Existing full backup/native
scope/failure cases remain required. Root alone builds and runs exact host tests
under build.lock, nice19 and `-LE gpu`, then a guarded model byte gate before use.

The first measured N24 capture took22.365s, while fill/restore took1.983/2.678s.
The FNV loop and pinned allocation are source-identified suspects, not separately
measured bottlenecks yet. A future diagnostic window must use a frozen workload,
same matching flags/cap/cache, price this instrumentation's own overhead and
retain full cold/warm lifecycle cost. Faster diagnostic hashing or batching legal
contiguous copies are conditional next leads only; literal strict memcmp and
full-byte all-rank restoration remain mandatory. Neither is implemented here.
