# Native DSpark hot-bank lending (default off, source only)

This instrument reuses actual static bank slots while the existing DSpark solo
rule suppresses bank work. It does not change proposals, verification, target
routing, or query shape. The intended gate is recovering the previously measured
concurrent bank-memory penalty; there is no measured gain for this implementation.

Integration dependencies: private quiescence `7fe96af6`, bounded backup helper
`0e95bd97`. Neither requires packed KV or a public ABI change. The new engine TU
and two new host test TUs are picked up by the existing CMake globs. No build,
test, emulator, device action or benchmark was run by the implementation lane.

## Configuration

`RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND=1` requires an explicit positive
`RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_BACKUP_MIB` cap. Unset or `0` keeps
the original execution order, performs no observation/capture/copy/allocation,
and prints no bank-lending markers. Ancillary values are ignored when off.
Every harness arm must state the flag explicitly; do not inherit a global enable.

The optional `RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_GRACE_MS` and
`RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_COOLDOWN_MS` are integers 0..600000,
both default1000. Grace requires consecutive inspected concurrent frames;
idle/noneligible frames reset it. Cooldown follows a verified restore/rollback.
Restoring is immediate when the existing solo hold releases, even if a cap or
cancellation leaves zero current query rows. No timer delays a bank reader.

Scope requires the existing native DSpark EP4 hot bank (`DRAFT_VRAM=N`, N1..32),
deterministic text-only block DSpark beside concurrency, and explicit
`RADIANCE_EXPERIMENTAL_DS41_DSPARK_CTX_DEFER=1` (Dx). Merely suppressing queries
leaves context passes reading bank weights. Initial scope refuses dynamic
placement, TBO, LoRA/media/generators/trees, generic KV/arena flex loans, and P1
ring/arena lenders. Ordinary idle P1 has a separate drain check. Existing
configuration, placement and packed-bank refusals still apply.

There is no cache/RAM capacity constant. Startup preserves the configured expert
cache budget and reserves its not-yet-backed bytes plus configured system
headroom before allocation. It first admits the controller metadata, then the
backup helper admits the entire cohort before any pinned allocation. The helper
charges four immutable bank snapshots, the three512-byte router mask planes/rank,
one shared relay/readback pair/rank, conservative AQL no-SVM allocation rounding,
all helper/controller metadata, and a shared source extent reader. Metadata
double charging is conservative. The source pair has18MiB bounded payload plus
4MiB conservative alignment overhead; these numbers derive from the unchanged
native GU/DN geometry and allocator alignment, not cache size. A cap such as
6144MiB at N24 is an example only; trust the explicit startup ledger.

## Ownership and byte contract

The transition seam is after the existing `block_begin`, before its context and
query passes and before `issue_ahead`. Scheduler captures that same frame's
existing `solo_held`, actual step and monotonic epoch, pending context/query row
counts, and sole actual query sequence ID. Its callback holds admission,
cancellation, reap and prefix operations out during the transaction. It rejects
stale/closed frames or an already planned next step; it never infers safety from
`n==0`. Cancellation after the callback can only remove old-frame work, not add a
new bank reader. A later request/slot reuse receives a new inspected frame.

The paired CPU release/join generation proves the issue threads cannot restart:
the scheduler does not enter the next barrier during reuse. Quiescence then checks
FileStager CPU jobs and P1 readers/packers, drains inserts and rowprefetch, waits
both Ctx lanes, mapped-source windows, the mover stream and tier copies. Idle
and teardown restore at the same ownership boundary, or after all worker joins.
No steady Ready/Lent step performs these extra drains.

The bank's raw slot addresses, offsets, three correction planes and every hot and
parked alias remain fixed. Startup captures each full18800640-byte hot slot,
including padding, after normal load/alias publication/router masking. Mask
planes are immutable invariants: strict bit checks reject a change rather than
silently repair it. The helper compares both Program and Residency pointers.

Target candidates are the highest profile-share native trunk file-tier pairs,
stable ties by unit identity. Every pair must match the original slot geometry,
weight order, offsets and stored sizes exactly; no packed/folded geometry is
introduced. One target is chosen per bank slot on every rank. Fill uses a leased
native BLOCK cache entry if available. The current packed CPU cache is never
interpreted as BLOCK: otherwise synchronous original-file extent reads and the
existing `relay_unit_block` reproduce the stored bytes in bounded scratch.
Source extents for all targets are preallocated/checked at startup, then reused
serially without growth. This first version does not use packed-RAM decode or
background restore; SSD, relay, copy and displaced-resident costs must be priced.

All target copies land before any `set_ring_resident` publication. Existing
FileStager resident handling sends those unchanged weights to the existing card
path. On failure, every target is unpublished and every original bank slot is
restored and read back byte-for-byte before a recoverable rollback returns Ready.
A failed invariant, stream drain, restore or readback is fatal: no later bank
query can be authorized. Restore removes all target publications, writes every
original raw hot slot and strictly checks all mask bytes and hot/parked pointers
before the all-rank Ready publication. Hashes are diagnostics; memcmp decides
readiness. No float equality, quantization, encoding or FMA/reduction change.

## Receipts and timing

Successful enable prints `DSBANKLEND startup` with rank count, N, raw/mask bytes,
relay/readback/shared scratch, pinned request/conservative charge, metadata,
total/cap, future cache/system headroom, source bound, hysteresis and `startup_ns`.
Capture includes full-bank bytewise diagnostic hashing and can be expensive;
do not omit it from cold-start/lifecycle reports.

Each transaction prints `DSBANKLEND transition epoch=... step=... seq=...
action=lend|restore phase=begin|ready pending_ctx=... pending_query=...
solo_held=... deferred=1 all_ranks=4 frame_epoch=...`. `epoch` is the stable
loan/copy identity from the lend frame through restore and resume; `frame_epoch`
is the current Scheduler observation (zero for idle/teardown). The controller's
completion ticket still checks the actual deciding frame, independently of the
stable receipt identity. Inside each interval every rank prints
`action=... stage=drained cpu_parked=1`, then `stage=lent slots=3*N
borrowed_bytes=... bank_bytes=... copied=1` or `stage=restored bytes=...
verified=1 aliases_ok=1 masks_ok=1 hash=... expected_hash=...
hash_scope=cohort strict_memcmp=1 selected_units=... restored_bytes=...
invariant_bytes=1536`. The hash is the same aggregate cohort diagnostic on each
rank; `bytes` includes checked mask invariants, while `restored_bytes` counts hot
bytes actually written. `selected_units` counts real selected resident skips on
normal/early FileStager paths during that interval, rather than just enabled slots.

After an actual successful query pass, `DSBANKLEND resume epoch=... step=...
seq=... query_rows=... ready=1` binds the restored cohort to Scheduler's numeric
request ID (not a random HTTP completion ID or a device pointer dereference).
Idle/teardown restore uses step−1,seq0 and need not produce a resume. `/stats`
empty is not by itself a restored receipt: idle restoration precedes idle
publication but can follow the last request's commit.

`DSBANKLEND totals` separates fill/source-drive/source-RAM/relay/restore/verify
bytes and startup/drain/fill/restore ns, lends/restores/rollbacks and current
borrowed bytes/readiness. Timing is inclusive where named: fill/restore includes
its drain; drain is also reported separately. Do not sum these overlapping
timers. End-to-end mixed-burst wall time is authoritative. Report startup and
post-workload/teardown restoration separately as well. Strict readback is part
of performance, not an untimed correctness-only option.

## Required serialized gates

Root owns all builds/tests/windows. Build under AGENT-LOAD-RULES and run exact
host names with `-LE gpu`: new `draft_bank_lend_test`, `draft_bank_backup_test`,
`sched_test`, existing native mover/stager/runtime tests. New scheduler cases
exercise precommit frame binding, stale/closed/issued-ahead refusal, explicit
solo/Dx admission, cancellation and a surviving anchor's catch-up query. Pure
controller cases cover off/reset/invalid flags, dwell/cooldown, double/stale
transitions, idle/end/cap, verified rollback and failed restore latching.

Before model enable, require a fresh library/cohort-bound real-card backup/fill/
restore receipt including each rank, short/partial copy failure and strict
readback/mask guards. Root owns that small DeviceIO/Image checker. Then one
stopped/restored QFN window, both locks: static bank+Dx reference versus lent
bank+Dx, with the SAME hotset, Db2 profile, query/verify settings and current
192GiB cache. Use the independent mixed-burst probe: long surviving anchor,
actual4-running interval, actual1-running interval and matching anchor resume.
Require per-rank positive borrowed bytes and selected_units, complete drain/
restore proofs, no parked draft expert routed, all reply IDs/text/counts exact,
and resumed proposal IDs matching static reference. Include repeated/cancelled
bursts and an ordinary prompt/prefix path; 96GiB then exercises functional miss/
spill fallback. Keep no-bank/no-Dx references where useful for the known penalty.
Price complete startup+mixed burst+restoration and plateau separately. No gain
claim is justified until bytes and end-to-end measurements pass.
