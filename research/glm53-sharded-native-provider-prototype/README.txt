P1/P2 once-bound sharded native provider — isolated synthetic CPU candidate
=========================================================================
SOURCE ONLY until parent validation. The child did not compile or run a test,
use a GPU, change a service, read a full model or apply any Core overlay.
Frozen S1/cache/bridge/B1/B2/source copies remain unchanged. No Core adapter,
Engine/CLI activation, learned model, TP execution or performance claim.

The earlier design is ROOT/GLM53_SHARDED_NATIVE_PROVIDER_DESIGN.txt. This folder
implements only its first CPU catalog/cache increment, with explicit buffered
I/O and ImmutableGenerationSelectedAudit. It does not silently substitute for
the frozen bridge's all-store mutation guard.

Pinned parser and normalized authority
--------------------------------------
candidate_native_record_store.{h,cpp} is a separately named/namespace S1 copy.
S1_BASELINE_PINS.json identifies the unchanged original. Candidate changes are
limited to explicit capture limits (the old overload still uses old defaults),
a directory/manifest recheck after startup, and FD-free immutable catalog export.
The original duplicate-key/schema/provenance/source/native-layout/extent/checksum/
padding parser remains the sole store parser. No second permissive manifest
parser or inferred packer offsets are used. All stores still declare synthetic
origin and checkpoint_headers_verified=false/checkpoint_payload_verified=false.

Reader::open still fully verifies every native record and padding at startup;
this is real setup I/O, not a lazy request miss. Stats separate unique verified
native bytes, manifest bytes and exact derived startup payload bytes including
the parser's separate zero-padding pass; fixture packing I/O is not a demand miss. export_catalog rechecks its
pinned descriptors and returns only private-constructor immutable normalized
segment stamps and record locations/checksums/member ranges. It retains neither
FDs nor the parser's JSON tree. The rank owner releases every startup Reader
before its reserved descriptor credits return. There is no duplicate per-record
Reader::read vector/copy in the steady provider.

RankNativeStore::open binds a specific actual Program pointer/rank and pure
Plan, common source ID/revision and one explicit nonwrapping epoch. Each exported
record is recaptured against the actual Program/Plan with explicit source caps
and compared exactly with its expected immutable RecordSpec; every expert unit
must be covered once across shards. Normalized handles are private-constructor
{strong generation identity, ordinal}; callers cannot supply or modify offsets,
lengths, class IDs or checksums on acquire. The Program/Plan must be immutable
and live through this provider's future Core use; the provider is not a second
runtime Plan validator. B2a's repaired authority checks remain mandatory for any
later actual Ctx adapter. Matching hashes do not authorize another generation.

The fixture uses actual Builder/Planner/weight_share metadata for rank 0 or 1
of a synthetic world=4 layout. No operator/collective/TP4 run occurs. Main native
records are 32 KiB; FP8/F32 records are 56 KiB, with all four actual member ranges
and independent coordinate/padding byte expectations. This is native metadata
compatibility, not actual R4D row resolution or checkpoint provenance.

Resource ownership and one rank-global ledger
---------------------------------------------
One ProcessBudget is registered process-wide for THIS experimental subsystem.
Creating another refuses while the first budget or any old quota token remains
alive. Every shard/class under a rank shares one HOT/COLD payload ledger. A
rank token owns its max HOT+COLD reservation and conservative metadata charge;
another owner for the same world/rank refuses while a store, bound handle,
lease, allocated Batch storage or accounting observer still retains that token.
All ranks share the process FD, metadata and cache-reservation limits. This does
not account for arbitrary unrelated allocations/Core budgets elsewhere.

The dense catalog has one weak/HOT cell per record, initialized once. A selected
expired cold weak pointer is a normal miss. Acquire performs no full-cell prune
and no string-key lookup; dedup uses bounded request stamps and first occurrence
order. Stamp exhaustion refuses rather than clearing the whole table/reloading.
HOT work counters depend on selected unique records/files, not unused layers.
Miss LRU eviction can scan its bounded HOT list and may bypass HOT admission if
live leases pin every candidate; there is no constant-time miss/latency claim.

The complete missing COLD union and live-record quota are reserved before any
payload read. Per-request vectors are pre-reserved except the bounded caller
result storage. Live lease references AND allocated provider Batch vector slots
have separate caps. Batch.leases.clear() returns lease references but intentionally
retains vector-storage credits; reset to Batch{} or destroy the Batch to free
that storage/credit. Moving a lease out still keeps its live-reference credit.
Arbitrary caller-created empty vectors/reserves are outside provider accounting.
Payload allocation is freed before its HOT/COLD charge is returned. Leases and
accounting remain valid and charged after close/destruction. No cache/gen/payload
ownership cycle exists.

All declared persistent catalog/cell/record-control/lease/work allowances and
one conservative worst-case startup parser charge are admitted before root I/O.
The startup parser charge is released after setup. Reservations are printed via
ProcessUsage/Accounting, not described as measured RSS. Drivers, C++ runtime,
filesystem buffers and arbitrary caller memory are outside these reservations.
The native fixture payload remains <8 MiB. Growing 6 to 36 real native records
uses actual verified files and opaque handles; no fake full-model ready catalog.

Bounded files and exact buffered read-into
-----------------------------------------
The root is an explicitly trusted absolute path, final component opened with
O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC. Intermediate root path components are trusted
configuration, not a hostile-path sandbox. Shards are restricted single leaf
names; all later directory/payload opens use openat with O_NOFOLLOW. Segment names
come from the exact S1 parser. Pinned and named device/inode/extent/mtime/ctime
must match. Per selected shard, one temporary directory FD is credited, checked
and closed; no model-wide directory FD set is retained.

Two payload FDs are cached in the tests. The growth test explicitly reserves
up to 20 rank FD credits to cover its 12-segment startup Reader plus root; the
ordinary runtime still retains only root plus at most two payload FDs. Its
startup peak is independently observed via bounded /proc/self/fd enumeration
at actual reads (the observer's own descriptor is excluded). Broker replacement closes the prior
unborrowed descriptor before opening another. Requests are serialized, so no FD
is evicted while its read is active. Startup Reader credits reserve the declared
max_segments+2 plus the root; the receipt separately reports its exact derived
actual peak segments+3 (Reader opens directory, manifest, all segments). There
is no uncounted second validation Reader or descriptor per catalog cell.
Logical credits do not reserve kernel RLIMIT slots against unrelated code;
EMFILE/read failures refuse without retries or leaks.

Buffered pread writes directly into PRIVATE unpublished COLD memory in bounded
chunks. EINTR/short positive reads are handled; EOF/error/cancellation/checksum/
nonzero padding refuses. No pinned memory, DMA alignment, RDMA or O_DIRECT path
is implemented. OS page-cache memory is EXCLUDED from the userspace HOT budget,
so this prototype is not a proof of production system RAM containment. A direct
I/O/aligned-window profile remains a separate refused implementation gate.

Whole-union publication and integrity contract
----------------------------------------------
Preflight validates ALL handles and selected file stamps before read. After
EVERY private fill/checksum/padding pass, selected files and root are audited
again INSIDE the serialized transaction, before any new weak/HOT cache cell or
caller Batch is published. A late failure discards private fills, releases their
charges and leaves the old caller result unchanged. Verified HOT bytes may be
read concurrently through existing immutable leases. Counters and FD LRU entries
may advance on refusal; no unverified or partial new ready payload commits.

Ordinary selected-file mutation, including HOT, refuses and seals the generation.
Foreign/empty caller handles refuse without sealing an otherwise valid source.
A transient pread error or allocation/cancellation error permits a later retry;
all newly filled records must be read/verified again. Unexpected EOF contradicts
the bound extent and seals. A selected corruption from a test read fault fails
checksum and seals. There is no silent repair/reload/in-place source update.

CRITICAL DIFFERENCE: mutation in an unselected separate shard/file may remain
undetected while an unrelated HOT request succeeds. audit_all performs an
explicit O(all files) ordinary stamp audit and seals on mutation; a selection
of the modified file also refuses. A changed unused record inside a selected
segment changes that selected segment's ordinary stamp and refuses. This is
explicit ImmutableGenerationSelectedAudit, not the stronger frozen bridge.
Root directory/manifest identity changes also refuse. FNV/stamps do not provide
authentication, concurrent-writer exclusion or hostile TOCTOU protection.
Published generations must remain immutable. Full payload re-scrub/reload is
unsupported; reload() returns unsupported. A numeric epoch alone cannot make
old opaque handles acceptable to another owner.

Close signals cancellation before waiting for the serialized active acquire.
It closes payload/root FDs only after that acquire leaves; immutable retained
lease bytes stay readable/charged. Same-thread reentry into mutating/bind APIs refuses before waiting. stats() is
an external observer that takes the transaction mutex; it must not be called
reentrantly through a test pread wrapper while acquire holds that mutex. The
provider exposes no user I/O callback.
No deadline is promised for an uninterruptible OS pread. Object destruction
must not race calls on the destroyed C++ object; explicit close may race a call
while the object remains alive. There is no GPU completion/failed-drain policy.

Staged tests and parent-only build
----------------------------------
Fourteen cases are staged, NOT child-executed:
1. Six vs 36 real native records across three shards; exact bytes, dedup order,
   fixed HOT work counters, zero payload I/O after two-FD eviction.
2. Global cross-class/shared HOT budget, live-lease protection, bytes/accounting
   after close, retained-token rejection of another rank/process quota.
3. Foreign rank/class/empty handles, cancellation and cap refusal before I/O.
4. Real late pread failure, no partial ready fill, exact retry and whole-union
   pressure refusal before further I/O.
5. Explicit unused-shard corruption policy, all-files audit sealing, and a
   stamp change after the LAST read proving final audit precedes ready commit.
6. Live lease/storage quotas and explicit request-stamp exhaustion.
7. Setup/miss ordinary allocation failures and complete resource rollback.
8. Six concurrent callers, shared exact immutable HOT bytes and no rereads.
9. Same-thread test-only reentry and selected read checksum refusal.
10. Old default capture refusal vs explicit caps, duplicate-key parser refusal,
    late truncated/corrupt startup stores and no partial catalog publication.
11. Selected HOT corruption/truncation/symlink refusal without reread.
12. A blocked actual read plus concurrent close/cancellation, no FD/private fill
    release before its transaction leaves.
13. Two hundred last cold lease release/acquire races, exact bytes/no false I/O.
14. Caller source/catalog/shard wrappers released after bind; the owned normalized
    generation remains exact and HOT does no fresh global metadata scan.
No implementation test hook is exposed: the executable wraps pread and ordinary
new for bounded failure injection; cache/provider source paths remain real.

Parent builds in a NEW external build directory with matching Core headers and
archives (no implicit Core compilation):
  cmake -S THIS -B NEW -DRADIANCE_SOURCE=MATCHING_SRC \
        -DRADIANCE_HOST_BUILD=MATCHING_HOST_BUILD
  cmake --build NEW --target glm53_sharded_native_store_test -j1
  NEW/glm53_sharded_native_store_test
Repeat with parent-selected sanitizer toolchains/instrumented matching Core.
No CTest/auto-run/install/GPU target is added. candidate S1 publication still
requires Linux renameat2 NOREPLACE; no overwrite fallback.

Not implemented: Core NativeRecordSource adapter, Engine loader/native-only
source skipping, production DirectFile/O_DIRECT/pinning, real checkpoint/native
reader provenance, full-size manifest/capture receipts, scalable B2 metadata
interning, TP4 READY/ABORT coordination, MTP draft lifecycle, GPU fault cleanup,
background prefetch, RDMA, reload, full payload audit and performance benchmarks.
