Verified REAL complete-main data reader/provider, source-only candidate

This NEW isolated package adds persisted complete-bundle reopen, actual once/rank
source/layout binding and a bounded RAM-cache NativeRecordSource. It does NOT
implement GlmMainTp4VerifiedSourceFactory, set Session source admission, modify
Core/serving, or execute model/GPU kernels. Current Engine activation stays CLOSED.
Parent owns all compilation and explicit validation. Authors did not build/run.

Writer contract is the frozen complete-main assembler cc6f93b4... . The copied
public header and passed v3 main_records.cpp POSIX/JSON/hash primitives are kept
exact; main_bundle_reader.cpp includes that cpp exactly once. It never calls
packing. Do not also compile producer/main_records.cpp into this library.

Strict complete-only authority:
* Trusted expected complete ID and exact persisted producer proof bytes.
* Exact72 pinned source files/hash transcript/current generation; same opened
  Checkpoint root/object. C++ trusts the parent cryptographic proof and does NOT
  claim to recompute those full file SHA hashes.
* Root bundle,4 rank templates,168 finalized rank/layer manifests,48384 records,
  no duplicate/missing/foreign keys; all native record CRCs and ordered checksum.
* Raw mapped checkpoint tensors are compared with their actual physical source
  bands once (76108 raw tensor entries). Raw full-tensor FNV witnesses independently
  bind every complete six-tensor component. Headers/gaps/non-tensor artifacts are
  covered by trusted producer hashes, not counted as mapped tensor bytes.
* bind_rank uses SAME Checkpoint/cache+actual finished Program/stable metadata
  Plan, max_tok1/seq1/noMTP. Actual geometry quote ONCE/rank; exact spec name/source/
  shape/dtype/encoding/shards/rowparts/cuts; independently hash actual selected
  source cuts and match native payload CRCs. This is real source-memory traffic,
  separately counted, not a metadata-only or runtime cache event.
* Compact NativeWeightAuthority capture validates later metadata and ignores only
  its established loader-published fields. No W WeightInfo intermediate/clone.
  Caller Program/Plan/Checkpoint/cache must remain immutable/alive while open;
  source close precedes their destruction. Leases afterward only use owned native
  byte buffers/key/accounting, never dangling Program/raw-checkpoint pointers.

The first validation is deliberately expensive: it scans complete native payload,
raw mapped-vs-physical tensor bytes, and rank-selected source cuts. All counters
are separate from cache I/O. No throughput/fast-start/production-prefill claim.
Input roots stay read-only. Initial same-root/proof binding is conservative;
portable proof rebinding and checkpoint-open file witnesses are future work.

Old synthetic Reader and RankNativeStore::open are not widened. The copied S1
Reader cpp and old generic open body remain exact. Only the LOCAL header adds a
friend for the strict VerifiedBundle member to construct normalized catalogs;
there is no public mutable POD->verified constructor. New provider admission
accepts only private immutable VerifiedRankMain, maps canonical ordinals to actual
units in O(N), and retains that authority in the same generation/budget token.
It reuses original cache/lease/FD/read/checksum/failure-sealing mechanics.
Original synthetic provider case bodies are unchanged and rebuilt against these
matching headers/archive, not an old library/private ABI.

Runtime acquire checks selected records/files plus root/bundle.json/source-proof
stamps (including in-place header changes). It does NOT rescan all weights,168
shards or raw sources on each of42 layer calls. Full audits are setup/explicit
lifecycle boundaries. Current NativeOwner separately checks header+selected unit
metadata before requesting bytes. Plain LRU can thrash on prefill sweeps; no claim
that HOT capacity is subtracted once from repeated-prefill disk traffic.

Initial source profile owns all12096 main experts/rank; resident mask is empty.
bootstrap() is UNSUPPORTED, finish_bootstrap() is required exactly once before
acquire and performs setup validation/trim. Requests<=8. For an8-record union,
COLD>=26738688B independently of HOT; HOT must separately hold that union for a
full repeat hit. NativeInventory landing/DEVICE backing is a later separate
Engine owner. Tiny synthetic cache defaults deliberately cannot admit this data.

Named budgets/retention:
Reader peak and immutable requested planning reservations are separate. Binder
quotes geometry before arrays/source inspection, caps authority scratch/capture
to remaining rank budget, and records both construction and retained sizes.
O(48384) normalized records/components plus4 numeric templates are owned once;
no full per-record JSON trees persist. Source data parser caps include duplicate
keys, bytes/nodes/depth/strings. Positive-path final shared owners/controlblocks
and audit scratch are prepared BEFORE final Program/source/file audits; only
noexcept output publication remains. Failure preserves caller outputs.
Provider token charges its own state +rank immutable authority +shared bundle
allowance +facade storage. Shared bundle allowance is intentionally reserved
PER rank token conservatively; that is not four physical copies. Parser/geometry
construction scratch is not retained/charged as steady authority. Caller-owned
NativeRecordBatch vector capacity is outside adapter storage (public ABI cannot
observe retained capacity); provider Batch slots and Holder lifetime are charged.
Quota retains SAME provider accounting token through Holder teardown. Whole
process cache/metadata/FD caps must cover all live rank tokens; leases keep those
credits after source close. No silent release/reuse while a lease survives.

Reader simultaneous owned FD upper8 (separate caller Checkpoint descriptors).
Reader keeps no FD after setup/audit. Provider keeps credited root/payload FDs;
setup normalized admission reads no payload and counts earlier verified bytes
as provenance only. Kernel page cache, source mmap working set, caller graph/
checkpoint metadata, driver/stdlib RSS are outside these requested reservations.
Parent allocator/FD/runtime witnesses are required; no full-process fit claim.

Manual targets (no CTest/auto-run/install/serve/source-proof/pack/download target):
 glm53_verified_main_reader_refusals --run-host-refusals
   Negative metadata only: scalar/cap/low-rank-budget no open/pread/device work,
   private-output OOM atomicity, deliberately untrusted NONREADY8-record subset
   header reaches exact domain refusal, wrong trusted ID. No fake complete data.
 glm53_verified_main_generic_regression
   Unchanged original generic/synthetic provider cases using this package copy.
 glm53_complete_main_source_check --run-complete-reader MODEL MATCHING_AQL_HOME \
   REAL_COMPLETE_BUNDLE PRODUCER_PROOF TRUSTED_BUNDLE_ID NEW_REPORT_JSON
   Expensive actual data validation plus actual4 logical-rank DEVICE layout
   selection, max_tok1. Parent complete-input/idle gate required. Uses one matching
   Core/private ABI throughout. No kernel instances/launches, NativeInventory,
   Session, Ctx, Pools or Engine. Verifies selected cache bytes, zero repeat payload
   rereads (metadata stats still run), then actually destroys sources/raw checkpoint/
   Program owners before checking retained native leases and full token retirement.

Configure with exact matching whole Core:
 cmake -S THIS_PACKAGE -B NEW_FIXTURE_BUILD -DRADIANCE_SOURCE=ABS_FROZEN_CORE \
   -DRADIANCE_BUILD=ABS_MATCHING_WHOLE_CORE [same sanitizer/ABI options]
For HIP/AQL Core also pass -DRADIANCE_AQL_CORE=ON (explicit ROCm link/compile
profile). HOST refuses that flag; non-HOST refuses omission. All source pins and
actual Core source-home must match. Parent builds targets separately and uses an
outer owned process-group timeout. Source-only hash verifier has no target imports,
compilers/device/model actions. Full data/generation gates remain pending until
parent actual results; passing HOST refusals is not complete-source success.
