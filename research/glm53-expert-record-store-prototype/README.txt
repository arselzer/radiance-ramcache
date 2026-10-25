Offline synthetic native expert record store — S1 source prototype
2026-10-25 UTC. Source only; child ran NO compiler/test executable/GPU/device/model.

This NEW folder implements packing/validation on tiny generated source files.
No previous prototype, SRC/Core, production configuration or serving file changed.
It is NOT an Engine/Provider adapter, checkpoint converter or deployment-ready
store. No real checkpoint header/payload is read or claimed verified.

API and provenance
-------------------
namespace rad::expert_store in native_record_store.h/.cpp:
 capture_unit(Program,Plan,unit,SourceCatalog,RecordSpec*) snapshots ACTUAL
  WeightInfo selected enc/role/dtype/shape/layout/shard spans and the ACTUAL
  MoveUnit member offsets/class stride. It checks scalar dimensions/products
  before calling REAL core/format/share.cpp weight_share/PlaneShare. The immutable
  RecordSpec cannot be edited by callers after capture.
 pack(specs,catalog,destination,Limits,Receipt*) validates actual source bytes,
  losslessly streams native records, then atomically publishes a complete store.
 Reader::open(dir,FRESH expected specs/catalog,published manifest checksum,Limits)
  validates identity, every segment/record/padding before returning a reader.
 Reader::read(Key,vector*) reads ONE bounded record into private storage and
  changes the caller's vector only on full checksum/stamp success.

SourceCatalog describes synthetic logical GU/DN tensors, their complete affine
encoding and global shape, exact source-plane file extents and expected FNV1a64
checksums. Source ID/revision plus sorted encoding/shape/plane checksum metadata
form the source fingerprint. Paths are transport locations, not content identity;
opening/checking every actual source plane is mandatory before packing. Immutable
file fstat stamps are checked again after copy. Metadata receipts alone cannot
create a store without matching actual source bytes.

Every manifest forces origin=synthetic_fixture, checkpoint_headers_verified=false,
checkpoint_payload_verified=false, and explicitly says layout provenance is
actual Builder/Planner footprint, NOT actual R4D row resolution. There is no API
flag to claim verified real checkpoints. Tests reuse the existing metadata-only
glm53_weight_probe::build helper: actual Builder/Planner accounting, with explicit
synthetic readability. Matching real native R4D callback/row footprint validation
is a SEPARATE parent-owned gate; this prototype never executes/registers a GPU
reader. Main/MTP class identity is checked independently of native execution.

Supported native selected planes
--------------------------------
Only canonical identity, no transform/widen/layout change:
 main = rad_enc_affine(FP4E2M1,E8M0,1,32);
 MTP  = rad_enc_affine(F8E4M3,F32,128,128).
Each unit must contain exactly GU codes/GU scale/DN codes/DN scale. Actual member
bytes, dimensions and class stride must agree with fresh Program/Plan. GU uses
ROW with two parts and explicit per-rank span; DN uses COL explicit span. Single-
span affine matrix profile is supported; arbitrary transforms/private layouts,
per-part heterogeneous spans and other encodings are explicitly refused.

The packer NEVER quantizes/dequantizes. It copies each REAL PlaneShare rectangle,
row by row, into its captured native member offset, preserving every byte.
Padding is explicitly zeroed and checksum-covered. Reader separately verifies
EVERY padding gap/tail, even if somebody recomputes a record/manifest checksum.
FNV1a64 is accidental-integrity evidence, NOT cryptographic authentication.

Test geometry and independent oracle
------------------------------------
All four TP ranks, hidden128/globalFF512/localFF128, two experts in each of main
layer0 and MTP layer1. Logical GU is[1024,128], DN[128,512]. Actual local records:
 main32KiB: offsets0/16384/20480/28672; member bytes16384/1024/8192/512.
 MTP56KiB: offsets0/32768/36864/53248; member bytes32768/8/16384/4.
These constants are ASSERTIONS against actual Builder/Planner-derived offsets;
they are not the packer's addressing logic.

Independent expected bytes do NOT call PlaneShare, weight_share or packer rect
helpers. They compute each GU half separately (rank*128 inside each full512-row
half), and every DN row's rank*128-column run. Scale axes shrink by32 or128 with
F32 byte width included. Coordinate-dependent source patterns distinguish rows,
columns, rank cuts, half boundaries, projections, classes and experts. MTP codes
avoid NaNs and F32 scales use finite positive patterns, but no inference is run.
Every byte of all16 output records is compared, including118688 padding bytes.
Expected native payload total720896B; source image stays under1MiB.

Segmented 64-bit format and bounds
----------------------------------
manifest.json schema radiance-synthetic-native-record-store-v1 contains strict
provenance, source fingerprint/catalog, native per-record Spec metadata,
segment index/offset/stride/checksum and segment names/ranks/exact extents.
Records never straddle segments or ranks. Segment names are generated basenames;
Reader refuses traversal, unknown geometry, gaps/overlap/extra trailing data,
class/rank/source/layout mismatch, truncation and corrupt checksums/padding.
Manifest byte/depth/node/string limits and duplicate-key rejection bound parsing.
Reader retains manifest AND segment descriptors/stamps and refuses later mutation.
Fresh specs/catalog and independently retained publication checksum are mandatory
on open; do not reuse an old reader as proof of a changed graph/source. A future
HOT cache must key/revalidate those same identities before its lookup.

This is a NEW format/validator, NOT a way around the old FileProvider's64MiB cap.
No old Provider API is used or compatibility claimed. Segment/file offsets and
length arithmetic use checked uint64 with off_t/INT64 boundaries. Size limits are
explicit profile parameters; source/capture/tests intentionally remain tiny.
This does not establish a production large-model packer/provider, huge-file test,
R4D native row compatibility, RDMA/pinning or O_DIRECT throughput semantics.

Defaults: max_total8MiB, segment96KiB, record1MiB, source2MiB, read1MiB,
manifest1MiB/depth16/nodes32768, records128,sources64,segments128,string512,
stream chunk65536B. Count/geometry/products/caps are checked before payload I/O;
manifest upper-size preparation precedes source reads/output creation. Payload
packing/whole-store validation stream through one bounded chunk. A Reader read
may allocate one record only after its allocation cap is checked. JSON/Builder/
metadata ownership is bounded separately; these caps are not a physical RSS or
exact allocator-overhead accounting claim. The test does not allocate pools or
query memory/device capacity.

Atomic publication and failure behavior
----------------------------------------
Destination must not already exist. A private0700 sibling staging directory uses
exclusive no-follow segment files. All source checksums validate before staging;
selected bytes/padding/checksums are verified, segments fsynced, manifest written
LAST and fsynced, staging directory fsynced. ALL allocating/fallible preparation
is complete before Linux renameat2(RENAME_NOREPLACE). There is no overwrite or
racy rename fallback; unavailable syscall/flag returns unsupported and cleans the
private staging directory. Existing destinations are never modified.

Only task-owned temporary segment/manifest names are removed on failure. Failed
pack leaves receipt unchanged and final destination absent. No failure is
returned AFTER successful publication. Parent-directory fsync is best effort:
a crash can lose namespace durability, but cannot expose a partially prepared
manifest/store. This is an explicit durability limit, not a durable-database claim.
Reader::open and read preserve their caller outputs on every refusal.

Seven staged CPU cases (not executed by child):
 1 all16 records round-trip across4ranks/bothclasses; exact source immutability
   and independent720896-byte comparison;12rank-separated96KiB-cappedsegments;
 2 malformed stored dtype/bytes/layout/encoding/span/member overlap/class stride,
   source offset/dimension overflow, capture output unchanged;
 3 stale source revision, late source-plane checksum failure and truncated source
   cannot publish a store or replace receipt;
 4 wrong rank/class/layout/source/header claim, offset overflow/overlap/negative
   length/path/segment count, duplicate JSON key/node cap; fresh expected bindings
   rechecked even when test deliberately recomputes the manifest hash;
 5 payload checksum/truncation/stamp refusal; recomputed checksums cannot hide
   nonzero padding; existing reader output vector remains unchanged;
 6 allocation/metadata/byte limits refuse before an intentionally missing source
   is opened; one-record read cap and existing-destination no-overwrite;
 7 actual late write failure using child-only RLIMIT_FSIZE40000 and ignored
   SIGXFSZ, forcing EFBIG after a partial segment. Parent forks IMMEDIATELY execs
   /proc/self/exe --late-write-child; no Core/allocator/lock operations occur in
   the fork child before exec. All Builder work/limit/signal changes occur in the
   fresh exec process, never the parent. Cleanup checks no published destination,
   unchanged receipt and no private staging remnants; normal child destruction
   removes its own synthetic source fixture. No device/production process touched.

Build/validation ownership
--------------------------
Source-only child; parent serializes normal/sanitizer builds and CPU execution.
CMake links matching EXISTING HOST Core/vendor archives, no implicit Core build,
no GPU/runtime plugin/library and no actual device enumeration. Parent must
select an archive built from matching source; default known location is provided
as a convenience, not an ABI-version assertion.

Parent-only example, NOT executed by child:
 cmake -S ROOT/glm53-expert-record-store-prototype \
  -B ROOT/build-glm53-expert-record-store -DCMAKE_BUILD_TYPE=Release \
  -DRADIANCE_HOST_BUILD=ROOT/radiance-src/build-iq2-host
 cmake --build ROOT/build-glm53-expert-record-store \
  --target glm53_native_record_store_test -j1
 ROOT/build-glm53-expert-record-store/glm53_native_record_store_test

No CTest/install/serving target. HOST-only sanitizer validation is parent-owned.
This artifact is S1 synthetic pack/verify evidence only; native R4D footprint,
actual checkpoint provenance, Engine ownership/current-route handoff, runtime
cache/provider, MTP inference and TP readiness/failure protocol remain separate.
