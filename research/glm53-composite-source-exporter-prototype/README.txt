Bounded complete synthetic physical-plane composite exporter

New isolated source-only derivative of frozen glm53-expert-record-store-prototype.
No original S1/provider/P3/Core/serving source was edited. Parent owns all compiler,
Release/ASan/TSan execution. No child compiler, GPU, model, service or network call.
No automatic CTest/install/convert/serve path exists here.

Scope and authority
  Closed fixture: logical H128/globalFF512; main layer0 plus MTP layer1, two
  experts/class, metadata TP4. Actual Registry/Builder/Planner/weight_share captures
  immutable native selected members for four ranks. The helper declares synthetic
  readability, not a resolved R4D reader. There is no Ctx, kernel, TP collective,
  NativeInventory TP4 enablement, full-model graph or learned checkpoint here.

  Each logical GU plane contains explicitly ordered complete physical GATE and
  UP pieces. DOWN contains one physical DOWN piece. Code and scale planes keep
  separate tensor identity/path/full extent/checksum, group, dtype, original
  row count and row byte width. Physical parts must exactly cover the logical
  encoding's plane geometry. Main scale rows use block1x32; MTP scale rows use
  block128x128. A complete=false marker, single saved row, missing part, relabelled
  part, wrong group/type, overflow or source extent shortfall refuses export.
  Neither a complete flag nor padding grants missing original bytes coverage.

  Catalog shape/group/class closure is deliberately fixed to eight logical
  projections (two layers x two experts x GU/DN). Each native record must have
  ordered GUcodes/GUscale/DNcodes/DNscale, truthful group slots, exact stored rank2
  shape/bytes, selected plane and identity encoding. All16 unique four-rank keys
  are required. Capture snapshots actual PlaneShare rectangles; it retains no
  borrowed Program/Plan pointer. Fingerprints include source revision, ordered
  physical pieces, tensor/path/full extent/checksums and native member metadata.
  FNV1a64 is an integrity checksum, not authentication or writer exclusion.

  The production AMD checkpoint has separate physical gate/up/down tensors, but
  this schema accepts ONLY the complete synthetic fixture. Saved actual partial
  rows are insufficient and must not be relabelled as a complete generation.
  No new actual payload is opened/downloaded here, nor does the test authenticate
  a real checkpoint. The negative one-row/partial metadata/extent cases cover
  that refusal category without pretending partial samples prove full coverage.

Streaming and resources
  One at-most4096B heap window streams physical checksums, ordered fused logical
  checksums, selected row rectangles, output checksums and padding. There is no
  full fused plane or whole native-record staging vector in pack(). The test
  independently materializes tiny originals/expected rows for its oracle.

  The input broker owns at most2 source FDs, closing the LRU BEFORE the next open.
  It retains bounded file identity stamps for all24 physical planes, validates
  each reopen and finally audits every source before publication. Four rank
  segments total720896B, each180224B;16 records (main32768B/MTP57344B). All118688
  padding bytes are initialized and independently compared, including MTP scales.
  The closed namespace bounds pack FD use by10 (two input, four output, temporary
  directory, manifest, parent directory, cleanup headroom); Reader owns at most6.
  Descriptor policy is a component reservation, not a process-wide FD quota.

  Default caps:8MiB output,2MiB complete source bytes/file extent,256KiB segment,
  1MiB record/read, four segments,128 records/64 logical sources/64 physical
  pieces,512B names/paths,1MiB manifest/32768 parser nodes. Closed fixture coverage
  imposes the smaller exact8 logical projections/16 records. Before payload I/O
  or window allocation, limits and metadata geometry are checked. Metadata has
  an explicit conservative16MiB logical/planning reservation covering bounded
  JSON/spec/source/stamp/cursor/allocator bookkeeping. Limits below16MiB or ten
  FD credits refuse. It is NOT an allocator-enforced heap/RSS quota; runtime,
  Core source metadata, allocator internals and OS buffered cache are separate.
  Caller-provided catalogs/specs and test-retained original source bytes are
  outside pack's quote. The tiny test retains <2MiB original files; at most one
  128KiB fixture joined plane plus one <=56KiB read buffer and one <=56KiB independent expected buffer per
  comparison are used. No full-model memory-fit or transport-speed claim follows.

  Core private structs are rebuilt against matching RADIANCE_SOURCE headers and
  RADIANCE_HOST_BUILD archives. CMake checks build source directory and selected
  ABI/PlaneShare/helper source pins. It does not rebuild Core or run a GPU.

Publication, errors and integrity
  Sources are fully verified before private output creation. All sources/output
  extents/checksums and every padding byte are verified before manifest write.
  One task-owned sibling temporary directory holds the whole four-rank generation.
  The final renameat2 NOREPLACE is the sole visibility commit. Unsupported atomic
  publication refuses; there is no overwrite/racy fallback. Failure preserves the
  caller Receipt/RecordSpec/Reader/read bytes and only removes the closed owned
  four segment names and manifest via allocation-/FD-free unlinkat then rmdir.
  Existing destinations are untouched, including a destination created after
  output work begins. OS unlink/rmdir failure itself cannot be guaranteed away;
  cleanup does not traverse or delete unowned paths.

  Segment/manifest fsync precedes commit. Parent-directory fsync is best effort
  after commit, so namespace crash durability is not a post-publication error
  guarantee. There is no fallible allocation/preparation after successful rename.
  Reader binds fresh expected catalog/spec/checksum, rejects duplicate JSON keys,
  verifies full output records/padding, and rechecks segment AND manifest stamps
  before publishing. Later read checks generation stamps/checksum transactionally.
  Ordinary immutability/stamp checks detect ordinary changes; no hostile concurrent
  writer or tamper-proof TOCTOU exclusion is claimed.

Eight test groups (source plans until parent executes)
  1. Every byte of all16 records against independent explicit physical part/row/
     column coordinates, all4 GU halves and down K/scales, F32 bit preservation,
     input immutability, FNV known answer, padding and resource receipt.
  2. Wrong native bytes/dtype/layout/transform/rank/same-byte shape, down block-
     splitting span, overlap. Caller immutable Spec unchanged.
  3. Missing/wrong-order/wrong-group/type/partial/one-row/overflow/oversized path
     physical pieces refuse before export. No actual partial payload is promoted.
  4. Late physical checksum, ordered logical checksum and complete-source extent
     failure, without destination or Receipt publication.
  5. Real wrapped pread and pwrite E_IO AFTER output writes begin; owned temporary
     files removed, caller Receipt unchanged, independent retry succeeds.
  6. Malformed output identity/rank/class/source/extents/provenance/path; changed
     native checksum; nonzero padding even after checksum recomputation; manifest
     stamp mutation during segment admission; Reader/output atomicity.
  7. Source/output/metadata/FD caps before payload reads, missing rank-key coverage,
     preexisting destination and actual late renameat2 collision NOREPLACE.
  8. Deterministic first allocation OOM before I/O, with complete new/delete hook
     family for sanitizer compatibility; no Receipt/temp publication, then good.
  Fault wrappers exist only in the checker, not exporter code. No model/serving
  allocator, file API, kernel or Core path is replaced by a fake implementation.

Parent handoff
  cmake -S HERE -B NEW_BUILD \
    -DRADIANCE_SOURCE=ROOT/radiance-src \
    -DRADIANCE_HOST_BUILD=MATCHING_HOST_CORE_BUILD
  Build target glm53_composite_exporter_test; invoke it with NO arguments.
  Link dependencies are matching Core/llama/httplib, dl/pthreads. Checker wraps
  pread/pwrite for deterministic failures. No automatic test/installation mode.
  SOURCE_RECEIPT.json and CORE_SOURCE_PINS.json record the source review boundary.

Remaining gates
  This distinct versioned synthetic format has NO S1/P1/P3 ready-store parser or
  Engine/provider compatibility. No reader is relaxed to accept it. Actual R4D
  layout resolution, checkpoint full-tensor/provenance export, scalable immutable
  authorities, Engine Loader activation, budgets, GPU last-reader/fault quarantine,
  TP READY/ABORT, MTP speculation, locality and performance remain separate work.
