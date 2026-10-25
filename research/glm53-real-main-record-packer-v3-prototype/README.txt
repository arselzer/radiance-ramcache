Real GLM main one-expert canonical record packer — FROZEN SOURCE CANDIDATE; PARENT VALIDATION PENDING

This separate stock-Core package implements a bounded next step after the real
main geometry quote: one COMPLETE actual expert at each of four logical TP4
ranks, with actual code/scale payload reads and canonical native record output.
No Core, old generic S1/composite/P3/provider, frozen source, serving/config file
or kernel is edited. No author compiler, GPU, numerical/model, payload read,
packing, network, service, deployment or restart was run. Parent owns validation
and must wait for complete original checkpoint + actual geometry gates before
any payload run. The explicit manual flag is --run-complete-record-pack; metadata
planning alone does NOT invoke it. This is not full-store/full-model caching.

API and ownership
  pack_main_records(actual Checkpoint&, actual CheckpointSources&, Program&,
      Plan&, span<one actual unit ID>, NEW destination, Limits, PublishedRecords*)
  MainReader::open(path, const PublishedRecords&, Limits, unique_ptr*, why)

The pack call synchronously requotes the actual full main graph/source/cache/
selected DEVICE layouts, then snapshots only the selected real unit. It owns all
needed strings, POD shapes/cuts, source evidence and JSON before returning. No
borrowed Checkpoint/Plan/Program/SrcTensor/CkptWeight pointer escapes. A private
PublishedRecords capture supplies the Reader's independent expected layout and
publication hashes; callers cannot fabricate it from mutable geometry/JSON.
Reader data and normalized VerifiedCatalog are NOT NativeRecordDescriptor,
NativeRecordSource, inventory, Ready or full-domain coverage. There is no new
provider/factory/generic cache framework. Source and graph stay alive during
pack; Reader retains the owned capture independently afterwards.

Original stock ABI ONLY: frozen geometry source and stock declaration helpers
are copied exactly and all translation units build against ONE matching original
Core headers/archive. Research private Native headers are absent. Future research
consumers must rebuild all helpers against their one Core or communicate solely
through saved manifest/segments; never mix stock/research Plan/Program objects.

Actual source authority and identity
Fresh quote validates TP4/rank, all42x288 main units, actual member layout/source
maps, exact SAME imported objects/raw spans and canonical DEVICE layout claims.
It refuses nonidentity/tags/widening; no kernel relayout is invented here. Output
source ranges come from the actual opened checkpoint directory's real index and
selected safetensors headers. Names/order/U8 raw shapes/extents/file offsets are
matched to the real mapped SrcTensor. FULL bytes of the selected six raw tensors
are streamed and compared to those mappings before packing. Mixed mapped A/pathB
with different selected payloads refuses. Separate GATE/UP and per-row DOWN
rectangles are copied directly; no huge fused-plane temporary is materialized.
Every source is an ordinary regular indexed leaf opened O_NOFOLLOW; no symlink,
hardlink manipulation, cloned partial checkpoint or user-source mutation occurs.

source_generation is explicitly the selected_raw_bundle identity: full six raw
tensor checksums, ordered logical planes, config/index/header fingerprints and
file witnesses. It is COMMON to all four rank cuts of the SAME expert and
DIFFERS for another expert/layer. It is NOT yet the common full-store/model
NativeInventory generation. Rank/world/member cuts are bound separately in the
local spec/record checksum and manifest_checksum. A future full-store assembler
must bind one checked common bundle revision plus these component hashes across
all records; this one-expert proof must not invent that authority.

The intended pinned AMD revision is recorded as an intended reference, never as
full LFS/SHA verification by this code. It validates complete SELECTED tensor
bytes, not every payload of the checkpoint. FNV/stat detect ordinary changes;
no authentication, hostile TOCTOU exclusion or global writer lock is claimed.
Root/index/config and physical source stamps are audited after ALL manifest/
result/parent preparation immediately BEFORE atomic rename, closing late-source
change/refusal. All result allocation precedes publication.

Format and reuse
Canonical S1 binary segment/member bytes, grouping, padding and record/manifest
semantics are retained. The source provenance is a strictly versioned actual
profile of the manifest schema radiance-native-record-store-v2, with selected
header/tensor verification scopes written explicitly. Old synthetic Readers
remain CLOSED to this profile. This does not relabel synthetic origins or
relax any frozen parser. Code reuses/reviews the S1 bounded copy/hash/Reader/
publication rules and composite exporter's ordered physical cursor/FD policy.
The new Reader verifies its private fresh recipe, exact actual-source profile,
independent manifest hash, full record checksum and all padding, then exports
normalized immutable file stamps/record fields without retaining descriptors.
Current one-record limit stays explicit; no claim of generic full-store parser
compatibility follows merely from the canonical record bytes.

Resources
Actual rank-local record is3342336B/3.1875MiB; selected complete raw input tensors
are13369344B/12.75MiB per pack. Copy/hash/padding readback use one64KiB window;
there is no whole fused tensor/whole model staging. Reader::read owns at most one
admitted record (4MiB cap, actual3342336B). The manual independent oracle owns a
separate same-size buffer using direct mapped physical coordinates. That oracle
uses neither packer cursor nor actual weight_share to choose its cuts.

Default bounds: ONE selected record, record/read4MiB, complete tensor reads16MiB,
physical file extent8GiB (separate from tensor-range reads), index16MiB, config2MiB, selected
header4MiB, manifest1MiB, metadata128MiB. A planning reservation using actual
source index/header byte counts is checked before each parse, includes the real
geometry quote peak and fixed temporary/control margins, and is printed. It is
not an allocator-enforced RSS/stdlib/plugin/driver quota; upstream full graph,
Checkpoint/import cache/mappings and test-owner objects are separate named owners.
There is no192MiB tiny ledger imposed on the real model. Low metadata/manifest
limits refuse before payload work; geometry construction gets the remaining
admitted metadata limit, not its unconditional default.

The two-source-payload-FD broker is only part of FD ownership. Pack peak bound8
consists of2 payload handles plus temporary directory, output segment, source
root/index, parent and one of manifest/config handles (manifest closes before
the final config audit). Capture may temporarily own root/index/config/header
handles; those are below8. Reader owns3 (directory/manifest/segment). Caller
Checkpoint's own source mapping descriptors and external users are separate.
peak_input_fds reports actual broker opens; owned_fd_bound=8 is a named upper
bound, not a measured process-wide descriptor count.

Canonical real profile has NO inter-member or tail padding: all member sizes
are page multiples and sum to3342336. Padding checks are retained but no actual
padding-mutation case/coverage is fabricated for this profile.

Explicit parent-run checker cases (source assertions, not yet observed results)
- Four actual stock Programs and uncapped layout-only Plans, instances disabled,
  real gfx1201 DEVICE metadata rows. This is not hardware fit or TP/kernel issue.
- Four complete rank-local source/record byte proofs against independent raw
  physical row/column coordinates; common selected bundle ID, separate native
  hashes; foreign rank key and record-read-minus-one refusals preserve output.
- Reader/publication ownership and normalized catalog survive their respective
  creator destruction; no escaping source borrow is required.
- Three real late I/O/publication errors after writes begin: pread EIO, pwrite
  EIO, and actual final rename collision creating a task-owned marker that must
  remain untouched. Caller prior publication and completed generations survive;
  private output staging must be removed.
- Fourth late fault: an ACTUAL successful source-index fstat is reported as EIO
  AFTER the real manifest pwrite. This is an explicitly semantic returned-error
  seam proving the final source audit is reached. It is NOT an actual source
  mutation/clone test. Real checkpoint bytes/inodes/links are never changed.
- Complete C++ allocation family first-OOM before source reads/writes; metadata
  budget before I/O; NOREPLACE on an already published sample. Hooks are checker
  only and delegate actual backend/libc calls; no fake Backend/private structs.

The checker stages all4 outputs under one task-owned private root and publishes
that root by NOREPLACE only after all cases pass. Failed driver work removes only
its own private tree. Prepared completion bytes use best-effort write after final
commit, not allocating/fallible result work. The final output contains four native
records plus manifests/receipt, not a model clone or whole checkpoint export.

Parent-only commands (not run by source author):
  python3 HERE/verify_source_pins.py
  cmake -S HERE -B NEW_BUILD -DRADIANCE_SOURCE=ABS_ORIGINAL_CORE \
    -DRADIANCE_BUILD=ABS_MATCHING_WHOLE_AQL_BUILD -DCMAKE_BUILD_TYPE=Release
  cmake --build NEW_BUILD --parallel 1
  NEW_BUILD/glm53_real_main_record_check --run-complete-record-pack \
    --checkpoint ABS_COMPLETE_PINNED_CHECKPOINT --home ABS_SHIPPED_HOME \
    --output ABS_NEW_TASK_SAMPLE_DIRECTORY
No CTest/automatic execution/implicit Core build/install/service target exists.
All parent compiler/runtime/source integrity/resource receipts remain separate.

Revision v2 / minimal actual compile repair
Parent v1 compilation failed because the manual checker called ::mkdir without
including <sys/stat.h>. The ONLY C++ difference is that one include in
sample_packer.cpp. Frozen v1 source and actual failed build evidence remain
unchanged. No helper/geometry/record/source/limit/admission/test-body behavior
is changed. Parent has separately PASSED actual full GLM metadata selection and
main geometry; this new packer itself still requires matching rebuild/runtime.

Revision v3 / bounded REAL configuration metadata repair
Parent v2 compiled but actual first pack refused the source-config1MiB cap.
The pinned verified original config is1716060B (SHA256
3a2c766e6a5256241b0b6ad82bc07e65bf4881b7211a0484cf18ce7020bfbd8f),
with924 per-layer quantization overrides. New explicit Limits.max_config_bytes
is2MiB, validated at no more than2MiB; smaller caller limits remain meaningful.
The source-config string was ALREADY charged at2x ACTUAL byte size before read;
that formula and the total128MiB metadata cap are unchanged. Receipt now exposes
actual config/index bytes. An actual-config-size-minus-one refusal is authored
with caller publication preservation. No payload/member/record/shard/source
acceptance geometry or other cap is changed; frozen v2 and failure stay intact.
SOURCE_CONFIG_REPAIR_QUOTE.json records exact metadata-only source byte/hash/
header witnesses and arithmetic for the selected actual expert. No author
payload/packing/compiler/runtime was run. Parent must rebuild and rerun v3.
