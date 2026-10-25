GLM complete-main canonical store assembler — SOURCE ONLY, execution pending

This NEW fork preserves passed real-main-record-packer-v3, its geometry and
stock-Core declaration helpers byte-for-byte. It adds one offline DATA assembler,
not a runtime cache/Engine source factory. No author compiler, target import,
Python test, payload hash/pack, model/kernel, GPU, network or service execution.
Only the parent may compile/run and schedule any complete source/hash/export.

Parent interface
  python3 HERE/verify_source_pins.py
  python3 -m unittest discover -s HERE -p test_source_proof.py
  cmake -S HERE -B NEW_BUILD -DRADIANCE_SOURCE=ROOT/radiance-src \
    -DRADIANCE_BUILD=ROOT/build-ds41-aql -DCMAKE_BUILD_TYPE=Release
  cmake --build NEW_BUILD --target glm53_main_store_assembler --parallel 1
  python3 HERE/source_proof.py --source ABS_COMPLETE_CHECKPOINT \
    --mode subset --output ABS_NEW_PROOF
  NEW_BUILD/glm53_main_store_assembler --run-subset-validation \
    --checkpoint ABS_COMPLETE_CHECKPOINT --home ABS_SHIPPED_HOME \
    --source-proof ABS_NEW_PROOF --output ABS_NEW_VALIDATION_DIRECTORY

The published validation directory contains data/ (the actual bundle) and
receipt.json. Schema is glm53-main-store-assembler-subset-check-v1. A parent
process-group deadline, current serving-health/idle gate and source/command
receipts remain separate. No CTest or implicit Core/model execution is installed.
Actual one-card gfx1201/AQL discovery selects real stock DEVICE metadata rows;
all4 ranks are logical declarations, not multi-GPU/collective execution. Builder
instances are disabled and tracked Core allocations must remain unchanged.

Subset is EXACT(layer3,expert0)+(layer44,expert1) at every rank:8 records,
8 layer shards,26738688 canonical bytes. Different layer/parity keys exercise
independent source files and DOWN slices. It ALWAYS emits subset_component_NONREADY,
even if the supplied producer transcript happens to verify all72 source files.
It cannot finish a complete-main generation or grant RuntimeReady.

Separately scheduled full mode — NOT run or approved by this source task
  python3 HERE/source_proof.py --source ABS_COMPLETE_CHECKPOINT \
    --mode complete --output ABS_NEW_COMPLETE_PROOF
  NEW_BUILD/glm53_main_store_assembler --assemble-complete-main-bundle \
    --checkpoint ABS_COMPLETE_CHECKPOINT --home ABS_SHIPPED_HOME \
    --source-proof ABS_NEW_COMPLETE_PROOF --output ABS_NEW_VALIDATION_DIRECTORY \
    --max-native-bytes 161715585024
The explicit cap is canonical native payload only; metadata is additional.
Full mode requires48384 records=12096 per rank,168 shards=42 per rank, layers3..44,
experts0..287 on EVERY rank, rank/world/cuts/member identity and a complete72-file
producer proof. Canonical payload161715585024B=150.609375GiB. Each full layer
segment is962592768B. The driver also requires free space for exact payload,
metadata upper bound and32GiB reserve BEFORE staging. No full packing is implicit
in metadata mode or the subset flag. Parent decides any full execution window.

Public DATA API / lifetime
bundle_records.h exposes Assembler::begin, append_rank, finish, PublishedBundle
and verify. One Checkpoint object and its cache stay alive/immutable throughout
the assembly; its identity is checked at every append. No Program/Plan pointer
survives append_rank. PublishedBundle owns compact Record/Shard arrays,4 immutable
rank layout templates and bounded proof/bundle strings. It retains no borrowed
source mappings, graph pointers, FDs or a NativeRecordSource. Its spans are DATA,
not acquirable expert leases or source-admission booleans. Exact closure can be
complete_main_domain=true without implying a complete model or Engine readiness.

The actual full main graph/source/layout quote runs ONCE per logical rank in
append_rank, never inside the expert loop. The quote checks all12096 records,
real imported objects/maps/raw views, actual weight_share rectangles and selected
DEVICE canonical layout callbacks. Packing retains only that rank's bounded
geometry, then releases it. No whole WeightInfo/fused-weight/source payload clone.
Data include main MXFP4/E8M0 experts only: dense layers0..2, nonexperts and MTP
are not exported. The retained stock declaration hook names11 attn_kv_b entries;
receipt records ACTUAL raw/planned encodings and whether they are already BF16.
It does not claim a needed conversion merely because a metadata hook exists.
No nonexpert payload is converted or loaded by this tool.

Source proof trust boundary
PINNED_UPSTREAM_MODEL_INFO.json and pinned_files.h contain the exact public
amd/GLM-5.3-Flash-Quark-MXFP4@b5688f25491202978c19c4d036eef579f61bbe07
72-name/size/SHA256-or-Git-blob-SHA1 inventory, total185107699217B. Their public
metadata SHA isefbd6ca6794f23f16113afb8e77f6dd68f8053f2b0c8f0fde4db5cb3be947596.
source_proof.py uses Python stdlib hashes in64KiB chunks. Complete mode hashes
ALL72 files, comparing expected identities with before/after fstat and final
reopen audits. Subset mode hashes non-safetensors metadata and captures all file
stamps/extents, WITHOUT reading tensor payloads. CLI has no arbitrary-profile
argument; tests use clearly synthetic overrides that C++ domain checks reject.

C++ checks exact72 ordered expected tuples, mode/counters, canonical source root
and every CURRENT stamp against the trusted transcript. It does NOT independently
compute those complete-file cryptographic hashes. Arbitrary JSON cannot prove a
producer ran: a trusted parent must run the pinned companion, receipt its bytes/
command/hash and keep source immutable. Transcript is not a security attestation
or RuntimeReady. Directory/file stamps and FNV protect ordinary changes/integrity;
no hostile writer exclusion, cryptographic FNV security or filesystem lock claim.
Proof output is canonicalized outside original checkpoint BEFORE temporary writes,
including symlink-parent/dotdot aliases; source is never modified by the tool.

C++ separately streams/hash-compares the COMPLETE six physical code/scale tensors
of each packed expert against that SAME opened mapping once, caching compact
raw ranges/checksums. Every rank's actual copied row band is also compared against
its current map. Source descriptors use a2-file LRU broker, opened O_NOFOLLOW and
rechecked. Header cache holds at most2 parsed headers; index is parsed once.
Final all72/root audits occur after all result/manifest preparation and BEFORE
atomic bundle publication. Read-only atime effects are outside the mutation claim.

Common identity / manifest format
Canonical binary member/record/segment semantics remain S1-style. Actual-profile
layer manifests use radiance-native-record-store-v2 with an explicit NEW actual
origin; old synthetic/generic Readers/providers remain CLOSED and unchanged.
Their accepting this new origin is not claimed or implemented. Shared Reader/
POSIX/checksum/padding primitives are reused by including the byte-identical v3
main_records.cpp in ONE assembler translation unit; CMake never compiles it twice.
The original one-record checker remains an optional separate regression target.

Each expert keeps a separate complete-six-raw-tensor component_source_id. The
common final bundle_id is computed only AFTER all4 ordered catalogs and finalized
shard manifest hashes exist. Basis is pinned source CONTENT identities, explicit
format/layout profile, all key/member/cut/spec/checksum data and the manifest set.
It excludes source path/inode/stat witnesses, runtime context/batch, cache budget
and resident mask. Those witnesses live in source-proof.json and a separate proof
checksum, not in the content-ID basis. It is not a hash of every final bundle.json
byte. Rank-specific offsets/cuts and manifest checksums are preserved. A shard
never gets rewritten to contain the final ID, avoiding checksum circularity.
FNV64 IDs are noncryptographic identifiers, not collision-resistant authentication.
Data may be reusable after later context/batch admission only if a FRESH actual
chosen-layout check still agrees; changed layouts receive no compatibility claim.

42 layer shards per rank bound each manifest/segment. Record specs carry real
member names/layout/ROW halves/COL cuts; complete closure never relabels a subset.
Every member must fill the exact3342336B record. This actual class has no padding;
padding validation is retained but no fabricated padding mutation is claimed.
COL copy bands read whole bounded rows and compact selected bytes in-place
forward. This avoids per-row tiny syscalls, with exact nibble/scale bytes and one
64KiB buffer. Source-band I/O is WIDER than selected/native bytes and is counted
separately (not a bandwidth/performance claim).

Publication / failures
All writes go to a private0700 task-owned tree; source/old outputs are untouched.
Only Linux renameat2 NOREPLACE publishes. A failure closes local descriptors and
removes known owned paths without allocating extra cleanup FDs/dir streams.
Any failed append/finish makes that assembler terminal. Caller publication changes
only after successful final commit, with all fallible result work already done.
The checker has an additional outer NOREPLACE transaction so failed oracle/fault
checks do not publish its final validation directory. File/directory fsync is
checked; parent fsync after successful rename is best effort, not a power-loss
transaction guarantee. The original checkpoints, frozen sources and prior stores
are not overwritten/cleaned.

verify is a strict DATA scan against the independently retained private capture:
all fixed root/layer metadata, manifest/spec/key/range/checksum values, every record
checksum and all padding. It holds/rechecks metadata stamps through the full scan.
It enforces caller metadata admission BEFORE I/O, including actual-minus-one
negative. Assembler is destroyed before verification, proving capture independence
and avoiding overlapping broker/window/FD ownership. This is NOT a standalone
fresh-process reopen from arbitrary saved JSON. A trusted reconstruction/Reader
handoff plus actual-profile provider adapter is still required for that future
factory; files/public Record spans alone must never be promoted to Ready.

Resource accounting
256MiB is a requested planning allowance, not allocator-enforced/RSS quota.
Quotes use real sizeof Record/Shard/RawGroup and selected count/capacities, actual
index/config sizes, bounded header/parser overlap, once/rank geometry construction
and one serialized+parsed layer manifest. Layers move their JSON arrays rather
than retaining a second full tree. Manifest/spec/component worst-case size is
pre-admitted before payload work. Caller Checkpoint/cache/Program/plugin/importer
metadata, mappings/OS page cache and driver allocations are SEPARATE owners.
The manual independent actual+expected record buffers are separately bounded8MiB.

Assembler owns at most2 cached source FDs plus bounded metadata/output/root FDs
(upper8); the independent verifier owns6, AFTER assembler destruction. Caller
Checkpoint mappings/descriptors and external process users are separate. One
64KiB assembler/hash window; no whole record staging in the packer. Independent
checker buffers are explicit. Byte/counter receipts separate trusted producer
hash bytes, own complete-tensor reads, wider source-band reads, native writes,
requested metadata and actual/upper metadata ON DISK. max_total_bytes/total_bytes
refer to canonical segments only. Disk metadata upper is169*4MiB+128KiB full or
9*4MiB+128KiB subset. Actual metadata_written_bytes is printed. Buffered I/O/OS
page cache and stdlib/plugin allocator internals are not hidden inside the ledger.
Full execution reads multiple passes beyond output size (proof/hash/copy/verify/
independent oracle);150.6GiB output is NOT a150.6GiB total-I/O claim.

Parent-run checker matrix (SOURCE AUTHORED, no observed PASS)
- Eight Python tiny stdlib tests: full/partial hash scope, mutation, missing/
  symlink/size, mode, checksum, Git blob hash, caps, final audit, early and late
  NOREPLACE and canonical output-parent alias refusal before any temp write.
- Real stock AQL metadata,8 records across4 ranks/two layers/parity; each actual
  native byte compared to independent mapped physical row/column coordinates,
  without packer cursor/weight_share reuse or observed-output expected seeding.
- Three exact counted semantic I/O faults after real native segment writes:
  pread EIO, pwrite EIO, and successful source-index fstat reported EIO AFTER a
  real layer-manifest write. No source mutation is performed or claimed.
- A separate complete4-rank subset assembly witnesses actual late final rename
  EEXIST preserving a destination marker and removes only its private output.
- First-allocation OOM before I/O; invalid mode/metadata cap, foreign Checkpoint,
  missing ranks, source-proof domain distinction, verifier cap/actual-minus-one,
  existing generation, caller previous-publication preservation, and actual
  task-owned record corruption/refusal/restore (good-bad-good).
Positive winner quotes4 graphs once each. Fault/collision cases are separate
transactions: subset driver totals9 declarations/11 full-geometry quotes, never
one quote per expert. Exact assertion count is printed by actual parent run.
Complete mode performs one4-rank assembly/verification/oracle pass, no repeated
fault exports. No actual model forward, TP collective, cache lease, server swap,
throughput/accuracy or whole-Engine startup is proved.

Full-scale I/O arithmetic (source-derived, not timing)
The64KiB row-band loop uses102 source-band preads and102 native pwrites per
record (GU34, DOWN codes64, DOWN scales4). Each pack/verify CRC takes51 preads
per record. Each expert's complete-six-tensor component proof is204 preads ONCE
across all4 ranks, not204 per rank. Full no-short-read baseline is4,935,168
native pwrites and roughly12,386,304 payload preads including pack/verify/oracle;
headers/metadata and short-read/EINTR retries add calls. C++read bytes are about
970,293,510,144 plus the separately scheduled185,107,699,217B full producer hash
pass; mapped comparisons/oracle touches and OS-cache physical I/O are separate.
This does not predict runtime from NVMe sequential bandwidth. The checker
reports actual total pread/pwrite counts and positive assembly+verify deltas
separately from its fault/collision transactions. No loop redesign/perf claim.
