QFN official FP8 shared-PLE-scale overlay — SOURCE CANDIDATE

Local metadata-only builder and verifier for ONE pinned release:
Qwen/Qwen3.8-Flash-Next-FP8, revision236dfdf285828023ca3bcd3f37366c58a3469b13.
No network/downloader, converter invocation, numerical code, Core patch, GPU,
Engine/serving integration or model payload copy is included. No source author
execution/import/testing was performed; only Python AST/hash inspection. Parent
owns CPU tests and any subsequent real-checkpoint invocation. All frozen parents
and the input checkpoint remain unchanged.

Parent commands (do not run against an incomplete download):
  python3 HERE/test_overlay.py
  python3 HERE/overlay.py build --source ABS_COMPLETE_PINNED_FP8_DIR --output ABS_NEW_OVERLAY
  python3 HERE/overlay.py verify --overlay ABS_NEW_OVERLAY
CLI accepts no synthetic-profile override. Unit tests use a separately labeled
small Profile through the Python API; they cannot establish real-model payload
correctness. The complete real FP8 checkpoint is not present in this package.

Practical output
NEW directory with symlinks to the resolved original131 weights and original
metadata/tokenizer/license artifacts, a new copied/extended index, a tiny
safetensors sidecar (128*2=256B scalar payload), one PLE conversion recipe, and
an explicit derived-overlay provenance receipt. Original shard data and index
are never written. Sidecar contains one disjoint BF16[1]0x3951 tensor named
...ngram_embedding.shard_N.weight_scale perN0..127. Parent scale is authenticated
from original ...ngram_embedding.weight_scale, not inferred from aliases.
Default fixed-global output rule writes scale0x3951; experts have no recipe rule
and retain their native E4M3/BF16block128 numbers. Nonexperts remain the official
FP8 checkpoint's forms, which differ from the controlled production-int8 planner.

Required preflight and honest scope
- Bundled108017B public model-info is pinned by SHA256; it gives exact file sizes,
  metadata identities and the expected LFS hashes for the release. Source config
  and17410140B full index are independently pinned by the parent's saved hashes.
- Every144 listed artifact must exist with the exact extent. Metadata bodies are
  SHA256/Git-blob-SHA1 checked, including config/tokenizer/index. All131 safetensors
  headers are read and structurally validated; every152089 index name must have
  exactly its declared owning file and contiguous, nonoverlapping shape-sized
  payload extent. No extra/uncovered header tensor or file gap is accepted.
- Every main+MTP expert header (150528 code/scale tensors) must have the exact
  source name/domain, E4M3 code shapes640x2560 or2560x640, and BF16block128 scales.
- All128 PLE headers must be complete F8_E4M3[2500012,160], plus one parentBF16[1].
  Only TWO source payload bytes are read: the authoritative parent0x3951. Existing
  per-shard _scale OR higher-precedence _scale_inv names refuse, as do global_inv
  and output-sidecar/name collisions. No alias can silently lose resolver priority.
- Complete file EXTENTS and all header/index/schema facts do NOT authenticate
  the185GB source payload. Expected LFS hashes are recorded but NOT computed or
  verified here. Arbitrary payload corruption/NaNs before admission may pass
  header inspection; tests explicitly preserve this distinction. Full payload
  verification must come from the independently trusted download/hash process.
  This builder cannot claim a completed/HF-verified download from file size alone.
- Each file/link's device/inode/extent/mode/mtime/ctime and resolved target are
  captured and rechecked before publication. verify rechecks the full source
  metadata/headers and exact saved generation, owned output bytes, all aliases,
  all links and the exact overlay file set. Source must remain immutable through
  conversion. Stamps/FNV/ordinary stat checks are NOT hostile-writer security or
  exclusion of concurrent writers; no TOCTOU/tamper-proof claim is made.

Atomicity/resources
Temporary directory is task-owned, under the destination parent; files use
O_EXCL/O_NOFOLLOW and bounded writes. The final step is Linux renameat2 NOREPLACE,
with no racy overwrite fallback; existing/late-created destinations stay intact.
Before publication, failures clean only the owned temporary directory. Owned
files and temporary directory are fsynced; no power-loss durability guarantee
about the parent directory or subsequent external modifications is claimed.
Source snapshot symlinks are supported by resolving, opening O_NOFOLLOW on the
exact regular target and recording/rechecking BOTH link and target identities.

Caps:20MiB per source index/metadata file,1MiB per header,64MiB AGGREGATE source
metadata/header bytes, bounded tensor/file counts and integer products,1MiB
receipt,4096-byte paths and512-byte tensor names. At most one source data FD is
held at once. Python dictionaries/strings/serialization create additional heap
storage; these are byte/count caps, NOT an allocator-enforced256MiB/RSS quota.
No full WeightInfo/model clone or expert payload buffer exists. A one-layer RAM
cache and GPU expert cache are unrelated to this offline mapping helper.

Existing bounded actual-converter evidence is separate:
validation-qfn8bit-scale-alias-20261025T160346Z passed3groups234checks5converter
calls7OUR-preflight refusals, exact all254 FINITE codes/signedzero/subnormals and
unchanged expert planes. That is preservation coverage, not arbitrary midpoint
rounding/model-quality proof. Its converter-only test architecture is not the
production QFN graph; real layouts are checked separately by the native metadata
planner. The source-only builder tests below do not rerun conversion.

Eight authored CPU unit tests cover exact small overlay/verification/input
immutability, missing/truncated files, owner mismatches/gaps/overlap, wrong scalar
dtype/shape/value, aliases and _scale_inv precedence, caps/duplicate JSON, early
and late NOREPLACE collision/owned-temp cleanup, source mutation before publish,
output tampering and explicit unverified-full-payload semantics. Parent execution
is pending; no runtime PASS is claimed for this new builder.
