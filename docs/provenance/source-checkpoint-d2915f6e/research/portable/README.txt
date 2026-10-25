Current pair: CoreV6 / EnginefixtureV8 / ComponentsV12 (source checkpoint)

FixtureV8 corrects the nested strictP3 privateCorepin map to exactCoreV6; its
C++/CMake/test/control bodies are byte-exactV7, including validbool sentinels.
All3active top/device/strictP3 maps now equal1535Core6pins83d57401. PriorV6/V7
fixtures remain exact historical sources; the initialCore6HOSTconfigurefailure
is preserved with its original scope. No Core change or inferred runtimepass.
Current portable fixture builddir portable-main-verified-v6-fixture-v8-host-MODE
avoids old artifacts. RunnerV4 admits only reviewed6/8 (or unchangedmatchinglegacy),
otherwise exactV3/V2 operational bodies; archivedsourceonly/no default use and
freshV4prepare required. Allcurrent runtime/prewalk/model gates stillpending.

Following fixtureV7/V3 notes are historical; current selection is V8/V4.

Portable CoreV6 / twelve explicitly requested HOST checks (current)

CoreV6 is exactly1535files; only ordinary-loader source eligibility differs from
CoreV5. Current manual gate uses Engine fixtureV7 on CoreV6 (fourgroups including the same production
source validator); the pure component HOST body is exact ComponentsV12. Defaults
use core-main-verified-v6-host-MODE and portable-main-verified-v6-fixture-v7-host-MODE, keeping
older private header/archive fixture caches separate. All compiler/ABI/env/mode/
ccache/source/archive guards are retained unchanged. No AQL, actual metadata,
model, maintenance runner, server or loader execution is added to defaults.

Matching CoreV6 builds, HOST/sanitizer runtime, portable12, actual full prewalk
and GPU/model test remain PENDING parent execution for this snapshot. CoreV5
passes retain their original archived source/manifest scopes, not a V6 pass.
The two replaced V5 loader sources are preserved under
research/historical-core-v5-overlay; all other1533Corefiles are byte-identical,
so original1535V5pins are reconstructible without duplicating weights/assets.

Following V5 instructions/results are historical; current source selection is
Engine fixtureV7/ComponentsV12 and current whole-Core pins are in Engine fixtureV7.

Portable verified-main Core V5 / twelve HOST source checks
========================================

This normalized entrypoint builds twelve explicitly invoked HOST checkers against
ONE matching private Core layout. The ten retained Foundation/contract/compact/
strict-P3/unified/tiny-Engine/TP4 checks keep their original source bodies. Two
new checks exercise actual Engine cold admission/Registry/geometry refusals and
pure actual-profile Sampler component quotes. No real checkpoint is required.
A tiny synthetic checker calls Engine.start/retire but refuses model steps. The
new real-source Engine gate refuses HOST startup before opening model data.
Actual AQL declaration, complete-data prewalk and one-token Engine execution
remain separate manual research paths, never automatic portable targets.
This V5 exported snapshot's build/runtime validation is pending parent execution.
Old source and failed/pass receipts retain their original scopes.

Expected layout:
  radiance/                 exact frozen verified-main V5 source (1535 files; pyc already absent)
  research/glm53-engine-native-foundation-v3-prototype/
  research/glm53-tiny-engine-native-transport-v4-prototype/
  research/ds41-unified-native-policy-host-prototype/
  research/glm53-sharded-native-provider-prototype/
  research/glm53-expert-record-store-prototype/
  research/glm53-record-cache-bridge-prototype/
  research/glm53-cache-prototype/
  research/glm53-main-verified-source-engine-v5-prototype/
  research/glm53-main-verified-components-v10-prototype/
  research/portable/        this normalized CMake entrypoint
  tools/{verify_snapshot.py,build_host.py}
  SNAPSHOT_MANIFEST.json    NEW complete exported file SHA256 map

Requirements: Linux/POSIX, Python 3.9+, CMake 3.24+, a C++20 compiler, pthreads
and ordinary build tools. Vendored JSON/llama/httplib source stays in radiance/.
No Torch, ROCm, downloaded tensor/tokenizer/model or network is needed for HOST.

From the repository root:
  python3 tools/verify_snapshot.py
  python3 tools/build_host.py --jobs 1

The default builds fresh out-of-tree HOST Core/reference/quantizer targets and
the twelve portable checker programs; it does NOT execute them. The experimental
DS41 source vocabulary is built explicitly to match the original fixture build
profile, not installed/activated as a service. The default build launches no
checker or Core executable; only --run-host-checks executes the tiny fixtures.
All generated files and the compiler serialization lock stay under repo/build/.
V5 uses core-main-verified-v5-host-MODE and portable-main-verified-v5-host-MODE;
older Foundation builds/evidence are left in place and are never silently reused
with the new Core headers. An explicit source mismatch still refuses.
Repeated invocations may reuse that snapshot's own verified build directory.

Explicit manual HOST checks (including intended fault/refusal subprocesses):
  python3 tools/build_host.py --jobs 1 --run-host-checks

Sanitizers use distinct matching Core+helper builds; checks remain enabled:
  python3 tools/build_host.py --mode asan --jobs 1 --run-host-checks
  python3 tools/build_host.py --mode tsan --jobs 1 --run-host-checks

Optional parent-owned already-built Core reuse:
  python3 tools/build_host.py --existing-core-build /ABS/MATCHING/HOST/BUILD --run-host-checks

Reuse is allowed only after the actual CMAKE_HOME_DIRECTORY source tree matches
EVERY exported radiance/ file SHA256, with no extra source files except the two
explicit original pyc files. The source-home path may differ; private source
bytes may not. Archives, HOST/no-FFmpeg profile, build type, sanitizer mode and
absolute compiler are checked. It does not silently rebuild an external Core.
An existing matching build must contain libref and libquant to run the checks.
No claim is made that source hashes authenticate arbitrary prebuilt binaries.
The accepted compiler cache profile is deliberately closed: empty global C/C++
flags; default Release -O3 -DNDEBUG, Debug -g, MinSizeRel -Os -DNDEBUG and
RelWithDebInfo -O2 -g -DNDEBUG; empty linker overrides. Extra ABI defines such as
_GLIBCXX_DEBUG, custom toolchain/compiler arguments, alternate launchers or LTO
are refused rather than silently compiling helpers with different layouts.
A reused matching Core may record ONLY the single /usr/bin/ccache system launcher
without arguments or a CMake list; it must be a regular executable, not a symlink.
This is the reviewed v4 compiler-cache front end, not permission for wrappers or
injected flags. Helpers never inherit it. All other launcher/toolchain/ABI checks
remain unchanged. Nonempty CCACHE_* or compiler-launcher environment overrides,
and flag/include/library-path overrides, are refused. System compiler/cache tool
trust is still required; cache/source inspection does not authenticate arbitrary
prebuilt archives. A machine without that exact launcher can use a fresh default
no-launcher Core build instead.
The same closed compiler-profile check applies to a reused portable fixture
cache, immediately after configure and again before any checker execution;
compiler, mode, source-home and Core-build paths must still match.

All subprocesses receive OMP_NUM_THREADS=1 and OMP_THREAD_LIMIT=1 with GPU
visibility variables empty. The tiny Engine, TP4 Session, verified-main Engine gate and pure component
checker remove ALL RADIANCE_* variables. Other retained checker and build
environments remain unchanged. ASan defaults (unless the caller already set the
variable) are ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 and
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1. TSan similarly defaults to
TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1. These match the original
research process controls. Address/undefined-behavior/thread errors halt; leak
detection follows the original detect_leaks=0 process setting.

Manifest verification checks all exported files, refuses missing/extra files,
symlinks and hash drift, and excludes only top-level .git/, build/ and the
manifest itself. It also checks the exported Core against the unchanged frozen
1535-file EngineV5 Core pin inventory exactly. The new pin manifest already
omits both historical pyc files; neither is subtracted twice or exported.
Source generation and verification assume an ordinarily immutable checkout;
these checks are not a hostile concurrent-writer or cryptographic-signature
security boundary. Builds may execute trusted CMake/compiler code as usual.

Archival verify_source_pins.py scripts still intentionally refer to historical
research parents and whole old Core trees. They are provenance artifacts, not
this portable entrypoint. Optional physical V3/composite exporter source is
archived separately; its older private-Core pin loops are not bypassed here and
it is NOT built by the initial portable target. DEVICE sources remain source
artifacts only; this entrypoint never configures HIP or executes a GPU checker.

The archived glm53-manual-maintenance-runner-v2-20261025 is server-specific
parent research source, not a public maintenance/deployment command. It requires
private build/data/prewalk/preparation receipts and exact hardware/container
baselines. It is neither imported, configured, built nor run by this entrypoint.
No shipped binary, checkpoint, complete native-store payload, service config,
API credential, Docker environment or preparation HOME is included.

ComponentV10 is current; its HOST checker is byte-exact V7. V7 remains archived
with a known actual-graph helper metadata lifetime defect; its complete prewalk
failed before Engine. V9 fixed ownership but an invented4096KV ceiling rejected
valid real metadata; V10 preserves ownership and checks exact observed counts.
The initial V5 portable12 Release pass is archived with its exact914c311b snapshot
manifest; a fresh check of this updated source map remains parent-owned.

Engine fixtureV6 is preserved historical source: its new refusal sentinel
memset produced an invalid bool representation. FixtureV7 uses a valid aggregate
and an unsigned-char memcpy snapshot, with allcases unchanged and exactCoreV6.
RunnerV3 is archived source only: same stop/restore/resource/window bodies asV2,
explicit reviewed Core6/fixture7 buildreceipt identity and full inventory binding.
Neither runner is imported or invoked by portable tools. FreshV3prepare required
for any separately authorized server-specific run in the original private setup.
