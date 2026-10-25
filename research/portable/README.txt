Portable Engine transport Core v4 / eight HOST source checks
========================================

This NEW entrypoint compiles manual Foundation checker v3, genuine startup
contract, compact authority normal/collision, compact inventory, strict P3 and
23-case unified-policy checker bodies plus the actual Engine transport v4
checker, with ONE matching private Core layout. Original checker bodies remain
byte-exact. The eighth checker calls actual Engine.start for closed synthetic
HOST transport, exercises startup/refusal/retirement, and explicitly refuses
serve/model/sampler steps. It is not real-checkpoint conversion, full-model/TP/
GPU serving or a deployment procedure. The older Foundation and earlier Engine
source/failed-run receipts remain unchanged; v4 is a distinct validated source
revision, not a relabeling of the older inventory.

Expected layout:
  radiance/                 selected frozen Engine transport v4 source minus TWO pyc
  research/glm53-engine-native-foundation-v3-prototype/
  research/glm53-tiny-engine-native-transport-v4-prototype/
  research/ds41-unified-native-policy-host-prototype/
  research/glm53-sharded-native-provider-prototype/
  research/glm53-expert-record-store-prototype/
  research/glm53-record-cache-bridge-prototype/
  research/glm53-cache-prototype/
  research/portable/        this NEW CMake entrypoint
  tools/{verify_snapshot.py,build_host.py}
  SNAPSHOT_MANIFEST.json    NEW complete exported file SHA256 map

Requirements: Linux/POSIX, Python 3.9+, CMake 3.24+, a C++20 compiler, pthreads
and ordinary build tools. Vendored JSON/llama/httplib source stays in radiance/.
No Torch, ROCm, downloaded tensor/tokenizer/model or network is needed for HOST.

From the repository root:
  python3 tools/verify_snapshot.py
  python3 tools/build_host.py --jobs 1

The default builds fresh out-of-tree HOST Core/reference/quantizer targets and
the eight portable checker programs; it does NOT execute them. The experimental
DS41 source vocabulary is built explicitly to match the original fixture build
profile, not installed/activated as a service. The default build launches no
checker or Core executable; only --run-host-checks executes the tiny fixtures.
All generated files and the compiler serialization lock stay under repo/build/.
V4 uses core-engine-v4-host-MODE and portable-engine-v4-host-MODE subdirectories;
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
visibility variables empty. Only the actual Engine checker additionally removes
ALL RADIANCE_* variables, matching its fixed clean-environment admission; seven
legacy checker environments and build environments remain unchanged. ASan defaults (unless the caller already set the
variable) are ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 and
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1. TSan similarly defaults to
TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1. These match the original
research process controls. Address/undefined-behavior/thread errors halt; leak
detection follows the original detect_leaks=0 process setting.

Manifest verification checks all exported files, refuses missing/extra files,
symlinks and hash drift, and excludes only top-level .git/, build/ and the
manifest itself. It also checks the exported Core against the unchanged frozen
1512-file Engine transport v4 Core pin inventory with exactly the two recorded pyc files removed.
Source generation and verification assume an ordinarily immutable checkout;
these checks are not a hostile concurrent-writer or cryptographic-signature
security boundary. Builds may execute trusted CMake/compiler code as usual.

Archival verify_source_pins.py scripts still intentionally refer to historical
research parents and whole old Core trees. They are provenance artifacts, not
this portable entrypoint. Optional physical V3/composite exporter source is
archived separately; its older private-Core pin loops are not bypassed here and
it is NOT built by the initial portable target. DEVICE sources remain source
artifacts only; this entrypoint never configures HIP or executes a GPU checker.
