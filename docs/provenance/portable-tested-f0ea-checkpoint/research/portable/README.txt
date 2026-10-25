Portable Core V6, fixture V8, Components V12

This entrypoint builds twelve explicitly requested HOST checkers against one
matching private Core layout. The retained tiny Engine, compact metadata,
source/provider, TP4 protocol and transfer checks keep their original bodies.
The actual Engine gate also tests cold admission, Registry/geometry and the same
production plain/affine source validator. The component HOST quote body is
unchanged. No model download, GPU, server or real-data path is invoked.

Current evidence: matching Core V6/fixture V8 Release, ASan/UBSan and TSan HOST
suites passed 20 commands each. The portable twelve-program Release build passed
against a verified source-identical existing Core at snapshot d2915f6e; execution
of that portable set remains pending. A separate actual-data prewalk passed
5,088 ordinary jobs and 193,536 omissions, without executing the loader, Engine
or model kernels. These are distinct scopes.

Requirements: Linux/POSIX, Python 3.10+, CMake 3.24+, a C++20 compiler, pthreads
and ordinary build tools. Vendored JSON/llama/httplib source remains in radiance/.
No Torch, ROCm, network, tokenizer or model is required for these HOST programs.

From the repository root:
  python3 tools/verify_snapshot.py
  python3 tools/build_host.py --mode release --jobs 1

The default builds fresh out-of-tree HOST Core/reference/quantizer targets and
twelve checker programs; it does NOT execute them. Generated files and locks stay
under repo/build/. Current directories are core-main-verified-v6-host-MODE and
portable-main-verified-v6-fixture-v8-host-MODE, preserving older caches/evidence.
Explicit execution:
  python3 tools/build_host.py --mode release --jobs 1 --run-host-checks
  python3 tools/build_host.py --mode asan --jobs 1 --run-host-checks
  python3 tools/build_host.py --mode tsan --jobs 1 --run-host-checks

Optional existing Core reuse:
  python3 tools/build_host.py --existing-core-build /ABS/MATCHING/HOST/BUILD --run-host-checks

Reuse requires every exported Core source SHA to match the actual source tree;
no extra files except the two recorded historical pyc files are allowed. The
current frozen 1,535-file inventory itself already omits both pyc files. Core,
headers, archives, compiler, HOST/no-FFmpeg profile, mode, sanitizer and source
home are checked. A reused build needs libref/libquant. External Core is not
silently rebuilt. Source hashes do not authenticate arbitrary prebuilt binaries.

The compiler profile is closed: empty global C/C++ flags and linker overrides;
standard Release -O3 -DNDEBUG, Debug -g, MinSizeRel -Os -DNDEBUG and
RelWithDebInfo -O2 -g -DNDEBUG. Extra ABI defines such as _GLIBCXX_DEBUG, custom
toolchains/compiler arguments, LTO and launchers refuse. The sole existing-Core
launcher exception is regular executable /usr/bin/ccache with no arguments,
CMake list or symlink; helper builds never inherit it. Nonempty CCACHE_*, compiler
launcher, flag, include and library-path environment overrides refuse. System
compiler/cache trust remains required. The same checks run on a reused fixture
cache, after configuration and before execution; no guard was relaxed.

All subprocesses receive OMP_NUM_THREADS=1 and OMP_THREAD_LIMIT=1, with GPU
visibility variables empty. The tiny Engine, TP4 Session, actual Engine gate and
component checker remove all RADIANCE_* variables. Defaults preserve a caller's
existing sanitizer settings; otherwise they use:
  ASAN_OPTIONS=detect_leaks=0:halt_on_error=1
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
  TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1
Address, UB and thread errors halt. Leak detection follows the original research
setting; these runs do not establish a separate leak-proof claim.

Snapshot verification covers the exact exported file set, refusing symlinks and
hash drift, and excludes only top-level .git/, build/ and the manifest itself.
Current radiance/ must equal fixture V8's CORE_SOURCE_PINS.json: exactly 1,535
Core V6 files. Private ABI, compiler, cache and environment checks are unchanged.
These source/inventory guards are not a hostile concurrent-writer or signature
boundary. Builds execute trusted CMake/compiler code as usual.

Archived packages retain original paths, maps and author-time READMEs; use this
guide for current public commands. Historical Core V5 is reconstructible using
its two-file overlay. Earlier manifests and failed/pass receipts retain their
individual scopes.

Runner V4 is archived server-specific source with no default, import, build or
checker action. It admits the reviewed Core V6/fixture V8 pair or matching legacy
revisions through V5. It requires fresh V4 preparation and original private
model, store, HOME, runtime, container and hardware gates. Its operational stop,
restore, resource, signal and window bodies are unchanged from V3/V2. No shipped
library, model, native store, tokenizer, serving configuration or API credential
is exported. Actual AQL declaration, Engine model execution, maintenance and
serving are never automatic actions of this portable entrypoint.

The later actual four-GPU Engine attempt failed at Core V6’s strict padded-vocab
guard before source admission (tokenizer154856/model154880); no model computation
or GLM token occurred. Same-container QFN restoration and an independent QFN
completion passed. This is separate from the passed CPU/prewalk evidence and
the still-pending portable runtime. No further GPU window is queued.
