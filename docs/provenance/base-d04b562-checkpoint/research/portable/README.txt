Portable Core V7, fixture V9, Components V13: thirteen HOST programs

Current public defaults build CPU-only Core/reference/quantizer targets and
thirteen HOST research checkers. The new thirteenth target is the exact fixture V9
vocabulary_checker.cpp with the same export-dynamic and GNU wrap flags. Its
--run-host-vocab HOME mode uses synthetic metadata and actual matching HOST
sampler/mask/top-k rows. It downloads no model and launches no GPU or API server.
The same binary's --model-vocab-check MODEL mode is explicit/manual only; the
portable driver NEVER invokes it. Archived Runner V5 is never imported or run.

Current fresh-build result: all13 Release HOST programs PASSED after a whole
Core build from the published source at manifest
4d9c9e4a8d662f90441c00c40519f35709ea4875347db7a42b0ae24edc9b6db1
(3287files). Checks used only that freshly built public archive; no private
research Core archive was used. The tested manifest and overwritten metadata
are preserved. Fresh-build/13check logs and receipts are under
evidence/glm53-main-verified-v7/portable13-fresh-{build,runtime}.

Earlier, all thirteen Release HOST programs passed at exact tested source manifest
8b966d37d75f96af9f1ad5f81d5f252f25072cc30726f68ba6a89601d4ec0d14
(3212files), using verified source-identical existing Core7 archives and newly
built portable fixtures. The final export adds evidence/normalized docs only;
no tested source, tool or CMake byte changed. Original Core7 full20 HOST commands
also passed Release/ASan/TSan. Focused mask+actualCPU vocabulary passed all modes;
the initial focused ASan ptrace/LSan executor failure remains separately failed,
with its unchanged host rerun passed. Full-data prewalk and a separate controlled
one-input/one-output GPU trial passed; none are automatic portable actions.

Requirements: Linux/POSIX, Python 3.10+, CMake 3.24+, C++20, pthreads and standard
build tools. All JSON/llama/httplib source is vendored. No Torch/ROCm/model/network
is required for the automatic HOST path.

Commands from the repository root:
  python3 tools/verify_snapshot.py
  python3 tools/build_host.py --mode release --jobs 1
  python3 tools/build_host.py --mode release --jobs 1 --run-host-checks
  python3 tools/build_host.py --mode asan --jobs 1 --run-host-checks
  python3 tools/build_host.py --mode tsan --jobs 1 --run-host-checks
The default builds only. The explicit flag runs all 13, including intended
refusal/fail-stop subprocesses. build/ contains generated files and locks.
Current directories core-main-verified-v7-host-MODE and
portable-main-verified-v7-fixture-v9-host-MODE never silently reuse old headers.

Optional existing matching HOST Core:
  python3 tools/build_host.py --existing-core-build /ABS/MATCHING/HOST/BUILD --run-host-checks
Every exported source SHA must match the actual CMake source tree. Whole-Core
archives, standard compiler, HOST/noFFmpeg/backend/mode/sanitizer/source-home and
fixture-cache identities are checked. The only recorded extra original source
files allowed for external reuse are two historicalpyc; frozenCore V7 already
omits both. Existing builds need matching libref/libquant. No arbitrary prebuilt
binary authenticity or hostile concurrent-writer protection is claimed.

Private ABI/compiler/environment/ccache/cache checks are unchanged from aacccd8:
empty globalC/C++/linker overrides; standardRelease -O3 -DNDEBUG/Debug -g/
MinSizeRel -Os -DNDEBUG/RelWithDebInfo -O2 -g -DNDEBUG. ExtraABIdefines/toolchains/
compilerargs/LTO/launchers refuse. Existing Core alone may name regular
/usr/bin/ccache with noargs/list/symlink; helpers never inherit it. Nonempty
CCACHE_*/launcher/flag/include/library-path environment overrides refuse.
Checks repeat afterconfigure and beforeexecution. Trusted compilers/CMake/source
are still required. Sanitizer errors halt; no diagnostic is disabled.

OMP_NUM_THREADS=1/OMP_THREAD_LIMIT=1 and emptyGPUvisibility are set. The tinyEngine,
TP4Session, actualEnginegate, component and newvocabulary checker remove ALL
RADIANCE_*variables. Existing ASan/UBSan/TSan default options and caller overrides
are unchanged (ASan detect_leaks=0:halt_on_error=1; UBSan halt_on_error=1:
print_stacktrace=1; TSan halt_on_error=1:second_deadlock_stack=1).

Exact snapshot file-set/hash/symlink verification excludes only top-level.git/,
build/and the manifest. CurrentCore pins are fixture V9/CORE_SOURCE_PINS.json,
exactly1535Core V7files; no second subtraction of oldpyc. All raw archives retain
original paths/READMEs/maps; use this guide for current normalized commands.
OldCore V6 is reconstructible via its six-file overlay; V5 then uses two more.
No raw package or old evidence/testedmanifest was modified to manufacture a pass.

Manual candidate profile is one prompt/output token, maxTok1/seq1/spec0/exactwire/
context512, nofixedresidents. Runner V5 requires originalprivate model/store/HOME/
focus+three HOST modes/prewalk/container/hardware gates and fresh V5preparation.
Its source is archived only; no maintenance action/default model/source admission,
weights/tokenizer/native-store payload/compiled library/APIkey/serving config is
included or invoked. Only the separate one-input/one-output GPU trial is validated. Subsequent decode,
quality/performance/normal chat readiness remain unvalidated.
