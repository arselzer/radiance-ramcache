Portable Core V8 / fixture V10 / Components V14: fourteen HOST programs

Current defaults build CPU-only Core/reference/quantizer targets and14 manual
HOST checkers. The new target glm53_main_bounded_decode_check uses exact frozen
Fixture10 decode_checker.cpp, including the unchanged gate source for its test
hooks. Its --run-host-decode mode tests HOST Scheduler/KV/BatchBuilder commits,
protocol-only PREFILL/DECODE frames and bounded admission/output preservation.
It grants no actual TP4 source authority and launches no GPU/model kernels.
The retained13 checker bodies are byte-identical to the preceding publication.

Fresh PUBLIC Core8 build/runtime is PENDING. Private matchingCore8 all20 HOST
commands and focused3 suites passed Release/ASan/TSan; a separate controlled
one-input/two-output GPU trial passed. Those do not replace a fresh public14
build/check. Core7's prior fresh-public13 PASS at manifest4d9c9e4a and its exact
source/docs checkpoint are preserved with their original scope.

Linux/POSIX, Python3.10+, CMake3.24+, C++20 and standard build tools are required.
All JSON/llama/httplib sources are vendored. No Torch/ROCm/model/network is
required for these automatic HOST targets. No CTest/auto-run/install/model/HTTP
or maintenance target is added. Default builds only:
  python3 tools/verify_snapshot.py
  python3 tools/build_host.py --mode release --jobs 1
  python3 tools/build_host.py --mode release --jobs 1 --run-host-checks
  python3 tools/build_host.py --mode asan --jobs 1 --run-host-checks
  python3 tools/build_host.py --mode tsan --jobs 1 --run-host-checks
Generated directories are core-main-verified-v8-host-MODE and
portable-main-verified-v8-fixture-v10-host-MODE. Old builds are preserved.

Optional exact existing HOST Core:
  python3 tools/build_host.py --existing-core-build /ABS/MATCHING/HOST/BUILD --run-host-checks
The actual source HOME must match every exported1535-file Core8 hash and closed
inventory. Whole archive/header/compiler/STL/backend/mode/sanitizer/cache source
identity is checked before and after configure. Only the two recorded historical
pyc paths may be additional external-source files; neither is exported. Cache
checks never admit a Core7 archive against Core8 private headers.

All prior compiler/ABI/environment safeguards remain: empty global C/C++/linker
injection; standard Release/Debug/MinSizeRel/RelWithDebInfo flags; no arbitrary
ABI defines/toolchain/compiler arguments/LTO. Existing Core alone may use exact
regular /usr/bin/ccache with no arguments/list; fresh builds and portable fixtures
use no launcher. Nonempty CCACHE_*, compiler/include/library/flag environment
overrides refuse. Source identity does not authenticate an arbitrary compiler or
prebuilt binary, and no hostile-writer or RSS-quota claim is made.

OMP_NUM_THREADS=1/OMP_THREAD_LIMIT=1 and empty GPU visibility are set. TinyEngine,
TP4Session, actual cold gates, component quotes, vocabulary and bounded-decode
checker commands remove ALL RADIANCE_* variables. ASan/UBSan/TSan defaults and
caller overrides remain unchanged; no sanitizer diagnostic is disabled here.
The vocabulary binary is invoked only with --run-host-vocab HOME. Its separate
actual-model CPU mode and archived manual Engine/Runner sources are NEVER
invoked by portable tools. No HOME binaries, checkpoint/native records, tokenizer
payload, serving configuration, API keys or parent deployment wrappers export.

The exact preceding guide/tools/CMake/manifest are retained in
 docs/provenance/base-d04b562-checkpoint/
Use docs/provenance/glm-core-v8-flight-wrapper.json for current source/target
mapping, guards and pending parent validation. Actual raw receipts stay under
 evidence/glm53-main-verified-v8/
No quality,16-output GPU, normal chat,262K,MTP,speed or production claim follows.
