Core12 / Fixture17: sixteen explicit portable HOST checks

Fresh public Release build/runtime is PENDING. Matching private Core12/Fixture17
Release, ASan/UBSan and TSan full20 plus focused5 suites passed; those do not
replace the fresh public build. No model or GPU result follows from these checks.

Linux/POSIX, Python3.10+, CMake3.24+ and C++20 build tools are required. The
HOST targets need no Torch, ROCm, model download or API server.

 python3 tools/verify_snapshot.py
 python3 tools/build_host.py --mode release --jobs 1
 python3 tools/build_host.py --mode release --jobs 1 --run-host-checks

The default builds without executing. --run-host-checks explicitly invokes the
sixteen programs. The first fourteen retain their prior bodies and arguments;
the two additions exercise normal-prefill HOST scheduling/protocol and the
HOST-only declaration refusal/terminal lifecycle. They grant no actual model
Source/Session/AQL admission or GPU readiness. No manual Engine, real-checkpoint
mode, maintenance runner, HTTP, Docker, CTest or install step is dispatched.

Fresh builds use build/core-main-verified-v12-host-MODE and
build/portable-main-verified-v12-fixture-v17-host-MODE. Modes release, asan and
tsan retain their existing flags and diagnostics. Optional --existing-core-build
ABS requires exact1536-file Core12 source identity and the closed matching
compiler/backend/ABI/mode/cache checks; a Core8 archive is refused. Fresh fixtures
use no compiler launcher. Existing safeguards against toolchain/environment/
packing/linker/LTO overrides remain. GPU visibility is disabled, OMP thread
limits remain set, and the relevant checks clear RADIANCE_* variables.

Core source must exactly match frozen map
c5c474c9ab5e8e96bd939069ee646937f99b9237d6cc1933bd5b8b69316eaf2a;
Fixture17 receipt is82665d51cccdb3bd55e16b4fa62282414737a2ae11dcd7db0f2024592f6c48b2.
Archived source-era CMake/pins/absolute paths remain historical; these portable
tools are the supported HOST entrypoint. Source identity alone does not establish
arbitrary binary/compiler provenance or an allocator/RSS quota.

The prior Core8 public14 and earlier checkpoints remain in evidence/provenance
with their original scopes. Exact overwritten Core8 files and manifest are
archived at docs/provenance/core8-before-core12/. No serving deployment or
configuration change is part of this source export.
