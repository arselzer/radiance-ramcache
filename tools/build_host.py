#!/usr/bin/env python3
"""Build the isolated HOST research checks; execute them only with an explicit flag."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("release", "asan", "tsan"), default="release")
    parser.add_argument("--jobs", type=int, default=1, help="bounded compile parallelism, 1..8 (default 1)")
    parser.add_argument("--existing-core-build", type=Path,
                        help="absolute existing HOST build; reuses it only after exact source/profile verification")
    parser.add_argument("--run-host-checks", action="store_true",
                        help="after building, explicitly execute the fourteen manual HOST checker programs, including bounded decode and real-source cold gates")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be in 1..8")
    root = args.repo_root.resolve(strict=True)
    verify = runpy.run_path(str(root / "tools/verify_snapshot.py"))
    refuse = verify["Refusal"]
    build_root = root / "build"
    if build_root.is_symlink():
        raise refuse("repo/build must not be a symlink")
    build_root.mkdir(exist_ok=True)
    # Only this portable repo's processes use this lock. A parent orchestrator
    # may additionally serialize its unrelated research compiler queue.
    lock = build_root / ".portable-host.lock"
    if lock.is_symlink():
        raise refuse("build lock must not be a symlink")
    with lock.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        files = verify["verify_snapshot"](root)
        env = dict(os.environ)
        for key in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CMAKE_TOOLCHAIN_FILE",
                    "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "LIBRARY_PATH",
                    "GCC_EXEC_PREFIX", "COMPILER_PATH",
                    "CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_C_COMPILER_LAUNCHER"):
            if env.get(key, "").strip():
                raise refuse("unsupported build environment override for closed HOST profile: " + key)
        # A reused Core may record the exact reviewed /usr/bin/ccache front
        # end, but caller-provided ccache/prefix/compiler overrides are outside
        # this closed build profile. Fresh Core/fixtures use no launcher.
        for key, value in env.items():
            if key.startswith("CCACHE_") and value.strip():
                raise refuse("unsupported compiler-cache environment override: " + key)
        env["OMP_NUM_THREADS"] = "1"
        env["OMP_THREAD_LIMIT"] = "1"
        for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            env[key] = ""
        if args.mode == "asan":
            env.setdefault("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=1")
            env.setdefault("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
        elif args.mode == "tsan":
            env.setdefault("TSAN_OPTIONS", "halt_on_error=1:second_deadlock_stack=1")

        def command(argv, clean_radiance=False):
            print("RUN", json.dumps([str(x) for x in argv]), flush=True)
            # Engine and the actual TP4 Session checker require a clean
            # RADIANCE_* namespace. Keep other checker/build environments unchanged.
            run_env = ({k: v for k, v in env.items() if not k.startswith("RADIANCE_")}
                       if clean_radiance else env)
            subprocess.run([str(x) for x in argv], cwd=build_root, env=run_env, check=True)

        build_type = "Release" if args.mode == "release" else "Debug"
        if args.existing_core_build is not None:
            if not args.existing_core_build.is_absolute():
                raise refuse("--existing-core-build must be absolute")
            core_build = args.existing_core_build.resolve(strict=True)
            cache = verify["verify_core_build"](root, core_build, args.mode, files)
            print("REUSE verified source-identical HOST Core:", core_build, flush=True)
        else:
            core_build = build_root / ("core-main-verified-v8-host-" + args.mode)
            if core_build.is_symlink():
                raise refuse("fresh Core build directory must not be a symlink")
            # Source and build directories are distinct. The default reuses only
            # this snapshot's prior verified out-of-tree build, never production.
            if (core_build / "CMakeCache.txt").exists():
                verify["verify_core_build"](root, core_build, args.mode, files, require_archives=False)
            command(["cmake", "-S", root / "radiance", "-B", core_build,
                     "-DCMAKE_BUILD_TYPE=" + build_type, "-DRAD_WITH_HIP=OFF", "-DRAD_WITH_FFMPEG=OFF",
                     "-DRAD_BUILD_TESTS=ON", "-DRAD_BUILD_DS41_EXPERIMENTAL=ON",
                     "-DRAD_BUILD_GLM_TP4_CACHE_EXPERIMENTAL=OFF",
                     "-DRAD_ASAN=" + ("ON" if args.mode == "asan" else "OFF"),
                     "-DRAD_TSAN=" + ("ON" if args.mode == "tsan" else "OFF")])
            command(["cmake", "--build", core_build, "--parallel", args.jobs, "--target",
                     "rad_core", "libref", "libquant"])
            cache = verify["verify_core_build"](root, core_build, args.mode, files)

        fixture_build = build_root / ("portable-main-verified-v8-fixture-v10-host-" + args.mode)
        if fixture_build.is_symlink():
            raise refuse("fixture build directory must not be a symlink")
        if (fixture_build / "CMakeCache.txt").exists():
            verify["verify_fixture_build"](root, fixture_build, args.mode, core_build, cache)
        command(["cmake", "-S", root / "research/portable", "-B", fixture_build,
                 "-DCMAKE_BUILD_TYPE=" + build_type,
                 "-DCMAKE_CXX_COMPILER=" + cache["CMAKE_CXX_COMPILER"],
                 "-DRADIANCE_SOURCE=" + str(root / "radiance"),
                 "-DRADIANCE_HOST_BUILD=" + str(core_build), "-DPORTABLE_MODE=" + args.mode])
        verify["verify_fixture_build"](root, fixture_build, args.mode, core_build, cache)
        command(["cmake", "--build", fixture_build, "--parallel", args.jobs])
        # Refuse changed source files before loading any research kernel/module.
        verify["verify_snapshot"](root)
        verify["verify_core_build"](root, core_build, args.mode, files)
        verify["verify_fixture_build"](root, fixture_build, args.mode, core_build, cache)
        if not args.run_host_checks:
            print("BUILT fourteen HOST checker programs; none executed. Add --run-host-checks for explicit manual execution.")
            return

        home = core_build / "radiance_home"
        for required in (home / "kernels/libref.so", home / "quantizers/libquant.so"):
            if not required.is_file() or required.is_symlink():
                raise refuse("missing regular freshly built HOST plugin: " + str(required))
        architecture = fixture_build / "research-architectures/libglm53_tiny_startup_arch.so"
        if not architecture.is_file() or architecture.is_symlink():
            raise refuse("missing freshly built research architecture module")
        executions = [
            [fixture_build / "glm53_tp4_cache_session_check", "--run-host-protocol"],
            [fixture_build / "glm53_tp4_mover_fence_check", "--radiance-home", home],
            [fixture_build / "tiny_native_foundation_check"],
            [fixture_build / "tiny_startup_metadata_check", "--run-host", "--radiance-home", home,
             "--architecture", architecture],
            [fixture_build / "compact_authority_test"],
            [fixture_build / "compact_authority_collision_test"],
            [fixture_build / "compact_native_inventory_test", "--radiance-home", home],
            [fixture_build / "glm53_native_source_strict_policy_test", "--radiance-home", home],
            [fixture_build / "ds41_unified_native_policy_host_test", "--radiance-home", home],
            [fixture_build / "tiny_engine_native_transport_check", "--run-host", "--radiance-home", home,
             "--architecture", architecture],
            [fixture_build / "glm53_main_verified_engine_gate", "--run-source-gates"],
            [fixture_build / "glm53_main_verified_component_quotes", "--run-host-quotes"],
            [fixture_build / "glm53_main_vocabulary_check", "--run-host-vocab", home],
            [fixture_build / "glm53_main_bounded_decode_check", "--run-host-decode"],
        ]
        clean_checkers = {fixture_build / "tiny_engine_native_transport_check",
                          fixture_build / "glm53_tp4_cache_session_check",
                          fixture_build / "glm53_main_verified_engine_gate",
                          fixture_build / "glm53_main_verified_component_quotes",
                          fixture_build / "glm53_main_vocabulary_check",
                          fixture_build / "glm53_main_bounded_decode_check"}
        for argv in executions:
            command(argv, clean_radiance=argv[0] in clean_checkers)
        print("PASS fourteen explicitly invoked HOST research checker programs, including bounded decode, tiny Engine, actual TP4 protocol/HostOnly upload fence and verified-main cold/component gates. These HOST checks grant no real-source capability and execute no GPU/model/serving path.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("REFUSED/FAILED:", error, file=sys.stderr)
        sys.exit(1)
