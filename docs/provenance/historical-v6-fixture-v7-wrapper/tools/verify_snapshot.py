#!/usr/bin/env python3
"""Verify the portable source snapshot; no build, model, device or network use."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


MANIFEST = "SNAPSHOT_MANIFEST.json"
CORE_CACHE_EXCLUSIONS = frozenset({
    "tests/__pycache__/gen_deepseek41_reference.cpython-314.pyc",
    "tests/data/golden/__pycache__/deepseek41-engram-generate.cpython-314.pyc",
})


class Refusal(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def relative_name(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise Refusal("invalid manifest path")
    p = PurePosixPath(value)
    if p.is_absolute() or str(p) != value or any(x in ("", ".", "..") for x in p.parts):
        raise Refusal("noncanonical manifest path: " + value)
    if p.parts[0] in (".git", "build") or value == MANIFEST:
        raise Refusal("manifest contains excluded/generated path: " + value)
    return value


def regular_path(root, name):
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink():
            raise Refusal("symlink in source path: " + str(path))
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            raise Refusal("source is not a regular file: " + name)
    except OSError as e:
        raise Refusal("missing/unreadable source: " + name) from e
    return path


def load_manifest(root):
    path = regular_path(root, MANIFEST)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise Refusal("manifest exceeds 16 MiB cap")
    # Duplicate JSON members must not shadow a path/digest.
    def object_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise Refusal("duplicate manifest member: " + key)
            obj[key] = value
        return obj
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs)
    if not isinstance(data, dict):
        raise Refusal("manifest root must be an object")
    files = data.get("files_sha256")
    if not isinstance(files, dict) or not files or len(files) > 100000:
        raise Refusal("manifest files_sha256 must contain 1..100000 file hashes")
    for name, h in files.items():
        relative_name(name)
        if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise Refusal("invalid SHA256 for " + name)
    return files


def source_files(root):
    found = set()
    for base, dirs, names in os.walk(root, followlinks=False):
        directory = Path(base)
        if directory == root:
            dirs[:] = [d for d in dirs if d not in (".git", "build")]
        for name in dirs:
            if (directory / name).is_symlink():
                raise Refusal("unlisted source directory symlink: " + str(directory / name))
        for name in names:
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if relative in (MANIFEST, ".git"):
                continue
            regular_path(root, relative)
            found.add(relative)
    return found


def verify_snapshot(root):
    root = root.resolve(strict=True)
    files = load_manifest(root)
    found = source_files(root)
    missing = set(files) - found
    extra = found - set(files)
    if missing or extra:
        raise Refusal("snapshot file-set drift; missing=" + repr(sorted(missing)[:8])
                      + " extra=" + repr(sorted(extra)[:8]))
    for name, expected in sorted(files.items()):
        if digest(regular_path(root, name)) != expected:
            raise Refusal("source SHA256 mismatch: " + name)
    core_files = {name.removeprefix("radiance/"): h
                  for name, h in files.items() if name.startswith("radiance/")}
    archived_pins = root / "research/glm53-main-verified-source-engine-v7-prototype/CORE_SOURCE_PINS.json"
    original = json.loads(archived_pins.read_text(encoding="utf-8"))
    # V6 retains the V5 inventory (which omitted both generated caches before adding20 source
    # files:1517-2+20=1535). Do not subtract them a second time or rewrite history.
    if not isinstance(original, dict) or len(original) != 1535 or CORE_CACHE_EXCLUSIONS & set(original):
        raise Refusal("unexpected frozen1535-file verified-main V6 Core pin inventory")
    if core_files != original:
        raise Refusal("exported Core differs from the exact frozen1535-file V6 inventory")
    if any(name in core_files for name in CORE_CACHE_EXCLUSIONS):
        raise Refusal("portable source snapshot must exclude the two recorded Python caches")
    return files


def cache_values(path):
    if path.stat().st_size > 4 * 1024 * 1024:
        raise Refusal("CMakeCache exceeds 4 MiB cap")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        left, value = line.split("=", 1)
        key, _kind = left.split(":", 1)
        if key in values:
            raise Refusal("duplicate CMakeCache entry: " + key)
        values[key] = value
    return values


def verify_compiler_profile(cache, languages=("CXX", "C"), allow_exact_ccache=False):
    # Closed research profile: source identity alone cannot establish matching
    # private STL/ABI layouts when e.g. -D_GLIBCXX_DEBUG or -fpack-struct differs.
    # Sanitizer flags are supplied only by the reviewed RAD_ASAN/RAD_TSAN CMake
    # branches, not arbitrary cache flags. Do not inherit an unbounded flag set.
    profiles = {"": (), "_DEBUG": ("-g",), "_RELEASE": ("-O3", "-DNDEBUG"),
                "_MINSIZEREL": ("-Os", "-DNDEBUG"),
                "_RELWITHDEBINFO": ("-O2", "-g", "-DNDEBUG")}
    permitted = {language + suffix: flags
                 for language in ("CMAKE_" + name + "_FLAGS" for name in languages)
                 for suffix, flags in profiles.items()}
    for key, flags in permitted.items():
        if key not in cache or tuple(cache[key].split()) != flags:
            raise Refusal("unsupported compiler/ABI profile: " + key + "=" + repr(cache.get(key)))
    for key, value in cache.items():
        if key.endswith("-ADVANCED"):
            continue
        if key.startswith(("CMAKE_CXX_FLAGS", "CMAKE_C_FLAGS")) and key not in permitted:
            raise Refusal("unsupported extra compiler profile: " + key)
        if any(key.startswith("CMAKE_" + kind + "_LINKER_FLAGS")
               for kind in ("EXE", "MODULE", "SHARED", "STATIC")) and value.strip():
            raise Refusal("unsupported linker flags: " + key)
        if key.startswith("CMAKE_INTERPROCEDURAL_OPTIMIZATION") and value not in ("", "OFF", "FALSE", "0"):
            raise Refusal("unsupported interprocedural optimization profile")
    for key in ("CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_C_COMPILER_LAUNCHER"):
        value = cache.get(key, "")
        if not value:
            continue
        # Existing reviewed v4 HOST archives used this ONE system compiler-cache
        # front end. Admit no argument/list/alternate launcher, and do not pass
        # it on to the fixture build. Compiler/flags/private source checks still
        # apply. System tool trust is not prebuilt-binary authentication.
        launcher = Path("/usr/bin/ccache")
        if (not allow_exact_ccache or value != "/usr/bin/ccache"
                or launcher.is_symlink() or not launcher.is_file()
                or not os.access(launcher, os.X_OK)):
            raise Refusal("unsupported compiler/toolchain injection: " + key)
    for key in ("CMAKE_CXX_COMPILER_ARG1", "CMAKE_C_COMPILER_ARG1",
                "CMAKE_TOOLCHAIN_FILE", "CMAKE_SYSROOT", "CMAKE_CXX_COMPILER_TARGET",
                "CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN", "CMAKE_PROJECT_TOP_LEVEL_INCLUDES"):
        if cache.get(key, "").strip():
            raise Refusal("unsupported compiler/toolchain injection: " + key)


def verify_core_build(root, build, mode, files=None, require_archives=True):
    if not build.is_absolute():
        raise Refusal("existing Core build path must be absolute")
    build = build.resolve(strict=True)
    cache = cache_values(regular_path(build, "CMakeCache.txt"))
    expected_type = "Release" if mode == "release" else "Debug"
    expected = {"RAD_WITH_HIP": "OFF", "RAD_WITH_FFMPEG": "OFF",
                "RAD_ASAN": "ON" if mode == "asan" else "OFF",
                "RAD_TSAN": "ON" if mode == "tsan" else "OFF",
                "CMAKE_BUILD_TYPE": expected_type}
    for key, value in expected.items():
        if cache.get(key) != value:
            raise Refusal("Core build profile mismatch: " + key + " expected " + value
                          + " observed " + repr(cache.get(key)))
    verify_compiler_profile(cache, allow_exact_ccache=True)
    if cache.get("RAD_BUILD_DS41_EXPERIMENTAL") not in ("ON", "OFF"):
        raise Refusal("Core must explicitly record its experimental-DS41 build option")
    if cache.get("RAD_BUILD_GLM_TP4_CACHE_EXPERIMENTAL") not in ("ON", "OFF"):
        raise Refusal("Core must explicitly record its experimental GLM TP4 module build option")
    home_value = cache.get("CMAKE_HOME_DIRECTORY", "")
    if not home_value or not Path(home_value).is_absolute():
        raise Refusal("Core build has no absolute CMAKE_HOME_DIRECTORY")
    home = Path(home_value).resolve(strict=True)
    if files is None:
        files = verify_snapshot(root)
    wanted = {name.removeprefix("radiance/"): h for name, h in files.items()
              if name.startswith("radiance/")}
    actual = set()
    for base, dirs, names in os.walk(home, followlinks=False):
        directory = Path(base)
        for name in dirs:
            if (directory / name).is_symlink():
                raise Refusal("Core source-home directory symlink is unsupported")
        for name in names:
            relative = (directory / name).relative_to(home).as_posix()
            regular_path(home, relative)
            actual.add(relative)
    missing = set(wanted) - actual
    extra = actual - set(wanted) - CORE_CACHE_EXCLUSIONS
    if missing or extra:
        raise Refusal("Core source-home file-set mismatch; missing=" + repr(sorted(missing)[:8])
                      + " extra=" + repr(sorted(extra)[:8]))
    for name, expected_hash in sorted(wanted.items()):
        if digest(regular_path(home, name)) != expected_hash:
            raise Refusal("built Core source-home differs from snapshot: " + name)
    if require_archives:
        for name in ("core/librad_core.a", "vendor/librad_vendor_llama.a",
                     "vendor/librad_vendor_httplib.a"):
            regular_path(build, name)
    compiler = cache.get("CMAKE_CXX_COMPILER", "")
    if not compiler or not Path(compiler).is_absolute():
        raise Refusal("Core must record an absolute C++ compiler")
    return cache



def verify_fixture_build(root, build, mode, core_build, core_cache):
    cache = cache_values(regular_path(build, "CMakeCache.txt"))
    verify_compiler_profile(cache, languages=("CXX",))
    expected_type = "Release" if mode == "release" else "Debug"
    if cache.get("CMAKE_BUILD_TYPE") != expected_type or cache.get("PORTABLE_MODE") != mode:
        raise Refusal("portable fixture mode/build-type mismatch")
    for key, target in (("CMAKE_HOME_DIRECTORY", root / "research/portable"),
                        ("RADIANCE_SOURCE", root / "radiance"),
                        ("RADIANCE_HOST_BUILD", core_build)):
        value = cache.get(key, "")
        if not value or not Path(value).is_absolute() or Path(value).resolve(strict=True) != target.resolve(strict=True):
            raise Refusal("portable fixture source/Core path mismatch: " + key)
    if cache.get("CMAKE_CXX_COMPILER") != core_cache.get("CMAKE_CXX_COMPILER"):
        raise Refusal("portable fixture compiler differs from verified Core")
    return cache

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--existing-core-build", type=Path)
    parser.add_argument("--mode", choices=("release", "asan", "tsan"), default="release")
    args = parser.parse_args()
    files = verify_snapshot(args.repo_root)
    if args.existing_core_build is not None:
        verify_core_build(args.repo_root, args.existing_core_build, args.mode, files)
    print("PASS source-only snapshot SHA256 verification:", len(files), "files; no runtime/device execution")


if __name__ == "__main__":
    try:
        main()
    except (Refusal, OSError, ValueError) as error:
        print("REFUSED:", error, file=sys.stderr)
        sys.exit(1)
