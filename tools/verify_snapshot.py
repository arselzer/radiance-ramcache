#!/usr/bin/env python3
"""Verify the ds41 source snapshot; no build, model, device or network use.

Every file in the tree must be listed in SNAPSHOT_MANIFEST.json with its SHA256, and nothing
else may be there. radiance/ must be exactly the engine commit the manifest names: the same
files, byte for byte, as docs/ds41/provenance/ENGINE_SOURCE_PINS.json, whose own SHA256 the
manifest records. With --engine-repo PATH the pins are also checked against that commit's git
tree (blob ids and executable bits); that needs a local copy of the engine repository.

Adapted from main's tools/verify_snapshot.py. Main's frozen Core12/Fixture17 identity and its
HOST build-profile checks belong to the GLM checkpoint and do not apply to this branch.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys


MANIFEST = "SNAPSHOT_MANIFEST.json"
ENGINE_DIR = "radiance"
ENGINE_PINS = "docs/ds41/provenance/ENGINE_SOURCE_PINS.json"
ENGINE_SOURCE = "docs/ds41/provenance/ENGINE_SOURCE.json"
HEX = frozenset("0123456789abcdef")


class Refusal(RuntimeError):
    pass


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def is_hex(value, length):
    return isinstance(value, str) and len(value) == length and all(c in HEX for c in value)


def canonical(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise Refusal("invalid path: " + repr(value))
    p = PurePosixPath(value)
    if p.is_absolute() or str(p) != value or any(x in ("", ".", "..") for x in p.parts):
        raise Refusal("noncanonical path: " + value)
    return p


def relative_name(value):
    p = canonical(value)
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


def strict_json(path, cap):
    if path.stat().st_size > cap:
        raise Refusal(str(path) + " exceeds its size cap")

    # Duplicate JSON members must not shadow a path/digest.
    def object_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise Refusal("duplicate JSON member in " + str(path) + ": " + key)
            obj[key] = value
        return obj

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=object_pairs)


def load_manifest(root):
    data = strict_json(regular_path(root, MANIFEST), 16 * 1024 * 1024)
    if not isinstance(data, dict):
        raise Refusal("manifest root must be an object")
    files = data.get("files_sha256")
    if not isinstance(files, dict) or not files or len(files) > 100000:
        raise Refusal("manifest files_sha256 must contain 1..100000 file hashes")
    for name, h in files.items():
        relative_name(name)
        if not is_hex(h, 64):
            raise Refusal("invalid SHA256 for " + name)
    return data, files


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
    data, files = load_manifest(root)
    found = source_files(root)
    missing = set(files) - found
    extra = found - set(files)
    if missing or extra:
        raise Refusal("snapshot file-set drift; missing=" + repr(sorted(missing)[:8])
                      + " extra=" + repr(sorted(extra)[:8]))
    for name, expected in sorted(files.items()):
        if digest(regular_path(root, name)) != expected:
            raise Refusal("source SHA256 mismatch: " + name)
    return data, files


def verify_engine(root, data, files):
    """radiance/ must be exactly the pinned engine tree the manifest names."""
    engine = data.get("engine")
    if not isinstance(engine, dict):
        raise Refusal("manifest has no engine record")
    commit, pins_sha256, count = engine.get("commit"), engine.get("pins_sha256"), engine.get("files")
    if not is_hex(commit, 40) or not is_hex(pins_sha256, 64) or not isinstance(count, int) or count < 1:
        raise Refusal("manifest engine record needs commit (40 hex), pins_sha256 and files")
    # verify_snapshot already held the pin file to the manifest's digest.
    if files.get(ENGINE_PINS) != pins_sha256:
        raise Refusal("engine pin file is not the one the manifest's engine record names")
    pins = strict_json(regular_path(root, ENGINE_PINS), 16 * 1024 * 1024)
    if not isinstance(pins, dict) or len(pins) != count:
        raise Refusal("engine pin file must map exactly " + str(count) + " files")
    for name, h in pins.items():
        canonical(name)
        if not is_hex(h, 64):
            raise Refusal("invalid engine pin for " + name)
    source = strict_json(regular_path(root, ENGINE_SOURCE), 1024 * 1024)
    if not isinstance(source, dict) or source.get("commit") != commit \
            or source.get("pins_sha256") != pins_sha256 or source.get("files") != count:
        raise Refusal("ENGINE_SOURCE.json disagrees with the manifest's engine record")
    prefix = ENGINE_DIR + "/"
    exported = {name[len(prefix):]: h for name, h in files.items() if name.startswith(prefix)}
    if exported != pins:
        missing = sorted(set(pins) - set(exported))
        extra = sorted(set(exported) - set(pins))
        changed = sorted(n for n in set(pins) & set(exported) if pins[n] != exported[n])
        raise Refusal("radiance/ differs from the pinned engine tree; missing=" + repr(missing[:8])
                      + " extra=" + repr(extra[:8]) + " changed=" + repr(changed[:8]))
    return commit, pins


def verify_engine_repo(root, repo, commit, pins):
    """Optional: the pins against the engine commit's own tree in a local git repository."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    try:
        git("cat-file", "-e", commit + "^{commit}")
        object_format = git("rev-parse", "--show-object-format").decode().strip() or "sha1"
        listing = git("ls-tree", "-r", "-z", "--full-tree", commit)
    except (OSError, subprocess.CalledProcessError) as e:
        raise Refusal("engine repository does not have commit " + commit) from e
    if object_format not in ("sha1", "sha256"):
        raise Refusal("unsupported git object format: " + object_format)
    entries = {}
    for record in listing.split(b"\0"):
        if not record:
            continue
        meta, path = record.split(b"\t", 1)
        mode, kind, blob = meta.split()
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise Refusal("engine tree entry is not a regular file: " + path.decode(errors="replace"))
        entries[path.decode()] = (mode == b"100755", blob.decode())
    if set(entries) != set(pins):
        raise Refusal("pins and the engine commit's tree name different files")
    base = root / ENGINE_DIR
    for name, (executable, blob) in sorted(entries.items()):
        path = regular_path(base, name)
        content = path.read_bytes()
        h = hashlib.new(object_format)
        h.update(b"blob %d\0" % len(content))
        h.update(content)
        if h.hexdigest() != blob:
            raise Refusal("radiance/" + name + " differs from the engine commit's blob")
        if bool(path.stat().st_mode & 0o111) != executable:
            raise Refusal("radiance/" + name + " executable bit differs from the engine commit")
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--engine-repo", type=Path,
                        help="local engine git repository holding the pinned commit (optional)")
    args = parser.parse_args()
    data, files = verify_snapshot(args.repo_root)
    commit, pins = verify_engine(args.repo_root.resolve(), data, files)
    note = ""
    if args.engine_repo is not None:
        checked = verify_engine_repo(args.repo_root.resolve(), args.engine_repo, commit, pins)
        note = "; " + str(checked) + " blobs and modes match the engine repository"
    print("PASS source-only snapshot SHA256 verification:", len(files), "files; radiance/ is engine",
          commit[:12], "(" + str(len(pins)) + " files" + note + "); no runtime/device execution")


if __name__ == "__main__":
    try:
        main()
    except (Refusal, OSError, ValueError) as error:
        print("REFUSED:", error, file=sys.stderr)
        sys.exit(1)
