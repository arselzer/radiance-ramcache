#!/usr/bin/env python3
"""Fresh bounded GLM file-hash/stamp proof; no packing, device or model execution."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

REPO = "amd/GLM-5.3-Flash-Quark-MXFP4"
REVISION = "b5688f25491202978c19c4d036eef579f61bbe07"
INFO_SHA = "efbd6ca6794f23f16113afb8e77f6dd68f8053f2b0c8f0fde4db5cb3be947596"
CHUNK = 65536
PROOF_CAP = 131072

class Refusal(ValueError):
    pass

def require(value, why):
    if not value:
        raise Refusal(why)

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

def stamp(value, directory=False):
    require((stat.S_ISDIR if directory else stat.S_ISREG)(value.st_mode), "regular source kind")
    return {"device": value.st_dev, "inode": value.st_ino, "bytes": value.st_size,
            "mtime": list(divmod(value.st_mtime_ns, 1_000_000_000)),
            "ctime": list(divmod(value.st_ctime_ns, 1_000_000_000))}

def profile():
    path = Path(__file__).with_name("PINNED_UPSTREAM_MODEL_INFO.json")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = stamp(os.fstat(fd))
        require(before["bytes"] <= CHUNK, "pinned model-info cap")
        data = os.read(fd, CHUNK + 1)
        require(len(data) == before["bytes"] and stamp(os.fstat(fd)) == before and
                hashlib.sha256(data).hexdigest() == INFO_SHA, "pinned model-info identity")
    finally:
        os.close(fd)
    value = json.loads(data)
    require(value["id"] == REPO and value["sha"] == REVISION and len(value["siblings"]) == 72,
            "exact upstream source domain")
    rows = []
    for item in value["siblings"]:
        name, size = item["rfilename"], item["size"]
        require(isinstance(name, str) and 0 < len(name) <= 160 and
                "/" not in name and name not in {".", ".."} and "\0" not in name and
                type(size) is int and size > 0, "bounded source leaf/size")
        lfs = item.get("lfs")
        rows.append((name, size, "sha256" if lfs else "git-sha1",
                     lfs["sha256"] if lfs else item["blobId"]))
    require(len({x[0] for x in rows}) == 72 and sum(x[1] for x in rows) == 185107699217,
            "exact upstream extent inventory")
    return tuple(sorted(rows))

def audit(root_fd, directory, rows):
    require(stamp(os.fstat(root_fd), True) == directory, "source directory changed")
    for row in rows:
        fd = os.open(row["name"], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            require(stamp(os.fstat(fd)) == row["stamp"], "source generation changed before proof commit")
        finally:
            os.close(fd)

def make_proof(source, mode, expected=None):
    """Tests may supply tiny expected rows; the public CLI never accepts them."""
    require(mode in {"complete", "subset"}, "explicit source verification mode")
    source = Path(source)
    require(source.is_absolute() and len(str(source)) <= 4096 and not source.is_symlink(),
            "absolute ordinary source directory")
    source = source.resolve(strict=True)
    synthetic = expected is not None
    expected = profile() if expected is None else tuple(expected)
    require(1 <= len(expected) <= 72, "source file cap")
    root = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        directory = stamp(os.fstat(root), True)
        rows, total = [], 0
        for name, size, algorithm, wanted in expected:
            require(0 < len(name) <= 160 and "/" not in name and name not in {".", ".."}, "source leaf")
            require(algorithm in {"sha256", "git-sha1"} and type(size) is int and size > 0,
                    "source identity")
            fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=root)
            try:
                before = stamp(os.fstat(fd))
                require(before["bytes"] == size, "complete source file extent")
                full = mode == "complete" or not name.endswith(".safetensors")
                observed, count = None, 0
                if full:
                    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
                    if algorithm == "git-sha1":
                        digest.update(f"blob {size}\0".encode())
                    while count < size:
                        block = os.read(fd, min(CHUNK, size - count))
                        require(block, "short complete hash read")
                        count += len(block)
                        digest.update(block)
                    require(not os.read(fd, 1), "source grew during hash")
                    observed = digest.hexdigest()
                    require(observed == wanted, "source file checksum mismatch: " + name)
                require(stamp(os.fstat(fd)) == before, "source changed during hash/stamp capture")
                rows.append({"name": name, "bytes": size, "algorithm": algorithm,
                             "expected_hash": wanted, "observed_hash": observed,
                             "hashed_bytes": count, "stamp": before})
                total += count
            finally:
                os.close(fd)
        audit(root, directory, rows)
        return {"schema": "glm53-source-file-hash-proof-v1", "repo": "synthetic_test" if synthetic else REPO,
                "revision": "fixture" if synthetic else REVISION, "mode": mode, "root": str(source),
                "root_stamp": directory, "upstream_info_sha256": "" if synthetic else INFO_SHA,
                "files": rows, "hashed_bytes": total, "window_bytes": CHUNK,
                "full_payload_hashes_verified": mode == "complete" and not synthetic,
                "authority": "trusted parent-run pinned stdlib proof; immutable-generation contract; not runtime Ready"}
    finally:
        os.close(root)

def noreplace(source, destination):
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    require(rename is not None, "Linux NOREPLACE required")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))

def publish(proof, destination):
    destination = Path(destination)
    require(destination.is_absolute() and len(str(destination)) <= 4096 and
            not os.path.lexists(destination), "NEW absolute proof path")
    require(destination.name not in {"", ".", ".."}, "new proof leaf")
    # Resolve parent aliases/dotdot BEFORE creating even a private temporary.
    # A lexical path guard could otherwise write into the original checkpoint.
    destination = destination.parent.resolve(strict=True) / destination.name
    source = Path(proof["root"]).resolve(strict=True)
    require(source != destination and source not in destination.parents, "proof outside original checkpoint")
    payload = encoded(proof)
    require(len(payload) <= PROOF_CAP, "proof metadata cap")
    fd, path = tempfile.mkstemp(prefix=".glm53-source-proof-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        root = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            audit(root, proof["root_stamp"], proof["files"])
        finally:
            os.close(root)
        noreplace(path, destination)
    finally:
        if os.path.lexists(path):
            os.unlink(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=("complete", "subset"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    proof = make_proof(args.source, args.mode)
    publish(proof, args.output)
    print(json.dumps({"mode": proof["mode"], "files": len(proof["files"]),
                      "hashed_bytes": proof["hashed_bytes"], "full_payload_hashes_verified": proof["full_payload_hashes_verified"],
                      "runtime_ready": False}, sort_keys=True))

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, MemoryError) as error:
        raise SystemExit("REFUSED: " + str(error))
