#!/usr/bin/env python3
"""Pinned QFN-only local metadata overlay. No conversion, network or inference."""
from __future__ import annotations
import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import tempfile
from dataclasses import dataclass

REPO = "Qwen/Qwen3.8-Flash-Next-FP8"
REVISION = "236dfdf285828023ca3bcd3f37366c58a3469b13"
INFO_SHA = "47f5198bad950052f4e8e9c8af2dcac17d1dcb5fedc64831df1037bc695de593"
INDEX_SHA = "0419e2c2dfbb925257d7409405433a793cf7ff7d96f3eba882a815ec6d9fe7a6"
CONFIG_SHA = "c22eb0a053eed62e18f0184d3ca62d3798f208c3e00b9feee939fc3dbacdc8ca"
PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
INDEX = "model.safetensors.index.json"
SIDECAR = "qfn8bit-ple-scale-aliases.safetensors"
RECEIPT = "qfn8bit-overlay-receipt.json"
RECIPE = "qfn8bit-ple-global.recipe"
RULE = ("blk.*.ple_ngram.weight rtn codes=fp8_e4m3 block=*x* scale=bf16 "
        "rule=fixed scale_value=0.00019931793212890625\n").encode()
SCALE = b"\x51\x39"
MAX_INDEX = 20 << 20
MAX_HEADER = 1 << 20
MAX_METADATA = 64 << 20
MAX_RECEIPT = 1 << 20
DTYPES = {"BF16": 2, "F8_E4M3": 1, "I64": 8}

class Refusal(ValueError):
    pass

def require(condition, why):
    if not condition:
        raise Refusal(why)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

def pairs(items):
    out = {}
    for key, value in items:
        require(key not in out, "duplicate JSON key")
        out[key] = value
    return out

def decoded(data):
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(Refusal("nonfinite JSON")))

def integer(value, lo=0, hi=(1 << 63) - 1):
    require(type(value) is int and lo <= value <= hi, "bounded integer required")
    return value

def leaf(value):
    require(type(value) is str and len(value) <= 128 and
            re.fullmatch(r"[A-Za-z0-9_.-]+", value) and value not in {".", ".."},
            "simple bounded local filename required")
    return value

def stamp(st):
    require(stat.S_ISREG(st.st_mode) and st.st_size >= 0, "regular file required")
    return [st.st_dev, st.st_ino, st.st_size, st.st_mode,
            st.st_mtime_ns, st.st_ctime_ns]

def capture(path):
    """Resolve legitimate HF snapshot symlinks, pin exact target and link identity."""
    link = os.lstat(path)
    require(stat.S_ISREG(link.st_mode) or stat.S_ISLNK(link.st_mode), "regular source or snapshot link")
    target = path.resolve(strict=True)
    require(len(str(path)) <= 4096 and len(str(target)) <= 4096, "source/link target path cap")
    fd = os.open(target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        target_stamp = stamp(os.fstat(fd))
    finally:
        os.close(fd)
    return {"name": path.name, "path": str(path), "target": str(target),
            "link": [link.st_dev, link.st_ino, link.st_size, link.st_mode,
                     link.st_mtime_ns, link.st_ctime_ns], "stamp": target_stamp}

def audit(record):
    require(capture(Path(record["path"])) == record, "source file/link generation changed")

def exact(fd, count, offset):
    require(0 <= count <= MAX_INDEX and 0 <= offset <= (1 << 63) - count, "read extent cap")
    out = bytearray()
    while len(out) < count:
        try:
            block = os.pread(fd, min(1 << 20, count - len(out)), offset + len(out))
        except InterruptedError:
            continue
        require(block, "short source read")
        out.extend(block)
    return bytes(out)

def bounded_file(record, cap):
    audit(record)
    require(record["stamp"][2] <= cap, "metadata byte cap")
    fd = os.open(record["target"], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        require(stamp(os.fstat(fd)) == record["stamp"], "opened source generation mismatch")
        data = exact(fd, record["stamp"][2], 0)
        require(stamp(os.fstat(fd)) == record["stamp"], "source changed during metadata read")
    finally:
        os.close(fd)
    audit(record)
    return data

@dataclass(frozen=True)
class Profile:
    label: str
    revision: str
    files: tuple
    index_sha: str
    config_sha: str
    n_tensors: int
    ng_rows: tuple
    ng_cols: int = 160
    official: bool = False

def official_profile():
    raw = bounded_file(capture(Path(__file__).with_name("PINNED_UPSTREAM_MODEL_INFO.json").resolve(strict=True)), 131072)
    require(len(raw) <= 131072 and digest(raw) == INFO_SHA, "pinned upstream metadata identity")
    info = decoded(raw)
    require(info["id"] == REPO and info["sha"] == REVISION, "pinned repo/revision")
    files = []
    for item in info["siblings"]:
        name = leaf(item["rfilename"])
        size = integer(item["size"], 1)
        lfs = item.get("lfs")
        if lfs:
            require(lfs["size"] == size and re.fullmatch(r"[0-9a-f]{64}", lfs["sha256"]), "LFS metadata")
            files.append((name, size, "sha256", lfs["sha256"]))
        else:
            require(re.fullmatch(r"[0-9a-f]{40}", item["blobId"]), "Git blob identity")
            files.append((name, size, "git-sha1", item["blobId"]))
    require(len(files) == 144, "pinned artifact inventory")
    return Profile(REPO, REVISION, tuple(files), INDEX_SHA, CONFIG_SHA,
                   152089, (2500012,) * 128, official=True)

def artifact_digest(data, algorithm):
    if algorithm == "sha256":
        return digest(data)
    require(algorithm == "git-sha1", "known metadata hash scheme")
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def alias_names(profile):
    return [PREFIX + f"shard_{i}.weight_scale" for i in range(len(profile.ng_rows))]

def tensor_header(data, payload_bytes):
    header = decoded(data)
    require(type(header) is dict and len(header) <= 32768, "bounded safetensors object")
    intervals = []
    tensors = {}
    for name, spec in header.items():
        if name == "__metadata__":
            require(type(spec) is dict and len(spec) <= 64 and all(type(k) is str and type(v) is str and len(k) <= 512 and len(v) <= 4096 for k, v in spec.items()), "bounded safetensors metadata")
            continue
        require(type(name) is str and 0 < len(name) <= 512 and "\0" not in name, "tensor name cap")
        require(type(spec) is dict and set(spec) == {"dtype", "shape", "data_offsets"}, "exact tensor header fields")
        dtype = spec["dtype"]
        require(dtype in DTYPES and type(spec["shape"]) is list and len(spec["shape"]) <= 8, "supported source header dtype/rank")
        n = 1
        for extent in spec["shape"]:
            extent = integer(extent, 1, 1 << 40)
            require(n <= ((1 << 63) - 1) // extent, "shape product overflow")
            n *= extent
        offsets = spec["data_offsets"]
        require(type(offsets) is list and len(offsets) == 2, "offset pair")
        lo, hi = (integer(x) for x in offsets)
        require(lo <= hi <= payload_bytes and n * DTYPES[dtype] == hi - lo, "tensor extent/shape/size")
        intervals.append((lo, hi))
        tensors[name] = spec
    cursor = 0
    for lo, hi in sorted(intervals):
        require(lo == cursor, "safetensors overlap or uncovered payload gap")
        cursor = hi
    require(cursor == payload_bytes, "safetensors trailing payload gap")
    return tensors

def inspect_source(source, profile):
    source = Path(source)
    require(source.is_absolute() and len(str(source)) <= 4096, "absolute bounded source path")
    source = source.resolve(strict=True)
    require(source.is_dir(), "source directory")
    declared = {leaf(name): (integer(size, 1), algorithm, expected) for name, size, algorithm, expected in profile.files}
    require(len(declared) == len(profile.files) and INDEX in declared and "config.json" in declared, "profile artifact uniqueness")
    records = {}
    index_data = None
    metadata_bytes = 0
    for name, (size, algorithm, expected) in sorted(declared.items()):
        record = capture(source / name)
        require(record["stamp"][2] == size, "complete source file extent: " + name)
        record["expected_payload_hash"] = expected
        record["hash_algorithm"] = algorithm
        records[name] = record
        if not name.endswith(".safetensors"):
            require(size <= MAX_INDEX and metadata_bytes <= MAX_METADATA - size, "aggregate metadata budget")
            data = bounded_file({k: v for k, v in record.items() if k not in {"expected_payload_hash", "hash_algorithm"}}, MAX_INDEX)
            require(artifact_digest(data, algorithm) == expected, "pinned metadata hash: " + name)
            metadata_bytes += size
            if name == INDEX:
                index_data = data
            elif name == "config.json":
                require(digest(data) == profile.config_sha, "pinned config hash")
    require(index_data is not None and digest(index_data) == profile.index_sha, "pinned full index hash")
    index = decoded(index_data)
    require(type(index) is dict and type(index.get("weight_map")) is dict and
            type(index.get("metadata")) is dict, "source index weight_map/metadata")
    owners = index["weight_map"]
    require(len(owners) == profile.n_tensors, "complete pinned tensor count")
    shards = set()
    for name, filename in owners.items():
        require(type(name) is str and 0 < len(name) <= 512 and "\0" not in name, "index tensor name")
        filename = leaf(filename)
        require(filename in declared and filename.endswith(".safetensors"), "known source shard owner")
        shards.add(filename)
    require(shards == {n for n in declared if n.endswith(".safetensors")}, "complete file/index shard coverage")
    if profile.official:
        require(shards == {f"model-{i:05d}-of-00131.safetensors" for i in range(1, 132)}, "all131 pinned shards")
    for name in alias_names(profile):
        require(name not in owners and name[:-6] + "_scale_inv" not in owners, "alias or higher-precedence _scale_inv collision")
    require(PREFIX + "weight_scale_inv" not in owners and SIDECAR not in declared and RECEIPT not in declared and RECIPE not in declared, "reserved overlay collision")
    ng = {}
    headers = []
    seen = payload_total = expert_headers = 0
    for filename in sorted(shards):
        record = records[filename]
        base_record = {k: v for k, v in record.items() if k not in {"expected_payload_hash", "hash_algorithm"}}
        audit(base_record)
        fd = os.open(record["target"], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            require(stamp(os.fstat(fd)) == record["stamp"], "opened shard identity")
            length = struct.unpack("<Q", exact(fd, 8, 0))[0]
            require(0 < length <= MAX_HEADER and length <= record["stamp"][2] - 8 and metadata_bytes <= MAX_METADATA - length - 8, "header/aggregate byte budget")
            raw = exact(fd, length, 8)
            metadata_bytes += length + 8
            payload = record["stamp"][2] - 8 - length
            tensors = tensor_header(raw, payload)
            for name, spec in tensors.items():
                require(owners.get(name) == filename, "index/header owner mismatch: " + name)
                seen += 1
                if name.startswith(PREFIX):
                    require(name not in ng, "duplicate ngram header")
                    ng[name] = spec
                    if name == PREFIX + "weight_scale":
                        require(spec["dtype"] == "BF16" and spec["shape"] == [1], "authoritative parent scalar dtype/shape")
                        require(exact(fd, 2, 8 + length + spec["data_offsets"][0]) == SCALE, "authoritative parent scalar must be0x3951")
                if profile.official and ".mlp.experts." in name:
                    match = re.fullmatch(r"(?:model\.language_model\.layers\.([0-9]+)|mtp\.layers\.([0-9]+))\.mlp\.experts\.([0-9]+)\.(gate|up|down)_proj\.weight(_scale_inv)?", name)
                    require(match is not None, "unexpected expert source spelling")
                    main, mtp, expert, projection, scale = match.groups()
                    require((main is not None and 0 <= int(main) < 48) or mtp == "0", "expert layer domain")
                    require(0 <= int(expert) < 512, "expert ID domain")
                    shape = [2560, 640] if projection == "down" else [640, 2560]
                    want = [x // 128 for x in shape] if scale else shape
                    require(spec["dtype"] == ("BF16" if scale else "F8_E4M3") and spec["shape"] == want, "complete native FP8 expert code/scale header")
                    expert_headers += 1
            payload_total += payload
            headers.append({"name": filename, "header_sha256": digest(raw), "header_bytes": length,
                            "tensor_count": len(tensors), "payload_bytes": payload})
            require(stamp(os.fstat(fd)) == record["stamp"], "shard changed during header/scalar read")
        finally:
            os.close(fd)
        audit(base_record)
    require(seen == profile.n_tensors and integer(index.get("metadata", {}).get("total_size")) == payload_total, "all source tensors and payload extents accounted")
    if profile.official:
        require(expert_headers == 49 * 512 * 6, "complete main+MTP expert header inventory")
    wanted = {PREFIX + f"shard_{i}.weight" for i in range(len(profile.ng_rows))} | {PREFIX + "weight_scale"}
    require(set(ng) == wanted, "exact PLE shard/parent name set")
    for i, count in enumerate(profile.ng_rows):
        spec = ng[PREFIX + f"shard_{i}.weight"]
        require(spec["dtype"] == "F8_E4M3" and spec["shape"] == [count, profile.ng_cols], "PLE complete shard geometry")
    for record in records.values():
        audit({k: v for k, v in record.items() if k not in {"expected_payload_hash", "hash_algorithm"}})
    return {"source": str(source), "profile": profile.label, "revision": profile.revision,
            "official": profile.official, "records": records, "index": index,
            "headers": headers, "source_index_sha256": digest(index_data),
            "metadata_bytes_read": metadata_bytes, "scalar_payload_bytes_read": 2,
            "expert_headers": expert_headers, "tensor_payload_hashes_verified": False}

def new_index(info, profile):
    index = dict(info["index"])
    index["weight_map"] = dict(index["weight_map"])
    index["metadata"] = dict(index.get("metadata", {}))
    for name in alias_names(profile):
        require(name not in index["weight_map"], "alias collision")
        index["weight_map"][name] = SIDECAR
    index["metadata"]["total_size"] += 2 * len(profile.ng_rows)
    return encoded(index)

def sidecar(profile):
    header = {name: {"dtype": "BF16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
              for i, name in enumerate(alias_names(profile))}
    raw = encoded(header).rstrip(b"\n")
    raw += b" " * (-len(raw) % 8)
    require(len(raw) <= MAX_HEADER, "alias header cap")
    return struct.pack("<Q", len(raw)) + raw + SCALE * len(profile.ng_rows)

def write_new(path, data):
    require(len(data) <= MAX_INDEX, "output metadata cap")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(data)
        while view:
            try:
                n = os.write(fd, view)
            except InterruptedError:
                continue
            require(n > 0, "short overlay write")
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)

def noreplace(source, target):
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    require(function is not None, "Linux renameat2 NOREPLACE required")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))

def build(source, destination, profile):
    destination = Path(destination)
    require(destination.is_absolute() and len(str(destination)) <= 4096 and not os.path.lexists(destination), "NEW bounded absolute output directory")
    parent = destination.parent.resolve(strict=True)
    require(parent.is_dir(), "existing output parent")
    destination = parent / destination.name
    require(len(str(destination)) <= 4096, "canonical destination path cap")
    info = inspect_source(source, profile)
    require(Path(info["source"]) != destination and Path(info["source"]) not in destination.parents, "output outside source checkpoint")
    index_data, alias_data = new_index(info, profile), sidecar(profile)
    summary = {k: v for k, v in info.items() if k != "index"}
    summary.update({"schema": "qfn8bit-global-PLE-overlay-v1", "aliases": len(profile.ng_rows),
                    "scale_bf16_hex": "3951", "scale_alias_payload_bytes": 2 * len(profile.ng_rows),
                    "derived_overlay_not_original_checkpoint": True,
                    "ordinary_mutation_detection_not_hostile_writer_security": True,
                    "full_payload_hashes_expected_but_NOT_verified": True,
                    "input_metadata_files_sha_verified": True,
                    "source_shard_payload_read_bytes": 2,
                    "resource_scope": "64MiB aggregate input metadata byte cap; Python object/allocator RSS not enforced; one source FD at a time, no expert payload or conversion",
                    "owned_sha256": {INDEX: digest(index_data), SIDECAR: digest(alias_data), RECIPE: digest(RULE)}})
    receipt = encoded(summary)
    require(len(receipt) <= MAX_RECEIPT, "receipt cap")
    temporary = Path(tempfile.mkdtemp(prefix=".qfn8bit-overlay-", dir=parent))
    published = False
    try:
        for name, record in info["records"].items():
            if name != INDEX:
                os.symlink(record["target"], temporary / name)
        write_new(temporary / INDEX, index_data)
        write_new(temporary / SIDECAR, alias_data)
        write_new(temporary / RECIPE, RULE)
        write_new(temporary / RECEIPT, receipt)
        for record in info["records"].values():
            audit({k: v for k, v in record.items() if k not in {"expected_payload_hash", "hash_algorithm"}})
        fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        noreplace(temporary, destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary)
    return summary

def verify(overlay, profile):
    overlay = Path(overlay)
    require(overlay.is_absolute() and not overlay.is_symlink(), "owned absolute overlay directory")
    receipt_record = capture(overlay / RECEIPT)
    receipt = decoded(bounded_file(receipt_record, MAX_RECEIPT))
    require(receipt.get("schema") == "qfn8bit-global-PLE-overlay-v1" and receipt.get("profile") == profile.label and
            receipt.get("revision") == profile.revision and receipt.get("official") == profile.official, "overlay profile/provenance")
    source = inspect_source(receipt["source"], profile)
    for key in ("records", "headers", "source_index_sha256"):
        require(source[key] == receipt[key], "source generation differs from sealed overlay")
    expected = {INDEX: new_index(source, profile), SIDECAR: sidecar(profile), RECIPE: RULE}
    for name, wanted in expected.items():
        require(not (overlay / name).is_symlink(), "owned overlay metadata must not be a link")
        raw = bounded_file(capture(overlay / name), MAX_INDEX)
        require(raw == wanted and receipt["owned_sha256"][name] == digest(wanted), "owned overlay bytes changed")
    names = {RECEIPT, *expected}
    for name, record in source["records"].items():
        if name == INDEX:
            continue
        names.add(name)
        require((overlay / name).is_symlink() and os.readlink(overlay / name) == record["target"], "overlay source link changed")
    require(set(os.listdir(overlay)) == names, "unexpected/missing overlay files (including scale precedence)")
    require(capture(overlay / RECEIPT) == receipt_record, "receipt changed during verification")
    return {"verified_overlay": str(overlay), "aliases": len(profile.ng_rows),
            "complete_headers_verified": True, "payload_hashes_verified": False,
            "conversion_executed": False, "official": profile.official}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--source", required=True)
    create.add_argument("--output", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--overlay", required=True)
    args = parser.parse_args()
    profile = official_profile()  # CLI has no synthetic-profile override.
    result = build(args.source, args.output, profile) if args.command == "build" else verify(args.overlay, profile)
    print(json.dumps({k: result[k] for k in result if k in {"profile", "revision", "official", "aliases", "metadata_bytes_read", "scalar_payload_bytes_read", "tensor_payload_hashes_verified", "verified_overlay", "complete_headers_verified", "payload_hashes_verified", "conversion_executed"}}, sort_keys=True))

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, MemoryError) as error:
        raise SystemExit("REFUSED: " + str(error))
