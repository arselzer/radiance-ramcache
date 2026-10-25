"""Parent-run CPU tests only. No real model, converter, network or GPU."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import overlay as o

def safetensors(tensors):
    header, payload = {}, bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    raw = o.encoded(header).rstrip(b"\n")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + payload

def profile(root, rows=(3, 5, 7)):
    files = []
    for path in sorted(root.iterdir()):
        data = path.read_bytes()
        files.append((path.name, len(data), "sha256", o.digest(data)))
    index = json.loads((root / o.INDEX).read_bytes())
    return o.Profile("SYNTHETIC CPU TEST ONLY", "synthetic-v1", tuple(files),
                     o.digest((root / o.INDEX).read_bytes()), o.digest((root / "config.json").read_bytes()),
                     len(index["weight_map"]), rows)

def fixture(root):
    root.mkdir()
    owners = {}
    total = 0
    for i, count in enumerate((3, 5, 7)):
        name = o.PREFIX + f"shard_{i}.weight"
        data = bytes((j * 37 + i) % 127 for j in range(count * 160))
        tensors = [(name, "F8_E4M3", [count, 160], data)]
        total += len(data)
        filename = f"tiny-{i}.safetensors"
        owners[name] = filename
        if i == 2:
            tensors.append((o.PREFIX + "weight_scale", "BF16", [1], o.SCALE))
            owners[o.PREFIX + "weight_scale"] = filename
            total += 2
        (root / filename).write_bytes(safetensors(tensors))
    (root / o.INDEX).write_bytes(o.encoded({"metadata": {"total_size": total}, "weight_map": owners}))
    (root / "config.json").write_bytes(b'{"scope":"synthetic test"}\n')
    return profile(root)

class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="qfn-overlay-unit-")
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.p = fixture(self.source)
        self.target = self.root / "overlay"

    def tearDown(self):
        self.directory.cleanup()

    def no_temporary(self):
        self.assertEqual(list(self.root.glob(".qfn8bit-overlay-*")), [])

    def refused(self, profile=None):
        with self.assertRaises((o.Refusal, OSError, ValueError)):
            o.build(self.source, self.target, profile or self.p)
        self.assertFalse(self.target.exists())
        self.no_temporary()

    def test_positive_exact_index_aliases_links_and_immutability(self):
        before = {x.name: x.read_bytes() for x in self.source.iterdir()}
        receipt = o.build(self.source, self.target, self.p)
        self.assertFalse(receipt["official"])
        self.assertEqual(receipt["scalar_payload_bytes_read"], 2)
        self.assertEqual(receipt["scale_alias_payload_bytes"], 6)
        result = o.verify(self.target, self.p)
        self.assertTrue(result["complete_headers_verified"])
        self.assertFalse(result["payload_hashes_verified"])
        self.assertEqual(before, {x.name: x.read_bytes() for x in self.source.iterdir()})
        for filename in before:
            if filename != o.INDEX:
                self.assertTrue((self.target / filename).is_symlink())
        data = (self.target / o.SIDECAR).read_bytes()
        length = struct.unpack("<Q", data[:8])[0]
        parsed = o.tensor_header(data[8:8+length], 6)
        self.assertEqual(set(parsed), set(o.alias_names(self.p)))
        self.assertEqual(data[8+length:], o.SCALE * 3)

    def test_missing_truncated_index_and_complete_header_refusals(self):
        path = self.source / "tiny-0.safetensors"
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        self.refused()
        path.write_bytes(original)
        index_path = self.source / o.INDEX
        index = json.loads(index_path.read_bytes())
        index["weight_map"][o.PREFIX + "shard_0.weight"] = "absent.safetensors"
        index_path.write_bytes(o.encoded(index))
        self.refused(profile(self.source))

    def test_overlap_wrong_dtype_shape_and_scalar_refusals(self):
        path = self.source / "tiny-2.safetensors"
        original = path.read_bytes()
        length = struct.unpack("<Q", original[:8])[0]
        header = json.loads(original[8:8+length])
        parent = o.PREFIX + "weight_scale"
        for change in ("overlap", "dtype", "shape", "scalar"):
            h = json.loads(json.dumps(header))
            payload = bytearray(original[8+length:])
            if change == "overlap":
                h[parent]["data_offsets"] = [0, 2]
            elif change == "dtype":
                h[parent]["dtype"], h[parent]["shape"] = "F8_E4M3", [2]
            elif change == "shape":
                h[parent]["shape"] = [1, 1]
            else:
                payload[-2:] = b"\x52\x39"
            raw = o.encoded(h).rstrip(b"\n")
            raw += b" " * (-len(raw) % 8)
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
            self.refused(profile(self.source))
        path.write_bytes(original)

    def test_adjacent_alias_and_scale_inv_precedence_collisions(self):
        index_path = self.source / o.INDEX
        original = index_path.read_bytes()
        for suffix in ("_scale", "_scale_inv"):
            name = o.PREFIX + "shard_0.weight" + suffix
            filename = "collision.safetensors"
            (self.source / filename).write_bytes(safetensors([(name, "BF16", [1], o.SCALE)]))
            index = json.loads(original)
            index["weight_map"][name] = filename
            index["metadata"]["total_size"] += 2
            index_path.write_bytes(o.encoded(index))
            self.refused(profile(self.source))
            (self.source / filename).unlink()
        index_path.write_bytes(original)

    def test_caps_duplicate_keys_and_index_header_ownership(self):
        with mock.patch.object(o, "MAX_HEADER", 8):
            self.refused()
        with mock.patch.object(o, "MAX_METADATA", 8):
            self.refused()
        path = self.source / o.INDEX
        old = path.read_bytes()
        path.write_bytes(b'{"weight_map":{},"weight_map":{}}')
        self.refused(dataclasses.replace(profile(self.source), n_tensors=4))
        path.write_bytes(old)
        index = json.loads(old)
        key = o.PREFIX + "shard_0.weight"
        index["weight_map"][key] = "tiny-1.safetensors"
        # Preserve all three file names in the index to reach owner validation.
        index["weight_map"][o.PREFIX + "shard_1.weight"] = "tiny-0.safetensors"
        path.write_bytes(o.encoded(index))
        self.refused(profile(self.source))

    def test_noreplace_existing_and_late_collision_cleanup(self):
        self.target.mkdir()
        (self.target / "other-owner").write_bytes(b"untouched")
        with self.assertRaises(o.Refusal):
            o.build(self.source, self.target, self.p)
        self.assertEqual((self.target / "other-owner").read_bytes(), b"untouched")
        self.no_temporary()
        (self.target / "other-owner").unlink()
        self.target.rmdir()
        def collision(_, target):
            target.mkdir()
            (target / "other-owner").write_bytes(b"late untouched")
            raise FileExistsError("simulated late renameat2 EEXIST")
        with mock.patch.object(o, "noreplace", collision), self.assertRaises(FileExistsError):
            o.build(self.source, self.target, self.p)
        self.assertEqual((self.target / "other-owner").read_bytes(), b"late untouched")
        self.no_temporary()

    def test_source_mutation_after_output_writes_no_publication(self):
        real = o.write_new
        def changed(path, data):
            real(path, data)
            if path.name == o.RECEIPT:
                (self.source / "config.json").write_bytes(b"changed")
        with mock.patch.object(o, "write_new", changed):
            self.refused()

    def test_verify_modified_sidecar_links_and_no_full_payload_hash_claim(self):
        o.build(self.source, self.target, self.p)
        path = self.target / o.SIDECAR
        raw = bytearray(path.read_bytes())
        raw[-2] ^= 1
        path.write_bytes(raw)
        with self.assertRaises(o.Refusal):
            o.verify(self.target, self.p)
        path.write_bytes(o.sidecar(self.p))
        self.assertFalse(o.verify(self.target, self.p)["payload_hashes_verified"])
        # An unselected source code payload changed BEFORE admission cannot be
        # authenticated by headers. The explicit false flag is essential: this
        # tool does not compute the expected full LFS payload hashes.
        new_source = self.root / "second-source"
        p = fixture(new_source)
        f = new_source / "tiny-0.safetensors"
        data = bytearray(f.read_bytes())
        data[-1] ^= 1
        f.write_bytes(data)
        inspected = o.inspect_source(new_source, p)
        self.assertFalse(inspected["tensor_payload_hashes_verified"])
        self.assertEqual(inspected["scalar_payload_bytes_read"], 2)

if __name__ == "__main__":
    unittest.main()
