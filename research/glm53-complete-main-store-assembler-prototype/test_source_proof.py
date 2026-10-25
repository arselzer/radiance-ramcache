"""Parent-run tiny stdlib tests only; never touch the real model or execute Core."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("glm53_local_source_proof", HERE / "source_proof.py")
proof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proof)

class SourceProofTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.root = Path(self.work.name)
        self.source = self.root / "checkpoint"
        self.source.mkdir()
        self.values = {"config.json": b'{"model":"tiny"}', "model.safetensors": bytes(range(256)) * 3}
        for name, value in self.values.items():
            (self.source / name).write_bytes(value)
        self.expected = tuple(sorted((name, len(value), "sha256", hashlib.sha256(value).hexdigest())
                                     for name, value in self.values.items()))

    def tearDown(self):
        self.work.cleanup()

    def capture(self, mode="complete"):
        return proof.make_proof(self.source, mode, self.expected)

    def test_complete_and_input_immutable(self):
        before = {x.name: x.read_bytes() for x in self.source.iterdir()}
        result = self.capture()
        self.assertEqual(result["repo"], "synthetic_test")
        self.assertFalse(result["full_payload_hashes_verified"])
        self.assertEqual(result["hashed_bytes"], sum(map(len, self.values.values())))
        self.assertTrue(all(x["observed_hash"] == x["expected_hash"] for x in result["files"]))
        output = self.root / "proof.json"
        proof.publish(result, output)
        self.assertEqual(output.read_bytes(), proof.encoded(result))
        self.assertEqual(before, {x.name: x.read_bytes() for x in self.source.iterdir()})

    def test_subset_never_hashes_tensor_payload(self):
        result = self.capture("subset")
        rows = {x["name"]: x for x in result["files"]}
        self.assertIsNone(rows["model.safetensors"]["observed_hash"])
        self.assertEqual(rows["model.safetensors"]["hashed_bytes"], 0)
        self.assertEqual(result["hashed_bytes"], len(self.values["config.json"]))

    def test_complete_checksum_and_extent_refusal(self):
        (self.source / "model.safetensors").write_bytes(b"x" * len(self.values["model.safetensors"]))
        with self.assertRaises(proof.Refusal):
            self.capture()
        (self.source / "model.safetensors").write_bytes(b"short")
        with self.assertRaises(proof.Refusal):
            self.capture()

    def test_symlink_missing_and_invalid_mode(self):
        file = self.source / "model.safetensors"
        file.unlink()
        file.symlink_to(self.source / "config.json")
        with self.assertRaises(OSError):
            self.capture()
        file.unlink()
        with self.assertRaises(OSError):
            self.capture()
        with self.assertRaises(proof.Refusal):
            self.capture("implicit")

    def test_final_source_audit_before_publication(self):
        result = self.capture()
        (self.source / "config.json").write_bytes(b"changed")
        with self.assertRaises(proof.Refusal):
            proof.publish(result, self.root / "never.json")
        self.assertFalse((self.root / "never.json").exists())
        self.assertFalse(list(self.root.glob(".glm53-source-proof-*")))

    def test_noreplace_and_late_collision(self):
        result = self.capture()
        existing = self.root / "existing.json"
        existing.write_bytes(b"original")
        with self.assertRaises(proof.Refusal):
            proof.publish(result, existing)
        destination = self.root / "late.json"
        real = proof.noreplace
        def collide(source, target):
            Path(target).write_bytes(b"original-late")
            return real(source, target)
        with mock.patch.object(proof, "noreplace", collide):
            with self.assertRaises(OSError):
                proof.publish(result, destination)
        self.assertEqual(destination.read_bytes(), b"original-late")
        self.assertFalse(list(self.root.glob(".glm53-source-proof-*")))

    def test_output_alias_dotdot_refused_before_temp(self):
        result = self.capture()
        alias = self.root / "alias"
        alias.symlink_to(self.source, target_is_directory=True)
        other = self.root / "other"
        other.mkdir()
        before = set(self.source.iterdir())
        with mock.patch.object(proof.tempfile, "mkstemp", side_effect=AssertionError("must refuse before temporary")):
            for destination in (alias / "proof.json", other / ".." / "checkpoint" / "proof.json"):
                with self.assertRaises(proof.Refusal):
                    proof.publish(result, destination)
        self.assertEqual(before, set(self.source.iterdir()))

    def test_metadata_cap_and_git_blob_scheme(self):
        result = self.capture()
        with mock.patch.object(proof, "PROOF_CAP", 1):
            with self.assertRaises(proof.Refusal):
                proof.publish(result, self.root / "never.json")
        value = self.values["config.json"]
        expected = (("config.json", len(value), "git-sha1", hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()),)
        result = proof.make_proof(self.source, "complete", expected)
        self.assertEqual(result["files"][0]["observed_hash"], expected[0][3])

if __name__ == "__main__":
    unittest.main()
