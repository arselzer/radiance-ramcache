#!/usr/bin/env python3
"""Fake host receipts/guards only. No library load, checker, server or GPU action."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("b16_gate", Path(__file__).with_name("ds41_index_b16_gate.py"))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)


def card_log(profile):
    return "".join(f"PASS index_b16 rank={r} profile={profile} score_cases=12 select_cases=12\n" for r in range(4)) + (
        f"PASS index_b16 coverage: profile={profile} ranks=4 score_cases=48 select_cases=48 all_score_bits=65536 all_input_bits=65536 max_M=4096 max_C=65536 score_values=123 selected_ids=456 paths={1023 if profile == 'unpinned' else 975} bad=0\n")


def server_log():
    return "".join(f"ds41 index b16 admitted: rank={r} rows=2047 columns=32769 bytes=134156286 legacy_bytes=268312572 same_runs=1 writers=24 candidates=1 topk=8\n" +
                   "".join(f"ds41 index b16 active: rank={r} role={role}\n" for role in ("score", "candidates", "topk")) for r in range(4))


class GateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        artifacts = {}
        for name in g.FILES:
            p = self.root / name; p.write_text("fake " + name)
            artifacts[name] = {"path": str(p), "sha256": g.sha(p)}
        self.cohort = self.root / "cohort.json"
        self.cohort.write_text(json.dumps({"schema": g.COHORT, "source_commit": "1" * 40, "artifacts": artifacts}))
        profiles = {}
        for profile in g.PROFILES:
            p = self.root / (profile + ".log"); p.write_text(card_log(profile)); profiles[profile] = {"path": str(p), "sha256": g.sha(p)}
        self.r = {"schema": g.RECEIPT, "window_id": "window1", "cohort_sha256": g.sha(self.cohort), "artifacts": artifacts, "profiles": profiles}
        self.receipt = self.root / "receipt.json"; self.write()

    def write(self):
        self.receipt.write_text(json.dumps(self.r))

    def test_complete_cohort_receipt_and_half_size_perrank_actual_paths(self):
        g.receipt(self.receipt, self.cohort, "window1"); g.server(server_log(), True); g.server("ordinary engine\n", False)

    def test_stale_window_and_replaced_artifact_or_receipt_authority_refuse(self):
        with self.assertRaises(g.Invalid): g.receipt(self.receipt, self.cohort, "oldwindow")
        (self.root / "library").write_text("different library")
        with self.assertRaises(g.Invalid): g.receipt(self.receipt, self.cohort, "window1")

    def test_missing_duplicate_or_wrong_profile_dispatch_receipts_refuse(self):
        for text in (card_log("exact").replace("rank=3", "rank=0"), card_log("exact").replace("paths=975", "paths=1023"), card_log("exact") + card_log("exact"), "FAIL\n" + card_log("exact")):
            with self.assertRaises(g.Invalid): g.oracle_log(text, "exact")
        del self.r["profiles"]["exact"]; self.write()
        with self.assertRaises(g.Invalid): g.receipt(self.receipt, self.cohort, "window1")

    def test_log_tampering_refuses_even_when_artifacts_unchanged(self):
        Path(self.r["profiles"]["exact"]["path"]).write_text(card_log("exact") + "tamper")
        with self.assertRaises(g.Invalid): g.receipt(self.receipt, self.cohort, "window1")

    def test_server_missing_duplicate_rank_or_role_and_wrong_bytes_refuse(self):
        original = server_log()
        for text in (original.replace("rank=3", "rank=0"), original.replace("role=topk", "role=score"), original.replace("bytes=134156286", "bytes=134156288"), original + original):
            with self.assertRaises(g.Invalid): g.server(text, True)

    def test_reference_off_leak_and_explicit_perarm_reset(self):
        with self.assertRaises(g.Invalid): g.server(server_log(), False)
        with self.assertRaises(g.Invalid): g.server(server_log(), True, reference=True)
        with mock.patch.dict(os.environ, {g.FLAG: "1"}):
            self.assertEqual(g.arm_env(False)[g.FLAG], "0"); self.assertEqual(g.arm_env(True)[g.FLAG], "1")

    def test_card_command_without_guard_does_not_reach_checker_or_window(self):
        argv = ["gate", "card", "--lock-holder-pid", "123", "--cohort", str(self.cohort), "--window-id", "window1", "--out", str(self.root / "new.json")]
        with mock.patch("sys.argv", argv), mock.patch.object(g, "window") as window, mock.patch.object(g.subprocess, "run") as run:
            with self.assertRaises(g.Invalid): g.main()
            window.assert_not_called(); run.assert_not_called()

    def test_running_qfn_free_lock_and_wrong_card_mask_refuse_before_gpu(self):
        for name in ("research-compiler.lock", "research-heavy-runtime.lock"): (self.root / name).touch()
        blocked = BlockingIOError()
        with mock.patch.object(g, "LOCK_ROOT", self.root), mock.patch.object(g.os, "kill"), mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(g.fcntl, "flock", side_effect=blocked), mock.patch.object(g.subprocess, "run", return_value=types.SimpleNamespace(stdout="true\n")):
                with self.assertRaises(g.Invalid): g.window(123)
            with mock.patch.object(g.fcntl, "flock", return_value=None), mock.patch.object(g.subprocess, "run") as run:
                with self.assertRaises(g.Invalid): g.window(123)
                run.assert_not_called()
            with mock.patch.object(g.fcntl, "flock", side_effect=blocked), mock.patch.object(g.subprocess, "run", return_value=types.SimpleNamespace(stdout="false\n")):
                with mock.patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0"}):
                    with self.assertRaises(g.Invalid): g.window(123)


if __name__ == "__main__": unittest.main()
