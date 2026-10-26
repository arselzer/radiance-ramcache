#!/usr/bin/env python3
"""Parent-owned pure fake-Docker controls; no actual CLI/service/model access."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("runner9_fake_docker_subject", HERE / "runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)  # definitions only; __main__ is not entered
CID = "a" * 64
IMAGE = "sha256:" + "b" * 64
PREFIX = ["/does-not-exist/fake-docker"]
ORIGINAL = dict(Id=CID, Image=IMAGE, Name="/radiance-flashnext", Running=True,
                Status="running", Paused=False, Restarting=False, Dead=False,
                ExitCode=0, StartedAt="2026-10-26T00:00:01.123456789Z",
                FinishedAt="2026-10-25T23:59:59.123456789Z")
KNOWN = ("Docker stop failed: Error response from daemon: cannot stop container: " + CID +
         ": tried to kill container, but did not receive an exit event\n")


def terminal(code=137):
    value = dict(ORIGINAL)
    value.update(Running=False, Status="exited", ExitCode=code,
                 FinishedAt="2026-10-26T00:01:01.123456789Z")
    return value


class FakeDocker:
    def __init__(self, error=KNOWN, wait=b"137\n", states=None):
        self.error, self.wait = error, wait
        self.states = copy.deepcopy(states if states is not None else [terminal(), terminal()])
        self.calls = []
        self.on_stop = self.on_wait = None

    def __call__(self, prefix, arguments, timeout=30):
        if prefix != PREFIX:
            raise AssertionError("fake prefix only; actual Docker forbidden")
        self.calls.append((list(arguments), timeout))
        if arguments[0] == "stop":
            if arguments != ["stop", "-t", "30", CID] or timeout != 90:
                raise AssertionError("original stop grace/client timeout changed")
            if self.on_stop:
                self.on_stop()
            if self.error is not None:
                raise RuntimeError(self.error)
            return (CID + "\n").encode()
        if arguments[0] == "wait":
            if arguments != ["wait", CID] or timeout != 120:
                raise AssertionError("captured full-ID bounded wait only")
            if self.on_wait:
                self.on_wait()
            if isinstance(self.wait, BaseException):
                raise self.wait
            return self.wait
        if arguments == ["inspect", "--type", "container", "--format", runner.INSPECT_FORMAT, CID]:
            if timeout != 30 or not self.states:
                raise AssertionError("unexpected inspect count/timeout")
            return json.dumps(self.states.pop(0)).encode()
        raise AssertionError("unexpected fake command; no extra stop/start permitted")

    def count(self, operation):
        return sum(row[0][0] == operation for row in self.calls)


class StopConfirmationTests(unittest.TestCase):
    def setUp(self):
        runner._INTERRUPTED = runner._RESTORING = runner._CLEANING = False
        self.receipt, self.snapshots = {}, []

    def tearDown(self):
        runner._INTERRUPTED = runner._RESTORING = runner._CLEANING = False

    def invoke(self, fake, original=None, sleep=None):
        def save():
            self.snapshots.append(json.loads(json.dumps(self.receipt)))
        with patch.object(runner, "docker_capture", fake), patch.object(runner.time, "sleep", sleep or (lambda _: None)):
            return runner.stop_with_terminal_ack(PREFIX, dict(original or ORIGINAL), self.receipt, save)

    def refused(self, fake, original=None, sleep=None, exception=RuntimeError):
        with self.assertRaises(exception):
            self.invoke(fake, original, sleep)
        status = self.receipt["stop_acknowledgement"]
        self.assertFalse(status["terminal_ACK"])
        self.assertIn("error", status)
        self.assertLessEqual(fake.count("stop"), 1)
        self.assertEqual(fake.count("start"), 0)
        return status

    def test_normal_ack_is_original_single_fresh_stopped(self):
        fake = FakeDocker(error=None, states=[terminal()])
        self.assertEqual(self.invoke(fake), terminal())
        status = self.receipt["stop_acknowledgement"]
        self.assertTrue(status["initial_ACK"])
        self.assertTrue(status["terminal_ACK"])
        self.assertEqual(status["kind"], "normal_stop_ACK_and_fresh_stopped")
        self.assertEqual((fake.count("stop"), fake.count("wait"), fake.count("inspect")), (1, 0, 1))

    def test_late_ack_retains_original_error_without_second_stop(self):
        fake = FakeDocker()
        self.assertEqual(self.invoke(fake), terminal())
        status = self.receipt["stop_acknowledgement"]
        self.assertEqual(status["stop_request_count"], 1)
        self.assertFalse(status["initial_ACK"])
        self.assertTrue(status["terminal_ACK"])
        self.assertEqual(status["kind"], "late_terminal_ACK_after_known_missed_exit_event")
        self.assertEqual(status["wait_exit_code"], 137)
        self.assertEqual(status["original_error"], "RuntimeError: " + KNOWN)
        self.assertEqual(len(status["stopped_samples"]), 2)
        self.assertEqual((fake.count("stop"), fake.count("wait"), fake.count("inspect")), (1, 1, 2))

    def test_wait_exit_code_edges(self):
        for code in (0, 255):
            with self.subTest(code=code):
                fake = FakeDocker(wait=str(code).encode(), states=[terminal(code), terminal(code)])
                self.invoke(fake)
                self.assertEqual(self.receipt["stop_acknowledgement"]["wait_exit_code"], code)

    def test_wait_timeout_refuses_without_inspection(self):
        fake = FakeDocker(wait=subprocess.TimeoutExpired(["fake", "wait", CID], 120))
        self.refused(fake, exception=subprocess.TimeoutExpired)
        self.assertEqual((fake.count("stop"), fake.count("wait"), fake.count("inspect")), (1, 1, 0))

    def test_malformed_wait_codes_refuse_before_inspection(self):
        for raw in (b"", b"-1\n", b"256\n", b"00\n", b"137\n0\n", b" 137\n", b"137\r\n", b"137oops", b"\xff"):
            with self.subTest(raw=raw):
                fake = FakeDocker(wait=raw)
                self.refused(fake)
                self.assertEqual(fake.count("inspect"), 0)

    def test_nonknown_stop_errors_never_wait(self):
        for message in ("Docker stop failed: permission denied", KNOWN.replace(CID, "c" * 64),
                        KNOWN.rstrip("\n") + " extra", "owned child deadline"):
            with self.subTest(message=message):
                fake = FakeDocker(error=message)
                state = self.refused(fake)
                self.assertEqual(state["original_error"], "RuntimeError: " + message)
                self.assertEqual((fake.count("stop"), fake.count("wait"), fake.count("inspect")), (1, 0, 0))

    def test_wrong_terminal_state_identity_or_generation_refuses(self):
        mutations = [dict(Running=True, Status="running"), dict(Id="c" * 64), dict(Image="sha256:" + "c" * 64),
                     dict(Name="/other"), dict(StartedAt="2026-10-26T00:00:02.123456789Z"),
                     dict(FinishedAt=ORIGINAL["FinishedAt"]), dict(FinishedAt="0001-01-01T00:00:00Z"),
                     dict(FinishedAt="not-a-time"), dict(FinishedAt=""), dict(FinishedAt=None),
                     dict(ExitCode=0), dict(ExitCode=True), dict(Paused=True), dict(Restarting=True), dict(Dead=True)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                state = terminal();state.update(mutation)
                fake = FakeDocker(states=[state, terminal()])
                self.refused(fake)
                self.assertEqual((fake.count("stop"), fake.count("wait"), fake.count("inspect")), (1, 1, 1))

    def test_second_fresh_sample_is_required_and_same_generation(self):
        for mutation in (dict(Running=True, Status="running"), dict(StartedAt="2026-10-26T00:00:03Z"),
                         dict(FinishedAt="2026-10-26T00:01:02.123456789Z"), dict(ExitCode=0)):
            with self.subTest(mutation=mutation):
                second = terminal();second.update(mutation)
                fake = FakeDocker(states=[terminal(), second])
                self.refused(fake)
                self.assertEqual(fake.count("inspect"), 2)

    def test_invalid_captured_generation_refuses_before_wait(self):
        for raw in ("", "0001-01-01T00:00:00Z", "bad", None):
            with self.subTest(raw=raw):
                original = dict(ORIGINAL);original["StartedAt"] = raw
                fake = FakeDocker()
                self.refused(fake, original=original)
                self.assertEqual((fake.count("stop"), fake.count("wait")), (1, 0))

    def test_cancel_before_stop(self):
        runner._INTERRUPTED = True
        fake = FakeDocker()
        status = self.refused(fake, exception=runner.Interrupted)
        self.assertEqual((status["stop_request_count"], len(fake.calls)), (0, 0))

    def test_cancel_after_known_error_never_waits(self):
        fake = FakeDocker();fake.on_stop = lambda: setattr(runner, "_INTERRUPTED", True)
        self.refused(fake, exception=runner.Interrupted)
        self.assertEqual((fake.count("stop"), fake.count("wait")), (1, 0))

    def test_cancel_after_wait_never_inspects(self):
        fake = FakeDocker();fake.on_wait = lambda: setattr(runner, "_INTERRUPTED", True)
        self.refused(fake, exception=runner.Interrupted)
        self.assertEqual((fake.count("wait"), fake.count("inspect")), (1, 0))

    def test_cancel_between_fresh_samples(self):
        fake = FakeDocker()
        self.refused(fake, sleep=lambda _: setattr(runner, "_INTERRUPTED", True), exception=runner.Interrupted)
        self.assertEqual(fake.count("inspect"), 1)

    def test_operational_sources_are_unchanged(self):
        def functions(path):
            source = path.read_text()
            return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        old = functions(ROOT / "glm53-manual-maintenance-runner-v6-20261026/runner.py")
        new = functions(HERE / "runner.py")
        self.assertEqual(set(new) - set(old), {"stop_with_terminal_ack"})
        self.assertEqual({name for name in old if old[name] != new[name]}, {"run_window", "verify_window_inputs"})
        reconstructed = new["run_window"].replace(
            "receipt['stopped_state'] = stop_with_terminal_ack(prefix, original, receipt, save)",
            "docker_capture(prefix, ['stop', '-t', str(STOP_SECONDS), original['Id']], STOP_CLIENT_SECONDS)\n"
            "                receipt['stopped_state'] = stopped(prefix, original)")
        self.assertEqual(reconstructed, old["run_window"])


if __name__ == "__main__":
    if sys.argv != [sys.argv[0], "--run-fake-docker-tests"]:
        raise SystemExit("explicit --run-fake-docker-tests required; no actual Docker is used")
    unittest.main(argv=[sys.argv[0]], verbosity=2)
