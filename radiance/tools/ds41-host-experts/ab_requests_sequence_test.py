"""HOST-only F5 client contract, no real server/model/GPU. Run under work/build.lock.

Exercises the real run()/one()/framed() path against an in-memory HTTP boundary: original
bodies, joined phases, different concurrency, full stats, and failure before a second phase.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_requests


class Sequence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ab-sequence-")
        self.addCleanup(self.tmp.cleanup)
        self.path = self.tmp.name
        self.long = self.receipt("long", [100])
        self.held = self.receipt("held", list(range(8)))
        self.out = os.path.join(self.path, "arm.json")
        self.before = os.path.join(self.path, "arm.before.json")
        self.sent = []
        self.done = set()
        self.mu = threading.Lock()
        self.must_finish = set()
        self.main_ids = set()
        self.fail_id = None

    def receipt(self, name, ids):
        reqs = [{"name": "%s-%d" % (name, x), "family": name,
                 "request": {"prompt": [x, 10], "temperature": 0, "max_tokens": 2,
                             "stream": False}} for x in ids]
        path = os.path.join(self.path, name + ".json")
        with open(path, "w") as f:
            json.dump({"benchmark_requests": reqs}, f)
        return path

    def post(self, url, path, body, timeout):
        self.assertEqual((url, path, timeout), ("http://same-server", "/v1/completions", 900.0))
        with self.mu:
            token = body["prompt"][0]
            if token in self.main_ids:
                self.assertTrue(self.must_finish <= self.done, "main started before the first wave joined")
            self.sent.append(copy.deepcopy(body))
            if token == self.fail_id:
                raise TimeoutError("prefix failed")
            self.done.add(token)
        return {"choices": [{"text": "reply-%d" % token}],
                "timings": {"predicted_n": 2, "predicted_ms": 20, "prompt_n": 2}}

    def stats(self, url, timeout):
        self.assertEqual((url, timeout), ("http://same-server/stats", 30))
        with self.mu:
            snapshot = {"total_requests": len(self.sent), "steps_decode": 2 * len(self.sent),
                        "in_flight": 0, "nested_observer": {"unfiltered": [5, 8, 13]}}
        return io.BytesIO(json.dumps(snapshot).encode())

    def execute(self, before, before_n, before_k, main, n, k):
        with mock.patch.dict(os.environ, {"AB_CONCURRENCY": str(k), "AB_REQUEST_TIMEOUT_S": "900",
                                          "AB_MAX_TOKENS": ""}), \
             mock.patch.object(ab_requests, "post", side_effect=self.post), \
             mock.patch.object(ab_requests.urllib.request, "urlopen", side_effect=self.stats), \
             contextlib.redirect_stdout(io.StringIO()):
            return ab_requests.before_run(before, before_n, before_k, main, "http://same-server", n, self.out)

    def read(self, path):
        with open(path) as f:
            return json.load(f)

    def test_long_then_original_eight_with_separate_metrics_and_full_snapshots(self):
        self.must_finish, self.main_ids = {100}, set(range(8))
        self.assertEqual(self.execute(self.long, 1, 1, self.held, 8, 8), 0)
        first, main = self.read(self.before), self.read(self.out)
        self.assertEqual([r["name"] for r in first["requests"]], ["long-100"])
        self.assertEqual([r["name"] for r in main["requests"]], ["held-%d" % x for x in range(8)])
        self.assertEqual((first["concurrency"], main["concurrency"]), (1, 8))
        self.assertEqual(([w["tokens"] for w in first["waves"]], [w["tokens"] for w in main["waves"]]), ([2], [16]))
        self.assertEqual((first["phase"]["stats_before"]["total_requests"], first["stats"]["total_requests"],
                          main["phase"]["stats_before"]["total_requests"], main["stats"]["total_requests"]), (0, 1, 1, 9))
        self.assertEqual(main["stats"]["nested_observer"], {"unfiltered": [5, 8, 13]})
        self.assertEqual(main["phase"]["before"], "arm.before.json")
        self.assertEqual(first["phase"]["next"], "arm.json")
        for report, receipt in ((first, self.long), (main, self.held)):
            with open(receipt, "rb") as f:
                self.assertEqual(report["phase"]["receipt_sha256"], hashlib.sha256(f.read()).hexdigest())
            self.assertGreaterEqual(report["phase"]["client_wall_s"], 0)
        expected = self.read(self.long)["benchmark_requests"] + self.read(self.held)["benchmark_requests"]
        self.assertEqual(sorted(self.sent, key=lambda x: x["prompt"]),
                         sorted([x["request"] for x in expected], key=lambda x: x["prompt"]))

    def test_eight_then_long_joins_all_original_requests(self):
        self.must_finish, self.main_ids = set(range(8)), {100}
        self.assertEqual(self.execute(self.held, 8, 8, self.long, 1, 1), 0)
        first, main = self.read(self.before), self.read(self.out)
        self.assertEqual((first["concurrency"], main["concurrency"]), (8, 1))
        self.assertEqual((len(first["requests"]), len(main["requests"])), (8, 1))
        self.assertEqual(main["phase"]["stats_before"]["total_requests"], 8)
        self.assertEqual(main["requests"][0]["text"], "reply-100")

    def test_failed_first_phase_is_retained_and_main_is_never_sent(self):
        self.fail_id = 100
        self.assertEqual(self.execute(self.long, 1, 1, self.held, 8, 8), 1)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(len(self.sent), 1)
        self.assertIn("prefix failed", self.read(self.before)["requests"][0]["error"])

    def test_incomplete_receipt_fails_before_any_http_request(self):
        with self.assertRaises(ValueError):
            self.execute(self.long, 1, 1, self.held, 9, 8)
        self.assertEqual(self.sent, [])
        self.assertFalse(os.path.exists(self.before))

    def test_plain_run_keeps_its_original_output_shape_and_body(self):
        with mock.patch.dict(os.environ, {"AB_CONCURRENCY": "1", "AB_REQUEST_TIMEOUT_S": "900", "AB_MAX_TOKENS": ""}), \
             mock.patch.object(ab_requests, "post", side_effect=self.post), \
             mock.patch.object(ab_requests.urllib.request, "urlopen", side_effect=self.stats), \
             contextlib.redirect_stdout(io.StringIO()):
            ab_requests.run(self.long, "http://same-server", 1, self.out)
        self.assertEqual(set(self.read(self.out)), {"requests", "waves", "concurrency", "request_timeout_s", "stats"})
        self.assertEqual(self.sent, [self.read(self.long)["benchmark_requests"][0]["request"]])


if __name__ == "__main__":
    unittest.main()
