#!/usr/bin/env python3
"""Fake HTTP/state and strict receipt source cases. No model, sockets, devices or services."""
import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest import mock

spec = importlib.util.spec_from_file_location("ds41_bank_lend_probe", Path(__file__).with_name("ds41_bank_lend_probe.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def workload():
    rows = []
    for i, name in enumerate(m.NAMES):
        body = dict(prompt=[0, 128803] + [7] * (i + 10) + [128804, 128822],
                    max_tokens=1024 if i == 0 else 256, temperature=0, seed=1, ignore_eos=True,
                    stream=True, timings_per_token=True, stream_options={"include_usage": True})
        rows.append(dict(name=name, body=body))
    return dict(schema=m.SCHEMA, endpoint="/v1/completions", requests=rows,
                trigger_output=16, minimum_solo_tail=32)


def row(i, output, draft=0, state="decode"):
    w = workload()["requests"][i]
    return dict(id=101+i, prompt=len(w["body"]["prompt"]), max=w["body"]["max_tokens"],
                output=output, draft=draft, state=state)


def stats(requests=(), running=0, total=0):
    return dict(started_at=1000, startup_s=3.25, total_requests=total, running=running,
                waiting=0, in_flight=running, requests=list(requests))


def trace():
    return [stats(), stats([row(0, 16, 2)], 1, 1),
            stats([row(0, 32)] + [row(i, 1) for i in range(1, 4)], 4, 4),
            stats([row(0, 400, 2)], 1, 4), stats([row(0, 432, 2)], 1, 4), stats(total=4)]


class FakeHTTP:
    def __init__(self):
        self.responses = trace()
        self.index = -1
        self.now = 100.
        self.sent = []

    def get(self, url):
        self.index += 1
        return copy.deepcopy(self.responses[min(self.index, len(self.responses)-1)])

    def sleep(self, seconds):
        self.now += seconds

    def stream(self, url, req, clock):
        outer = self
        class FakeStream:
            thread = None
            def start(self):
                outer.sent.append(req["name"])
                self.thread = mock.Mock()
            def cancel(self):
                raise AssertionError("valid fake workload must not cancel")
            def snapshot(self):
                i = m.NAMES.index(req["name"])
                done = outer.index >= 5
                return dict(name=req["name"], text="reply " + req["name"],
                            timings=dict(predicted_n=req["body"]["max_tokens"],
                                         prompt_n=len(req["body"]["prompt"]), cache_n=0),
                            error=None, done=done, http_id="cmpl-random-" + req["name"],
                            finish_reason="length" if done else None, cancelled=False,
                            submitted_s=100., first_chunk_s=100.01 if i == 0 else 100.11,
                            finished_s=(100.39 if i == 0 else 100.19) if done else None)
        return FakeStream()


def receipt(role="lend", settle_index=5):
    fake = FakeHTTP()
    class Watch:
        text = property(lambda self: logs())
        def poll(self): pass
        def settled(self): return fake.index >= settle_index
    result = m.run_probe("http://fake", workload(), role, timeout=2, interval=.1,
                         get=fake.get, stream_factory=fake.stream,
                         clock=lambda: fake.now, sleep=fake.sleep,
                         log_watch=Watch() if role == "lend" else None)
    guard = dict(production_container="radiance-flashnext", production_stopped=True,
                 both_research_locks=True, lock_holder_pid=123)
    result["window_guard"] = result["window_guard_end"] = guard
    return result, fake


def logs(epoch=1, seq=101):
    lines = []
    def emit(kind="rank", **kw):
        lines.append("DSBANKLEND " + (kind + " " if kind != "rank" else "") +
                     " ".join(str(k) + "=" + str(v) for k, v in kw.items()))
    def phase(action, phase, step, seq=0):
        emit("transition", epoch=epoch, step=step, seq=seq, action=action, phase=phase,
             pending_ctx=0, pending_query=0, solo_held=1, deferred=1, all_ranks=4)
    phase("lend", "begin", 10)
    for rank in range(4):
        emit(epoch=epoch, rank=rank, action="lend", stage="drained", cpu_parked=1)
        emit(epoch=epoch, rank=rank, stage="lent", slots=72, borrowed_bytes=100, bank_bytes=120, copied=1)
    phase("lend", "ready", 10)
    phase("restore", "begin", 300, seq)
    for rank in range(4):
        emit(epoch=epoch, rank=rank, action="restore", stage="drained", cpu_parked=1)
        emit(epoch=epoch, rank=rank, stage="restored", bytes=124, verified=1,
             aliases_ok=1, masks_ok=1, hash="aabb1234", expected_hash="aabb1234")
    phase("restore", "ready", 300, seq)
    if seq:
        emit("resume", epoch=epoch, step=301, seq=seq, query_rows=2, ready=1)
    return "\n".join(lines)


class BankProbeTest(unittest.TestCase):
    def test_idle_stats_waits_for_late_restore_and_prices_that_drain(self):
        r, _ = receipt(settle_index=7)
        self.assertTrue(r["valid"], r.get("invalid_reason"))
        self.assertAlmostEqual(r["workload_drain_s"], .6)
        m.check_receipt(r)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "server.log"
            path.write_text("DSBANKLEND transition epoch=1 action=lend phase=begin\n")
            watch = m.BankLogWatch(path); watch.poll()
            self.assertFalse(watch.settled())
            with path.open("a") as f:
                f.write("DSBANKLEND transition epoch=1 action=restore phase=ready")
            watch.poll(); self.assertFalse(watch.settled())
            with path.open("a") as f: f.write("\n")
            watch.poll(); self.assertTrue(watch.settled())
            path.write_text("")
            with self.assertRaises(m.Invalid): watch.poll()

    def test_fake_http_uses_exactly_four_requests_and_keeps_e2e_load_and_drain(self):
        r, f = receipt()
        self.assertTrue(r["valid"], r.get("invalid_reason"))
        self.assertEqual(f.sent, m.NAMES)
        self.assertEqual(r["seq_by_name"], dict(zip(m.NAMES, range(101, 105))))
        self.assertEqual(r["peak_running"], 4)
        self.assertTrue(r["solo_tail_observed"])
        self.assertTrue(r["solo_draft_observed"])
        self.assertAlmostEqual(r["load_workload_drain_s"], 3.65)
        m.check_receipt(r)

    def test_fresh_process_refuses_hidden_warmup_or_restart_identity(self):
        for key, value in (("total_requests", 1), ("running", 1), ("in_flight", 1),
                           ("requests", [row(0, 1)]), ("started_at", None), ("startup_s", float("nan"))):
            bad = stats(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(m.Invalid):
                m.fresh(bad)

    def test_real_concurrency_tail_and_positive_solo_draft_are_required(self):
        for edit in ("concurrency", "tail", "draft"):
            r, _ = receipt()
            if edit == "concurrency":
                r["stats_trace"][1]["requests"][3]["state"] = "prefill"
            elif edit == "tail":
                r["stats_trace"][3]["requests"][0]["output"] = 431
            else:
                for s in r["stats_trace"][2:]:
                    for q in s["requests"]:
                        q["draft"] = 0
            with self.subTest(edit=edit), self.assertRaises(m.Invalid):
                m.check_receipt(r)

    def test_bindings_reject_foreign_duplicate_changed_and_regressing_sequences(self):
        for change in ("foreign", "duplicate", "id", "regress"):
            t = m.Traffic(workload())
            t.observe(stats([row(0, 16)], 1, 1), 0)
            bad = stats([row(0, 20)], 1, 1)
            if change == "foreign": bad["requests"][0]["prompt"] = 999
            elif change == "duplicate": bad["requests"].append(row(0, 20))
            elif change == "id": bad["requests"][0]["id"] = 900
            else: bad["requests"][0]["output"] = 15
            with self.subTest(change=change), self.assertRaises(m.Invalid):
                t.observe(bad, 1)

    def test_sse_chunks_preserve_bytes_and_reject_error_duplicate_finish_or_id_change(self):
        s = m.Stream("http://fake", workload()["requests"][0])
        def chunk(text, reason=None, ident="cmpl-random"):
            return dict(id=ident, choices=[dict(index=0, text=text, finish_reason=reason)])
        s.feed(chunk("first\nλ"), 1)
        s.feed(chunk("\x00tail", "length"), 2)
        self.assertEqual(s.snapshot()["text"], "first\nλ\x00tail")
        for bad in (chunk("", "length"), chunk("", ident="other"), {"error": "failure"}):
            with self.assertRaises(m.Invalid): s.feed(bad, 3)

    def test_strict_copy_restore_and_anchor_query_gate(self):
        r, _ = receipt()
        summary = m.validate_markers(logs(), r)
        self.assertEqual(summary[0]["anchor"], 101)
        self.assertEqual(summary[0]["borrowed_bytes"], [100]*4)
        m.validate_markers(logs() + "\n" + logs(epoch=2, seq=0), r)
        for old, new in (("borrowed_bytes=100", "borrowed_bytes=0"), ("copied=1", "copied=0"),
                         ("verified=1", "verified=0"), ("aliases_ok=1", "aliases_ok=0"),
                         ("masks_ok=1", "masks_ok=0"), ("expected_hash=aabb1234", "expected_hash=bb"),
                         ("cpu_parked=1", "cpu_parked=0"), ("pending_ctx=0", "pending_ctx=1"),
                         ("query_rows=2", "query_rows=0"), ("seq=101", "seq=999")):
            with self.subTest(old=old), self.assertRaises(m.Invalid):
                m.validate_markers(logs().replace(old, new, 1), r)
        with self.assertRaises(m.Invalid): m.validate_markers(logs(seq=0), r)
        missing = "\n".join(s for s in logs().splitlines() if not ("rank=3" in s and "stage=restored" in s))
        with self.assertRaises(m.Invalid): m.validate_markers(missing, r)

    def test_comparison_requires_all_three_roles_and_every_text_count_prompt(self):
        receipts = {role: receipt(role)[0] for role in ("no-bank", "static", "lend")}
        server_logs = {"no-bank": "plain", "static": "bank Dx", "lend": logs()}
        self.assertTrue(m.validate(receipts, server_logs)["valid"])
        for field in ("text", "count", "prompt", "order", "metric"):
            bad = copy.deepcopy(receipts)
            row0 = bad["lend"]["requests"][0]
            if field == "text": row0["text"] += "!"
            elif field == "count": row0["timings"]["predicted_n"] -= 1
            elif field == "prompt": bad["lend"]["workload"]["requests"][0]["body"]["prompt"][2] += 1
            elif field == "order": bad["lend"]["requests"].reverse()
            else: bad["lend"]["load_workload_drain_s"] -= 1
            with self.subTest(field=field), self.assertRaises(m.Invalid):
                m.validate(bad, server_logs)
        with self.assertRaises(m.Invalid): m.validate(receipts, {**server_logs, "static": logs()})

    def test_window_attestation_fails_before_any_runtime_access(self):
        with mock.patch.object(m, "load", return_value={}), mock.patch.object(m.os, "kill") as kill:
            with self.assertRaises(m.Invalid): m.check_window("unused")
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
