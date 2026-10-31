#!/usr/bin/env python3
"""Separate N1 -> N4 -> N1 bank-lending receipt. Run only in a guarded GPU window.

No warmup, endpoint invention, service control or existing prompt-set edits. `prepare` only
freezes tokenized prompts; `run` submits exactly four requests; `validate` is offline.
"""
import argparse
import fcntl
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.parse

import ab_requests

SCHEMA = "ds41-bank-lend-probe-v1"
LOCK_ROOT = Path("/home/as/radiance-setup/research/deepseek-v41-1017-20261024")
LOCKS = [LOCK_ROOT / "research-compiler.lock", LOCK_ROOT / "research-heavy-runtime.lock"]
NAMES = ["anchor", "burst-1", "burst-2", "burst-3"]
ACTIVE = {"waiting", "prefill", "decode", "preempted"}


class Invalid(ValueError):
    pass


def need(ok, why):
    if not ok:
        raise Invalid(why)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def check_window(path):
    """Read-only runtime guard: the launcher owns services and both long-lived locks."""
    proof = load(path)
    need(proof.get("production_container") == "radiance-flashnext" and
         proof.get("production_stopped") is True and proof.get("both_research_locks") is True,
         "window attestation must name stopped production and both research locks")
    pid = proof.get("lock_holder_pid")
    need(type(pid) is int and pid > 1, "missing lock holder PID")
    os.kill(pid, 0)
    for path in LOCKS:
        with path.open("rb") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            fcntl.flock(f, fcntl.LOCK_UN)
            raise Invalid("research lock is not held: " + str(path))
    state = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}",
                            "radiance-flashnext"], check=True, capture_output=True,
                           text=True, timeout=10).stdout.strip()
    need(state == "false", "production QFN must already be stopped by the window launcher")
    return proof


def fresh(stats):
    need(stats.get("total_requests") == 0 and stats.get("running") == 0 and
         stats.get("waiting") == 0 and stats.get("in_flight") == 0 and
         not stats.get("requests"), "probe requires a fresh idle server; no hidden warmup")
    need(type(stats.get("startup_s")) in (float, int) and
         math.isfinite(stats["startup_s"]) and stats["startup_s"] >= 0,
         "missing finite server startup_s")
    need(type(stats.get("started_at")) is int and stats["started_at"] > 0,
         "missing server process-start identity")


def prepare(url):
    need(not os.environ.get("AB_MAX_TOKENS"), "AB_MAX_TOKENS would alter this frozen workload")
    prompts = [
        "Write a complete, carefully commented Python implementation of a persistent B-tree, "
        "including insertion, deletion, iteration, crash recovery, and at least forty detailed "
        "unit tests. Explain each invariant in code comments. Produce at least 5000 words of "
        "code and comments; continue until the implementation and all tests are complete.",
        "Implement a Python trie supporting insert, erase, exact lookup and prefix iteration. "
        "Include extensive comments and twenty tests. Produce at least 1500 words of code.",
        "Implement a Python interval tree with rotations, deletion and overlap queries, then "
        "explain the invariants and include thirty tests. Produce at least 1800 words of code. "
        "Use descriptive function names and comment every boundary condition carefully.",
        "Implement a Python disk-backed append-only log, with checksums, recovery, compaction, "
        "and concurrent-reader snapshots. Include detailed comments, examples, and forty unit "
        "tests. Produce at least 2000 words of code and explanations. Discuss torn writes, "
        "truncated records, empty files and generation numbers in the comments.",
    ]
    rows = []
    lengths = set()
    for name, text in zip(NAMES, prompts):
        body, chat = ab_requests.framed({"request": {"chat_text": text}}, url)
        need(not chat and len(body["prompt"]) not in lengths,
             "default prompt lengths collided; change this separate receipt, not existing sets")
        lengths.add(len(body["prompt"]))
        body.update(max_tokens=1024 if name == "anchor" else 256, temperature=0, seed=1,
                    ignore_eos=True, stream=True, timings_per_token=True,
                    stream_options={"include_usage": True})
        rows.append({"name": name, "chat_text": text, "body": body})
    result = {"schema": SCHEMA, "endpoint": "/v1/completions", "requests": rows,
              "trigger_output": 16, "minimum_solo_tail": 32}
    check_workload(result)
    return result


def check_workload(w):
    need(w.get("schema") == SCHEMA and w.get("endpoint") == "/v1/completions",
         "unknown workload schema/endpoint")
    rows = w.get("requests", [])
    need([r.get("name") for r in rows] == NAMES, "request order/names must be exact")
    need(w.get("trigger_output") == 16 and w.get("minimum_solo_tail") == 32,
         "changed transition trigger/tail")
    keys = set()
    for r in rows:
        b = r["body"]
        ids = b.get("prompt")
        need(isinstance(ids, list) and len(ids) > 4 and ids[:2] == [0, 128803] and
             ids[-2:] == [128804, 128822] and all(type(i) is int and i >= 0 for i in ids),
             "request is not the existing DS41 framed token completion")
        cap = 1024 if r["name"] == "anchor" else 256
        need(b.get("max_tokens") == cap and b.get("temperature") == 0 and b.get("seed") == 1 and
             b.get("ignore_eos") is True and b.get("stream") is True and
             b.get("timings_per_token") is True and b.get("stream_options") == {"include_usage": True},
             "generation/stream parameters changed")
        need(set(b) == {"prompt", "max_tokens", "temperature", "seed", "ignore_eos", "stream",
                        "timings_per_token", "stream_options"}, "unexpected request parameter")
        key = (len(ids), cap)
        need(key not in keys, "ambiguous /stats prompt/max binding")
        keys.add(key)


class Stream:
    """One bounded HTTP SSE reader; socket disconnect is the supported cancellation path."""
    def __init__(self, url, row, clock=time.monotonic):
        self.url, self.request, self.clock = url, row, clock
        self.lock = threading.Lock()
        self.connection = None
        self.thread = None
        self.result = {"name": row["name"], "text": "", "timings": {}, "error": None,
                       "done": False, "http_id": None, "finish_reason": None,
                       "submitted_s": None, "first_chunk_s": None, "finished_s": None,
                       "cancelled": False}

    def feed(self, obj, at):
        need(isinstance(obj, dict) and "error" not in obj, "SSE error: " + str(obj.get("error")))
        with self.lock:
            choices = obj.get("choices", [])
            need(isinstance(choices, list) and len(choices) <= 1, "unexpected SSE choice count")
            if obj.get("id"):
                need(self.result["http_id"] in (None, obj["id"]), "SSE HTTP ID changed")
                self.result["http_id"] = obj["id"]
            if choices:
                ch = choices[0]
                need(ch.get("index") == 0 and isinstance(ch.get("text"), str), "malformed completion chunk")
                if ch["text"] and self.result["first_chunk_s"] is None:
                    self.result["first_chunk_s"] = at
                self.result["text"] += ch["text"]
                if ch.get("finish_reason") is not None:
                    need(self.result["finish_reason"] is None, "duplicate final SSE choice")
                    self.result["finish_reason"] = ch["finish_reason"]
            if "timings" in obj:
                self.result["timings"] = obj["timings"]
            if "usage" in obj:
                self.result["usage"] = obj["usage"]

    def start(self):
        self.thread = threading.Thread(target=self.read, name="bank-probe-" + self.request["name"], daemon=True)
        self.thread.start()

    def cancel(self):
        with self.lock:
            self.result["cancelled"] = True
            conn = self.connection
        if conn and conn.sock:
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if conn:
            conn.close()

    def snapshot(self):
        with self.lock:
            return dict(self.result)

    def read(self):
        conn = None
        try:
            p = urllib.parse.urlsplit(self.url)
            need(p.scheme in ("http", "https") and not p.query and not p.fragment, "invalid server URL")
            cls = http.client.HTTPSConnection if p.scheme == "https" else http.client.HTTPConnection
            conn = cls(p.hostname, p.port, timeout=15)
            with self.lock:
                self.connection = conn
                self.result["submitted_s"] = self.clock()
            conn.request("POST", p.path.rstrip("/") + "/v1/completions",
                         json.dumps(self.request["body"]).encode(), {"Content-Type": "application/json"})
            response = conn.getresponse()
            need(response.status == 200, "completion HTTP status " + str(response.status))
            need(response.getheader("Content-Type", "").startswith("text/event-stream"), "completion is not SSE")
            fields, terminal = [], False
            while True:
                line = response.readline(4 * 1024 * 1024 + 1)
                need(len(line) <= 4 * 1024 * 1024, "oversized SSE line")
                if not line:
                    break
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    fields.append(line[5:].lstrip(" "))
                elif not line and fields:
                    data = "\n".join(fields)
                    fields = []
                    if data == "[DONE]":
                        terminal = True
                        break
                    self.feed(json.loads(data), self.clock())
            with self.lock:
                need(terminal and self.result["finish_reason"] is not None, "truncated SSE without final/DONE")
        except Exception as e:
            with self.lock:
                self.result["error"] = type(e).__name__ + ": " + str(e)
        finally:
            if conn:
                conn.close()
            with self.lock:
                self.result["done"] = True
                self.result["finished_s"] = self.clock()


class Traffic:
    def __init__(self, workload):
        check_workload(workload)
        self.workload = workload
        self.keys = {(len(r["body"]["prompt"]), r["body"]["max_tokens"]): r["name"]
                     for r in workload["requests"]}
        self.ids = {}
        self.last_outputs = {}
        self.trace = []
        self.peak = 0
        self.bursts_started_s = None
        self.solo_s = None
        self.solo_output = None
        self.initial_solo = False
        self.concurrent = False
        self.solo_tail = False
        self.solo_draft = False

    def observe(self, s, at):
        need(type(s.get("running")) is int and 0 <= s["running"] <= 4, "invalid running count")
        need(isinstance(s.get("requests"), list), "missing live request table")
        need(type(s.get("total_requests")) is int and 0 <= s["total_requests"] <= 4,
             "hidden/foreign generation request")
        rows, seen_ids = {}, set()
        for row in s["requests"]:
            key = (row.get("prompt"), row.get("max"))
            need(key in self.keys, "foreign/ambiguous /stats row")
            name, seq = self.keys[key], row.get("id")
            need(type(seq) is int and seq > 0 and name not in rows and seq not in seen_ids,
                 "duplicate/malformed request binding")
            need(self.ids.get(name, seq) == seq, "request numeric sequence ID changed")
            self.ids[name] = seq
            seen_ids.add(seq)
            output = row.get("output")
            need(type(output) is int and self.last_outputs.get(name, 0) <= output <= key[1],
                 "output counter regressed/exceeded cap")
            self.last_outputs[name] = output
            rows[name] = dict(row)
        active = {name: row for name, row in rows.items() if row.get("state") in ACTIVE}
        self.peak = max(self.peak, s["running"])
        a = active.get("anchor")
        ready = False
        if self.bursts_started_s is None:
            ready = bool(a and set(active) == {"anchor"} and s["running"] == 1 and
                         a["state"] == "decode" and a["output"] >= 16 and a["output"] <= 128)
            self.initial_solo |= ready
        else:
            if set(active) == set(NAMES) and s["running"] == 4 and all(
                    r["state"] == "decode" and r["output"] > 0 for r in active.values()):
                self.concurrent = True
            if self.concurrent and a and set(active) == {"anchor"} and s["running"] == 1 and a["state"] == "decode":
                if self.solo_s is None:
                    self.solo_s, self.solo_output = at, a["output"]
                    need(a["max"] - a["output"] >= 32, "anchor has insufficient remaining solo generation")
                self.solo_tail |= a["output"] - self.solo_output >= 32
                self.solo_draft |= type(a.get("draft")) is int and a["draft"] > 0
        self.trace.append({"at_s": at, "running": s["running"], "waiting": s.get("waiting"),
                           "in_flight": s.get("in_flight"), "total_requests": s["total_requests"],
                           "started_at": s.get("started_at"), "requests": list(rows.values())})
        return ready

    def verify(self, role):
        need(self.initial_solo and self.concurrent and self.peak == 4 and self.solo_tail,
             "actual N1->N4->N1 with a 32-token solo tail was not observed")
        need(set(self.ids) == set(NAMES), "not all four names have actual sequence bindings")
        if role != "no-bank":
            need(self.solo_draft, "no actual outstanding draft observed after the burst")


def get_stats(url):
    import urllib.request
    with urllib.request.urlopen(url.rstrip("/") + "/stats", timeout=10) as response:
        return json.load(response)


class BankLogWatch:
    """Incremental, bounded proof: /stats idle alone can precede an idle bank restore."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.partial = b""
        self.lines = []

    @property
    def text(self):
        return "\n".join(self.lines)

    def poll(self):
        need(self.path.stat().st_size >= self.offset, "server log rotated/truncated during probe")
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read(64 * 1024 * 1024 - self.offset + 1)
        self.offset += len(data)
        need(self.offset <= 64 * 1024 * 1024, "live server log exceeds bounded64MiB probe scan")
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        self.lines.extend(line.decode("utf-8") for line in lines if b"DSBANKLEND " in line)

    def settled(self):
        opened, seen = set(), False
        for row in marker_rows(self.text):
            if row["kind"] != "transition":
                continue
            key = int(row["epoch"])
            if row.get("action") == "lend" and row.get("phase") == "begin":
                opened.add(key); seen = True
            if row.get("action") == "restore" and row.get("phase") == "ready":
                opened.discard(key)
        return seen and not opened


def run_probe(url, workload, role, timeout=240, interval=.1, get=get_stats,
              stream_factory=Stream, clock=time.monotonic, sleep=time.sleep, log_watch=None):
    need(role in {"no-bank", "static", "lend"}, "invalid probe role")
    need(1 <= timeout <= 900 and .05 <= interval <= 1, "unbounded polling parameters")
    need(role != "lend" or log_watch is not None, "lend role requires a live server-log restoration proof")
    before = get(url)
    fresh(before)
    traffic = Traffic(workload)
    streams = [stream_factory(url, row, clock) for row in workload["requests"]]
    start, wall_start = clock(), time.time_ns()
    result = {"schema": SCHEMA, "role": role, "workload": workload, "workload_sha256": digest(workload),
              "initial_stats": before, "started_s": start, "started_unix_ns": wall_start, "valid": False}
    streams[0].start()
    launched, drained = False, False
    try:
        polls = 0
        while clock() - start < timeout:
            polls += 1
            need(polls <= math.ceil(timeout / interval) + 2, "poll bound exceeded")
            now = clock()
            stats = get(url)
            if log_watch:
                log_watch.poll()
            need(stats.get("started_at") == before.get("started_at"), "server restarted during workload")
            ready = traffic.observe(stats, now)
            if ready and not launched:
                traffic.bursts_started_s = now
                for stream in streams[1:]:
                    stream.start()
                launched = True
            snapshots = [stream.snapshot() for stream in streams]
            for row in snapshots:
                need(not row["error"], row["name"] + ": " + str(row["error"]))
            if snapshots[0]["done"] and not launched:
                raise Invalid("anchor ended before the 16-token launch trigger")
            if all(row["done"] for row in snapshots):
                need(launched, "burst was never launched")
                need(max(row["finished_s"] for row in snapshots[1:]) < snapshots[0]["finished_s"],
                     "anchor ended before all bursts; solo restoration was not exercised")
                active = [r for r in stats["requests"] if r.get("state") in ACTIVE]
                if stats["running"] == 0 and stats.get("waiting") == 0 and stats.get("in_flight") == 0 and not active:
                    if role == "lend":
                        if not log_watch.settled():
                            sleep(interval)
                            continue
                        validate_markers(log_watch.text, {"seq_by_name": traffic.ids})
                    drained = True
                    break
            sleep(interval)
        need(drained, "workload/drain deadline exceeded")
        traffic.verify(role)
        result["valid"] = True
    except Exception as e:
        result["invalid_reason"] = type(e).__name__ + ": " + str(e)
    finally:
        for stream in streams:
            if stream.thread and not stream.snapshot()["done"]:
                stream.cancel()
        for stream in streams:
            if stream.thread:
                stream.thread.join(timeout=16)
        ended = clock()
        result.update(requests=[stream.snapshot() for stream in streams], stats_trace=traffic.trace,
                      seq_by_name=traffic.ids, peak_running=traffic.peak,
                      bursts_started_s=traffic.bursts_started_s, solo_resumed_s=traffic.solo_s,
                      solo_resumed_output=traffic.solo_output, solo_tail_observed=traffic.solo_tail,
                      solo_draft_observed=traffic.solo_draft, drained=drained,
                      drained_s=ended, workload_drain_s=ended - start, finished_unix_ns=time.time_ns())
        result["load_workload_drain_s"] = before["startup_s"] + result["workload_drain_s"]
        if log_watch:
            result["observed_lender_log"] = log_watch.text
    return result


def marker_rows(text):
    rows = []
    for line in text.splitlines():
        if "DSBANKLEND " not in line:
            continue
        tail = line.split("DSBANKLEND ", 1)[1]
        parts = tail.split()
        row = {"kind": parts.pop(0)} if parts and "=" not in parts[0] else {"kind": "rank"}
        for part in parts:
            need("=" in part, "malformed DSBANKLEND field")
            key, value = part.split("=", 1)
            need(key not in row, "duplicate DSBANKLEND field")
            row[key] = value
        rows.append(row)
    return rows


def validate_markers(text, receipt):
    rows = marker_rows(text)
    need(rows, "missing DSBANKLEND activity")
    transitions = [r for r in rows if r["kind"] == "transition"]
    epochs = sorted({int(r["epoch"]) for r in transitions})
    need(epochs and all(epoch > 0 for epoch in epochs), "missing/invalid loan epochs")
    for row in rows:
        if row.get("epoch") is not None and int(row["epoch"]) > 0:
            need(int(row["epoch"]) in epochs, "unpaired lender epoch")
    anchor = str(receipt["seq_by_name"]["anchor"])
    summaries = []
    for epoch in epochs:
        er = [r for r in rows if int(r.get("epoch", -1)) == epoch]
        phases = [r for r in er if r["kind"] == "transition"]
        need([(r.get("action"), r.get("phase")) for r in phases] ==
             [("lend", "begin"), ("lend", "ready"), ("restore", "begin"), ("restore", "ready")],
             "missing/duplicate/out-of-order loan/restore transition")
        for row in phases:
            need(row.get("deferred") == "1" and row.get("all_ranks") == "4", "unsafe lender cohort")
        need(all(r.get("pending_ctx") == "0" and r.get("pending_query") == "0" and
                 r.get("solo_held") == "1" for r in phases[:2]), "bank work was not completely deferred while lent")
        lend_step, restore_step = int(phases[1]["step"]), int(phases[3]["step"])
        idle = phases[3].get("seq") == "0"
        need(lend_step >= 0 and ((idle and restore_step == -1) or
             (not idle and restore_step >= lend_step)) and phases[3].get("seq") in (anchor, "0"),
             "restoration was bound to an unknown solo sequence")
        need(phases[0].get("seq") == phases[1].get("seq") == "0" and
             phases[2].get("seq") == phases[3].get("seq"), "transition sequence binding changed")
        borrowed, selected_units, hashes = [], [], []
        for rank in range(4):
            rr = [r for r in er if r.get("rank") == str(rank)]
            for stage in ("lent", "restored"):
                match = [r for r in rr if r.get("stage") == stage]
                need(len(match) == 1, "missing/duplicate per-rank " + stage)
                row = match[0]
                if stage == "lent":
                    n = int(row["borrowed_bytes"])
                    slots = int(row["slots"])
                    need(0 < n <= int(row["bank_bytes"]) and 0 < slots <= 96 and slots % 3 == 0 and
                         row.get("copied") == "1", "no actual bank copy/lending")
                    borrowed.append(n)
                else:
                    need(int(row["bytes"]) > 0 and row.get("verified") == "1" and
                         row.get("aliases_ok") == "1" and row.get("masks_ok") == "1" and
                         row.get("hash_scope") == "cohort" and row.get("strict_memcmp") == "1" and
                         re.fullmatch(r"[0-9a-fA-F]+", row.get("hash", "")) and
                         row["hash"] == row.get("expected_hash"), "unproven bank restoration")
                    selected_units.append(int(row["selected_units"]))
                    hashes.append(row["hash"])
            for action in ("lend", "restore"):
                drained = [r for r in rr if r.get("stage") == "drained" and r.get("action") == action]
                need(len(drained) == 1 and drained[0].get("cpu_parked") == "1",
                     "missing/duplicate/unsafe drained rank for " + action)
        resume = [r for r in er if r["kind"] == "resume"]
        selected = phases[3].get("seq") == anchor
        if selected:
            need(len(resume) == 1 and resume[0].get("seq") == anchor and resume[0].get("ready") == "1" and
                 int(resume[0]["query_rows"]) > 0 and int(resume[0]["step"]) >= restore_step,
                 "no actual surviving-anchor query after verified restore")
        else:
            need(not resume and epoch == epochs[-1], "idle restore must be terminal with no query")
        positions = {id(row): i for i, row in enumerate(rows)}
        need(positions[id(phases[0])] < positions[id(phases[1])] < positions[id(phases[2])] <
             positions[id(phases[3])], "transition phases out of order")
        if selected:
            need(positions[id(phases[3])] < positions[id(resume[0])], "resume preceded restored publication")
        for rank in range(4):
            rr = [r for r in er if r.get("rank") == str(rank)]
            for r in rr:
                if r.get("stage") == "lent":
                    need(positions[id(phases[0])] < positions[id(r)] < positions[id(phases[1])], "lent proof outside transaction")
                if r.get("stage") == "restored":
                    need(positions[id(phases[2])] < positions[id(r)] < positions[id(phases[3])], "restore proof outside transaction")
                if r.get("stage") == "drained":
                    lo, hi = (phases[:2] if r.get("action") == "lend" else phases[2:])
                    need(positions[id(lo)] < positions[id(r)] < positions[id(hi)], "drain proof outside transaction")
                    stage = "lent" if r["action"] == "lend" else "restored"
                    done = next(x for x in rr if x.get("stage") == stage)
                    need(positions[id(r)] < positions[id(done)], "copy/restore proof preceded its drain")
        need(all(n >= 0 for n in selected_units) and len(set(hashes)) == 1,
             "inconsistent cohort restoration receipt")
        if selected:
            need(sum(selected_units) > 0, "lent residents were never actually selected")
        summaries.append({"epoch": epoch, "borrowed_bytes": borrowed, "selected_units": selected_units,
                          "anchor": int(anchor) if selected else 0})
    need(any(row["anchor"] for row in summaries), "only idle restoration observed; anchor never resumed")
    return summaries


def check_receipt(r):
    need(r.get("schema") == SCHEMA and r.get("valid") is True and r.get("drained") is True,
         "invalid/incomplete client receipt")
    check_workload(r["workload"])
    fresh(r["initial_stats"])
    guard, end_guard = r.get("window_guard", {}), r.get("window_guard_end", {})
    for g in (guard, end_guard):
        need(g.get("production_container") == "radiance-flashnext" and g.get("production_stopped") is True and
             g.get("both_research_locks") is True and type(g.get("lock_holder_pid")) is int and
             g["lock_holder_pid"] > 1, "missing start/end guarded-window receipt")
    need(guard == end_guard, "window attestation changed during probe")
    if r["role"] == "lend":
        validate_markers(r.get("observed_lender_log", ""), r)
    need(r.get("workload_sha256") == digest(r["workload"]), "workload digest mismatch")
    need(all(type(r.get(k)) in (float, int) and math.isfinite(r[k]) for k in
             ("started_s", "drained_s", "workload_drain_s", "load_workload_drain_s")), "invalid wall-clock metrics")
    need(r["workload_drain_s"] > 0 and abs(r["drained_s"] - r["started_s"] - r["workload_drain_s"]) < .001 and
         abs(r["initial_stats"]["startup_s"] + r["workload_drain_s"] - r["load_workload_drain_s"]) < .001,
         "end-to-end timings do not include startup and full workload/drain")
    traffic = Traffic(r["workload"])
    for sample in r["stats_trace"]:
        need(r["started_s"] <= sample["at_s"] <= r["drained_s"] and
             sample["started_at"] == r["initial_stats"]["started_at"], "trace outside process/time bounds")
        if r["bursts_started_s"] is not None and sample["at_s"] > r["bursts_started_s"]:
            traffic.bursts_started_s = r["bursts_started_s"]
        traffic.observe(sample, sample["at_s"])
    traffic.verify(r["role"])
    last = r["stats_trace"][-1]
    need(last["running"] == 0 and last["waiting"] == 0 and last["in_flight"] == 0 and
         last["total_requests"] == 4 and not any(q.get("state") in ACTIVE for q in last["requests"]),
         "receipt has no observed fully idle four-request drain")
    need(traffic.ids == r["seq_by_name"], "stored /stats bindings do not match trace")
    need([row.get("name") for row in r["requests"]] == NAMES, "reply order changed")
    for req, row in zip(r["workload"]["requests"], r["requests"]):
        t = row.get("timings", {})
        need(row.get("done") is True and not row.get("error") and not row.get("cancelled") and
             isinstance(row.get("text"), str) and row.get("finish_reason") == "length",
             "request did not complete normally to its frozen cap")
        need(t.get("predicted_n") == req["body"]["max_tokens"] and
             ab_requests.prompt_tokens(t) == len(req["body"]["prompt"]), "reply prompt/generated count mismatch")
        need(row.get("http_id"), "missing SSE HTTP ID")
        need(r["started_s"] <= row["submitted_s"] <= row["first_chunk_s"] <= row["finished_s"] <= r["drained_s"],
             "request wall-clock ordering is invalid")
    need(len({row["http_id"] for row in r["requests"]}) == 4, "duplicate HTTP IDs")
    need(max(row["finished_s"] for row in r["requests"][1:]) < r["requests"][0]["finished_s"],
         "no surviving anchor")
    return traffic


def validate(receipts, logs):
    need(set(receipts) == {"no-bank", "static", "lend"} and set(logs) == set(receipts),
         "need exactly no-bank/static/lend receipts and complete server logs")
    for role, receipt in receipts.items():
        need(receipt.get("role") == role, "reference role mismatch")
        check_receipt(receipt)
    ref = receipts["no-bank"]
    for role in ("static", "lend"):
        other = receipts[role]
        need(other["workload"] == ref["workload"], "reference workload differs")
        for a, b in zip(ref["requests"], other["requests"]):
            need(a["name"] == b["name"] and a["text"].encode() == b["text"].encode() and
                 a["timings"]["predicted_n"] == b["timings"]["predicted_n"] and
                 ab_requests.prompt_tokens(a["timings"]) == ab_requests.prompt_tokens(b["timings"]),
                 role + " reply/count differs: " + a["name"])
    need(not marker_rows(logs["no-bank"]) and not marker_rows(logs["static"]),
         "lender markers present in a lender-off reference")
    observed = marker_rows(receipts["lend"]["observed_lender_log"])
    need(observed == marker_rows(logs["lend"])[:len(observed)],
         "live restored receipt does not belong to the complete server log")
    activity = validate_markers(logs["lend"], receipts["lend"])
    return {"schema": SCHEMA, "valid": True, "byte_identical": 4, "lender_activity": activity,
            "metrics": {role: {"workload_drain_s": r["workload_drain_s"],
                                "startup_s": r["initial_stats"]["startup_s"],
                                "load_workload_drain_s": r["load_workload_drain_s"],
                                "generated_tokens": sum(x["timings"]["predicted_n"] for x in r["requests"]),
                                "end_to_end_tokens_s": sum(x["timings"]["predicted_n"] for x in r["requests"]) /
                                                       r["load_workload_drain_s"]}
                        for role, r in receipts.items()},
            "server_log_sha256": {role: hashlib.sha256(text.encode()).hexdigest() for role, text in logs.items()},
            "proposal_identity": "not required against no-bank; separate static/capture gate"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("url"); prep.add_argument("output"); prep.add_argument("--window-guard", required=True)
    run = sub.add_parser("run")
    run.add_argument("url"); run.add_argument("workload"); run.add_argument("output")
    run.add_argument("--role", choices=("no-bank", "static", "lend"), required=True)
    run.add_argument("--window-guard", required=True)
    run.add_argument("--server-log", help="required live research-server log for lend role")
    run.add_argument("--timeout", type=float, default=240); run.add_argument("--poll-s", type=float, default=.1)
    val = sub.add_parser("validate")
    for role in ("no-bank", "static", "lend"):
        val.add_argument("--" + role, required=True); val.add_argument("--" + role + "-log", required=True)
    val.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "prepare":
        check_window(args.window_guard)
        fresh(get_stats(args.url))
        workload = prepare(args.url)
        check_window(args.window_guard)
        save(args.output, workload)
    elif args.command == "run":
        guard = check_window(args.window_guard)
        result = run_probe(args.url, load(args.workload), args.role, args.timeout, args.poll_s,
                           log_watch=BankLogWatch(args.server_log) if args.server_log else None)
        result["window_guard"] = guard
        try:
            result["window_guard_end"] = check_window(args.window_guard)
            need(guard == result["window_guard_end"], "window attestation changed during probe")
        except Exception as e:
            result["valid"] = False
            result["invalid_reason"] = "end window guard: " + str(e)
        save(args.output, result)
        need(result["valid"], result.get("invalid_reason", "invalid run"))
    else:
        save(args.output, {"schema": SCHEMA, "valid": False, "invalid_reason": "validation incomplete"})
        receipts = {role: load(getattr(args, role.replace("-", "_"))) for role in ("no-bank", "static", "lend")}
        logs = {role: Path(getattr(args, role.replace("-", "_") + "_log")).read_text()
                for role in receipts}
        save(args.output, validate(receipts, logs))
        print("DSBANKLEND_PROBE_PASS roles=3 replies=4 peak_running=4 anchor_resumed=1")


if __name__ == "__main__":
    try:
        main()
    except (Invalid, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        raise SystemExit("INVALID: " + str(exc))
