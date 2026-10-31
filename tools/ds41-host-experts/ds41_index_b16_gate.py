#!/usr/bin/env python3
"""SHA-bound B16 oracle/arm guard. Metadata commands never load a GPU library.

Only `card` launches the checker, after read-only stopped-QFN/two-lock checks.
The window launcher retains sole responsibility for stopping/restoring QFN.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

FLAG = "RADIANCE_EXPERIMENTAL_DS41_INDEX_SCORE_BF16"
COHORT = "ds41-index-b16-cohort-v1"
RECEIPT = "ds41-index-b16-card-v1"
FILES = ("engine", "architecture", "libref", "library", "checker")
PROFILES = ("unpinned", "exact", "lcp4096")
LOCK_ROOT = Path("/home/as/radiance-setup/research/deepseek-v41-1017-20261024")


class Invalid(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise Invalid(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    p = Path(path)
    need(not p.exists(), "refuse replacing an existing frozen manifest/receipt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cohort(path):
    c = load(path)
    need(c.get("schema") == COHORT and re.fullmatch(r"[0-9a-f]{40}", c.get("source_commit", "")), "explicit cohort/source authority missing")
    need(set(c) == {"schema", "source_commit", "artifacts"} and set(c["artifacts"]) == set(FILES), "unknown/missing cohort fields")
    for name in FILES:
        a = c["artifacts"][name]
        need(set(a) == {"path", "sha256"} and re.fullmatch(r"[0-9a-f]{64}", a["sha256"]), "invalid artifact metadata")
        need(sha(a["path"]) == a["sha256"], "artifact changed: " + name)
    return c


def window(holder):
    need(type(holder) is int and holder > 1, "explicit live lock-holder PID required")
    os.kill(holder, 0)
    for name in ("research-compiler.lock", "research-heavy-runtime.lock"):
        with (LOCK_ROOT / name).open("rb") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            fcntl.flock(f, fcntl.LOCK_UN)
            raise Invalid("both research locks must already be held")
    state = subprocess.run(["/usr/bin/sudo", "-n", "/usr/bin/docker", "inspect", "--format", "{{.State.Running}}", "radiance-flashnext"],
                           capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    need(state == "false", "production QFN must already be stopped by window launcher")
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        need(key not in os.environ or os.environ[key] == "0,1,2,3", "oracle requires all four cards: " + key)


def oracle_log(text, profile):
    need(not re.search(r"\bFAIL\b|\bbad=[1-9]", text), "oracle failure marker")
    rows = re.findall(r"^PASS index_b16 rank=(\d+) profile=(\w+) score_cases=12 select_cases=12$", text, re.M)
    need(sorted(rows) == [(str(r), profile) for r in range(4)], "missing/duplicate card rank coverage")
    pattern = (r"^PASS index_b16 coverage: profile=" + re.escape(profile) +
               r" ranks=4 score_cases=48 select_cases=48 all_score_bits=65536 all_input_bits=65536 max_M=4096 max_C=65536 "
               r"score_values=([1-9]\d*) selected_ids=([1-9]\d*) paths=(\d+) bad=0$")
    matched = re.findall(pattern, text, re.M)
    need(len(matched) == 1, "missing/duplicate complete oracle receipt")
    paths = int(matched[0][2]); required = 1023 if profile == "unpinned" else 975
    need(paths == required, "actual scalar/WMMA/selection dispatch coverage mismatch")


def arm_env(enabled):
    need(type(enabled) is bool, "explicit enabled boolean required")
    return {FLAG: "1" if enabled else "0"}


def receipt(path, manifest, window_id):
    c = cohort(manifest); r = load(path)
    need(r.get("schema") == RECEIPT and r.get("window_id") == window_id, "stale/different-window oracle receipt")
    need(r.get("cohort_sha256") == sha(manifest) and r.get("artifacts") == c["artifacts"], "receipt/cohort authority mismatch")
    need(set(r.get("profiles", {})) == set(PROFILES), "missing/extra oracle profile")
    for profile, log in r["profiles"].items():
        need(set(log) == {"path", "sha256"} and sha(log["path"]) == log["sha256"], "oracle log changed")
        oracle_log(Path(log["path"]).read_text(), profile)
    return c


def server(text, enabled, reference=False):
    start = re.findall(r"ds41 index b16 admitted: rank=(\d+) rows=(\d+) columns=(\d+) bytes=(\d+) legacy_bytes=(\d+) same_runs=1 writers=(\d+) candidates=(\d+) topk=(\d+)", text)
    roles = re.findall(r"ds41 index b16 active: rank=(\d+) role=(score|candidates|topk)", text)
    if not enabled:
        need(not start and not roles and "ds41 index b16" not in text, "B16 leaked into disabled/reference arm")
        return
    need(not reference, "byte-reference arm must keep B16 off")
    need(len(start) == 4 and sorted(int(x[0]) for x in start) == list(range(4)), "missing/duplicate startup rank proof")
    shapes = set()
    for rank, rows, cols, size, old, writers, cand, top in start:
        rows, cols, size, old, writers, cand, top = map(int, (rows, cols, size, old, writers, cand, top))
        need(0 < rows <= 4096 and cols > 0 and size == rows * cols * 2 and old == size * 2, "wrong B16 plane byte ledger")
        need(writers > 0 and cand > 0 and top > 0, "incomplete index graph")
        shapes.add((rows, cols, writers, cand, top))
    need(len(shapes) == 1, "rank graph/run geometry differs")
    need(sorted(roles) == sorted((str(r), role) for r in range(4) for role in ("score", "candidates", "topk")), "missing/duplicate actual per-rank reader/writer proof")


def selected_files(c, engine, home):
    need(sha(engine) == c["artifacts"]["engine"]["sha256"], "arm binary differs from oracle cohort")
    for name, relative in (("architecture", "architectures/deepseek41.so"), ("library", "kernels/libr4d.so"), ("libref", "kernels/libref.so")):
        paths = [Path(h) / relative for h in home.split(":") if (Path(h) / relative).is_file()]
        need(paths and sha(paths[0]) == c["artifacts"][name]["sha256"], "arm home differs from oracle cohort: " + name)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--source-commit", required=True); prep.add_argument("--out", required=True)
    for name in FILES:
        prep.add_argument("--" + name, required=True)
    card = sub.add_parser("card"); card.add_argument("--inside-guarded-window", action="store_true"); card.add_argument("--lock-holder-pid", type=int, required=True)
    card.add_argument("--cohort", required=True); card.add_argument("--window-id", required=True); card.add_argument("--out", required=True)
    for command in ("receipt", "server"):
        q = sub.add_parser(command); q.add_argument("--cohort"); q.add_argument("--receipt"); q.add_argument("--window-id"); q.add_argument("--engine"); q.add_argument("--home")
        if command == "server":
            q.add_argument("--log", required=True); q.add_argument("--enabled", choices=("0", "1"), required=True); q.add_argument("--reference", action="store_true")
    a = p.parse_args()
    if a.command == "prepare":
        need(re.fullmatch(r"[0-9a-f]{40}", a.source_commit), "explicit source SHA required")
        save(a.out, {"schema": COHORT, "source_commit": a.source_commit, "artifacts": {name: {"path": str(Path(getattr(a, name)).resolve()), "sha256": sha(getattr(a, name))} for name in FILES}})
    elif a.command == "card":
        need(a.inside_guarded_window and a.window_id and not Path(a.out).exists(), "fresh explicit guarded-window oracle required")
        window(a.lock_holder_pid); c = cohort(a.cohort); logs = {}
        for profile in PROFILES:
            window(a.lock_holder_pid); cohort(a.cohort)
            env = os.environ.copy()
            env.update({FLAG: "0", "RADIANCE_EXPERIMENTAL_DS41_KV_PACKED": "0", "RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST": "0",
                        "RADIANCE_EXPERIMENTAL_DS41_INDEX_SCORE_COMPACT": "0", "RADIANCE_EXPERIMENTAL_DS41_P1": "0",
                        "RADIANCE_EXPERIMENTAL_DS41_EXACT_TALL": "1" if profile == "exact" else "0",
                        "RADIANCE_EXPERIMENTAL_DS41_LCP": "4096" if profile == "lcp4096" else "0"})
            run = subprocess.run([c["artifacts"]["checker"]["path"], "--inside-guarded-window", c["artifacts"]["library"]["path"], profile, "4"],
                                 env=env, capture_output=True, text=True, timeout=180)
            text = run.stdout + run.stderr; log = Path(a.out + "." + profile + ".log")
            need(not log.exists(), "refuse stale oracle log"); log.parent.mkdir(parents=True, exist_ok=True); log.write_text(text)
            need(run.returncode == 0, "card checker failed"); oracle_log(text, profile)
            cohort(a.cohort); logs[profile] = {"path": str(log.resolve()), "sha256": sha(log)}
        window(a.lock_holder_pid); cohort(a.cohort)
        save(a.out, {"schema": RECEIPT, "window_id": a.window_id, "lock_holder_pid": a.lock_holder_pid,
                     "cohort_sha256": sha(a.cohort), "artifacts": c["artifacts"], "profiles": logs})
    else:
        enabled = a.command == "receipt" or a.enabled == "1"
        if a.command == "server":
            server(Path(a.log).read_text(), enabled, a.reference)
        if enabled:
            need(all((a.receipt, a.cohort, a.window_id, a.engine, a.home)), "explicit receipt/cohort/window/serving paths required")
            c = receipt(a.receipt, a.cohort, a.window_id); selected_files(c, a.engine, a.home)


if __name__ == "__main__":
    try:
        main()
    except (Invalid, OSError, subprocess.SubprocessError, ValueError, KeyError) as e:
        raise SystemExit("index_b16 gate refused: " + str(e))
