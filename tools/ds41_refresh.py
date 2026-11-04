#!/usr/bin/env python3
"""Rebuild this branch's ds41 content from an engine commit, the report folder and A/B windows.

A later update is one command plus a commit:

    python3 tools/ds41_refresh.py --engine-commit <rev> [--add-window ab-...] [--add-server-logs ab-...]
    git commit

Defaults come from tools/ds41_refresh.json, which is rewritten with what a successful run used.
Nothing is built, run or pushed: no GPU, model or network use.

  radiance/        `git archive <commit>` of the engine repository: its tracked files, exact.
  docs/ds41/       the report files the config names, byte for byte (README.md there is kept);
                   provenance/ENGINE_SOURCE.json and ENGINE_SOURCE_PINS.json record the commit
                   and every radiance/ file's SHA256.
  research/ds41/   the small measurement programs and outputs the config names (README.md kept).
  evidence/ds41/   per A/B window: window.log, the per-arm *.json results, *.mem traces and the
                   small check logs; *.server.log only for the config's server_log_windows,
                   xz-compressed. Subdirectories (access traces, routing traces, KLD records) and
                   binary captures stay out; index.json lists each with its reason. README.md kept.

Everything new is built in build/ (ignored, outside the manifest), scanned for secrets, private
hosts and addresses, e-mail addresses, binaries and oversize files, and only then moved into
place. A finding outside radiance/ refuses the refresh; radiance/ is the exact engine tree, so its
findings are listed for review, never edited (a secret pattern there refuses too). Then
SNAPSHOT_MANIFEST.json is regenerated, tools/verify_snapshot.py runs against the engine
repository, the refreshed paths are staged and a summary printed.

--manifest-only regenerates the manifest after hand edits (README.md, STATUS.md) and verifies.
"""

import argparse
import datetime
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools" / "ds41_refresh.json"
MANIFEST = "SNAPSHOT_MANIFEST.json"
ENGINE_DIR = "radiance"
DOCS_DIR = "docs/ds41"
PROV_DIR = "docs/ds41/provenance"
RESEARCH_DIR = "research/ds41"
EVIDENCE_DIR = "evidence/ds41"
PINS_NAME = "ENGINE_SOURCE_PINS.json"
SOURCE_NAME = "ENGINE_SOURCE.json"
KEEP = "README.md"                     # hand-written index kept in docs/, research/, evidence/
STAGING = "build/ds41-refresh"         # build/ is ignored and outside the manifest
MANAGED = (ENGINE_DIR, DOCS_DIR, RESEARCH_DIR, EVIDENCE_DIR, MANIFEST, "tools/ds41_refresh.json")
MB = 1000 * 1000


class Refusal(RuntimeError):
    pass


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def run(cmd, cwd=None, capture=True):
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Refusal(" ".join(map(str, cmd)) + " failed: "
                      + result.stderr.decode(errors="replace").strip()[-400:])
    return result.stdout if capture else b""


def git(*args, repo=ROOT):
    return run(["git", "-C", str(repo), *args])


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def canonical(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or str(p) != name or any(x in ("", ".", "..") for x in p.parts):
        raise Refusal("noncanonical path: " + repr(name))
    return p


def tree_files(base):
    """Regular files under base as {posix path: Path}; refuses symlinks and special files."""
    out = {}
    for directory, dirs, names in os.walk(base, followlinks=False):
        d = Path(directory)
        for name in dirs + names:
            if (d / name).is_symlink():
                raise Refusal("symlink: " + str(d / name))
        for name in names:
            path = d / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise Refusal("not a regular file: " + str(path))
            out[path.relative_to(base).as_posix()] = path
    return out


def human(n):
    return "%.1f MB" % (n / MB) if n >= 100 * 1000 else "%.1f kB" % (n / 1000)


# ---------------------------------------------------------------------------------------- box


def processes_matching(*needles):
    hits = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            cmdline = Path("/proc", entry, "cmdline").read_bytes()
        except OSError:
            continue
        if all(n.encode() in cmdline for n in needles):
            hits.append(int(entry))
    return hits


def wait_for_quiet_box(cfg, no_wait):
    """The box's load rule: no heavy step while a GPU A/B window is pending or running."""
    quiet = cfg.get("quiet_box") or {}
    flag, proc = quiet.get("flag"), quiet.get("process")
    announced = False
    while (flag and Path(flag).exists()) or (proc and processes_matching(proc)):
        if no_wait:
            raise Refusal("a GPU A/B window is pending or running (" + str(flag) + ", " + str(proc)
                          + "); rerun when it is done")
        if not announced:
            print("waiting for the GPU A/B window to finish (" + str(flag) + ", " + str(proc) + ") ...",
                  flush=True)
            announced = True
        time.sleep(20)


# ----------------------------------------------------------------------------------- engine


def build_engine(cfg, stage):
    repo = Path(cfg["engine_repo"])
    commit = git("rev-parse", "--verify", cfg["engine_commit"] + "^{commit}", repo=repo).decode().strip()
    tree = git("rev-parse", commit + "^{tree}", repo=repo).decode().strip()
    listing = git("ls-tree", "-r", "-z", "--full-tree", commit, repo=repo)
    tracked = {}
    for record in listing.split(b"\0"):
        if record:
            meta, path = record.split(b"\t", 1)
            mode, kind, _blob = meta.split()
            if kind != b"blob" or mode not in (b"100644", b"100755"):
                raise Refusal("engine tree has a non-regular entry (symlink/submodule): " + path.decode())
            tracked[path.decode()] = mode == b"100755"
    archive = git("archive", "--format=tar", commit, repo=repo)
    dest = stage / ENGINE_DIR
    pins, total, executables = {}, 0, 0
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        if tf.pax_headers.get("comment") not in (None, commit):
            raise Refusal("archive comment names another commit")
        for member in tf.getmembers():
            if member.isdir():
                continue
            if not member.isreg():
                raise Refusal("engine archive member is not a regular file: " + member.name)
            canonical(member.name)
            data = tf.extractfile(member).read()
            target = dest / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            executable = bool(member.mode & 0o111)
            os.chmod(target, 0o755 if executable else 0o644)
            pins[member.name] = sha256_bytes(data)
            total += len(data)
            executables += executable
            if tracked.get(member.name) is None or tracked[member.name] != executable:
                raise Refusal("archive and tree disagree on " + member.name)
    if set(pins) != set(tracked):
        raise Refusal("git archive left tracked files out (export-ignore?): "
                      + repr(sorted(set(tracked) - set(pins))[:8]))
    pins = dict(sorted(pins.items()))
    pins_text = json.dumps(pins, indent=1) + "\n"
    (stage / PROV_DIR).mkdir(parents=True, exist_ok=True)
    (stage / PROV_DIR / PINS_NAME).write_text(pins_text, encoding="utf-8")
    subject, committed = git("log", "-1", "--format=%s%x00%cI", commit, repo=repo).decode().rstrip("\n").split("\0")
    upstream = {}
    for label, rev in (cfg.get("engine_upstream") or {}).get("commits", {}).items():
        full = git("rev-parse", "--verify", rev + "^{commit}", repo=repo).decode().strip()
        ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", full, commit],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        if ancestor not in (0, 1):
            raise Refusal("git merge-base --is-ancestor failed for " + full)
        upstream[label] = {"commit": full, "contained": ancestor == 0,
                           "commits_after": int(git("rev-list", "--count", commit, "^" + full, repo=repo))}
    source = {
        "commit": commit,
        "tree": tree,
        "branch": cfg.get("engine_branch"),
        "subject": subject,
        "committed_utc": committed,
        "files": len(pins),
        "bytes": total,
        "executable_files": executables,
        "pins": PROV_DIR + "/" + PINS_NAME,
        "pins_sha256": sha256_bytes(pins_text.encode()),
        "export": "git archive --format=tar " + commit + " (tracked files only, modes kept)",
        "upstream": {"repository": (cfg.get("engine_upstream") or {}).get("repository"),
                     "note": (cfg.get("engine_upstream") or {}).get("note"),
                     "commits": upstream},
    }
    write_json(stage / PROV_DIR / SOURCE_NAME, source)
    return source


# ------------------------------------------------------------------------------ docs, research


def build_copies(cfg, stage, key, subdir):
    src_dir = Path(cfg["docs_dir"])
    copied = []
    for name in cfg.get(key, []):
        canonical(name)
        src = src_dir / name
        if src.is_symlink() or not src.is_file():
            raise Refusal("missing " + key + " file: " + str(src))
        dst = stage / subdir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o644)
        copied.append(name)
    return copied


def unclassified_docs(cfg):
    known = set(cfg.get("docs", [])) | set(cfg.get("research", [])) | set(cfg.get("docs_left_out", {}))
    return sorted(p.name for p in Path(cfg["docs_dir"]).iterdir() if p.is_file() and p.name not in known)


# ---------------------------------------------------------------------------------- windows


TEXT_SUFFIXES = (".json", ".mem", ".log", ".jsonl", ".txt", ".csv", ".md")


def window_reason(name, is_dir):
    if is_dir:
        for prefix, why in (("trace-", "per-rank expert access trace (binary records)"),
                            ("rtrace-", "routing trace (binary records)"),
                            ("kld-", "KLD logit records (binary f16/f32/i32)")):
            if name.startswith(prefix):
                return why
        return "subdirectory"
    if name.endswith(".server.log"):
        return "server log; kept (xz) only for the windows the README's headline numbers come from"
    if name.endswith(".bin"):
        return "binary capture"
    return "not a text result, log or memory trace"


def build_window(src, dst, keep_server_logs, max_text):
    """Copies one window's text evidence; returns (record, left_out)."""
    record = {"files": 0, "bytes": 0, "server_logs": 0, "server_log_bytes": 0, "server_log_xz_bytes": 0}
    left = []
    for entry in sorted(src.iterdir(), key=lambda p: p.name):
        name = entry.name
        if entry.is_symlink():
            left.append({"path": name, "reason": "symlink"})
            continue
        if entry.is_dir():
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file() and not f.is_symlink())
            left.append({"path": name + "/", "bytes": size, "reason": window_reason(name, True)})
            continue
        size = entry.stat().st_size
        if name.endswith(".server.log"):
            if not keep_server_logs:
                left.append({"path": name, "bytes": size, "reason": window_reason(name, False)})
                continue
            data = entry.read_bytes()
            if b"\0" in data:
                raise Refusal("binary content in server log " + str(entry))
            packed = lzma.compress(data, preset=6)
            dst.mkdir(parents=True, exist_ok=True)
            (dst / (name + ".xz")).write_bytes(packed)
            record["server_logs"] += 1
            record["server_log_bytes"] += len(data)
            record["server_log_xz_bytes"] += len(packed)
            continue
        if name.startswith("chain-") and name.endswith(".csv"):
            left.append({"path": name, "bytes": size, "reason": "raw per-kernel chain trace (Ct part); its analysis is a text file beside it"})
            continue
        if name == "window.log" or name.endswith(TEXT_SUFFIXES):
            if size > max_text:
                left.append({"path": name, "bytes": size, "reason": "over the text size cap"})
                continue
            data = entry.read_bytes()
            if b"\0" in data:
                left.append({"path": name, "bytes": size, "reason": "binary content"})
                continue
            dst.mkdir(parents=True, exist_ok=True)
            (dst / name).write_bytes(data)
            record["files"] += 1
            record["bytes"] += size
            continue
        left.append({"path": name, "bytes": size, "reason": window_reason(name, False)})
    log = src / "window.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    first = re.search(r"^\[(\d{4}-\d\d-\d\dT[\d:]+Z)\]", text, re.M)
    record["started_utc"] = first.group(1) if first else None
    record["arms_started"] = len(re.findall(r"^\[[^\]]+\] arm [^:]+: serving after", text, re.M))
    record["qfn_restored"] = "QFN up after" in text
    return record, left


def build_evidence(cfg, stage, names):
    root = Path(cfg["windows_dir"])
    server = cfg.get("server_log_windows", {})
    index = {"layout": "evidence/ds41/<window>/: window.log, <arm>.json, <arm>.mem, check logs; "
                       "<arm>.server.log.xz in the server_log_windows only",
             "source": str(root), "server_log_windows": server, "windows": {}}
    for name in names:
        src = root / name
        if not src.is_dir() or src.is_symlink():
            raise Refusal("window not found: " + str(src))
        record, left = build_window(src, stage / EVIDENCE_DIR / name, name in server,
                                    int(cfg.get("max_text_bytes", 20 * MB)))
        record["left_out"] = left
        index["windows"][name] = record
    write_json(stage / EVIDENCE_DIR / "index.json", index)
    return index


def finished_windows(cfg, requested):
    root = Path(cfg["windows_dir"])
    if requested == ["all"]:
        requested = sorted(p.name for p in root.iterdir()
                           if p.is_dir() and re.fullmatch(r"ab-\d{8}T\d{6}Z", p.name))
    running = [w for w in requested if processes_matching("ab_window.sh", w)]
    return [w for w in requested if w not in running], running


# ------------------------------------------------------------------------------------- scan


# Token formats: a match is a finding anywhere, the exact engine tree included.
SECRETS = [
    ("private key block", rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    ("ssh key", rb"\b(?:ssh-(?:rsa|ed25519|dss)|ecdsa-sha2-nistp\d+) AAAA[0-9A-Za-z+/]{20,}"),
    ("AWS access key", rb"\bAKIA[0-9A-Z]{16}\b"),
    ("GitHub token", rb"\bgh[pousr]_[A-Za-z0-9]{36,}|\bgithub_pat_[A-Za-z0-9_]{20,}"),
    ("Slack token", rb"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    ("API key (sk-)", rb"\bsk-(?:ant-)?[A-Za-z0-9_-]{24,}"),
    ("Hugging Face token", rb"\bhf_[A-Za-z0-9]{30,}"),
    ("Google API key", rb"\bAIza[0-9A-Za-z_-]{35}"),
    ("JWT", rb"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ("credentials in a URL", rb"\b[a-z][a-z0-9+.-]*://[^/\s:@\"'<>]+:[^/\s@\"'<>]+@"),
    ("bearer token", rb"(?i)authorization:[ \t]*bearer[ \t]+[A-Za-z0-9._~+/=-]{12,}"),
]
SECRETS = [(label, re.compile(pattern)) for label, pattern in SECRETS]
# Assignments that look like credentials: findings in what this branch adds; in the engine's
# source code (variables named password, api_key ...) they are listed for review instead.
ASSIGNMENTS = [
    ("password value", rb"(?i)\b(?:password|passwd)[ \t]*[:=][ \t]*[\"']?[^\s\"']{6,}"),
    ("key/token value", rb"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)"
                        rb"[ \t]*[:=][ \t]*[\"']?[A-Za-z0-9_\-]{16,}"),
]
ASSIGNMENTS = [(label, re.compile(pattern)) for label, pattern in ASSIGNMENTS]
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
IPV4 = re.compile(rb"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.])")
PRIVATE_DOMAIN = re.compile(rb"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:local|lan|internal|intranet|corp|home\.arpa|ts\.net)\b")
HOME_PATH = re.compile(rb"/home/[a-z_][a-z0-9_-]*/")
BINARY_SUFFIXES = (".so", ".a", ".o", ".bin", ".rad", ".safetensors", ".gguf", ".pt", ".pth", ".onnx",
                   ".npy", ".npz", ".f16", ".f32", ".i32", ".hsaco", ".exe", ".dll", ".dylib", ".zip",
                   ".tar", ".tgz", ".gz", ".7z", ".rkb")
MAGIC = ((b"\x7fELF", "ELF executable/object"), (b"GGUF", "GGUF model file"),
         (b"\xcf\xfa\xed\xfe", "Mach-O"), (b"PK\x03\x04", "zip archive"))


def scan_bytes(data, cfg, hostnames):
    """Returns {category: sorted set of matched strings}."""
    found = {}

    def add(category, value):
        found.setdefault(category, set()).add(value.decode(errors="replace")[:120])

    for label, pattern in SECRETS:
        for m in pattern.finditer(data):
            add("secret: " + label, m.group(0))
    for label, pattern in ASSIGNMENTS:
        for m in pattern.finditer(data):
            add("credential-like assignment: " + label, m.group(0))
    allowed = {e.lower().encode() for e in cfg.get("allowed_emails", [])}
    for m in EMAIL.finditer(data):
        if m.group(0).lower() not in allowed:
            add("e-mail address", m.group(0))
    for m in IPV4.finditer(data):
        octets = [int(x) for x in m.groups()]
        if max(octets) > 255 or octets[0] == 127 or octets == [0, 0, 0, 0]:
            continue
        add("IPv4 address", m.group(0))
    lowered = data.lower()
    for host in hostnames:
        if host and host.lower().encode() in lowered:
            add("private hostname", host.encode())
    for m in PRIVATE_DOMAIN.finditer(lowered):
        add("private domain", m.group(0))
    return found


def classify(path, rel, data):
    notes = []
    if rel.endswith(BINARY_SUFFIXES):
        notes.append("binary/payload suffix")
    for magic, label in MAGIC:
        if data.startswith(magic):
            notes.append(label)
    if b"\0" in data[:65536]:
        notes.append("binary content")
    if path.stat().st_mode & 0o111:
        notes.append("executable bit")
    return notes


def scan_tree(stage, cfg, published_paths):
    """Scans the staged files. Returns (fatal list, radiance notes, info)."""
    hostnames = set(cfg.get("private_hostnames", [])) | {socket.gethostname()}
    hostnames.discard("localhost")
    max_file = int(cfg.get("max_file_bytes", 50 * MB))
    fatal, engine_notes = [], {}
    info = {"files": 0, "bytes": 0, "home_path_files": 0}
    for rel, path in sorted(published_paths.items()):
        data = path.read_bytes()
        info["files"] += 1
        info["bytes"] += len(data)
        in_engine = rel.startswith(ENGINE_DIR + "/")
        if len(data) > max_file:
            fatal.append((rel, "file over " + human(max_file), human(len(data))))
        notes = classify(path, rel, data)
        text = data
        if rel.endswith(".server.log.xz") and rel.startswith(EVIDENCE_DIR + "/"):
            text = lzma.decompress(data)          # our own compressed logs: scan the text
            notes = [n for n in notes if n != "binary content"]
            if b"\0" in text:
                notes.append("binary content in the compressed log")
        if HOME_PATH.search(text):
            info["home_path_files"] += 1
        found = scan_bytes(text, cfg, hostnames)
        if in_engine:
            for category, values in found.items():
                if category.startswith("secret"):
                    fatal.append((rel, category, ", ".join(sorted(values))))
                else:
                    engine_notes.setdefault(category, {})[rel] = sorted(values)
            for note in notes:
                engine_notes.setdefault(note, {})[rel] = [human(len(data))]
            continue
        executable_ok = rel.startswith("tools/") and rel.endswith(".py")
        for note in notes:
            if note == "executable bit" and executable_ok:
                continue
            fatal.append((rel, note, human(len(data))))
        for category, values in found.items():
            fatal.append((rel, category, ", ".join(sorted(values))))
    return fatal, engine_notes, info


# --------------------------------------------------------------------------- manifest, git


def write_manifest(cfg, source):
    path = ROOT / MANIFEST
    old = {}
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
    files = {}
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        d = Path(directory)
        if d == ROOT:                    # what tools/verify_snapshot.py walks
            dirs[:] = [x for x in dirs if x not in (".git", "build")]
        for name in dirs:
            if (d / name).is_symlink():
                raise Refusal("symlink: " + str(d / name))
        for name in names:
            p = d / name
            rel = p.relative_to(ROOT).as_posix()
            if rel in (".git", MANIFEST):
                continue
            if p.is_symlink() or not stat.S_ISREG(p.lstat().st_mode):
                raise Refusal("not a regular file: " + rel)
            files[rel] = sha256_file(p)
    keep_created = old.get("branch") == "ds41"
    data = {
        "created_utc": old.get("created_utc") if keep_created else now_utc(),
        "updated_utc": now_utc(),
        "branch": "ds41",
        "scope": cfg.get("scope"),
        "source_checkpoint": "radiance engine " + source["commit"][:12] + " (" + str(source.get("branch")) + ")",
        "engine": {"commit": source["commit"], "files": source["files"], "pins_sha256": source["pins_sha256"],
                   "pins": source["pins"]},
        "preceding_branch": old.get("preceding_branch") if keep_created else "main",
        "preceding_published_commit": old.get("preceding_published_commit") if keep_created
        else git("rev-parse", "HEAD").decode().strip(),
        "preceding_manifest_sha256": old.get("preceding_manifest_sha256") if keep_created
        else (sha256_file(path) if path.is_file() else None),
        "files_sha256": dict(sorted(files.items())),
    }
    write_json(path, data)
    return data


def current_source():
    path = ROOT / PROV_DIR / SOURCE_NAME
    if not path.is_file():
        raise Refusal("no " + str(path) + "; run a full refresh first")
    return json.loads(path.read_text(encoding="utf-8"))


def verify(cfg):
    cmd = [sys.executable, "-I", "-B", str(ROOT / "tools" / "verify_snapshot.py"), "--repo-root", str(ROOT)]
    if cfg.get("engine_repo") and Path(cfg["engine_repo"]).is_dir():
        cmd += ["--engine-repo", cfg["engine_repo"]]
    return run(cmd).decode().strip()


def stage_paths(paths):
    present = [p for p in paths if (ROOT / p).exists() or git("ls-files", "--", p).strip()]
    git("add", "-A", "--", *present)
    ignored = [line for line in git("ls-files", "--others", "--ignored", "--exclude-standard", "-z",
                                    "--", *present).decode().split("\0") if line]
    if ignored:                         # a nested .gitignore must not drop part of the snapshot
        git("add", "-f", "--", *ignored)
    return ignored


def dirty_managed():
    """Uncommitted changes to generated files (hand-kept README.md files and the config excepted)."""
    out = git("status", "--porcelain", "-z", "--untracked-files=all", "--",
              ENGINE_DIR, DOCS_DIR, RESEARCH_DIR, EVIDENCE_DIR).decode()
    return [r for r in out.split("\0") if r and not r.endswith("/" + KEEP)]


def blob_sizes(blobs):
    if not blobs:
        return 0
    out = subprocess.run(["git", "-C", str(ROOT), "cat-file", "--batch-check=%(objectsize)"],
                         input="\n".join(sorted(blobs)).encode() + b"\n", stdout=subprocess.PIPE,
                         check=True).stdout
    return sum(int(x) for x in out.split())


def git_summary(base_ref):
    index = {}                          # path -> (mode, blob)
    for record in git("ls-files", "-s", "-z").decode().split("\0"):
        if record:
            meta, path = record.split("\t", 1)
            mode, blob = meta.split()[:2]
            index[path] = (mode, blob)
    base = {}
    for record in git("ls-tree", "-r", "-z", base_ref).decode().split("\0"):
        if record:
            meta, path = record.split("\t", 1)
            mode, _kind, blob = meta.split()
            base[path] = (mode, blob)
    base_blobs = {blob for _mode, blob in base.values()}
    new_blobs = {blob for _mode, blob in index.values() if blob not in base_blobs}
    return {
        "index_files": len(index),
        "added": len(set(index) - set(base)),
        "removed": len(set(base) - set(index)),
        "changed": sum(1 for p in set(index) & set(base) if index[p] != base[p]),
        "tree_bytes": blob_sizes({blob for _mode, blob in index.values()}),
        "new_blob_bytes": blob_sizes(new_blobs),
        "shortstat": git("diff", "--cached", "--shortstat", base_ref).decode().strip(),
    }


# ------------------------------------------------------------------------------------- main


def load_config(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--engine-repo")
    p.add_argument("--engine-commit", help="engine revision to export (default: the config's)")
    p.add_argument("--engine-branch", help="branch name recorded with the commit")
    p.add_argument("--docs-dir")
    p.add_argument("--windows-dir")
    p.add_argument("--windows", nargs="+", metavar="W", help="the full window list, or 'all' finished ones")
    p.add_argument("--add-window", nargs="+", default=[], metavar="W")
    p.add_argument("--add-server-logs", nargs="+", default=[], metavar="W=REASON",
                   help="keep these windows' server logs (xz), with the README number they back")
    p.add_argument("--manifest-only", action="store_true", help="regenerate the manifest and verify")
    p.add_argument("--no-stage", action="store_true")
    p.add_argument("--no-wait", action="store_true", help="refuse instead of waiting for a GPU window")
    p.add_argument("--force", action="store_true", help="refresh over uncommitted changes in managed paths")
    p.add_argument("--base-ref", default="origin/main", help="what the summary compares against")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        os.nice(19)
    except OSError:
        pass
    cfg = load_config(args.config)
    for key in ("engine_repo", "engine_commit", "engine_branch", "docs_dir", "windows_dir"):
        value = getattr(args, key)
        if value:
            cfg[key] = value
    if args.windows:
        cfg["windows"] = args.windows
    for w in args.add_window:
        if w not in cfg["windows"] and cfg["windows"] != ["all"]:
            cfg["windows"].append(w)
    for item in args.add_server_logs:
        name, _, reason = item.partition("=")
        cfg.setdefault("server_log_windows", {})[name] = reason or "named on the command line"
        if name not in cfg["windows"] and cfg["windows"] != ["all"]:
            cfg["windows"].append(name)

    if git("rev-parse", "--abbrev-ref", "HEAD").decode().strip() != "ds41":
        print("note: the current branch is not ds41")
    wait_for_quiet_box(cfg, args.no_wait)

    if args.manifest_only:
        source = current_source()
        manifest = write_manifest(cfg, source)
        print(verify(cfg))
        if not args.no_stage:
            stage_paths([MANIFEST, "README.md", "STATUS.md"])
        print("manifest:", len(manifest["files_sha256"]), "files")
        return

    if not args.force:
        dirty = dirty_managed()
        if dirty:
            raise Refusal("uncommitted changes in managed paths (commit them or pass --force): "
                          + repr(dirty[:6]))

    stage = ROOT / STAGING
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        source = build_engine(cfg, stage)
        docs = build_copies(cfg, stage, "docs", DOCS_DIR)
        research = build_copies(cfg, stage, "research", RESEARCH_DIR)
        windows, running = finished_windows(cfg, cfg["windows"])
        for missing in set(cfg.get("server_log_windows", {})) - set(windows):
            raise Refusal("server_log_windows names a window not in the list: " + missing)
        index = build_evidence(cfg, stage, windows)

        published = {}
        for sub in (ENGINE_DIR, DOCS_DIR, RESEARCH_DIR, EVIDENCE_DIR):
            for rel, path in tree_files(stage / sub).items() if (stage / sub).exists() else ():
                published[sub + "/" + rel] = path
        for keep_dir in (DOCS_DIR, RESEARCH_DIR, EVIDENCE_DIR):
            kept = ROOT / keep_dir / KEEP
            if kept.is_file():
                published[keep_dir + "/" + KEEP] = kept
        for rel in ("README.md", "STATUS.md", "LICENSE", ".gitignore", "tools/verify_snapshot.py",
                    "tools/ds41_refresh.py", "tools/ds41_refresh.json", "tools/ds41_reply_check.py"):
            if (ROOT / rel).is_file():
                published[rel] = ROOT / rel
        fatal, engine_notes, info = scan_tree(stage, cfg, published)
        if fatal:
            for rel, what, detail in fatal:
                print("SCAN:", rel, "--", what, "--", detail, file=sys.stderr)
            raise Refusal(str(len(fatal)) + " scan finding(s) (a token format anywhere, or any finding"
                          " outside the exact engine tree); nothing moved")

        added = sum(p.stat().st_size for r, p in published.items() if not r.startswith(ENGINE_DIR + "/"))
        if added > int(cfg.get("max_added_bytes", 60 * MB)):
            raise Refusal("docs, research and evidence total " + human(added) + ", over max_added_bytes")

        # Move into place: the engine tree and the generated parts of docs/research/evidence.
        for sub in (ENGINE_DIR,):
            old = ROOT / sub
            if old.exists():
                shutil.rmtree(old)
            os.rename(stage / sub, old)
        for sub in (DOCS_DIR, RESEARCH_DIR, EVIDENCE_DIR):
            target = ROOT / sub
            readme = (target / KEEP).read_bytes() if (target / KEEP).is_file() else None
            if target.exists():
                shutil.rmtree(target)
            if (stage / sub).exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.rename(stage / sub, target)
            if readme is not None:
                target.mkdir(parents=True, exist_ok=True)
                (target / KEEP).write_bytes(readme)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if (ROOT / "build").is_dir() and not any((ROOT / "build").iterdir()):
            (ROOT / "build").rmdir()

    cfg["engine_commit"] = source["commit"]
    cfg["windows"] = sorted(set(windows) | set(running))   # a running window stays listed for next time
    write_json(args.config if args.config.is_absolute() else Path.cwd() / args.config, cfg)
    manifest = write_manifest(cfg, source)
    verified = verify(cfg)
    ignored = [] if args.no_stage else stage_paths(list(MANAGED) + ["README.md", "STATUS.md"])

    # ------------------------------------------------------------------------------ summary
    win = index["windows"]
    left = [x for w in win.values() for x in w["left_out"]]
    reasons = {}
    for x in left:
        r = reasons.setdefault(x["reason"], [0, 0])
        r[0] += 1
        r[1] += x.get("bytes", 0)
    print("engine   ", source["commit"], "(" + str(source.get("branch")) + "):", source["files"], "files,",
          human(source["bytes"]) + ",", source["executable_files"], "executable")
    print("docs     ", len(docs), "reports ->", DOCS_DIR + "/;", len(research), "research files ->", RESEARCH_DIR + "/")
    for name in unclassified_docs(cfg):
        print("          not classified in the config (left out):", name)
    print("evidence ", len(win), "windows,", sum(w["files"] for w in win.values()), "text files,",
          human(sum(w["bytes"] for w in win.values())) + ";",
          sum(w["server_logs"] for w in win.values()), "server logs kept from",
          sum(1 for w in win.values() if w["server_logs"]), "windows,",
          human(sum(w["server_log_bytes"] for w in win.values())), "->",
          human(sum(w["server_log_xz_bytes"] for w in win.values())), "xz")
    if running:
        print("          still running, left for a later refresh:", ", ".join(running))
    for reason, (count, size) in sorted(reasons.items(), key=lambda kv: -kv[1][1]):
        print("          left out:", count, "x", human(size) + ":", reason)
    print("scan     ", info["files"], "files,", human(info["bytes"]) + ": nothing outside radiance/;",
          info["home_path_files"], "files name local /home paths")
    for category, files in sorted(engine_notes.items()):
        sample = ", ".join(sorted(files)[:4]) + (" ..." if len(files) > 4 else "")
        print("          radiance/ (exact engine tree, kept):", category + ":", len(files), "files:", sample)
    if ignored:
        print("          force-added (matched a .gitignore):", ", ".join(ignored[:8]))
    print("manifest ", len(manifest["files_sha256"]), "files")
    print(verified)
    try:
        s = git_summary(args.base_ref)
        print("git      ", "staged:" if not args.no_stage else "worktree:", s["shortstat"])
        print("          against", args.base_ref + ":", s["added"], "added,", s["removed"], "removed,",
              s["changed"], "changed;", s["index_files"], "files,", human(s["tree_bytes"]),
              "in the tree,", human(s["new_blob_bytes"]), "in blobs", args.base_ref, "does not have")
    except Refusal as e:
        print("git summary skipped:", e)
    print("next: review, then git commit")


if __name__ == "__main__":
    try:
        main()
    except Refusal as error:
        print("REFUSED:", error, file=sys.stderr)
        sys.exit(1)
