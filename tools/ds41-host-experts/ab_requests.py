"""The A/B's requests and its comparison (ab_window.sh).

  ab_requests.py RECEIPT URL N OUT.json   send the first N held-out requests of RECEIPT (the loop's
                                          receipt.json) to URL, one at a time (AB_CONCURRENCY=K:
                                          in waves of K at once); save each response
  ab_requests.py --compare DIR ARM...     per request: whether each arm's text equals the first
                                          arm's, and each arm's decode tokens/s
  ab_requests.py --hashes DIR ARM...      per prompt, in the order the server ended their prefills:
                                          whether each arm's prefill hashes (RADIANCE_EXPERIMENTAL_DS41_
                                          PREFILL_HASH, the Kh part) equal the first arm's, every KV
                                          group of every rank and the last row's logits; on a
                                          difference, the groups, ranks and window layers that differ
"""
import re
import json
import os
import sys
import threading
import time
import urllib.request


def one(b, url):
    """Sends one request of the receipt; the row saved for it."""
    body = dict(b["request"])
    if "chat_text" in body:
        # DeepSeek-V4.1's chat framing around the server's own tokens for the text, as the
        # held-out prompts are built: BOS, <|User|>, the text, <|Assistant|>, </think>.
        text = body.pop("chat_text")
        tq = urllib.request.Request(url + "/tokenize", data=json.dumps({"content": text}).encode(),
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(tq, timeout=60) as r:
            ids = json.load(r)["tokens"]
        body["prompt"] = [0, 128803] + ids + [128804, 128822]
    if os.environ.get("AB_MAX_TOKENS"):
        body["max_tokens"] = int(os.environ["AB_MAX_TOKENS"])
    t0 = time.time()
    chat = "messages" in body   # a chat request goes through the server's chat template
    req = urllib.request.Request(url + ("/v1/chat/completions" if chat else "/v1/completions"),
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
    except Exception as e:  # recorded, the next request still runs
        resp = {"error": repr(e)}
    wall = time.time() - t0
    t = resp.get("timings", {})
    choice = resp.get("choices", [{}])[0] if "choices" in resp else None
    text = None if choice is None else (choice.get("message", {}).get("content") if chat else choice.get("text"))
    print("%s: %.1f s, decode %.2f tok/s, prefill %.0f ms%s" % (
        b["name"], wall, t.get("predicted_per_second") or 0.0, t.get("prompt_ms") or 0.0,
        "" if text is not None else " ERROR %s" % resp.get("error")), flush=True)
    return {"name": b["name"], "wall_s": wall, "text": text, "timings": t, "error": resp.get("error")}


def run(receipt, url, n, out):
    """One request at a time, or with AB_CONCURRENCY=K in waves of K sent together; each wave's
    wall time and generated tokens are kept for the aggregate rate."""
    reqs = json.load(open(receipt))["benchmark_requests"][:n]
    k = max(1, int(os.environ.get("AB_CONCURRENCY", "1")))
    rows, waves = [None] * len(reqs), []
    for w in range(0, len(reqs), k):
        idx = list(range(w, min(len(reqs), w + k)))
        t0 = time.time()
        if k == 1:
            rows[idx[0]] = one(reqs[idx[0]], url)
        else:
            def go(i):
                rows[i] = one(reqs[i], url)
            th = [threading.Thread(target=go, args=(i,)) for i in idx]
            for x in th: x.start()
            for x in th: x.join()
        waves.append({"wall_s": time.time() - t0,
                      "tokens": sum((rows[i]["timings"].get("predicted_n") or 0) for i in idx)})
    if k > 1:
        tok = sum(x["tokens"] for x in waves); wall = sum(x["wall_s"] for x in waves)
        print("aggregate: %d tokens in %.1f s of waves of %d: %.2f tok/s" % (tok, wall, k, tok / wall if wall else 0.0), flush=True)
    stats = None
    try:
        with urllib.request.urlopen(url + "/stats", timeout=30) as r:
            stats = json.load(r)
    except Exception as e:
        stats = {"error": repr(e)}
    json.dump({"requests": rows, "waves": waves, "concurrency": k, "stats": stats}, open(out, "w"), indent=1)


def compare(d, arms):
    data = {}
    for a in arms:
        try:
            data[a] = {r["name"]: r for r in json.load(open("%s/%s.json" % (d, a)))["requests"]}
        except OSError:
            print("arm %s: no results" % a)
    if not data:
        return
    first = arms[0]
    names = list(data.get(first, next(iter(data.values()))).keys())
    print("%-12s" % "request" + "".join("%18s" % a for a in data))
    tot = {a: [0.0, 0] for a in data}
    for n in names:
        cells = []
        for a in data:
            r = data[a].get(n)
            if not r or r["text"] is None:
                cells.append("%18s" % "-")
                continue
            same = r["text"] == data[first][n]["text"] if n in data[first] else False
            tps = r["timings"].get("predicted_per_second") or 0.0
            tot[a][0] += r["timings"].get("predicted_ms") or 0.0
            tot[a][1] += r["timings"].get("predicted_n") or 0
            cells.append("%12.2f %5s" % (tps, "same" if same else "DIFF"))
        print("%-12s" % n + "".join(cells))
    print("%-12s" % "decode" + "".join(
        "%12.2f tok/s" % (tot[a][1] / tot[a][0] * 1e3 if tot[a][0] else 0.0) for a in data))


_ALL = re.compile(r"prefill hash: seq (\d+), (\d+) tokens: all ranks ([0-9a-f]{16})")
_RANK = re.compile(r"prefill hash: seq (\d+), (\d+) tokens \(([^)]*)\), rank (\d+): (.*)$")
_LAYERS = re.compile(r"prefill hash: seq (\d+) rank (\d+) (\S+) by layer: (.*)$")


def hash_log(path):
    """One record a prompt, in log order: tokens, the digest over every rank, and per rank its
    groups' hashes (name -> hex) and the window group's hashes by layer."""
    out, cur = [], {}
    for line in open(path, errors="replace"):
        m = _RANK.search(line)
        if m:
            seq, rank = m.group(1), int(m.group(4))
            groups = dict(g.split("=", 1) for g in m.group(5).split() if "=" in g)
            cur.setdefault(seq, {"ranks": {}, "layers": {}, "cut": {}})
            cur[seq]["ranks"][rank] = groups
            cur[seq]["cut"][rank] = m.group(3)
            continue
        m = _LAYERS.search(line)
        if m:
            seq, rank = m.group(1), int(m.group(2))
            cur.setdefault(seq, {"ranks": {}, "layers": {}, "cut": {}})
            cur[seq]["layers"][rank] = dict(x.split(":", 1) for x in m.group(4).split())
            continue
        m = _ALL.search(line)
        if m:
            seq = m.group(1)
            rec = cur.pop(seq, {"ranks": {}, "layers": {}, "cut": {}})
            rec.update({"seq": seq, "tokens": int(m.group(2)), "all": m.group(3)})
            out.append(rec)
    return out


def hashes(d, arms):
    """Each arm's prompts against the first arm's, by position: the same tokens and digest, or what
    differs. Exit status 1 when anything differs, 2 when an arm has no hashes."""
    logs = {}
    for a in arms:
        try:
            logs[a] = hash_log("%s/%s.server.log" % (d, a))
        except OSError:
            logs[a] = []
        if not logs[a]:
            print("arm %s: no prefill hashes (run it with the Kh part)" % a)
    arms = [a for a in arms if logs[a]]
    if len(arms) < 2:
        return 2
    first = arms[0]
    ref = logs[first]
    bad = 0
    print("prefill hashes against %s: %d prompts" % (first, len(ref)))
    for a in arms[1:]:
        got = logs[a]
        same = 0
        for i, r in enumerate(ref):
            if i >= len(got):
                print("  %s: prompt %d (%d tokens) missing" % (a, i, r["tokens"]))
                bad += 1
                continue
            g = got[i]
            if g["tokens"] != r["tokens"]:
                print("  %s: prompt %d is %d tokens, %s's %d: not the same prompts" %
                      (a, i, g["tokens"], first, r["tokens"]))
                bad += 1
                continue
            if g["all"] == r["all"]:
                same += 1
                continue
            bad += 1
            diffs = []
            for rank in sorted(r["ranks"]):
                for name, h in r["ranks"][rank].items():
                    if g["ranks"].get(rank, {}).get(name) != h:
                        diffs.append("rank %d %s" % (rank, name))
                lr, lg = r["layers"].get(rank, {}), g["layers"].get(rank, {})
                ls = [l for l in lr if lg.get(l) != lr[l]]
                if ls:
                    diffs.append("rank %d window layers %s" % (rank, ",".join(ls)))
            print("  %s: prompt %d (%d tokens, %s): DIFFERS: %s" %
                  (a, i, r["tokens"], g["cut"].get(0, "?"), "; ".join(diffs) or "the digest"))
        cut = sum(1 for g in got if g["cut"].get(0, "").startswith("cut"))
        print("  %s: %d of %d prompts identical (%d of its prefills cut)" %
              (a, same, len(ref), cut))
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "--hashes":
        sys.exit(hashes(sys.argv[2], sys.argv[3:]))
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
