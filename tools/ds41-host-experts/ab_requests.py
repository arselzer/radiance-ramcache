"""The A/B's requests and its comparison (ab_window.sh).

  ab_requests.py RECEIPT URL N OUT.json   send the first N held-out requests of RECEIPT (the loop's
                                          receipt.json) to URL, one at a time (AB_CONCURRENCY=K:
                                          in waves of K at once); save each response
  ab_requests.py --compare DIR ARM...     per request: whether each arm's text equals the first
                                          arm's, and each arm's decode tokens/s
  ab_requests.py --kld-corpus OUT MAXCHARS RECEIPT...
                                          a --kld-corpus (JSONL) of the receipts' text prompts
                                          (chat_text, or a string prompt) of at most MAXCHARS
                                          characters, as plain text, every position scored
  ab_requests.py --kld-compare REF DIR    two --kld-record directories byte for byte: every
                                          position's log-probabilities, top entries and row values
                                          (exit 1 when any differ, naming the first positions)
"""
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


def kld_corpus(out, maxchars, receipts):
    """The receipts' text prompts as a --kld-corpus; returns the number of documents."""
    docs = []
    for path in receipts:
        for b in json.load(open(path))["benchmark_requests"]:
            r = b["request"]
            text = r.get("chat_text", r.get("prompt"))
            if isinstance(text, str) and 0 < len(text) <= maxchars:
                docs.append({"prompt": text, "source": b["name"]})
    with open(out, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print("kld corpus: %d documents, %d characters, from %s" % (
        len(docs), sum(len(d["prompt"]) for d in docs), " ".join(receipts)), flush=True)
    return len(docs)


def kld_compare(ref, cand):
    """Every file a --kld-record writes but its manifest (which holds times), position by position."""
    a, b = json.load(open(ref + "/kld.json")), json.load(open(cand + "/kld.json"))
    key = lambda m: (m["n_vocab"], m["top"], m["rows"], [(d["tokens"], d["score_from"], d["row0"]) for d in m["docs"]])
    if key(a) != key(b):
        print("kld DIFFER: %s and %s record different documents or shapes" % (ref, cand))
        return 1
    rows, docs = a["rows"], a["docs"]

    def where(r):
        for i, d in enumerate(docs):
            n = d["tokens"] - d["score_from"]
            if d["row0"] <= r < d["row0"] + n:
                return "doc %d (%s) position %d" % (i, d.get("source", ""), d["score_from"] + r - d["row0"])
        return "row %d" % r

    bad = 0
    if open(ref + "/tokens.i32", "rb").read() != open(cand + "/tokens.i32", "rb").read():
        print("kld DIFFER: tokens.i32 (the corpus tokenised differently)")
        bad += 1
    for name, width in (("logp.f16", a["n_vocab"] * 2), ("top.bin", a["top"] * 8), ("rows.f32", 8)):
        diff, first = 0, []
        with open(ref + "/" + name, "rb") as fa, open(cand + "/" + name, "rb") as fb:
            r = 0
            while r < rows:
                n = min(256, rows - r)
                xa, xb = fa.read(n * width), fb.read(n * width)
                if len(xa) != n * width or len(xb) != n * width:
                    print("kld DIFFER: %s is short" % name)
                    return 1
                if xa != xb:
                    for k in range(n):
                        if xa[k * width:(k + 1) * width] != xb[k * width:(k + 1) * width]:
                            diff += 1
                            if len(first) < 4:
                                first.append(where(r + k))
                r += n
        if diff:
            print("kld DIFFER: %s: %d of %d positions, first %s" % (name, diff, rows, "; ".join(first)))
            bad += 1
    if not bad:
        print("kld identical: %d positions over %d documents (logp.f16, top.bin, rows.f32, tokens.i32), %s against %s"
              % (rows, len(docs), cand, ref))
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "--kld-corpus":
        sys.exit(0 if kld_corpus(sys.argv[2], int(sys.argv[3]), sys.argv[4:]) > 0 else 1)
    elif sys.argv[1] == "--kld-compare":
        sys.exit(kld_compare(sys.argv[2], sys.argv[3]))
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
