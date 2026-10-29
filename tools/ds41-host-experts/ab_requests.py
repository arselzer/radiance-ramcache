"""The A/B's requests and its comparison (ab_window.sh).

  ab_requests.py RECEIPT URL N OUT.json   send the first N held-out requests of RECEIPT (the loop's
                                          receipt.json) to URL, one at a time; save each response
  ab_requests.py --compare DIR ARM...     per request: whether each arm's text equals the first
                                          arm's, and each arm's decode tokens/s
"""
import json
import os
import sys
import time
import urllib.request


def run(receipt, url, n, out):
    reqs = json.load(open(receipt))["benchmark_requests"][:n]
    rows = []
    for b in reqs:
        body = dict(b["request"])
        if os.environ.get("AB_MAX_TOKENS"):
            body["max_tokens"] = int(os.environ["AB_MAX_TOKENS"])
        t0 = time.time()
        req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.load(r)
        except Exception as e:  # recorded, the next request still runs
            resp = {"error": repr(e)}
        wall = time.time() - t0
        t = resp.get("timings", {})
        text = resp.get("choices", [{}])[0].get("text") if "choices" in resp else None
        rows.append({"name": b["name"], "wall_s": wall, "text": text, "timings": t,
                     "error": resp.get("error")})
        print("%s: %.1f s, decode %.2f tok/s, prefill %.0f ms%s" % (
            b["name"], wall, t.get("predicted_per_second") or 0.0, t.get("prompt_ms") or 0.0,
            "" if text is not None else " ERROR %s" % resp.get("error")), flush=True)
    stats = None
    try:
        with urllib.request.urlopen(url + "/stats", timeout=30) as r:
            stats = json.load(r)
    except Exception as e:
        stats = {"error": repr(e)}
    json.dump({"requests": rows, "stats": stats}, open(out, "w"), indent=1)


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


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3:])
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
