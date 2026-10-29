"""The A/B's requests and its comparison (ab_window.sh).

  ab_requests.py RECEIPT URL N OUT.json   send the first N held-out requests of RECEIPT (the loop's
                                          receipt.json) to URL, one at a time (AB_CONCURRENCY=K:
                                          in waves of K at once); save each response
  ab_requests.py --compare DIR ARM...     per request: whether each arm's text equals the first
                                          arm's, and each arm's decode tokens/s

The quality harness for the lossy track (RESEARCH-3-LOSSY.md, "Recommended order" 3; ab_window.sh's
Lq part). Every distribution is the raw one, read from the logits before any sampler stage
(sampler.cpp's sample_logprobs), K = AB_LOGPROBS_K (20) alternatives a position:

  ab_requests.py --lp RECEIPT URL N OUT.json      the reference: the first N requests free-running
                                                  (greedy, as above) with every reply token's id,
                                                  its logprob and the top K at each position
  ab_requests.py --force REF.json URL OUT.json    teacher-forced decode: each of REF's prompts with a
                                                  grammar of REF's reply token ids (`<[id]> <[id]>
                                                  ...`, so tokenisation cannot drift), so every
                                                  position runs a decode pass over REF's own prefix
                                                  and reports the arm's distribution there; a reply
                                                  that stopped on its own is forced up to its stop
                                                  token, which the finished grammar leaves the model
                                                  to pick among the end-of-generation ids
  ab_requests.py --kl REF.json FORCED.json...     per position KL(ref || arm) over REF's top K plus a
                                                  tail bucket, top-1 agreement, |d logprob| of REF's
                                                  token, and bit-identical positions; overall, without
                                                  position 0 (a prefill pass's), and by prompt family
  ab_requests.py --kl-dir DIR                     --kl for every DIR/*.lpforce.json against the
                                                  reference it was forced along

Validation (exact arm against itself): every forced token followed, every position bit-identical
(ids, logprobs and top K), KL 0 and top-1 agreement 100%.
"""
import glob
import json
import math
import os
import sys
import threading
import time
import urllib.request

LP_K = int(os.environ.get("AB_LOGPROBS_K", "20"))


def post(url, path, body, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def framed(b, url):
    """The request's body as the server gets it: chat_text framed into token ids. A `messages`
    request goes through the chat endpoint (True)."""
    body = dict(b["request"])
    if "chat_text" in body:
        # DeepSeek-V4.1's chat framing around the server's own tokens for the text, as the
        # held-out prompts are built: BOS, <|User|>, the text, <|Assistant|>, </think>.
        text = body.pop("chat_text")
        ids = post(url, "/tokenize", {"content": text}, 60)["tokens"]
        body["prompt"] = [0, 128803] + ids + [128804, 128822]
    if os.environ.get("AB_MAX_TOKENS"):
        body["max_tokens"] = int(os.environ["AB_MAX_TOKENS"])
    return body, "messages" in body


def family(b):
    """A request's prompt family: the receipt's `domain` (the held-out prompts), else `family`,
    else its name without the trailing number (chat-03: chat)."""
    return b.get("domain") or b.get("family") or b["name"].rsplit("-", 1)[0]


def one(b, url):
    """Sends one request of the receipt; the row saved for it."""
    body, chat = framed(b, url)   # a chat request goes through the server's chat template
    t0 = time.time()
    try:
        resp = post(url, "/v1/chat/completions" if chat else "/v1/completions", body, 600)
    except Exception as e:  # recorded, the next request still runs
        resp = {"error": repr(e)}
    wall = time.time() - t0
    t = resp.get("timings", {})
    choice = resp.get("choices", [{}])[0] if "choices" in resp else None
    text = None if choice is None else (choice.get("message", {}).get("content") if chat else choice.get("text"))
    print("%s: %.1f s, decode %.2f tok/s, prefill %.0f ms%s" % (
        b["name"], wall, t.get("predicted_per_second") or 0.0, t.get("prompt_ms") or 0.0,
        "" if text is not None else " ERROR %s" % resp.get("error")), flush=True)
    return {"name": b["name"], "family": family(b), "wall_s": wall, "text": text, "timings": t,
            "error": resp.get("error")}


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


# ---- the quality harness -----------------------------------------------------------------------

def lp_rows(choice):
    """[[id, logprob, [[id, logprob], ...]], ...] from a completion's logprobs (its `content`)."""
    content = ((choice or {}).get("logprobs") or {}).get("content") or []
    return [[e["id"], e["logprob"], [[a["id"], a["logprob"]] for a in e.get("top_logprobs") or []]]
            for e in content]


def lp_run(receipt, url, n, out):
    """The reference: free-running greedy replies with logprobs (top LP_K) at every position."""
    with open(receipt) as f:
        reqs = json.load(f)["benchmark_requests"][:n]
    rows = []
    for b in reqs:
        row = {"name": b["name"], "family": family(b), "lp": [], "error": None}
        try:
            body, chat = framed(b, url)
            if chat:
                raise ValueError("a `messages` request: the harness sends token prompts only")
            body["logprobs"] = LP_K
            resp = post(url, "/v1/completions", body)
            ch = resp["choices"][0]
            row.update(prompt=body["prompt"], lp=lp_rows(ch), finish=ch.get("finish_reason"),
                       text=ch.get("text"), model=body.get("model"), timings=resp.get("timings", {}))
        except Exception as e:  # recorded; the next request still runs
            row["error"] = repr(e)
        print("%s: reference, %d tokens%s" % (row["name"], len(row["lp"]),
              "" if row["error"] is None else " ERROR " + row["error"]), flush=True)
        rows.append(row)
    with open(out, "w") as f:
        json.dump({"requests": rows, "logprobs": LP_K, "receipt": receipt}, f)


def force_grammar(ids):
    """GBNF for exactly these token ids, in order (core/sample/gbnf.cpp: `<[id]>` is one token)."""
    return "root ::= " + (" ".join("<[%d]>" % t for t in ids) if ids else '""')


def grammar_probe(ref, url):
    """Whether the server forces what the grammar names: the first reference reply's first position
    forced to its second-likeliest token (one the model would not pick), one token. An exact arm
    forced along its own reference would follow it even if the grammar were ignored; this cannot."""
    for r in ref["requests"]:
        if r.get("error") or not r.get("lp"):
            continue
        stop = r["lp"][-1][0] if r.get("finish") == "stop" else None
        alts = [t for t, _lp in r["lp"][0][2] if t != r["lp"][0][0] and t != stop]
        if not alts:
            continue
        body = {"prompt": r["prompt"], "max_tokens": 1, "temperature": 0, "stream": False,
                "logprobs": ref.get("logprobs", LP_K), "grammar": force_grammar([alts[0]])}
        if r.get("model"):
            body["model"] = r["model"]
        try:
            got = lp_rows(post(url, "/v1/completions", body)["choices"][0])
            res = {"name": r["name"], "token": alts[0], "got": got[0][0] if got else None}
        except Exception as e:
            res = {"name": r["name"], "token": alts[0], "got": None, "error": repr(e)}
        res["ok"] = res["got"] == alts[0]
        print("grammar probe (%s position 0 forced to %d): %s" % (
            r["name"], alts[0], "forced" if res["ok"] else "NOT FORCED, got %r" % res["got"]), flush=True)
        return res
    return None


def force_run(ref_path, url, out):
    """Each reference reply forced through decode passes: its prompt, a grammar of its token ids,
    max_tokens its length. A reply that stopped on its own ends with its stop token, which a
    grammar cannot name (an end-of-generation id is admitted only once the grammar completes):
    it is left out of the grammar, and the completed grammar leaves the model only the
    end-of-generation ids at that position, where the raw distribution is still reported."""
    with open(ref_path) as f:
        ref = json.load(f)
    probe = grammar_probe(ref, url)
    rows = []
    for r in ref["requests"]:
        row = {"name": r["name"], "family": r.get("family"), "lp": [], "error": None}
        if r.get("error") or not r.get("lp"):
            row["error"] = "no reference reply"
            rows.append(row)
            continue
        ids = [e[0] for e in r["lp"]]
        forced = ids[:-1] if r.get("finish") == "stop" else ids
        body = {"prompt": r["prompt"], "max_tokens": len(ids), "temperature": 0, "stream": False,
                "logprobs": ref.get("logprobs", LP_K), "grammar": force_grammar(forced)}
        if r.get("model"):
            body["model"] = r["model"]
        try:
            resp = post(url, "/v1/completions", body)
            ch = resp["choices"][0]
            row.update(lp=lp_rows(ch), finish=ch.get("finish_reason"), forced=len(forced),
                       timings=resp.get("timings", {}))
        except Exception as e:
            row["error"] = repr(e)
        follows = sum(1 for i, e in enumerate(row["lp"]) if i < len(ids) and e[0] == ids[i])
        print("%s: forced %d tokens, %d of %d positions follow the reference%s" % (
            row["name"], len(forced), follows, len(ids),
            "" if row["error"] is None else " ERROR " + row["error"]), flush=True)
        rows.append(row)
    with open(out, "w") as f:
        json.dump({"requests": rows, "logprobs": ref.get("logprobs", LP_K), "forced_along": ref_path,
                   "probe": probe}, f)


def dist(e):
    """{id: p} of a position's top list (its own token added), and the mass outside it."""
    d = {}
    for tid, lp in e[2]:
        d[tid] = max(d.get(tid, 0.0), math.exp(lp))
    if e[0] not in d:
        d[e[0]] = math.exp(e[1])
    return d, max(0.0, 1.0 - sum(d.values()))


def kl_topk(pe, qe):
    """KL(p || q) over p's listed tokens plus one tail bucket each side. A token q does not list
    gets min(q's smallest listed probability, q's tail mass): an estimate, slightly low when the
    lists disagree badly (with K = 20 on greedy replies the lists hold nearly all the mass). 0
    exactly when the two positions are identical."""
    if pe == qe:
        return 0.0
    p, pt = dist(pe)
    q, qt = dist(qe)
    floor = max(min(min(q.values()) if q else 1e-12, qt), 1e-12)
    s = 0.0
    for tid, pi in p.items():
        if pi > 0:
            s += pi * (math.log(pi) - math.log(max(q.get(tid, floor), 1e-12)))
    if pt > 1e-9:
        s += pt * (math.log(pt) - math.log(max(qt, 1e-12)))
    return max(0.0, s)


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def kl_compare(ref_path, forced_paths):
    """Each forced file against the reference it was forced along: overall, decode positions only
    (position 0 comes from the prompt's prefill pass), and by prompt family. Printed, and returned
    as {forced file: {"positions", "followed", "identical", "groups": {group: summary}}}."""
    with open(ref_path) as f:
        ref = {r["name"]: r for r in json.load(f)["requests"]}
    out = {}
    for fp in forced_paths:
        with open(fp) as f:
            arm = json.load(f)
        groups = {}
        followed = positions = identical = 0
        short = []
        for a in arm["requests"]:
            r = ref.get(a["name"])
            if not r or not r.get("lp") or a.get("error"):
                short.append(a["name"])
                continue
            rl, al = r["lp"], a["lp"]
            if len(al) != len(rl):
                short.append("%s (%d of %d)" % (a["name"], len(al), len(rl)))
            fam = a.get("family") or r.get("family") or "?"
            for i in range(min(len(rl), len(al))):
                k = kl_topk(rl[i], al[i])
                top_r = rl[i][2][0][0] if rl[i][2] else rl[i][0]
                top_a = al[i][2][0][0] if al[i][2] else al[i][0]
                cell = (k, int(top_r == top_a), abs(rl[i][1] - al[i][1]) if al[i][0] == rl[i][0] else None)
                for g in ("all", "decode" if i else "prefill", "family " + fam):
                    groups.setdefault(g, []).append(cell)
                positions += 1
                followed += int(al[i][0] == rl[i][0])
                identical += int(al[i] == rl[i])
        probe = arm.get("probe") or {}
        print("== %s against %s" % (fp, ref_path))
        print("  %d positions: %d follow the reference's token, %d bit-identical (ids, logprobs, top %s); "
              "grammar probe %s" % (positions, followed, identical, arm.get("logprobs", LP_K),
                                     "forced" if probe.get("ok") else "FAILED: %r" % probe))
        if short:
            print("  incomplete: " + ", ".join(short))
        print("  %-26s %6s %10s %10s %10s %10s %9s %10s" % (
            "positions", "n", "KL mean", "p50", "p90", "max", "top-1", "|d lp|"))
        order = ["all", "decode", "prefill"] + sorted(g for g in groups if g.startswith("family "))
        summary = {}
        for g in order:
            if g not in groups:
                continue
            cs = groups[g]
            ks = [c[0] for c in cs]
            dl = [c[2] for c in cs if c[2] is not None]
            summary[g] = dict(n=len(cs), kl_mean=sum(ks) / len(ks), kl_p50=pct(ks, .5),
                              kl_p90=pct(ks, .9), kl_max=max(ks),
                              top1=sum(c[1] for c in cs) / len(cs),
                              dlp=sum(dl) / len(dl) if dl else float("nan"))
            x = summary[g]
            print("  %-26s %6d %10.5f %10.5f %10.5f %10.4f %8.2f%% %10.5f" % (
                g, x["n"], x["kl_mean"], x["kl_p50"], x["kl_p90"], x["kl_max"], 100.0 * x["top1"], x["dlp"]))
        out[fp] = dict(positions=positions, followed=followed, identical=identical,
                       incomplete=short, groups=summary, probe_ok=bool(probe.get("ok")))
    return out


def kl_dir(d):
    """Every forced file in d, grouped by the reference each was forced along."""
    by_ref = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.lpforce.json"))):
        try:
            with open(fp) as f:
                by_ref.setdefault(json.load(f)["forced_along"], []).append(fp)
        except (OSError, ValueError, KeyError) as e:
            print("%s: %r" % (fp, e))
    for ref, fps in by_ref.items():
        kl_compare(ref, fps)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--compare":
        compare(a[1], a[2:])
    elif a[0] == "--lp":
        lp_run(a[1], a[2], int(a[3]), a[4])
    elif a[0] == "--force":
        force_run(a[1], a[2], a[3])
    elif a[0] == "--kl":
        kl_compare(a[1], a[2:])
    elif a[0] == "--kl-dir":
        kl_dir(a[1])
    else:
        run(a[0], a[1], int(a[2]), a[3])
