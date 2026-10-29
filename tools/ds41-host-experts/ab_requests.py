"""The A/B's requests and its comparison (ab_window.sh).

  ab_requests.py RECEIPT URL N OUT.json   send the first N held-out requests of RECEIPT (the loop's
                                          receipt.json) to URL, one at a time (AB_CONCURRENCY=K:
                                          in waves of K at once); save each response
  ab_requests.py --compare DIR ARM...     per request: whether each arm's text equals the first
                                          arm's, and each arm's decode tokens/s (arms that sent
                                          other requests, a U part's receipt, are compared apart,
                                          against the first of their own)
  ab_requests.py --ttft DIR ARM...        per request: each arm's time to first token, the prompt
                                          tokens the prefix cache supplied, and whether its text
                                          equals the first arm's (prefix-cache A/Bs; grouped the
                                          same way), then repeats within an arm and replies sent
                                          back

A request is a completion body, with its prompt as text or token ids, or one of:
  chat_text   one user message in DeepSeek-V4.1's chat framing (BOS, <|User|>, the text,
              <|Assistant|>, </think>), the text tokenized by the server;
  chat_turns  a conversation in the same framing (the official encoding's chat mode): BOS, then per
              turn <|User|> user <|Assistant|></think>, and after every turn but the last its reply
              and <|end_of_sentence|>. A reply is "assistant" text, or "assistant_from": the name of
              an earlier request of the receipt, whose reply in this run is used, as a chat client
              sends the model's own answer back. Every text is tokenized on its own, so each turn's
              prompt starts with the previous turn's, token for token. The prompt ids are saved.
"""
import json
import os
import sys
import threading
import time
import urllib.request

# DeepSeek-V4.1's chat tokens, as the held-out prompts use them.
BOS, USER, ASSISTANT, END_THINK = 0, 128803, 128804, 128822
_eos = {}


def tokenize(url, text):
    """The server's tokens for `text` (no BOS)."""
    tq = urllib.request.Request(url + "/tokenize", data=json.dumps({"content": text}).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(tq, timeout=60) as r:
        return json.load(r)["tokens"]


def eos_id(url):
    """<|end_of_sentence|>: the special token's own id (the server parses special tokens in
    /tokenize's content), else the tokenizer's EOS."""
    if url not in _eos:
        ids = tokenize(url, "<｜end▁of▁sentence｜>")
        eos = ids[0] if len(ids) == 1 else None
        if eos is None:
            try:
                with urllib.request.urlopen(url + "/get_tokenizer_info", timeout=60) as r:
                    eos = json.load(r).get("eos_token_id")
            except Exception:
                eos = None
        _eos[url] = eos if isinstance(eos, int) else 1
        print("chat_turns: <|end_of_sentence|> is token %d" % _eos[url], flush=True)
    return _eos[url]


def chat_prompt(turns, url, replies, sent_back=None):
    """The token ids of a chat_turns conversation (see the module's docstring). `replies`: earlier
    requests' rows by name. For each reply sent back, `sent_back` gets {from, ids, generated}: the
    reply text's own token count against the tokens the server generated for it. Equal counts
    mean the turn most likely repeats the generated ids, so its prompt runs on through the blocks
    the earlier request's decode steps wrote (the prefix cache's chain); fewer or more mean
    re-tokenization changed them and the chain parts there."""
    ids = [BOS]
    for i, t in enumerate(turns):
        ids += [USER] + tokenize(url, t["user"]) + [ASSISTANT, END_THINK]
        if "assistant" in t:
            ids += tokenize(url, t["assistant"]) + [eos_id(url)]
        elif "assistant_from" in t:
            prev = replies.get(t["assistant_from"])
            if prev is None or prev.get("text") is None:
                raise ValueError("turn %d: no reply of %s in this run" % (i, t["assistant_from"]))
            rid = tokenize(url, prev["text"])
            if sent_back is not None:
                sent_back.append({"from": t["assistant_from"], "ids": len(rid),
                                  "generated": (prev.get("timings") or {}).get("predicted_n")})
            ids += rid + [eos_id(url)]
        elif i + 1 < len(turns):
            raise ValueError("turn %d of %d has no reply" % (i, len(turns)))
    return ids


def one(b, url, replies=None):
    """Sends one request of the receipt; the row saved for it. `replies`: earlier requests' rows
    by name, for chat_turns' assistant_from."""
    body = dict(b["request"])
    prompt_ids = None
    sent_back = []
    if "chat_text" in body:
        # DeepSeek-V4.1's chat framing around the server's own tokens for the text, as the
        # held-out prompts are built: BOS, <|User|>, the text, <|Assistant|>, </think>.
        text = body.pop("chat_text")
        ids = tokenize(url, text)
        body["prompt"] = [0, 128803] + ids + [128804, 128822]
    if "chat_turns" in body:
        try:
            prompt_ids = chat_prompt(body.pop("chat_turns"), url, replies or {}, sent_back)
        except Exception as e:  # recorded, the next request still runs
            print("%s: ERROR building the prompt: %r" % (b["name"], e), flush=True)
            return {"name": b["name"], "wall_s": 0.0, "text": None, "timings": {}, "error": repr(e)}
        body["prompt"] = prompt_ids
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
    print("%s: %.1f s, decode %.2f tok/s, prefill %.0f ms%s%s" % (
        b["name"], wall, t.get("predicted_per_second") or 0.0, t.get("prompt_ms") or 0.0,
        ", %d of %d prompt tokens cached" % (t.get("cache_n") or 0, t.get("prompt_n") or 0)
        if t.get("cache_n") else "",
        "" if text is not None else " ERROR %s" % resp.get("error")), flush=True)
    row = {"name": b["name"], "wall_s": wall, "text": text, "timings": t, "error": resp.get("error")}
    if prompt_ids is not None:
        row["prompt_ids"] = prompt_ids
    if sent_back:
        row["sent_back"] = sent_back
    return row


def run(receipt, url, n, out):
    """One request at a time, or with AB_CONCURRENCY=K in waves of K sent together; each wave's
    wall time and generated tokens are kept for the aggregate rate."""
    reqs = json.load(open(receipt))["benchmark_requests"][:n]
    k = max(1, int(os.environ.get("AB_CONCURRENCY", "1")))
    rows, waves = [None] * len(reqs), []
    replies = {}   # earlier requests' rows by name, for chat_turns' assistant_from
    for w in range(0, len(reqs), k):
        idx = list(range(w, min(len(reqs), w + k)))
        t0 = time.time()
        if k == 1:
            rows[idx[0]] = one(reqs[idx[0]], url, replies)
        else:
            def go(i):
                rows[i] = one(reqs[i], url, dict(replies))
            th = [threading.Thread(target=go, args=(i,)) for i in idx]
            for x in th: x.start()
            for x in th: x.join()
        for i in idx:
            if rows[i]["text"] is not None:
                replies[rows[i]["name"]] = rows[i]
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


def load(d, arms):
    """Each arm's rows by request name, and its /stats, in the order given."""
    data, stats = {}, {}
    for a in arms:
        try:
            j = json.load(open("%s/%s.json" % (d, a)))
        except (OSError, ValueError):
            print("arm %s: no results" % a)
            continue
        data[a] = {r["name"]: r for r in j["requests"]}
        stats[a] = j.get("stats") or {}
    return data, stats


def by_receipt(data):
    """Arms grouped by the requests they sent (a U part's receipt, or the held-out prompts), in the
    order given; texts are compared within a group, against its first arm."""
    groups = []
    for a, rows in data.items():
        for g in groups:
            if g[0] == tuple(rows):
                g[1].append(a)
                break
        else:
            groups.append((tuple(rows), [a]))
    return groups


def compare(d, arms):
    data, _ = load(d, arms)
    for names, cols in by_receipt(data):
        first = cols[0]
        if len(data) > len(cols):
            print("arms sending %s .. %s (%d requests), against %s:" % (names[0], names[-1],
                                                                        len(names), first))
        print("%-12s" % "request" + "".join("%18s" % a for a in cols))
        tot = {a: [0.0, 0] for a in cols}
        for n in names:
            cells = []
            for a in cols:
                r = data[a].get(n)
                if not r or r["text"] is None:
                    cells.append("%18s" % "-")
                    continue
                same = r["text"] == data[first][n]["text"]
                tps = r["timings"].get("predicted_per_second") or 0.0
                tot[a][0] += r["timings"].get("predicted_ms") or 0.0
                tot[a][1] += r["timings"].get("predicted_n") or 0
                cells.append("%12.2f %5s" % (tps, "same" if same else "DIFF"))
            print("%-12s" % n + "".join(cells))
        print("%-12s" % "decode" + "".join(
            "%12.2f tok/s" % (tot[a][1] / tot[a][0] * 1e3 if tot[a][0] else 0.0) for a in cols))


def ttft(d, arms):
    """Prefix-cache A/Bs, per group of arms that sent the same requests (by_receipt), against the
    group's first arm (put the arm without Pc first: every request cold). Per request and arm: the
    time to first token, the prompt tokens the cache supplied (cache_n), and the text: 'same' or
    'DIFF' against the first arm, 'pdif' where a chat_turns prompt differs from the first arm's (an
    earlier reply that differed was sent back, so the texts are not comparable; the first DIFF
    above it is the one to read). Then per arm: the sums; repeats (requests with the same prompt
    ids in one arm, e.g. a cold request and its hit) whose texts differ, want none; replies sent
    back whose own tokens are not the count the server generated (the chain parts there, so the
    next turn adopts fewer of the decode steps' blocks); and the /stats prefix counters."""
    data, stats = load(d, arms)
    for names, cols in by_receipt(data):
        first = cols[0]
        print("%d requests (%s .. %s), against [0]:" % (len(names), names[0], names[-1]))
        for i, a in enumerate(cols):
            print("[%d] %s" % (i, a))
        print("%-14s %7s" % ("request", "prompt") +
              "".join("  [%d] ttft_s cached text" % i for i in range(len(cols))))
        tot = {a: [0.0, 0, 0, 0] for a in cols}   # ttft s, cached tokens, same texts, requests
        for n in names:
            p = None
            cells = []
            ref = data[first][n]
            for a in cols:
                r = data[a].get(n)
                if not r or r.get("text") is None:
                    cells.append("%24s" % "-")
                    continue
                t = r.get("timings") or {}
                p = p if p is not None else t.get("prompt_n")
                s = (t.get("ttft_ms") if t.get("ttft_ms") is not None else t.get("prompt_ms") or 0.0) / 1e3
                c = t.get("cache_n") or 0
                same = r["text"] == ref.get("text")
                mark = "same" if same else "DIFF"
                if not same and r.get("prompt_ids") is not None and \
                        r.get("prompt_ids") != ref.get("prompt_ids"):
                    mark = "pdif"
                tot[a][0] += s; tot[a][1] += c; tot[a][2] += same; tot[a][3] += 1
                cells.append("  %9.2f %6d %4s" % (s, c, mark))
            print("%-14s %7s" % (n, p if p is not None else "-") + "".join(cells))
        print("%-14s %7s" % ("sum", "") + "".join(
            "  %9.2f %6d %2d/%-2d" % (tot[a][0], tot[a][1], tot[a][2], tot[a][3]) for a in cols))
        for i, a in enumerate(cols):
            rows = [data[a][n] for n in names if n in data[a]]
            seen, reps = {}, []
            for r in rows:
                if r.get("prompt_ids") is None or r.get("text") is None:
                    continue
                k = tuple(r["prompt_ids"])
                if k in seen:
                    reps.append("%s=%s %s" % (r["name"], seen[k]["name"],
                                              "same" if r["text"] == seen[k]["text"] else "DIFF"))
                else:
                    seen[k] = r
            back = ["%s<-%s %s/%s" % (r["name"], x["from"], x["ids"], x["generated"])
                    for r in rows for x in r.get("sent_back") or []]
            print("[%d] repeats: %s; replies sent back (own tokens/generated): %s" % (
                i, ", ".join(reps) or "-", ", ".join(back) or "-"))
            s = stats.get(a) or {}
            print("[%d] linear_hit_rate %s linear_cached_tokens %s checkpoints_written %s hit_cuts %s "
                  "hit_cut_tokens %s publish_held_tokens %s prefix_evictions %s ckpt_evictions %s "
                  "ckpt_refused %s" % (
                      i, s.get("linear_hit_rate"), s.get("linear_cached_tokens"),
                      s.get("checkpoints_written"), json.dumps(s.get("hit_cuts")),
                      s.get("hit_cut_tokens"), s.get("publish_held_tokens"),
                      s.get("prefix_evictions"), json.dumps(s.get("ckpt_evictions")),
                      s.get("ckpt_refused")))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "--ttft":
        ttft(sys.argv[2], sys.argv[3:])
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
