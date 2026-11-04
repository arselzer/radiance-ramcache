"""The A/B's requests and its comparison (ab_window.sh).

  ab_requests.py RECEIPT URL N OUT.json   send the first N held-out requests of RECEIPT (the loop's
                                          receipt.json) to URL, one at a time (AB_CONCURRENCY=K:
                                          in waves of K at once); save each response
  ab_requests.py --compare DIR ARM...     per request: whether each arm's text equals that of the
                                          first arm that sent the request (by name), and each arm's
                                          decode tokens/s (arms with no request in common, a U
                                          part's receipt beside the held-out prompts, are compared
                                          apart)
  ab_requests.py --ttft DIR ARM...        per request: each arm's time to first token, the prompt
                                          tokens the prefix cache supplied, and whether its text
                                          equals the first arm's (prefix-cache A/Bs; grouped the
                                          same way), then repeats within an arm and replies sent
                                          back
  ab_requests.py --hashes DIR ARM...      per prompt, in the order the server ended their prefills:
                                          whether each arm's prefill hashes (RADIANCE_EXPERIMENTAL_DS41_
                                          PREFILL_HASH, the Kh part) equal the first arm's, every KV
                                          group of every rank and the last row's logits; on a
                                          difference, the groups, ranks and window layers that differ
  ab_requests.py --hashes-ref KHREF DIR ARM...
                                          the same against a reference recorded in another run (a
                                          <name>-khref.json, make_longctx_receipts.py), and each of
                                          its requests' replies against the reference's text
  ab_requests.py --kld-corpus OUT MAXCHARS RECEIPT...
                                          a --kld-corpus (JSONL) of the receipts' text prompts
                                          (chat_text, or a string prompt) of at most MAXCHARS
                                          characters, as plain text, every position scored
  ab_requests.py --draft-stats DIR A B     per request: DSpark's draft_n and draft_n_accepted and the
                                          text of arm B against arm A (exit 1 on any difference): a
                                          Dk arm against the same arm without Dk, whose exact draft
                                          GEMV must propose the same tokens
  ab_requests.py --acceptance DIR ARM...  DSpark arms by prompt set: each arm's drafts verified and accepted
                                          (draft_n, draft_n_accepted summed over its requests), the
                                          acceptance and its ratio to the set's first arm with the same
                                          Ds and Db parts (r: a masked draft's share of the unmasked
                                          one's at the same verify cap and block), tokens a step
                                          (predicted_n / (predicted_n - accepted)) and decode tokens/s,
                                          and whether its texts equal the first arm's
  ab_requests.py --kld-compare REF DIR    two --kld-record directories byte for byte: every
                                          position's log-probabilities, top entries and row values
                                          (exit 1 when any differ, naming the first positions)

Each timed request (RECEIPT URL N OUT) times out after AB_REQUEST_TIMEOUT_S seconds (600 by default;
ab_window.sh raises it for a receipt that declares long prompts, its max_context_tokens). A receipt's
`provenance` may give each request's prompt_tokens (the model's tokenizer, make_longctx_receipts.py):
the server's count is printed beside it.

A request is a completion body, with its prompt as text or token ids, or one of:
  chat_text   one user message in DeepSeek-V4.1's chat framing (BOS, <|User|>, the text,
              <|Assistant|>, </think>), the text tokenized by the server;
  chat_turns  a conversation in the same framing (the official encoding's chat mode): BOS, then per
              turn <|User|> user <|Assistant|></think>, and after every turn but the last its reply
              and <|end_of_sentence|>. A reply is "assistant" text, or "assistant_from": the name of
              an earlier request of the receipt, whose reply in this run is used, as a chat client
              sends the model's own answer back. Every text is tokenized on its own, so each turn's
              prompt starts with the previous turn's, token for token. The prompt ids are saved.
              `chat_system` beside it: a system prompt, right after BOS (an agent harness's).

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
import hashlib
import json
import math
import os
import re
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


def request_timeout():
    """The timed requests' timeout in seconds: AB_REQUEST_TIMEOUT_S, 600 when unset or empty (a
    100K-token prefill takes ~5 min at large chunks, a 32K one ~15 min at 16-token passes, so
    ab_window.sh sets it for a receipt of long prompts). Anything but a positive number raises
    ValueError."""
    v = os.environ.get("AB_REQUEST_TIMEOUT_S", "").strip()
    if not v:
        return 600.0
    t = float(v)
    if not (math.isfinite(t) and t > 0):
        raise ValueError("AB_REQUEST_TIMEOUT_S=%r is not a positive number of seconds" % v)
    return t


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


def chat_prompt(turns, url, replies, sent_back=None, system=None):
    """The token ids of a chat_turns conversation (see the module's docstring). `replies`: earlier
    requests' rows by name. For each reply sent back, `sent_back` gets {from, ids, generated}: the
    reply text's own token count against the tokens the server generated for it. Equal counts
    mean the turn most likely repeats the generated ids, so its prompt runs on through the blocks
    the earlier request's decode steps wrote (the prefix cache's chain); fewer or more mean
    re-tokenization changed them and the chain parts there. `system`: a system prompt (the
    request's chat_system), right after BOS, where the official encoding puts it."""
    ids = [BOS]
    if system:
        ids += tokenize(url, system)
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


def framed(b, url, prompt_ids=None):
    """The request's body as the server gets it: chat_text framed into token ids, a chat_turns
    conversation as `prompt_ids` (chat_prompt's ids for it). A `messages` request goes through the
    chat endpoint (True)."""
    body = dict(b["request"])
    if "chat_text" in body:
        # DeepSeek-V4.1's chat framing around the server's own tokens for the text, as the
        # held-out prompts are built: BOS, <|User|>, the text, <|Assistant|>, </think>.
        text = body.pop("chat_text")
        ids = tokenize(url, text)
        body["prompt"] = [0, 128803] + ids + [128804, 128822]
    if "chat_turns" in body:
        if prompt_ids is None:
            raise ValueError("a chat_turns request: its prompt needs the earlier replies (one())")
        body.pop("chat_turns")
        body.pop("chat_system", None)
        body["prompt"] = prompt_ids
    if os.environ.get("AB_MAX_TOKENS"):
        body["max_tokens"] = int(os.environ["AB_MAX_TOKENS"])
    return body, "messages" in body


def family(b):
    """A request's prompt family: the receipt's `domain` (the held-out prompts), else `family`,
    else its name without the trailing number (chat-03: chat)."""
    return b.get("domain") or b.get("family") or b["name"].rsplit("-", 1)[0]


def one(b, url, replies=None, timeout=600.0):
    """Sends one request of the receipt; the row saved for it. `replies`: earlier requests' rows
    by name, for chat_turns' assistant_from; `timeout` in seconds (request_timeout())."""
    prompt_ids = None
    sent_back = []
    if "chat_turns" in b["request"]:
        try:
            prompt_ids = chat_prompt(b["request"]["chat_turns"], url, replies or {}, sent_back,
                                     b["request"].get("chat_system"))
        except Exception as e:  # recorded, the next request still runs
            print("%s: ERROR building the prompt: %r" % (b["name"], e), flush=True)
            return {"name": b["name"], "family": family(b), "wall_s": 0.0, "text": None, "timings": {},
                    "error": repr(e)}
    body, chat = framed(b, url, prompt_ids)   # a chat request goes through the server's chat template
    t0 = time.time()
    try:
        resp = post(url, "/v1/chat/completions" if chat else "/v1/completions", body, timeout)
    except Exception as e:  # recorded, the next request still runs
        resp = {"error": repr(e)}
    wall = time.time() - t0
    t = resp.get("timings", {})
    choice = resp.get("choices", [{}])[0] if "choices" in resp else None
    text = None if choice is None else (choice.get("message", {}).get("content") if chat else choice.get("text"))
    print("%s: %.1f s, decode %.2f tok/s, prefill %.0f ms%s%s" % (
        b["name"], wall, t.get("predicted_per_second") or 0.0, t.get("prompt_ms") or 0.0,
        # The server's prompt_n is the prompt tokens it computed: the prompt less the cached ones.
        ", %d of %d prompt tokens cached" % (t.get("cache_n") or 0, prompt_tokens(t) or 0)
        if t.get("cache_n") else "",
        "" if text is not None else " ERROR %s" % resp.get("error")), flush=True)
    row = {"name": b["name"], "family": family(b), "wall_s": wall, "text": text, "timings": t,
           "error": resp.get("error")}
    if prompt_ids is not None:
        row["prompt_ids"] = prompt_ids
    if sent_back:
        row["sent_back"] = sent_back
    return row


def run(receipt, url, n, out):
    """One request at a time, or with AB_CONCURRENCY=K in waves of K sent together; each wave's
    wall time and generated tokens are kept for the aggregate rate."""
    with open(receipt) as f:
        rec = json.load(f)
    reqs = rec["benchmark_requests"][:n]
    k = max(1, int(os.environ.get("AB_CONCURRENCY", "1")))
    try:
        timeout = request_timeout()
    except ValueError as e:
        sys.exit("ab_requests.py: %s" % e)
    if os.environ.get("AB_REQUEST_TIMEOUT_S", "").strip():
        print("requests time out after %g s (AB_REQUEST_TIMEOUT_S)" % timeout, flush=True)
    rows, waves = [None] * len(reqs), []
    replies = {}   # earlier requests' rows by name, for chat_turns' assistant_from
    for w in range(0, len(reqs), k):
        idx = list(range(w, min(len(reqs), w + k)))
        t0 = time.time()
        if k == 1:
            rows[idx[0]] = one(reqs[idx[0]], url, replies, timeout)
        else:
            def go(i):
                rows[i] = one(reqs[i], url, dict(replies), timeout)
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
    # The receipt's own count of each prompt (provenance, the model's tokenizer) beside the server's.
    want = {q.get("name"): q.get("prompt_tokens")
            for q in ((rec.get("provenance") or {}).get("requests") or []) if isinstance(q, dict)}
    for r in rows:
        if want.get(r["name"]) is None:
            continue
        got = prompt_tokens(r["timings"] or {})
        print("%s: %s prompt tokens by the server's count, %d by the receipt's (%s)" % (
            r["name"], "no" if got is None else got, want[r["name"]],
            "same" if got == want[r["name"]] else "DIFFERENT"), flush=True)
    stats = None
    try:
        with urllib.request.urlopen(url + "/stats", timeout=30) as r:
            stats = json.load(r)
    except Exception as e:
        stats = {"error": repr(e)}
    with open(out, "w") as f:
        json.dump({"requests": rows, "waves": waves, "concurrency": k, "request_timeout_s": timeout, "stats": stats},
                  f, indent=1)


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
    """Arms grouped by the requests they sent, in the order given: arms whose request names overlap,
    directly or through another arm, are one group (a verify arm that sent the first three
    held-out prompts goes with the arms that sent all eight); only arms with no name in common are
    apart (a U part's receipt beside the held-out prompts). A group's names come in the order its
    arms first sent them. An arm with no requests is left out."""
    groups = []   # [names, their set, arms]
    for a, rows in data.items():
        if not rows:
            print("arm %s: no requests" % a)
            continue
        mine = set(rows)
        hit = [g for g in groups if g[1] & mine]
        if hit:
            g = hit[0]
            for h in hit[1:]:   # this arm's names join groups that were apart
                g[0] += [n for n in h[0] if n not in g[1]]
                g[1] |= h[1]
                g[2] += h[2]
                groups.remove(h)
        else:
            g = [[], set(), []]
            groups.append(g)
        g[0] += [n for n in rows if n not in g[1]]
        g[1] |= mine
        g[2].append(a)
    order = {a: i for i, a in enumerate(data)}
    return [(tuple(g[0]), sorted(g[2], key=order.get)) for g in groups]


def ref_arm(data, cols, n):
    """The arm request n is compared against: the first of `cols` that sent it."""
    return next(a for a in cols if n in data[a])


def compare(d, arms):
    """Per request, each arm's text against the first arm that sent that request (by name), and
    each arm's decode tokens/s; arms with no request name in common are compared apart."""
    data, _ = load(d, arms)
    groups = by_receipt(data)
    for names, cols in groups:
        if len(groups) > 1:
            print("arms sending %s .. %s (%d requests), each against the first arm that sent it (%s):" % (
                names[0], names[-1], len(names), cols[0]))
        print("%-12s" % "request" + "".join("%18s" % a for a in cols))
        tot = {a: [0.0, 0] for a in cols}
        for n in names:
            ref = data[ref_arm(data, cols, n)][n]
            cells = []
            for a in cols:
                r = data[a].get(n)
                if not r or r["text"] is None:
                    cells.append("%18s" % "-")
                    continue
                same = r["text"] == ref["text"]
                tps = r["timings"].get("predicted_per_second") or 0.0
                tot[a][0] += r["timings"].get("predicted_ms") or 0.0
                tot[a][1] += r["timings"].get("predicted_n") or 0
                cells.append("%12.2f %5s" % (tps, "same" if same else "DIFF"))
            print("%-12s" % n + "".join(cells))
        print("%-12s" % "decode" + "".join(
            "%12.2f tok/s" % (tot[a][1] / tot[a][0] * 1e3 if tot[a][0] else 0.0) for a in cols))


def prompt_tokens(t):
    """A request's prompt tokens from its timings: the server's prompt_n counts only the ones it
    computed (the prompt less cache_n, core/server/server.cpp timings_of), so the two together. None
    without timings."""
    if t.get("prompt_n") is None:
        return None
    return (t.get("prompt_n") or 0) + (t.get("cache_n") or 0)


def ttft(d, arms):
    """Prefix-cache A/Bs, per group of arms that sent the same requests (by_receipt), each request
    against the group's first arm that sent it (put the arm without Pc first: every request cold).
    Per request and arm: the time to first token, the prompt tokens the cache supplied (cache_n),
    and the text: 'same' or 'DIFF' against that arm, 'pdif' where a chat_turns prompt differs from its (an
    earlier reply that differed was sent back, so the texts are not comparable; the first DIFF
    above it is the one to read). Then per arm: the sums; repeats (requests with the same prompt
    ids in one arm, e.g. a cold request and its hit) whose texts differ, want none; replies sent
    back whose own tokens are not the count the server generated (the chain parts there, so the
    next turn adopts fewer of the decode steps' blocks); and the /stats prefix counters."""
    data, stats = load(d, arms)
    for names, cols in by_receipt(data):
        print("%d requests (%s .. %s), each against the first arm that sent it ([0] when it did):" % (
            len(names), names[0], names[-1]))
        for i, a in enumerate(cols):
            print("[%d] %s" % (i, a))
        print("%-14s %7s" % ("request", "prompt") +
              "".join("  [%d] ttft_s cached text" % i for i in range(len(cols))))
        tot = {a: [0.0, 0, 0, 0] for a in cols}   # ttft s, cached tokens, same texts, requests
        for n in names:
            p = None
            cells = []
            ref = data[ref_arm(data, cols, n)][n]
            for a in cols:
                r = data[a].get(n)
                if not r or r.get("text") is None:
                    cells.append("%24s" % "-")
                    continue
                t = r.get("timings") or {}
                p = p if p is not None else prompt_tokens(t)
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
        # Per session (a receipt family: a conversation, e.g. the harness receipt's hxA / hxB):
        # turns, the prompt tokens sent, and per arm the TTFT summed, the tokens the cache
        # supplied and the share of the prompt tokens they are, and the turns whose text is the
        # first arm's. A session's later turns are what a harness pays for.
        fams = []
        for n in names:
            f = next((data[a][n].get("family") for a in cols if n in data[a]), None)
            if f is not None and f not in fams:
                fams.append(f)
        if len(fams) > 1 or any(data[a][n].get("prompt_ids") for a in cols for n in names if n in data[a]):
            print("%-14s %7s %8s" % ("session", "turns", "prompt") +
                  "".join("  [%d] ttft_s  cached  share same" % i for i in range(len(cols))))
            for f in fams:
                ns = [n for n in names if any(n in data[a] and data[a][n].get("family") == f for a in cols)]
                prompt = 0
                cells = []
                for a in cols:
                    st, sc, sp, ss, k = 0.0, 0, 0, 0, 0   # ttft, cached and prompt tokens, same, turns
                    for n in ns:
                        r = data[a].get(n)
                        if not r or r.get("text") is None:
                            continue
                        t = r.get("timings") or {}
                        st += (t.get("ttft_ms") if t.get("ttft_ms") is not None else t.get("prompt_ms") or 0.0) / 1e3
                        sc += t.get("cache_n") or 0
                        sp += prompt_tokens(t) or 0
                        ss += r["text"] == data[ref_arm(data, cols, n)][n].get("text")
                        k += 1
                    prompt = max(prompt, sp)
                    cells.append("  %9.2f %7d %5.1f%% %2d/%-2d" % (st, sc, 100.0 * sc / sp if sp else 0.0, ss, k))
                print("%-14s %7d %8d" % (f, len(ns), prompt) + "".join(cells))
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
            if s.get("eors") is not None or s.get("snapshot_store") is not None:
                print("[%d] eors %s snapshot_store %s" % (
                    i, json.dumps(s.get("eors")), json.dumps(s.get("snapshot_store"))))


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


def acceptance(d, arms):
    """DSpark arms grouped by the requests they sent (by_receipt): per arm the drafts verified and
    accepted over its requests, the acceptance, its ratio to the group's first arm with the same Ds
    and Db parts (the verify cap and query block), tokens a step and decode tokens/s, and its texts
    against the first arm's
    (they must be the same: the target verifies every draft). 1 when a text differs."""
    data, _ = load(d, arms)
    bad = 0

    def cap(a):   # the verify cap and the query block: the Ds part and any DbN part
        return tuple(p for p in a.split("-") if re.fullmatch(r"Ds[0-9a]?|Db[0-9]+", p))

    for names, cols in by_receipt(data):
        print("DSpark acceptance, %d requests (%s .. %s); r against the first arm with the same Ds and Db parts:" % (
            len(names), names[0], names[-1]))
        print("  %-8s %-9s %7s %7s %6s %10s %7s  %s" % ("drafts", "accepted", "accept", "r", "tok/st", "tok/s", "texts", "arm"))
        ref = {}
        for a in cols:
            dn = acc = pn = 0
            pms = 0.0
            same = 0
            for n in names:
                r = data[a].get(n)
                if not r or r.get("text") is None:
                    continue
                t = r.get("timings") or {}
                dn += t.get("draft_n") or 0
                acc += t.get("draft_n_accepted") or 0
                pn += t.get("predicted_n") or 0
                pms += t.get("predicted_ms") or 0.0
                same += r["text"] == data[ref_arm(data, cols, n)][n]["text"]
            rate = acc / dn if dn else float("nan")
            ref_acc = ref.setdefault(cap(a), rate)
            ratio = rate / ref_acc if dn and ref_acc and ref_acc == ref_acc else float("nan")
            tps = pn / pms * 1e3 if pms else 0.0
            step = pn / (pn - acc) if pn > acc else float("nan")
            sent = sum(1 for n in names if data[a].get(n))
            bad += same != sent
            print("  %-8d %-9d %6.1f%% %7.3f %6.3f %10.2f %3d/%-3d  %s" % (dn, acc, 100.0 * rate, ratio, step, tps,
                                                                       same, sent, a))
    return 1 if bad else 0


def draft_stats(d, a, b):
    """Arm b's requests against arm a's, by name: the drafter's draft_n and draft_n_accepted and the
    reply text must be the same (the K1 review's draft_stats_cmp.py). 1 on any difference."""
    data, _ = load(d, [a, b])
    if a not in data or b not in data:
        return 1
    bad = 0
    for n, x in data[a].items():
        y = data[b].get(n)
        if y is None:
            print("  %s: not in %s" % (n, b))
            bad += 1
            continue
        tx, ty = x.get("timings") or {}, y.get("timings") or {}
        sx = (tx.get("draft_n"), tx.get("draft_n_accepted"))
        sy = (ty.get("draft_n"), ty.get("draft_n_accepted"))
        same = x.get("text") == y.get("text")
        ok = same and sx == sy and sx[0] is not None
        bad += not ok
        print("  %s: drafts %s/%s against %s/%s, text %s%s" % (n, sy[0], sy[1], sx[0], sx[1],
              "same" if same else "DIFFERS", "" if ok else "  <-- MISMATCH"))
    extra = [n for n in data[b] if n not in data[a]]
    if extra:
        print("  not in %s: %s" % (a, ", ".join(extra)))
        bad += 1
    print("draft stats %s against %s: %s" % (b, a, "PASS" if not bad else "FAIL, %d request(s)" % bad))
    return 1 if bad else 0


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
    """Every forced file in d, grouped by the reference each was forced along (a path as the window
    wrote it, relative to the tools directory; else the reference beside the forced file)."""
    by_ref = {}
    for fp in sorted(glob.glob(os.path.join(d, "*.lpforce.json"))):
        try:
            with open(fp) as f:
                ref = json.load(f)["forced_along"]
            if not os.path.exists(ref):
                ref = os.path.join(d, os.path.basename(ref))
            by_ref.setdefault(ref, []).append(fp)
        except (OSError, ValueError, KeyError) as e:
            print("%s: %r" % (fp, e))
    for ref, fps in by_ref.items():
        kl_compare(ref, fps)


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


def hash_logs(d, arms):
    """Each arm's hash_log records (none for an arm without a log or hashes, said so)."""
    logs = {}
    for a in arms:
        try:
            logs[a] = hash_log("%s/%s.server.log" % (d, a))
        except OSError:
            logs[a] = []
        if not logs[a]:
            print("arm %s: no prefill hashes (run it with the Kh part)" % a)
    return logs


def hashes(d, arms):
    """Each arm's prompts against the first arm's, by position: the same tokens and digest, or what
    differs. Requests sent together (CK) reach the server in an order of their own in each arm, so a
    prompt whose position differs is matched to the arm's prompt of the same tokens and digest
    elsewhere in its log (each used once; counted as reordered). A prompt that matches none is then
    compared with one of the arm's prompts of its length that matched none either (the one at its own
    position first, each used once), naming the groups, ranks and window layers that differ. Exit
    status 1 when anything differs, 2 when an arm has no hashes."""
    logs = hash_logs(d, arms)
    arms = [a for a in arms if logs[a]]
    if len(arms) < 2:
        return 2
    return hash_compare(arms[0], logs[arms[0]], arms[1:], logs)


def hash_compare(first, ref, arms, logs):
    """`arms`' records (logs) against `ref`, the records of `first` (an arm, or a reference), as
    hashes() describes; 1 when anything differs, else 0."""
    bad = 0
    print("prefill hashes against %s: %d prompts" % (first, len(ref)))
    for a in arms:
        got = logs[a]
        same = 0
        moved = 0
        used = [False] * len(got)
        unmatched = []
        for i, r in enumerate(ref):
            j = i if (i < len(got) and not used[i] and got[i]["tokens"] == r["tokens"] and
                      got[i]["all"] == r["all"]) else None
            if j is None:
                j = next((k for k, g in enumerate(got) if not used[k] and g["tokens"] == r["tokens"] and
                          g["all"] == r["all"]), None)
                moved += j is not None
            if j is not None:
                used[j] = True
                same += 1
            else:
                unmatched.append(i)
        # What differs, once every match is taken: in a reordered arm the prompt at a position is often
        # another one, so each unmatched prompt is compared with an unmatched one of its length.
        for i in unmatched:
            r = ref[i]
            bad += 1
            cands = [k for k, g in enumerate(got) if not used[k] and g["tokens"] == r["tokens"]]
            if not cands:
                left = [str(g["tokens"]) for k, g in enumerate(got) if not used[k]]
                if left:
                    print("  %s: prompt %d (%d tokens): no unmatched prompt of that length (%s tokens "
                          "left): not the same prompts" % (a, i, r["tokens"], ", ".join(left)))
                else:
                    print("  %s: prompt %d (%d tokens) missing" % (a, i, r["tokens"]))
                continue
            j = i if i in cands else cands[0]
            used[j] = True
            g = got[j]
            diffs = []
            for rank in sorted(r["ranks"]):
                for name, h in r["ranks"][rank].items():
                    if g["ranks"].get(rank, {}).get(name) != h:
                        diffs.append("rank %d %s" % (rank, name))
                lr, lg = r["layers"].get(rank, {}), g["layers"].get(rank, {})
                ls = [l for l in lr if lg.get(l) != lr[l]]
                if ls:
                    diffs.append("rank %d window layers %s" % (rank, ",".join(ls)))
            print("  %s: prompt %d (%d tokens, %s%s): DIFFERS: %s" %
                  (a, i, r["tokens"], g["cut"].get(0, "?"), "" if j == i else "; its prompt %d" % j,
                   "; ".join(diffs) or "the digest"))
        # Each prefill as rank 0's line names it: "cut for N", "not cut", or "not cut: shared step" (a
        # whole prompt admitted beside others into one step, which the consumer skip runs uncut).
        cut = sum(1 for g in got if g["cut"].get(0, "").startswith("cut"))
        shared = sum(1 for g in got if g["cut"].get(0, "").startswith("not cut: shared"))
        print("  %s: %d of %d prompts identical (%d of its prefills cut%s%s)" %
              (a, same, len(ref), cut, ", %d uncut in a shared step" % shared if shared else "",
               ", %d matched out of order" % moved if moved else ""))
    return 1 if bad else 0


def load_khref(path):
    """A prefill-hash reference (<name>-khref.json): the file and its prompts as hash_log's records,
    each with the request's name and reply text."""
    with open(path) as f:
        k = json.load(f)
    if k.get("schema") != "ds41-prefill-hash-reference-v1":
        raise ValueError("%s: schema %r" % (path, k.get("schema")))
    recs = []
    for p in k["prompts"]:
        recs.append({"name": p.get("name"), "text": p.get("text"), "tokens": int(p["tokens"]), "all": p["all"],
                     "ranks": {int(r): g for r, g in (p.get("ranks") or {}).items()},
                     "layers": {int(r): v for r, v in (p.get("layers") or {}).items()},
                     "cut": {int(r): c for r, c in (p.get("cut") or {}).items()}})
    return k, recs


def hashes_ref(path, d, arms):
    """Each arm's prefill hashes against a reference recorded in another run (load_khref), as
    hashes() compares two arms, and each reference request's reply against the arm's (its
    <arm>.json row of that name). The reference names its receipt (beside it) by sha256: a receipt
    that changed since makes the reference stale. Exit status 1 when anything differs, 2 when no
    arm has hashes."""
    k, ref = load_khref(path)
    bad = 0
    rp = os.path.join(os.path.dirname(os.path.abspath(path)), k.get("receipt") or "")
    if k.get("receipt"):
        try:
            with open(rp, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            sha = None
        if sha != k.get("receipt_sha256"):
            print("reference %s: made for %s with sha256 %s, which now has %s: stale" %
                  (path, k["receipt"], str(k.get("receipt_sha256"))[:16], str(sha)[:16]))
            bad = 1
    logs = hash_logs(d, arms)
    have = [a for a in arms if logs[a]]
    if not have:
        return 2
    bad |= hash_compare("the reference %s" % os.path.basename(path), ref, have, logs)
    for a in have:
        try:
            with open("%s/%s.json" % (d, a)) as f:
                rows = {r["name"]: r for r in json.load(f)["requests"]}
        except (OSError, ValueError, KeyError):
            rows = {}
        same = 0
        named = [r for r in ref if r.get("name") and r.get("text") is not None]
        for r in named:
            got = (rows.get(r["name"]) or {}).get("text")
            if got == r["text"]:
                same += 1
            else:
                bad = 1
                print("  %s: %s: reply %s" % (a, r["name"], "missing" if got is None else "DIFFERS: %r" % got[:80]))
        if named:
            print("  %s: %d of %d replies identical to the reference's" % (a, same, len(named)))
    return 1 if bad else 0


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
    elif a[0] == "--hashes":
        sys.exit(hashes(a[1], a[2:]))
    elif a[0] == "--hashes-ref":
        sys.exit(hashes_ref(a[1], a[2], a[3:]))
    elif a[0] == "--ttft":
        ttft(a[1], a[2:])
    elif a[0] == "--draft-stats":
        sys.exit(draft_stats(a[1], a[2], a[3]))
    elif a[0] == "--acceptance":
        sys.exit(acceptance(a[1], a[2:]))
    elif a[0] == "--kld-corpus":
        sys.exit(0 if kld_corpus(a[1], int(a[2]), a[3:]) > 0 else 1)
    elif a[0] == "--kld-compare":
        sys.exit(kld_compare(a[1], a[2]))
    else:
        run(a[0], a[1], int(a[2]), a[3])
