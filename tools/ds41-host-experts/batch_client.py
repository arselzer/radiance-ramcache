#!/usr/bin/env python3
"""Sends a batch of prompts to a radiance server (serve-batch.sh), N at a time, and writes the replies.

  batch_client.py PROMPTS.jsonl OUT.jsonl [--url http://127.0.0.1:8000] [--concurrency 256] [--max-tokens 128]
                  [--chat]

PROMPTS.jsonl: one JSON object a line, {"prompt": "...", "system": "...", "id": "..."}; `system` and `id` are
optional. Put the part prompts share (a schema, instructions) in `system`, the same text in every line that shares
it: the prefix cache then prefills it once and the other streams resume after it (the measured batches shared a
few long heads). A line may also give its own "max_tokens" and "model". OUT.jsonl gets {"id", "text",
"completion_tokens", "error"} a line, in input order.

--receipt reads PROMPTS as an ab_window.sh receipt instead (JSON with benchmark_requests; --first N of them):
a single-turn chat_turns request with its chat_system, or a chat_text one, becomes a line with the request's
name as its id and its max_tokens (AB_MAX_TOKENS overrides, as in ab_requests.py) and model, framed as
ab_requests.py frames it; any other request is refused. --prompts-out FILE keeps the lines made.

By default a prompt is framed as the measured runs framed it, DeepSeek-V4.1's non-thinking chat form in token ids
(BOS, the system text, <|User|>, the prompt, <|Assistant|>, </think>; the text tokenized by the server's /tokenize),
and sent to /v1/completions, greedy. --chat sends OpenAI chat messages to /v1/chat/completions instead, which needs
a container with a chat template baked in (rad-convert --chat-template; without one the server answers 501). Python
standard library only.
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.request

BOS, USER, ASSISTANT, END_THINK = 0, 128803, 128804, 128822


def post(url, path, body, timeout):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Framer:
    """Token ids for (system, prompt); each distinct system text tokenized once, before the batch starts."""

    def __init__(self, url, systems):
        self.url = url
        self.heads = {t: self.tokens(t) for t in systems if t}

    def tokens(self, text):
        return post(self.url, "/tokenize", {"content": text}, 120)["tokens"]

    def ids(self, system, prompt):
        head = self.heads.get(system, []) if system else []
        return [BOS] + head + [USER] + self.tokens(prompt) + [ASSISTANT, END_THINK]


def from_receipt(path, first):
    """A receipt's first `first` requests as prompt lines (see the module's docstring); ValueError otherwise."""
    with open(path) as f:
        reqs = json.load(f)["benchmark_requests"]
    rows = []
    for b in reqs[:first] if first else reqs:
        q = b["request"]
        row = {"id": b["name"]}
        if "chat_turns" in q and len(q["chat_turns"]) == 1 and set(q["chat_turns"][0]) == {"user"}:
            row["prompt"] = q["chat_turns"][0]["user"]
            if q.get("chat_system"):
                row["system"] = q["chat_system"]
        elif "chat_text" in q:
            row["prompt"] = q["chat_text"]
        else:
            raise ValueError("%s: not a single-turn chat_turns or chat_text request" % b["name"])
        for k in ("max_tokens", "model"):
            if k in q:
                row[k] = q[k]
        extra = {k: v for k, v in q.items() if k not in ("chat_turns", "chat_system", "chat_text", "max_tokens", "model")}
        if extra:
            row["extra"] = extra   # the request's other fields (temperature, stream, ...) as ab_requests.py sends them
        if os.environ.get("AB_MAX_TOKENS"):
            row["max_tokens"] = int(os.environ["AB_MAX_TOKENS"])
        rows.append(row)
    return rows


def one(row, args, framer):
    try:
        if args.chat:
            msgs = ([{"role": "system", "content": row["system"]}] if row.get("system") else []) + \
                   [{"role": "user", "content": row["prompt"]}]
            body = {"model": row.get("model", args.model), "messages": msgs,
                    "max_tokens": row.get("max_tokens", args.max_tokens), "temperature": 0,
                    "stream": False, "enable_thinking": False}
            resp = post(args.url, "/v1/chat/completions", body, args.timeout)
            text = resp["choices"][0]["message"]["content"]
        else:
            body = {"model": row.get("model", args.model), "prompt": framer.ids(row.get("system"), row["prompt"]),
                    "max_tokens": row.get("max_tokens", args.max_tokens), "temperature": 0, "stream": False}
            body.update({k: v for k, v in (row.get("extra") or {}).items() if k != "prompt"})
            resp = post(args.url, "/v1/completions", body, args.timeout)
            text = resp["choices"][0]["text"]
        n = (resp.get("usage") or {}).get("completion_tokens")
        if n is None:
            n = (resp.get("timings") or {}).get("predicted_n")
        return {"id": row.get("id"), "text": text, "completion_tokens": n, "error": None}
    except Exception as e:  # recorded; the other requests go on
        return {"id": row.get("id"), "text": None, "completion_tokens": None, "error": repr(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=256, help="requests in flight (serve-batch.sh --streams)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--model", default="ds41", help="the served model name (serve-batch.sh: ds41)")
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds a request may take")
    ap.add_argument("--chat", action="store_true", help="OpenAI chat messages to /v1/chat/completions")
    ap.add_argument("--receipt", action="store_true", help="PROMPTS is an ab_window.sh receipt")
    ap.add_argument("--first", type=int, default=0, help="with --receipt: its first N requests (0: all)")
    ap.add_argument("--prompts-out", help="write the prompt lines sent as JSONL")
    args = ap.parse_args()
    if args.receipt:
        try:
            rows = from_receipt(args.prompts, args.first)
        except (KeyError, ValueError) as e:
            print("batch_client.py: %s" % e)
            return 2
    else:
        with open(args.prompts) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    if args.prompts_out:
        with open(args.prompts_out, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    for i, r in enumerate(rows):
        r.setdefault("id", i)
    framer = None if args.chat else Framer(args.url, {r.get("system") for r in rows})
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        out = list(ex.map(lambda r: one(r, args, framer), rows))
    wall = time.time() - t0
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    toks = sum(r["completion_tokens"] or 0 for r in out)
    errs = sum(1 for r in out if r["error"])
    print("%d prompts in %.1f s, %d tokens generated (%.1f tok/s over the wall), %d errors"
          % (len(rows), wall, toks, toks / wall if wall > 0 else 0.0, errs))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
