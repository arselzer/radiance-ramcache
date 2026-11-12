#!/usr/bin/env python3
"""batch_client.py against a stub server on the loopback (no radiance, no model): the framing in token ids the
measured runs used (BOS, system, <|User|>, prompt, <|Assistant|>, </think>), each system text tokenized once, the
requests in flight together, the replies in input order, and an error recorded without stopping the batch."""
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def tok(text):
    return [1000 + ord(c) for c in text]


class Stub(http.server.BaseHTTPRequestHandler):
    seen, lock, live, most, tokenized = [], threading.Lock(), [0], [0], []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            with self.lock:
                self.tokenized.append(body["content"])
            out = {"tokens": tok(body["content"])}
        else:
            with self.lock:
                self.live[0] += 1
                self.most[0] = max(self.most[0], self.live[0])
                self.seen.append((self.path, body))
            time.sleep(0.2)
            with self.lock:
                self.live[0] -= 1
            if body["prompt"][-3] == tok("x")[0]:
                self.send_response(500); self.end_headers(); return
            out = {"choices": [{"text": "reply %d" % len(body["prompt"])}], "usage": {"completion_tokens": 7}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % srv.server_address[1]
    fail = 0

    def check(cond, what):
        nonlocal fail
        print(("ok   " if cond else "FAIL ") + what)
        fail |= not cond

    with tempfile.TemporaryDirectory() as d:
        rows = [{"system": "SCHEMA", "prompt": "q%d" % i} for i in range(8)] + [{"prompt": "x"}]
        with open(os.path.join(d, "in.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "batch_client.py"), os.path.join(d, "in.jsonl"),
                            os.path.join(d, "out.jsonl"), "--url", url, "--concurrency", "4", "--max-tokens", "16"],
                           capture_output=True, text=True, timeout=60)
        check(r.returncode == 1 and "9 prompts" in r.stdout and "1 errors" in r.stdout, "exit 1 with one error: %s" % r.stdout.strip())
        out = [json.loads(l) for l in open(os.path.join(d, "out.jsonl"))]
        check([o["id"] for o in out] == list(range(9)), "replies in input order")
        check(all(o["text"] == "reply %d" % (1 + 6 + 1 + 2 + 2) and o["completion_tokens"] == 7 for o in out[:8]),
              "replies and token counts")
        check(out[8]["text"] is None and out[8]["error"], "the failed request recorded")
    first = [b for p, b in Stub.seen if b["prompt"][-4:-2] == tok("q0")][0]
    check(first["prompt"] == [0] + tok("SCHEMA") + [128803] + tok("q0") + [128804, 128822],
          "the measured framing in token ids")
    check(first["temperature"] == 0 and first["max_tokens"] == 16 and first["model"] == "ds41", "greedy, max_tokens, model")
    check(all(p == "/v1/completions" for p, _ in Stub.seen), "sent to /v1/completions")
    check(Stub.tokenized.count("SCHEMA") == 1, "the shared system text tokenized once")
    check([b for p, b in Stub.seen if b["prompt"][:2] == [0, 128803]] != [], "no system: BOS then <|User|>")
    check(Stub.most[0] == 4, "4 requests in flight at most and at once (%d)" % Stub.most[0])
    srv.shutdown()
    print("batch_client_test: all passed" if not fail else "batch_client_test: FAILED")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
