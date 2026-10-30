"""ab_requests_test.py -- the quality harness (ab_requests.py --lp, --force, --kl) against a fake
server that decodes greedily from a deterministic toy model and honours a token-id grammar the way
the engine's GBNF does (`<[id]>` terminals; an end-of-generation id only once the grammar is done).

  python3 -I ab_requests_test.py [-v]

The validation the window runs on the exact arm, here offline: forced along its own reference, an
exact model follows every token with every position bit-identical, KL 0 and top-1 agreement 100%;
a perturbed model still follows every forced token (teacher forcing does not depend on the arm's
own choice), with KL above 0 and top-1 agreement below 100%.
"""
import http.server
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_requests  # noqa: E402

V, EOS = 300, 1


def logits(prefix, noise):
    """A toy model: deterministic in the prefix; EOS once the reply is 9 tokens long when the prompt
    asks for it (its first id odd)."""
    n, last = len(prefix), prefix[-1]
    z = [5.0 * math.sin(v * 0.37 + n * 1.3 + last * 0.11) + noise * math.cos(v * 1.7 + n) for v in range(V)]
    z[EOS] = -50.0
    return z


class Model:
    def __init__(self, noise=0.0):
        self.noise = noise
        self.grammars = []

    def complete(self, body):
        prompt = list(body["prompt"])
        k = body.get("logprobs") or 0
        g = body.get("grammar")
        forced = None
        if g is not None:
            self.grammars.append(g)
            m = re.fullmatch(r'root ::= ((?:<\[\d+\]> ?)*|"")', g)
            assert m, g
            forced = [int(x) for x in re.findall(r"<\[(\d+)\]>", g)]
        out, content, finish = [], [], "length"
        for i in range(body["max_tokens"]):
            z = logits(prompt + out, self.noise)
            if prompt[0] % 2 and len(out) == 9:
                z[EOS] = 50.0
            mx = max(z)
            lse = mx + math.log(sum(math.exp(x - mx) for x in z))
            lp = [x - lse for x in z]
            if forced is not None and i < len(forced):
                tok = forced[i]
            elif forced is not None:
                tok = EOS   # the grammar is complete: only end-of-generation ids remain
            else:
                tok = max(range(V), key=lambda v: (z[v], -v))
            top = sorted(range(V), key=lambda v: (-z[v], v))[:k]
            content.append({"id": tok, "logprob": lp[tok],
                            "top_logprobs": [{"id": v, "logprob": lp[v]} for v in top]})
            out.append(tok)
            if tok == EOS:
                finish = "stop"
                break
        return {"choices": [{"text": " ".join(map(str, out)), "finish_reason": finish,
                             "logprobs": {"content": content} if k else None}],
                "timings": {"predicted_n": len(out), "predicted_ms": 10.0 * len(out)}}


def serve(model):
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                out = {"tokens": [ord(c) % 250 + 2 for c in body["content"]]}
            else:
                out = model.complete(body)
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


class Harness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="abreq-test-")
        cls.receipt = os.path.join(cls.dir, "receipt.json")
        reqs = [{"name": "heldout-01", "domain": "sql",
                 "request": {"prompt": [3, 10, 20], "max_tokens": 16, "temperature": 0, "stream": False}},
                {"name": "heldout-02", "domain": "code",
                 "request": {"prompt": [4, 11, 21, 31], "max_tokens": 12, "temperature": 0, "stream": False}},
                {"name": "chat-01",
                 "request": {"chat_text": "Explain.", "max_tokens": 8, "temperature": 0, "stream": False}}]
        with open(cls.receipt, "w") as f:
            json.dump({"benchmark_requests": reqs}, f)
        cls.exact, cls.noisy = Model(), Model(noise=0.6)
        cls.srv, cls.url = serve(cls.exact)
        cls.srv2, cls.url2 = serve(cls.noisy)
        cls.ref = os.path.join(cls.dir, "lpref.json")
        ab_requests.lp_run(cls.receipt, cls.url, 3, cls.ref)
        cls.same = os.path.join(cls.dir, "exact.lpforce.json")
        cls.other = os.path.join(cls.dir, "noisy.lpforce.json")
        ab_requests.force_run(cls.ref, cls.url, cls.same)
        ab_requests.force_run(cls.ref, cls.url2, cls.other)

    @classmethod
    def tearDownClass(cls):
        for srv in (cls.srv, cls.srv2):
            srv.shutdown()
            srv.server_close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_the_reference(self):
        with open(self.ref) as f:
            ref = json.load(f)["requests"]
        self.assertEqual([r["family"] for r in ref], ["sql", "code", "chat"])
        self.assertEqual(ref[0]["finish"], "stop")          # odd first prompt id: EOS after 9 tokens
        self.assertEqual(len(ref[0]["lp"]), 10)
        self.assertEqual(ref[0]["lp"][-1][0], EOS)
        self.assertEqual((ref[1]["finish"], len(ref[1]["lp"])), ("length", 12))
        self.assertEqual(ref[2]["prompt"][:2], [0, 128803])  # chat_text framed as the A/B frames it
        self.assertEqual(len(ref[2]["lp"][0][2]), ab_requests.LP_K)

    def test_the_grammar_forces_the_reference_ids(self):
        self.assertEqual(ab_requests.force_grammar([5, 17]), "root ::= <[5]> <[17]>")
        self.assertEqual(ab_requests.force_grammar([]), 'root ::= ""')
        with open(self.ref) as f:
            ref = json.load(f)["requests"]
        ids = [e[0] for e in ref[0]["lp"]]
        g0 = self.exact.grammars[1]
        self.assertEqual(self.exact.grammars[1], ab_requests.force_grammar(ids[:-1]))   # [0]: the probe
        self.assertNotIn("<[%d]>" % EOS, g0)

    def test_the_grammar_probe(self):
        with open(self.same) as f:
            probe = json.load(f)["probe"]
        self.assertTrue(probe["ok"])
        self.assertEqual(probe["got"], probe["token"])
        # a server that ignores grammars fails it (and an exact arm alone could not tell)
        ignoring = Model()
        ignoring.complete_orig = ignoring.complete
        ignoring.complete = lambda body: ignoring.complete_orig({k: v for k, v in body.items() if k != "grammar"})
        srv, url = serve(ignoring)
        try:
            with open(self.ref) as f:
                res = ab_requests.grammar_probe(json.load(f), url)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertFalse(res["ok"])

    def test_exact_against_itself(self):
        res = ab_requests.kl_compare(self.ref, [self.same])[self.same]
        self.assertTrue(res["probe_ok"])
        self.assertEqual(res["positions"], 10 + 12 + 8)
        self.assertEqual(res["followed"], res["positions"])
        self.assertEqual(res["identical"], res["positions"])
        self.assertEqual(res["incomplete"], [])
        for g, x in res["groups"].items():
            self.assertEqual(x["kl_max"], 0.0, g)
            self.assertEqual(x["top1"], 1.0, g)
        self.assertEqual(sorted(res["groups"]), ["all", "decode", "family chat", "family code",
                                                 "family sql", "prefill"])
        self.assertEqual(res["groups"]["prefill"]["n"], 3)

    def test_a_perturbed_model(self):
        res = ab_requests.kl_compare(self.ref, [self.other])[self.other]
        self.assertEqual(res["followed"], res["positions"])   # forced, whatever its own argmax
        self.assertEqual(res["positions"], 30)
        self.assertLess(res["identical"], res["positions"])
        a = res["groups"]["all"]
        self.assertGreater(a["kl_mean"], 0.0)
        self.assertLess(a["top1"], 1.0)
        self.assertGreater(a["dlp"], 0.0)

    def test_kl_is_zero_only_for_identical_positions(self):
        e = [7, -0.1, [[7, -0.1], [8, -2.5], [9, -3.0]]]
        self.assertEqual(ab_requests.kl_topk(e, list(e)), 0.0)
        f = [7, -0.2, [[7, -0.2], [8, -2.0], [9, -3.5]]]
        self.assertGreater(ab_requests.kl_topk(e, f), 0.0)
        g = [8, -0.3, [[8, -0.3], [10, -1.8], [11, -4.0]]]   # top-1 moved, listed tokens differ
        self.assertGreater(ab_requests.kl_topk(e, g), ab_requests.kl_topk(e, f))

    def test_kl_dir_groups_by_reference(self):
        out = os.path.join(self.dir, "kldir")
        os.makedirs(out, exist_ok=True)
        for p in (self.same, self.other):
            shutil.copy(p, out)
        ab_requests.kl_dir(out)   # prints both against the one reference; must not raise
        # moved with its reference (the window's relative path no longer resolves): found beside
        moved = os.path.join(self.dir, "moved")
        os.makedirs(moved, exist_ok=True)
        for p in (self.same, self.ref):
            shutil.copy(p, moved)
        with open(os.path.join(moved, os.path.basename(self.same))) as f:
            js = json.load(f)
        js["forced_along"] = "build/ab-nowhere/" + os.path.basename(self.ref)
        with open(os.path.join(moved, os.path.basename(self.same)), "w") as f:
            json.dump(js, f)
        ab_requests.kl_dir(moved)


if __name__ == "__main__":
    unittest.main()
