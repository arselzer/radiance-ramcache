"""ab_requests_test.py -- the quality harness (ab_requests.py --lp, --force, --kl) against a fake
server that decodes greedily from a deterministic toy model and honours a token-id grammar the way
the engine's GBNF does (`<[id]>` terminals; an end-of-generation id only once the grammar is done).

  python3 -I ab_requests_test.py [-v]

The validation the window runs on the exact arm, here offline: forced along its own reference, an
exact model follows every token with every position bit-identical, KL 0 and top-1 agreement 100%;
a perturbed model still follows every forced token (teacher forcing does not depend on the arm's
own choice), with KL above 0 and top-1 agreement below 100%.
"""
import contextlib
import hashlib
import http.server
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
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
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass   # a client that timed out (RequestTimeout) has gone

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


class HarnessTurns(unittest.TestCase):
    """The agent-harness receipt's shape (make_prefix_receipt.py --harness): chat_turns with a
    chat_system, replies sent back with assistant_from, and --ttft's per-session sums."""

    def test_a_system_prompt_follows_bos_and_every_turn_begins_with_the_last(self):
        srv, url = serve(Model())
        d = tempfile.mkdtemp(prefix="abreq-hx-")
        try:
            reqs, turns = [], []
            for i, u in enumerate(["the task", "tool output one", "tool output two"]):
                name = "hxT-%d" % (i + 1)
                reqs.append({"name": name, "family": "hxT",
                             "request": {"max_tokens": 6, "temperature": 0, "stream": False,
                                         "chat_system": "the rules", "chat_turns": turns + [{"user": u}]}})
                turns = turns + [{"user": u, "assistant_from": name}]
            receipt = os.path.join(d, "receipt.json")
            with open(receipt, "w") as f:
                json.dump({"benchmark_requests": reqs}, f)
            ab_requests.run(receipt, url, 99, os.path.join(d, "cold.json"))
            ab_requests.run(receipt, url, 99, os.path.join(d, "warm.json"))
            with open(os.path.join(d, "cold.json")) as f:
                rows = json.load(f)["requests"]
            self.assertEqual([r["family"] for r in rows], ["hxT"] * 3)
            system = ab_requests.tokenize(url, "the rules")
            for r in rows:
                self.assertIsNone(r["error"])
                self.assertEqual(r["prompt_ids"][:1 + len(system)], [ab_requests.BOS] + system)
                self.assertEqual(r["prompt_ids"][1 + len(system)], ab_requests.USER)
            for a, b in zip(rows, rows[1:]):
                self.assertEqual(b["prompt_ids"][:len(a["prompt_ids"])], a["prompt_ids"])
                self.assertEqual(b["sent_back"][-1]["from"], a["name"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ab_requests.ttft(d, ["cold", "warm"])
            text = out.getvalue()
            self.assertIn("session", text)
            self.assertRegex(text, r"\nhxT +3 ")
            self.assertIn("3/3", text)
        finally:
            srv.shutdown()
            srv.server_close()
            shutil.rmtree(d, ignore_errors=True)

    def test_a_sessions_cached_share_is_of_every_prompt_token_it_sent(self):
        """The server's prompt_n is the prompt less what the cache supplied: two turns of 1,000 and
        2,200 tokens, the second resuming at 1,000, sent 3,200 prompt tokens of which 1,000 (31.2%)
        came from the cache; cached over computed would say 1,000 of 2,200 (45.5%)."""
        d = tempfile.mkdtemp(prefix="abreq-share-")
        try:
            def arm(name, cached):
                rows = [{"name": "hxS-%d" % (i + 1), "family": "hxS", "text": "t%d" % i,
                         "prompt_ids": [1] * n,
                         "timings": {"ttft_ms": 100.0, "cache_n": c, "prompt_n": n - c}}
                        for i, (n, c) in enumerate(zip((1000, 2200), cached))]
                with open(os.path.join(d, name + ".json"), "w") as f:
                    json.dump({"requests": rows}, f)
            arm("cold", (0, 0))
            arm("warm", (0, 1000))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ab_requests.ttft(d, ["cold", "warm"])
            text = out.getvalue()
            session = re.search(r"\nhxS +2 +(\d+) .*", text)
            self.assertIsNotNone(session, text)
            self.assertEqual(session.group(1), "3200")
            self.assertIn(" 1000  31.2%  2/2", session.group(0))
            self.assertIn("    0   0.0%  2/2", session.group(0))
            self.assertRegex(text, r"\nhxS-2 +2200 ")      # a request's prompt, cached or not
        finally:
            shutil.rmtree(d, ignore_errors=True)


class PrefillHashes(unittest.TestCase):
    """--hashes: requests sent together reach the server in an order of their own, so an arm's prompts
    are matched to the first arm's by tokens and digest wherever they sit in its log; a real difference
    (another digest) still fails, named against the arm's unmatched prompt of that length rather than
    whichever prompt sits at its position; a prefill run uncut in a shared step is not counted cut."""

    @staticmethod
    def log(d, arm, prompts, label=None):
        """Each prompt (tokens, digest) or (tokens, digest, window group, logits); rank 0's label is
        `label`, by default "cut for <tokens>"."""
        with open(os.path.join(d, arm + ".server.log"), "w") as f:
            for seq, p in enumerate(prompts, 1):
                n, digest = p[0], p[1]
                win, lg = (p[2], p[3]) if len(p) > 2 else (digest, digest)
                f.write("I[0] prefill hash: seq %d, %d tokens (%s), rank 0: win[40]=%016x logits=%016x\n"
                        % (seq, n, label or "cut for %d" % n, win, lg))
                f.write("I prefill hash: seq %d, %d tokens: all ranks %016x\n" % (seq, n, digest))

    def run_hashes(self, d, arms):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = ab_requests.hashes(d, arms)
        return rc, out.getvalue()

    def test_reordered_prompts_match_and_a_different_digest_does_not(self):
        d = tempfile.mkdtemp(prefix="abreq-hash-")
        try:
            wave = [(46, 1), (40, 2), (39, 3), (40, 7)]           # two prompts of 40 tokens
            self.log(d, "alone", wave)
            self.log(d, "batched", [wave[1], wave[0], wave[3], wave[2]])
            self.log(d, "swapped40", [wave[3], wave[1], wave[2], wave[0]])
            self.log(d, "wrong", [wave[0], (40, 9), wave[2], wave[3]])
            rc, text = self.run_hashes(d, ["alone", "batched", "swapped40"])
            self.assertEqual(rc, 0, text)
            self.assertIn("batched: 4 of 4 prompts identical", text)
            self.assertIn("matched out of order", text)
            self.assertIn("swapped40: 4 of 4 prompts identical", text)
            rc, text = self.run_hashes(d, ["alone", "wrong"])
            self.assertEqual(rc, 1, text)
            self.assertIn("DIFFERS", text)
            self.assertIn("wrong: 3 of 4 prompts identical", text)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_a_difference_is_named_against_the_unmatched_prompt_of_its_length(self):
        d = tempfile.mkdtemp(prefix="abreq-hash-")
        try:
            a, b, c, x = (46, 1, 1, 1), (40, 2, 2, 2), (39, 3, 3, 3), (40, 4, 4, 4)
            x2 = (40, 5, 5, 4)      # x with another window group (so another digest), the same logits
            self.log(d, "alone", [a, b, c, x])
            self.log(d, "moved", [b, a, x2, c])       # x2 sits where c was, and c where x was
            rc, text = self.run_hashes(d, ["alone", "moved"])
            self.assertEqual(rc, 1, text)
            self.assertIn("moved: prompt 3 (40 tokens, cut for 40; its prompt 2): DIFFERS: rank 0 win[40]\n", text)
            self.assertNotIn("not the same prompts", text)
            self.assertIn("moved: 3 of 4 prompts identical", text)
            # Twins: the first matches the second's record; the second is named against x2, not its twin.
            self.log(d, "twins", [x, x])
            self.log(d, "twin2", [x2, x])
            rc, text = self.run_hashes(d, ["twins", "twin2"])
            self.assertEqual(rc, 1, text)
            self.assertIn("twin2: prompt 1 (40 tokens, cut for 40; its prompt 0): DIFFERS: rank 0 win[40]\n", text)
            # No unmatched prompt of the length left: not the same prompts; none left at all: missing.
            self.log(d, "other", [a, b, c, (41, 4)])
            rc, text = self.run_hashes(d, ["alone", "other"])
            self.assertEqual(rc, 1, text)
            self.assertIn("other: prompt 3 (40 tokens): no unmatched prompt of that length (41 tokens left)", text)
            self.log(d, "short", [b, a, c])
            rc, text = self.run_hashes(d, ["alone", "short"])
            self.assertEqual(rc, 1, text)
            self.assertIn("short: prompt 3 (40 tokens) missing", text)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_prompts_run_uncut_in_a_shared_step_are_not_counted_cut(self):
        d = tempfile.mkdtemp(prefix="abreq-hash-")
        try:
            wave = [(46, 1), (40, 2), (39, 3)]
            self.log(d, "alone", wave)
            self.log(d, "batched", [wave[1], wave[0], wave[2]], label="not cut: shared step")
            rc, text = self.run_hashes(d, ["alone", "batched"])
            self.assertEqual(rc, 0, text)
            self.assertIn("batched: 3 of 3 prompts identical (0 of its prefills cut, 3 uncut in a shared step, "
                          "2 matched out of order)", text)
            rc, text = self.run_hashes(d, ["batched", "alone"])
            self.assertEqual(rc, 0, text)
            self.assertIn("alone: 3 of 3 prompts identical (3 of its prefills cut, 2 matched out of order)", text)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Slow(Model):
    """The exact toy model, `delay` seconds late, with the server's prompt count in its timings."""
    def __init__(self, delay=0.0):
        super().__init__()
        self.delay = delay

    def complete(self, body):
        time.sleep(self.delay)
        out = super().complete(body)
        out["timings"]["prompt_n"] = len(body["prompt"])
        return out


@contextlib.contextmanager
def env(**kv):
    """os.environ with these values (None: unset), restored afterwards."""
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class RequestTimeout(unittest.TestCase):
    """AB_REQUEST_TIMEOUT_S: the timed requests' timeout (600 s by default), which ab_window.sh raises
    for a receipt of long prompts; the receipt's own prompt counts printed beside the server's."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="abreq-timeout-")
        self.receipt = os.path.join(self.dir, "long-receipt.json")
        reqs = [{"name": "long-1", "request": {"prompt": [3, 10, 20, 30], "max_tokens": 4, "temperature": 0}},
                {"name": "long-2", "request": {"prompt": [5, 11], "max_tokens": 4, "temperature": 0}}]
        with open(self.receipt, "w") as f:
            json.dump({"benchmark_requests": reqs, "max_context_tokens": 8,
                       "provenance": {"requests": [{"name": "long-1", "prompt_tokens": 4},
                                                   {"name": "long-2", "prompt_tokens": 3}]}}, f)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_requests(self, model, **kv):
        srv, url = serve(model)
        out, text = os.path.join(self.dir, "arm.json"), io.StringIO()
        try:
            with env(**kv), contextlib.redirect_stdout(text):
                ab_requests.run(self.receipt, url, 9, out)
        finally:
            srv.shutdown()
            srv.server_close()
        with open(out) as f:
            return json.load(f), text.getvalue()

    def test_the_default_is_600_seconds_and_a_value_must_be_positive(self):
        with env(AB_REQUEST_TIMEOUT_S=None):
            self.assertEqual(ab_requests.request_timeout(), 600.0)
        with env(AB_REQUEST_TIMEOUT_S=""):
            self.assertEqual(ab_requests.request_timeout(), 600.0)
        with env(AB_REQUEST_TIMEOUT_S="1302"):
            self.assertEqual(ab_requests.request_timeout(), 1302.0)
        for v in ("0", "-5", "10m", "nan", "inf"):
            with env(AB_REQUEST_TIMEOUT_S=v), self.assertRaises(ValueError, msg=v):
                ab_requests.request_timeout()

    def test_each_request_gets_the_timeout(self):
        seen = []
        real = ab_requests.post

        def post(url, path, body, timeout=900):
            seen.append(timeout)
            return real(url, path, body, timeout)
        ab_requests.post = post
        try:
            j, _ = self.run_requests(Slow(), AB_REQUEST_TIMEOUT_S="1302")
            j0, _ = self.run_requests(Slow(), AB_REQUEST_TIMEOUT_S=None)
        finally:
            ab_requests.post = real
        self.assertEqual(seen, [1302.0, 1302.0, 600.0, 600.0])
        self.assertEqual((j["request_timeout_s"], j0["request_timeout_s"]), (1302.0, 600.0))

    def test_a_slow_reply_times_out_and_the_next_request_still_runs(self):
        j, text = self.run_requests(Slow(delay=1.5), AB_REQUEST_TIMEOUT_S="0.3")
        self.assertIn("requests time out after 0.3 s (AB_REQUEST_TIMEOUT_S)", text)
        self.assertEqual([r["text"] for r in j["requests"]], [None, None])
        self.assertTrue(all("timed out" in (r["error"] or "") for r in j["requests"]), j["requests"])
        j, text = self.run_requests(Slow(delay=0.2), AB_REQUEST_TIMEOUT_S="5")
        self.assertTrue(all(r["text"] is not None for r in j["requests"]), j["requests"])

    def test_an_invalid_timeout_sends_nothing(self):
        model = Slow()
        with self.assertRaises(SystemExit):
            self.run_requests(model, AB_REQUEST_TIMEOUT_S="10m")
        self.assertEqual(model.grammars, [])

    def test_the_receipts_prompt_counts_beside_the_servers(self):
        _, text = self.run_requests(Slow(), AB_REQUEST_TIMEOUT_S=None)
        self.assertIn("long-1: 4 prompt tokens by the server's count, 4 by the receipt's (same)", text)
        self.assertIn("long-2: 2 prompt tokens by the server's count, 3 by the receipt's (DIFFERENT)", text)


class PrefillHashReference(unittest.TestCase):
    """--hashes-ref: a window's Kh arms against a reference recorded in another run (a khref file, as
    make_longctx_receipts.py writes it), and their replies against the reference's."""

    LINES = ("I prefill hash: seq {s}, {n} tokens (not cut), rank 0: kv_window[40]={g:016x} logits={l:016x}\n"
             "I prefill hash: seq {s} rank 0 kv_window by layer: 0:{g:08x} 1:{l:08x}\n"
             "I prefill hash: seq {s}, {n} tokens: all ranks {a:016x}\n")

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="abreq-khref-")
        self.ref = [(32768, 0x71eaab9a74338840, 1, 2, "fill-1", " terms and"), (82, 0x112530d7d4d5b367, 3, 4, "fill-2", "ft licence")]
        self.write_log("reference", self.ref)
        recs = ab_requests.hash_log(os.path.join(self.dir, "reference.server.log"))
        receipt = os.path.join(self.dir, "fill-receipt.json")
        with open(receipt, "w") as f:
            json.dump({"benchmark_requests": []}, f)
        with open(receipt, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        self.khref = os.path.join(self.dir, "fill-khref.json")
        with open(self.khref, "w") as f:
            json.dump({"schema": "ds41-prefill-hash-reference-v1", "receipt": "fill-receipt.json",
                       "receipt_sha256": sha,
                       "prompts": [dict(r, name=x[4], text=x[5]) for r, x in zip(recs, self.ref)]}, f)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write_log(self, arm, prompts, texts=None):
        with open(os.path.join(self.dir, arm + ".server.log"), "w") as f:
            for s, (n, a, g, l, _, _) in enumerate(prompts, 1):
                f.write(self.LINES.format(s=s, n=n, a=a, g=g, l=l))
        rows = [{"name": x[4], "text": (texts or {}).get(x[4], x[5])} for x in prompts]
        with open(os.path.join(self.dir, arm + ".json"), "w") as f:
            json.dump({"requests": rows}, f)

    def check(self, *arms):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = ab_requests.hashes_ref(self.khref, self.dir, list(arms))
        return rc, out.getvalue()

    def test_identical_hashes_and_replies(self):
        self.write_log("c2", self.ref)
        rc, text = self.check("c2")
        self.assertEqual(rc, 0, text)
        self.assertIn("c2: 2 of 2 prompts identical", text)
        self.assertIn("c2: 2 of 2 replies identical to the reference's", text)

    def test_a_different_digest_names_its_group(self):
        bad = [self.ref[0], (82, 0x2b28b67525bb7f18, 3, 5, "fill-2", "ft licence")]
        self.write_log("live", bad)
        rc, text = self.check("live")
        self.assertEqual(rc, 1, text)
        self.assertIn("DIFFERS: rank 0 logits; rank 0 window layers 1", text)
        self.assertIn("live: 1 of 2 prompts identical", text)

    def test_a_different_reply_fails(self):
        self.write_log("c4", self.ref, {"fill-1": " terms or"})
        rc, text = self.check("c4")
        self.assertEqual(rc, 1, text)
        self.assertIn("fill-1: reply DIFFERS", text)
        self.assertIn("c4: 1 of 2 replies identical", text)

    def test_a_changed_receipt_makes_the_reference_stale(self):
        self.write_log("c2", self.ref)
        with open(os.path.join(self.dir, "fill-receipt.json"), "w") as f:
            json.dump({"benchmark_requests": [{"name": "other"}]}, f)
        rc, text = self.check("c2")
        self.assertEqual(rc, 1, text)
        self.assertIn("stale", text)

    def test_no_hashes(self):
        rc, text = self.check("missing")
        self.assertEqual(rc, 2, text)


if __name__ == "__main__":
    unittest.main()
