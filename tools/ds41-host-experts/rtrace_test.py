"""rtrace_test.py -- rtrace.py on synthetic traces whose answers are known (rtrace.py synth).

  python3 -I rtrace_test.py [-v]

The synthetic trace (see synth's docstring): two requests of families alpha and beta, 4 decode
tokens each plus a post-stop pass behind alpha's last, 2 misses a token: layer 3's at slot 5
with weight share 0.05 and a substitute on the card 2% below the 6th key, layer 17's at slot 1
with share 0.25 and the nearest held expert 10% below; 238 CPU hits a token; 60 ms a token
measured. A trace is also read from the C++ writer when RTRACE_CPP_PREFIX names one
(route_trace_test's dump: RTRACE_DUMP=<dir> route_trace_test).
"""
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtrace  # noqa: E402


class SynthTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="rtrace-test-")
        cls.prefix = os.path.join(cls.dir, "rt")
        cls.arm = os.path.join(cls.dir, "arm.json")
        rtrace.synth(cls.prefix, cls.arm)
        cls.d = rtrace.load(cls.prefix, cls.arm)
        cls.g = rtrace.gates(cls.d, tau=0.1, delta=0.05)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_tokens_requests_and_the_post_stop_pass(self):
        d = self.d
        self.assertEqual(d["problems"], [])
        self.assertEqual(len(d["reqs"]), 2)
        self.assertEqual(sorted(r["name"] for r in d["reqs"].values()), ["alpha-01", "beta-01"])
        self.assertEqual([r["prompt"] for r in sorted(d["reqs"].values(), key=lambda r: r["index"])], [5, 5])
        self.assertEqual(len(d["tokens"]), 9)
        post = [t for t in d["tokens"] if t["post_stop"]]
        self.assertEqual(len(post), 1)
        self.assertEqual((post[0]["name"], post[0]["token"]), ("alpha-01", 4))
        self.assertEqual([t["token"] for t in d["tokens"] if t["name"] == "beta-01"], [0, 1, 2, 3])
        self.assertTrue(all(len(t["layers"]) == 40 for t in d["tokens"]))

    def test_the_gates(self):
        a = self.g["all"].row(0.1)
        self.assertEqual(a["misses"], 16)
        self.assertAlmostEqual(a["per_token"], 2.0)
        self.assertAlmostEqual(a["share_mean"], 0.15, places=6)
        self.assertAlmostEqual(a["below_tau"], 0.5)
        self.assertAlmostEqual(a["assignable"], 0.5)
        self.assertAlmostEqual(a["any_sub"], 0.5)
        self.assertAlmostEqual(a["slot2plus"], 0.5)
        self.assertEqual(a["known"], 16)
        self.assertEqual(self.g["status"], {"H": 8 * 238, "M": 16})
        self.assertEqual(len(self.g["post"].misses), 2)

    def test_families_and_layers(self):
        for f in ("alpha", "beta"):
            r = self.g["families"][f].row(0.1)
            self.assertEqual((self.g["families"][f].tokens, r["misses"]), (4, 8))
            self.assertAlmostEqual(r["below_tau"], 0.5)
        l3, l17 = self.g["layers"][3].row(0.1), self.g["layers"][17].row(0.1)
        self.assertEqual((l3["misses"], l17["misses"]), (8, 8))
        self.assertAlmostEqual(l3["share_mean"], 0.05, places=6)
        self.assertAlmostEqual(l17["share_mean"], 0.25, places=6)
        self.assertEqual((l3["below_tau"], l3["assignable"], l3["slot2plus"]), (1.0, 1.0, 1.0))
        self.assertEqual((l17["below_tau"], l17["assignable"], l17["slot2plus"]), (0.0, 0.0, 0.0))
        self.assertEqual(self.g["layers"][0].row(0.1)["misses"], 0)

    def test_substitutes_follow_delta(self):
        # 1%: layer 3's held near-tie (2% below) is out; 15%: layer 17's (10% below) is in. The
        # 1%-below expert is on the file tier and not cached: never a substitute.
        self.assertAlmostEqual(rtrace.gates(self.d, 0.1, 0.01)["all"].row(0.1)["assignable"], 0.0)
        self.assertAlmostEqual(rtrace.gates(self.d, 0.1, 0.15)["all"].row(0.1)["assignable"], 1.0)
        t = self.d["tokens"][0]
        subs = rtrace.substitutes(self.d, 3, t["layers"][3], 0.05)
        self.assertEqual([(e, h) for e, h, _ in subs], [(200, "V")])
        self.assertAlmostEqual(subs[0][2], 0.02, places=5)

    def test_the_replay(self):
        rp = rtrace.replay(self.g, self.d, cap=6, tau=0.1, insert_ms=0.15)
        self.assertEqual(rp["tokens"], 8)
        self.assertAlmostEqual(rp["measured_ms"], 60.0)
        a, b, c = rtrace.MODELS["fit"]
        h = 238
        fit = rp["fit"]
        self.assertAlmostEqual(fit["exact"]["ms"], a + 2 * b + c * h)
        self.assertAlmostEqual(fit["drop-all"]["ms"], a + c * h + 2 * 0.15)
        self.assertAlmostEqual(fit["drop<tau"]["ms"], a + b + c * h + 0.15)
        self.assertAlmostEqual(fit["cap"]["ms"], a + c * h + 2 * 0.15)
        cal = [k for k in rp if k.startswith("fit a=")]
        self.assertEqual(len(cal), 1)
        self.assertAlmostEqual(rp[cal[0]]["exact"]["ms"], 60.0)
        self.assertAlmostEqual(rp[cal[0]]["drop-all"]["ms"], 60.0 - 2 * b + 0.3)
        # protect-top-k: slot >= 2 and slot >= 4 drop layer 3's slot-5 miss, not layer 17's slot 1
        self.assertAlmostEqual(fit["slot>=2"]["ms"], a + b + c * h + 0.15)
        self.assertAlmostEqual(fit["slot>=4"]["drops"], 1.0)
        # a cap of 1 drops the first miss in layer order (layer 3's), not the lighter one
        rp1 = rtrace.replay(self.g, self.d, cap=1, tau=0.1, insert_ms=0.0)
        self.assertAlmostEqual(rp1["fit"]["cap"]["ms"], a + b + c * h)

    def test_the_fit_on_token_times(self):
        # T = 20 + 2.5 x misses + 0.1 x hits, exactly: the fit gives it back with R^2 1.
        toks = [dict(misses=[(0, 0.1, 3)] * w, hits=h, ms=20 + 2.5 * w + 0.1 * h)
                for w in range(0, 12) for h in (80, 95, 110)]
        f = rtrace.fit_tokens(toks)
        self.assertAlmostEqual(f["a"], 20.0, places=6)
        self.assertAlmostEqual(f["b"], 2.5, places=6)
        self.assertAlmostEqual(f["c"], 0.1, places=6)
        self.assertAlmostEqual(f["r2"], 1.0, places=9)
        self.assertIsNone(rtrace.fit_tokens(toks[:10]))   # too few tokens
        # the synthetic trace's token times: 50 ms apart, no spread in misses: no fit
        self.assertIsNone(rtrace.fit_tokens(self.g["per_token"], least=1))
        self.assertEqual([t["ms"] for t in self.d["tokens"] if t["name"] == "beta-01"], [50.0, 50.0, 50.0, None])

    def test_requests_past_the_arms_list_are_left_out(self):
        one = os.path.join(self.dir, "one.json")
        with open(self.arm) as f:
            arm = json.load(f)
        arm["requests"] = arm["requests"][:1]
        with open(one, "w") as f:
            json.dump(arm, f)
        d = rtrace.load(self.prefix, one)
        g = rtrace.gates(d, 0.1, 0.05)
        self.assertEqual(g["extra"], 4)                     # beta's 4 decode tokens
        self.assertEqual(g["all"].row(0.1)["misses"], 8)    # alpha's alone
        self.assertEqual(sorted(g["families"]), ["alpha"])

    def test_a_join_failure_is_reported(self):
        # Rank 1 says it selected an expert rank 0 did not route: a problem, not a silent pass.
        bad = os.path.join(self.dir, "bad")
        for s in range(5):
            shutil.copy(rtrace.path_of(self.prefix, s), rtrace.path_of(bad, s))
        h, recs, b = rtrace.read_records(rtrace.path_of(bad, 1))
        b = bytearray(b)
        for t, _fl, _n, o in recs:
            if t == rtrace.T_LAYER and struct.unpack_from("<H", b, o + 8)[0] == 5:
                n_sel = struct.unpack_from("<H", b, o + 12)[0]
                self.assertGreater(n_sel, 0)
                struct.pack_into("<H", b, o + rtrace.LAYER.size, 100)   # a pick rank 0 never routed
                break
        with open(rtrace.path_of(bad, 1), "wb") as f:
            f.write(bytes(b))
        d = rtrace.load(bad, self.arm)
        self.assertTrue(any("rank 1 selected 100" in p for p in d["problems"]))

    def test_report_runs_and_writes_json(self):
        js = os.path.join(self.dir, "out.json")
        rtrace.report(self.prefix, self.arm, out_json=js)
        with open(js) as f:
            out = json.load(f)
        self.assertEqual(out["tokens"], 8)
        self.assertAlmostEqual(out["all"]["below_tau"], 0.5)
        self.assertEqual(out["slots"], [0, 8, 0, 0, 0, 8])


@unittest.skipUnless(os.environ.get("RTRACE_CPP_PREFIX"), "RTRACE_CPP_PREFIX names no C++-written trace")
class CppTrace(unittest.TestCase):
    """The C++ writer's file (route_trace_test with RTRACE_DUMP=<dir>), read by this reader: the
    values that test encodes (one pass, one decode layer on each rank, a bias, tiers, a step and an
    end record) come back as written."""

    def test_the_cpp_layout_reads_back(self):
        p = os.environ["RTRACE_CPP_PREFIX"]
        r0 = rtrace.Rank(p, 0)
        self.assertEqual(r0.passes[7]["step"], 41)
        self.assertEqual(r0.passes[7]["n_tok"], 1)
        L = r0.layers[(7, 12)]
        self.assertEqual(L["step"], 41)
        self.assertEqual(L["flags"], rtrace.F_DECIDED | rtrace.F_ROUTE | rtrace.F_LOGITS | rtrace.F_RAM)
        self.assertEqual(L["picks"], {3: ("M", 1), 40: ("V", 1)})
        ids, w, keys = L["route"][0]
        self.assertEqual(ids, (3, 140, 40, 250, 99, 383))
        self.assertAlmostEqual(w[0], 0.5)
        self.assertAlmostEqual(keys[5], 2.5)
        self.assertAlmostEqual(L["logits"][383], 3.25)
        self.assertEqual(L["ram"], (1 << 7) | (1 << 70))
        self.assertAlmostEqual(r0.bias[12][7], -0.125)
        self.assertEqual(r0.tiers[12][40], 0)
        self.assertEqual(r0.tiers[12][3], 3)
        steps, ends = rtrace.read_steps(p)
        self.assertEqual(steps[41]["entries"][0]["seq"], 77)
        self.assertEqual(steps[41]["entries"][0]["ctx"], 63)
        self.assertEqual(ends, [dict(step=41, ahead_step=42, seq=77, n_out=64)])
        r1 = rtrace.Rank(p, 1)
        self.assertEqual(r1.layers[(7, 12)]["picks"], {140: ("H", 1)})
        self.assertEqual(r1.layers[(7, 12)]["route"], [])


if __name__ == "__main__":
    unittest.main()
