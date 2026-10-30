"""draft_hotset_test.py -- draft_hotset.py (RADIANCE_EXPERIMENTAL_DS41_DRAFT_VRAM's hot sets) on
synthetic --expert-stats captures.

  python3 -I draft_hotset_test.py [-v]
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draft_hotset as dh  # noqa: E402


def capture(counts, passes=10, how="container", n_layers=43, layer_experts=128, top_k=3):
    """A radiance-expert-stats/1 file of a DSpark run: `counts` {layer: {expert: count}}."""
    draft = [[0] * (384 if l < 40 else 128) for l in range(n_layers)]
    for L, c in counts.items():
        for g, x in c.items():
            if L < n_layers:
                draft[L][g] = x
    return {
        "format": "radiance-expert-stats/1", "tp": 4, "n_layers": n_layers, "n_trunk_layers": 40,
        "n_experts": 384,
        "layer_experts": [384] * 40 + [layer_experts] * (n_layers - 40),
        "top_k": [6] * 40 + [top_k] * (n_layers - 40),
        "map": ["container"] * 40 + [how] * (n_layers - 40),
        "kinds": ["prefill", "decode", "mixed", "draft"],
        "steps": {"prefill": 1, "decode": passes, "mixed": 0, "draft": 2 * passes},
        "histograms": {"draft": [0] * 40 + [passes] * (n_layers - 40)},
        "ranks": {"credited": [1, 1, 1, 1], "unmapped": [0] * 4, "unclassified": [0] * 4, "dropped": [0] * 4},
        "counts": {"prefill": [], "decode": [], "mixed": [], "draft": draft},
    }


class HotsetTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def put(self, name, doc):
        p = os.path.join(self.dir, name)
        with open(p, "w") as f:
            json.dump(doc, f)
        return p

    def quiet(self, fn, *a):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = fn(*a)
        return rc, out.getvalue()

    def test_write_ranks_each_rank_by_count_ties_to_the_lower_id(self):
        # stage 40: rank 0 has 5 (x9), 2 (x9), 30 (x1); rank 3 has 127 (x4). Stage 41: only rank 1.
        c = self.put("a.json", capture({40: {5: 9, 2: 9, 30: 1, 127: 4}, 41: {40: 7}, 42: {100: 3}}))
        out = os.path.join(self.dir, "hot.json")
        d = dh.write(out, [c])
        self.assertEqual(d["hot"][0][0][:4], [2, 5, 30, 0])      # 2 and 5 tie at 9: lower id first
        self.assertEqual(d["hot"][0][3][:2], [127, 96])          # then rank 3's zero-count by id
        self.assertEqual(d["hot"][1][1][:2], [40, 32])
        for s in range(3):
            for r in range(4):
                l = d["hot"][s][r]
                self.assertEqual(sorted(l), list(range(32 * r, 32 * r + 32)))   # every expert, each once
        with open(out) as f:
            again = json.load(f)
        self.assertEqual(again["format"], dh.FORMAT)
        self.assertEqual(again["layers"], [40, 41, 42])
        self.assertEqual(again["ranks"], 4)
        self.assertEqual(again["experts"], 128)
        self.assertEqual(again["hot"], d["hot"])
        dh.load_hotset(out)

    def test_write_sums_captures_and_honours_top(self):
        a = self.put("a.json", capture({40: {1: 5}, 41: {33: 1}, 42: {64: 1}}))
        b = self.put("b.json", capture({40: {3: 9}, 41: {33: 1}, 42: {64: 1}}))
        out = os.path.join(self.dir, "hot.json")
        d = dh.write(out, [a, b], top=2)
        self.assertEqual(d["hot"][0][0], [3, 1])
        self.assertTrue(all(len(l) == 2 for s in d["hot"] for l in s))
        self.assertEqual(d["counts"][0][3], 9)
        self.assertEqual(len(d["sources"]), 2)
        with self.assertRaises(dh.Refused):
            dh.write(out, [a], top=0)
        with self.assertRaises(dh.Refused):
            dh.write(out, [a], top=33)

    def test_refuses_captures_that_are_not_a_dspark_run(self):
        out = os.path.join(self.dir, "hot.json")
        good = {40: {1: 1}, 41: {1: 1}, 42: {1: 1}}
        bad = [
            dict(capture(good), format="other"),
            capture(good, how="rank order"),
            capture(good, layer_experts=96),
            capture(good, top_k=6),
            capture(good, n_layers=40),
            capture({40: {1: 1}, 41: {1: 1}}),      # stage 42 never selected
            capture({}, passes=0),
        ]
        for i, doc in enumerate(bad):
            p = self.put("bad%d.json" % i, doc)
            with self.assertRaises(dh.Refused, msg=str(i)):
                dh.write(out, [p])

    def test_cover_and_check(self):
        c = self.put("a.json", capture({40: {0: 10, 1: 5, 32: 4}, 41: {64: 3}, 42: {96: 2, 97: 1}}))
        hot = os.path.join(self.dir, "hot.json")
        dh.write(hot, [c])
        rc, out = self.quiet(dh.cover, hot, c, (1, 2))
        self.assertEqual(rc, 0)
        self.assertIn("N= 1", out)
        self.assertIn("N= 2", out)
        self.assertIn("all 100.0%", out.splitlines()[-1])
        # Under the mask at N=1 the capture must select only each rank's first expert.
        masked = self.put("m.json", capture({40: {0: 10, 32: 4}, 41: {64: 3}, 42: {96: 2}}))
        rc, out = self.quiet(dh.check, hot, 1, masked)
        self.assertEqual(rc, 0, out)
        self.assertIn("PASS", out)
        leaked = self.put("l.json", capture({40: {0: 10, 1: 1}, 41: {64: 3}, 42: {96: 2}}))
        rc, out = self.quiet(dh.check, hot, 1, leaked)
        self.assertEqual(rc, 1)
        self.assertIn("1 (rank 0) x1", out)

    def test_load_hotset_refuses_an_expert_its_rank_does_not_hold(self):
        c = self.put("a.json", capture({40: {0: 1}, 41: {0: 1}, 42: {0: 1}}))
        hot = os.path.join(self.dir, "hot.json")
        d = dh.write(hot, [c], top=4)
        d["hot"][0][1][0] = 0      # expert 0 is rank 0's
        p = self.put("bad.json", d)
        with self.assertRaises(dh.Refused):
            dh.load_hotset(p)

    def test_main_exit_codes(self):
        c = self.put("a.json", capture({40: {0: 1}, 41: {0: 1}, 42: {0: 1}}))
        hot = os.path.join(self.dir, "hot.json")
        rc, _ = self.quiet(dh.main, ["write", hot, c, "--top", "8"])
        self.assertEqual(rc, 0)
        rc, out = self.quiet(dh.main, ["show", hot, "2"])
        self.assertEqual(rc, 0)
        self.assertIn("stage 42 rank 3", out)
        with contextlib.redirect_stderr(io.StringIO()):
            rc, _ = self.quiet(dh.main, ["cover", hot, self.put("x.json", {"format": "no"})])
        self.assertEqual(rc, 2)
        with contextlib.redirect_stderr(io.StringIO()):
            rc, _ = self.quiet(dh.main, ["bogus"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
