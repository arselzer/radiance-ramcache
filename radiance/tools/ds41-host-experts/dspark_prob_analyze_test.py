#!/usr/bin/env python3
"""Source-only prepared cases. Run through the box-wide build lock; no GPU/server needed."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("dspark_prob_analyze", Path(__file__).with_name("dspark_prob_analyze.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def epoch(seq=1, number=1, first=True, second=False):
    rows = []
    for depth in (1, 2):
        p = dict(v=1, event="proposal", seq=seq, epoch=number, anchor_pos=40+number*3,
                 pos=40+number*3+depth, depth=depth, token=9+depth, predecessor=8+depth,
                 candidate_count=16, candidate_p=.8, score=4., margin=2., margin_unbounded=False,
                 signal_valid=True, status="pending", verified=None, reached=None, accepted=None, cancelled=False)
        o = dict(p, event="outcome", verified=True, reached=depth == 1 or first,
                 accepted=(first if depth == 1 else second) if depth == 1 or first else None)
        o["status"] = "unreached" if not o["reached"] else "accepted" if o["accepted"] else "rejected"
        rows.append((p, o))
    return [rows[0][0], rows[1][0], rows[0][1], rows[1][1]]


def pair(events, incomplete=False):
    return mod.pair_events("run", ["I x " + mod.MARKER + json.dumps(e) for e in events], incomplete)


def attach(records, split="train", group="prompt-A"):
    for row in records:
        row["meta"] = dict(family="code", prompt_key=group, split_group=group)
        row["split"] = split
    return records


class AnalysisTest(unittest.TestCase):
    def test_cli_calibrates_and_incomplete_capture_cannot_fit(self):
        with tempfile.TemporaryDirectory(prefix="ds41-prob-analysis-cli-") as tmp:
            log, meta, out = [Path(tmp)/name for name in ("trace.log", "meta.json", "report.json")]
            events = epoch(seq=1, number=1, second=True) + epoch(seq=2, number=2, first=False)
            log.write_text("\n".join(mod.MARKER+json.dumps(e) for e in events)+"\n")
            meta.write_text(json.dumps(dict(schema="ds41-proposal-metadata-v1", requests=[
                dict(run="r", seq=1, family="code", prompt_key="p1", split="train"),
                dict(run="r", seq=2, family="code", prompt_key="p2", split="heldout")])))
            argv = ["analyze", "--trace", "r="+str(log), "--metadata", str(meta), "--out", str(out)]
            self.assertEqual(mod.main(argv), 0)
            result = json.loads(out.read_text())
            self.assertEqual(result["validation"], "passed")
            self.assertEqual(result["candidates"]["status"], "awaiting_matched_measured_C0_C1_C2")
            self.assertEqual(sum(r["n"] for r in result["calibration"] if r["family"] == "code"), 3)
            log.write_text("\n".join(mod.MARKER+json.dumps(e) for e in events[:-1])+"\n")
            self.assertEqual(mod.main(argv+["--allow-incomplete"]), 0)
            result = json.loads(out.read_text())
            self.assertEqual(result["validation"], "incomplete")
            self.assertEqual(result["candidates"]["status"], "disabled_for_incomplete_or_mismatched_capture")

    def test_pairing_and_unreached_are_not_false_trials(self):
        records, missing = pair(epoch(first=False))
        self.assertEqual(missing, 0)
        bins = mod.calibration(attach(records), 10)
        self.assertEqual(sum(b["n"] for b in bins if b["family"] == "code"), 1)
        self.assertTrue(all(b["depth"] == 1 for b in bins))
        replay, _ = mod.replay_rows(records, 0)
        self.assertEqual(replay[0]["a1"], 0)
        self.assertEqual(replay[0]["a2"], 0)  # known prefix length, NOT a conditional trial

    def test_duplicate_or_changed_identity_or_prefix_is_rejected(self):
        events = epoch()
        for field in ("token", "pos", "epoch", "candidate_p"):
            bad = copy.deepcopy(events)
            bad[2][field] += 1
            with self.assertRaises(mod.Invalid):
                pair(bad)
        with self.assertRaises(mod.Invalid):
            pair(events + [events[2]])
        with self.assertRaises(mod.Invalid):
            pair([events[2]])
        bad = epoch()
        bad[1]["predecessor"] = bad[3]["predecessor"] = 777
        with self.assertRaises(mod.Invalid):
            pair(bad)
        bad = epoch(first=False)
        bad[3].update(status="rejected", reached=True, accepted=False)
        with self.assertRaises(mod.Invalid):
            pair(bad)

    def test_missing_outcomes_and_engine_mismatches_are_not_silently_used(self):
        events = epoch()
        with self.assertRaises(mod.Invalid):
            pair(events[:-1])
        _, missing = pair(events[:-1], True)
        self.assertEqual(missing, 1)
        bad = epoch()
        bad[2].update(status="proposal_mismatch", verified=False, reached=False, accepted=None)
        with self.assertRaises(mod.Invalid):
            pair(bad)

    def test_discarded_cancelled_and_invalid_signals_stay_out_of_calibration(self):
        events = epoch()
        for p in events:
            p["cancelled"] = True
        records, _ = pair(events)
        self.assertFalse(mod.calibration(attach(records), 10))
        events = epoch()
        events[3].update(status="verify_cap", verified=False, reached=False, accepted=None)
        records, _ = pair(events)
        bins = mod.calibration(attach(records), 10)
        self.assertTrue(all(b["depth"] == 1 for b in bins))
        rows, excluded = mod.replay_rows(records, 0)
        self.assertFalse(rows)
        self.assertEqual(excluded["not_fixed_depth2_verified"], 1)

    def test_group_split_is_stable_across_runs_and_requests(self):
        a = dict(split_group="same-prompt", run="arm-A", seq=1)
        b = dict(a, run="arm-B", seq=99)
        self.assertEqual(mod.split_of(a, "seed", .25), mod.split_of(b, "seed", .25))
        self.assertEqual(mod.split_of(dict(a, split="heldout"), "seed", .25), "heldout")

    def test_heldout_outcomes_never_select_thresholds_or_reference(self):
        rows = []
        for split, count in (("train", 4), ("heldout", 2)):
            for i in range(count):
                for high in (False, True):
                    rows.append(dict(family="code", group=f"{split}-{i}", split=split,
                                     p1=.8 if high else .2, p2=.8 if high else .2,
                                     a1=int(high), a2=int(high)))
        costs = {"families": {"code": dict(c0_ms=1., c1_ms=1.6, c2_ms=2.)}}
        a = mod.fit_candidates(rows, costs, 3, 2, 2, 1, 0, "seed")[0]
        changed = copy.deepcopy(rows)
        for r in changed:
            if r["split"] == "heldout":
                r["a1"] = r["a2"] = 1-r["a1"]
        b = mod.fit_candidates(changed, costs, 3, 2, 2, 1, 0, "seed")[0]
        self.assertEqual(a["thresholds"], b["thresholds"])
        self.assertEqual(a["fixed_reference_chosen_on_train"], b["fixed_reference_chosen_on_train"])
        self.assertNotEqual(a["heldout"]["tokens_per_ms"], b["heldout"]["tokens_per_ms"])
        self.assertEqual(mod.fit_candidates(rows, costs, 3, 99, 2, 1, 0, "seed")[0]["status"],
                         "insufficient_request_group_support")

    def test_replay_trims_whole_request_boundaries(self):
        records, _ = pair([e for i in range(4) for e in epoch(number=i+1)])
        kept, excluded = mod.replay_rows(attach(records), 1)
        self.assertEqual([r["epoch"] for r in kept], [2, 3])
        self.assertEqual(excluded["request_boundary_epochs"], 2)

    def test_metadata_split_leakage_and_unmatched_costs_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="ds41-prob-analysis-test-") as tmp:
            path = Path(tmp)/"input.json"
            rows = [dict(run="a", seq=1, family="code", prompt_key="same", split="train"),
                    dict(run="b", seq=2, family="code", prompt_key="same", split="heldout")]
            path.write_text(json.dumps(dict(schema="ds41-proposal-metadata-v1", requests=rows)))
            with self.assertRaises(mod.Invalid):
                mod.load_metadata(path)
            costs = dict(schema="ds41-proposal-costs-v1", same_drafter_observer_wait=False,
                         measurement_profile="synthetic unit fixture", families={"code":
                         dict(c0_ms=1., c1_ms=1.6, c2_ms=2., source="synthetic, not a measurement")})
            path.write_text(json.dumps(costs))
            with self.assertRaises(mod.Invalid):
                mod.load_costs(path)
            costs["same_drafter_observer_wait"] = True
            path.write_text(json.dumps(costs))
            self.assertEqual(mod.load_costs(path)["families"]["code"]["c0_ms"], 1.)
            results = dict(concurrency=1, requests=[dict(name="p", family="code", text="ok")],
                           stats=dict(requests=[dict(id=1)]))
            path.write_text(json.dumps(results))
            args = SimpleNamespace(assert_sequential=False, results=["r="+str(path)], first_seq=1)
            with self.assertRaises(mod.Invalid):
                mod.make_metadata(args)
            args.assert_sequential = True
            self.assertEqual(mod.make_metadata(args)["requests"][0]["seq"], 1)
            results["concurrency"] = 4
            path.write_text(json.dumps(results))
            with self.assertRaises(mod.Invalid):
                mod.make_metadata(args)


if __name__ == "__main__":
    unittest.main()
