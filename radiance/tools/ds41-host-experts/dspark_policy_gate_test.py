#!/usr/bin/env python3
"""Synthetic policy guards only; no model, fitting data or observed timings."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("dspark_policy_gate", Path(__file__).with_name("dspark_policy_gate.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
CONFIG = "v1:0.25:0.75"


def fixture():
    actions, cycles = [], []
    for epoch, (k, ctx, signal, mask) in enumerate(((0, 50, [.125, .875], 0),
                                                  (1, 51, [.5, .5], 0),
                                                  (2, 53, [.5, None], 2)), 1):
        actions.append(dict(seq=7, epoch=epoch, step=epoch, card=0, ctx=ctx, anchor=20+epoch,
                            query_rows=2, ids=[30+epoch, 40+epoch], k=k, candidate_p=signal,
                            invalid_depths=mask, fallback=bool(mask), wait_ns=10, pure_decode=True))
        cycles.append(dict(seq=7, epoch=epoch-1, verify_step=epoch, next_epoch=epoch,
                           k=epoch-2, next_k=k, cycle_ns=100 if epoch > 1 else 0, wait_ns=10,
                           cost_valid=epoch > 1, live=True, published=True,
                           accepted=1 if epoch == 3 else 0, output=(1, 2, 4)[epoch-1],
                           proposal_invalid_depths=0, next_invalid_depths=mask))
    return actions, cycles


def log(actions=None, cycles=None):
    a, c = fixture()
    actions = a if actions is None else actions
    cycles = c if cycles is None else cycles
    return "\n".join([mod.START + CONFIG + mod.SUFFIX] +
                     [mod.ACTION + json.dumps(row) for row in actions] +
                     [mod.CYCLE + json.dumps(row) for row in cycles] +
                     [mod.TOTALS + "queries=3 query_rows=6 observers=3 waits=3 costs=[1,1,0]"])


class PolicyGateTest(unittest.TestCase):
    def test_explicit_config_off_and_arm_scope(self):
        for off in (None, "off", "0"):
            self.assertIsNone(mod.parse_config(off))
        self.assertEqual(mod.parse_config(CONFIG), (.25, .75))
        self.assertEqual(mod.parse_config("v1:0:1e0"), (0., 1.))
        for bad in ("", "1", "v2:0.25:0.75", "v1:.25:0.75", "v1:0.25:0.75:",
                    "v1:NaN:0.5", "v1:0.5:inf", "v1:-0.1:0.5", "v1:0.5:1.01",
                    "v1: 0.2:0.5", "v1:0x1:0.5", "v1:0.2:0.5\n"):
            with self.subTest(bad=bad), self.assertRaises(mod.Invalid):
                mod.parse_config(bad)
        mod.check_arm(CONFIG, "Ds", "2", "1")
        for args in (("off", "Ds", "2", "1", False), (CONFIG, "Ds0", "2", "1", False),
                     (CONFIG, "Ds", "3", "1", False), (CONFIG, "Ds", "2", "2", False),
                     (CONFIG, "Ds", "2", "1", True)):
            with self.subTest(args=args), self.assertRaises(mod.Invalid):
                mod.check_arm(*args)

    def test_strict_startup_both_ways_and_fixed_measurement_exclusion(self):
        startup = mod.START + CONFIG + mod.SUFFIX
        mod.check_log(startup, CONFIG)
        mod.check_log("ordinary server", "off", True)
        for text, config in ((startup, "off"), ("ordinary server", CONFIG),
                             (startup+"\n"+startup, CONFIG),
                             (startup.replace(CONFIG, "v1:0.5:0.75"), CONFIG),
                             (startup.replace("query_rows=2", "query_rows=1"), CONFIG),
                             (startup+"\nDSpark matched early verify: on k=2", CONFIG)):
            with self.subTest(text=text), self.assertRaises(mod.Invalid):
                mod.check_log(text, config)

    def test_changing_actions_fallback_costs_and_epoch_context_guards(self):
        mod.check_log(log(), CONFIG, True)
        actions, cycles = fixture()
        finite_invalid = copy.deepcopy(actions); finite_invalid[2]["candidate_p"] = [.125, .875]
        mod.check_log(log(actions=finite_invalid), CONFIG, True)  # another observer field was invalid
        terminal = copy.deepcopy(cycles)
        terminal[2].update(live=False, published=False, cost_valid=False)
        mod.check_log(log(cycles=terminal).replace("costs=[1,1,0]", "costs=[1,0,0]"), CONFIG, True)
        for field, value in (("k", 0), ("fallback", False), ("invalid_depths", 0)):
            bad = copy.deepcopy(actions); bad[2][field] = value
            with self.subTest(field=field), self.assertRaises(mod.Invalid):
                mod.check_log(log(actions=bad), CONFIG, True)
        for field, value in (("epoch", 0), ("k", 0), ("accepted", 2), ("cycle_ns", 0),
                             ("next_k", 0), ("live", False), ("proposal_invalid_depths", 2)):
            bad = copy.deepcopy(cycles); bad[2][field] = value
            with self.subTest(field=field), self.assertRaises(mod.Invalid):
                mod.check_log(log(cycles=bad), CONFIG, True)
        bad = copy.deepcopy(actions); bad[2]["ctx"] += 1
        with self.assertRaises(mod.Invalid): mod.check_log(log(actions=bad), CONFIG, True)
        with self.assertRaises(mod.Invalid): mod.check_log(log(actions=actions[:2]), CONFIG, True)
        with self.assertRaises(mod.Invalid): mod.check_log(log(cycles=cycles+cycles[:1]), CONFIG, True)
        with self.assertRaises(mod.Invalid): mod.check_log(log().replace("queries=3", "queries=4"), CONFIG, True)


if __name__ == "__main__":
    unittest.main()
