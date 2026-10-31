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


def concurrent_fixture():
    a, c = fixture()
    for action, cycle in zip(a, c):
        action.update(frame=action['epoch'], solo_span=1, owner_slot=2)
        cycle.update(frame=action['frame'], solo_span=1, previous_span=1 if cycle['epoch'] else 0)
    first = copy.deepcopy(a)
    for i, original in enumerate(first):
        epoch = i + 4
        action = dict(original, epoch=epoch, step=10+i, frame=10+i, solo_span=2, ctx=(100, 101, 103)[i])
        a.append(action)
        c.append(dict(seq=7, epoch=0 if i == 0 else epoch-1, verify_step=10+i, next_epoch=epoch,
                      k=-1 if i == 0 else i-1, next_k=action['k'], cycle_ns=0 if i == 0 else 100,
                      wait_ns=10, cost_valid=i > 0, live=True, published=True, accepted=1 if i == 2 else 0,
                      output=(10, 11, 13)[i], proposal_invalid_depths=0, next_invalid_depths=action['invalid_depths'],
                      frame=10+i, solo_span=2, previous_span=0 if i == 0 else 2))
    scope = {'max_seqs': 4, 'tp': 4, 'dsolo': True, 'deferred': True, 'query_rows': 2,
             'full_rows': 5, 'depth': 5, 'from': 1, 'tbo': False, 'native_bank': True, 'bank_hot_n': 24}
    pause = dict(seq=7, epoch=3, step=4, frame=4, previous_span=1, next_span=2, reason='no_query', query_rows=0)
    resume = dict(seq=7, epoch=4, step=10, frame=10, solo_span=2, card=0, owner_slot=2, ctx=100,
                  previous_seq=7, previous_epoch=3, query_rows=2, ready=True)
    return a, c, scope, pause, resume


def concurrent_log(a=None, c=None, scope=None, pause=None, resume=None):
    defaults = concurrent_fixture()
    a = defaults[0] if a is None else a
    c = defaults[1] if c is None else c
    scope = defaults[2] if scope is None else scope
    pause = defaults[3] if pause is None else pause
    resume = defaults[4] if resume is None else resume
    rows = [mod.START + CONFIG + mod.SUFFIX, mod.CONCURRENT + json.dumps(scope)]
    rows += [mod.ACTION + json.dumps(x) for x in a]
    rows += [mod.CYCLE + json.dumps(x) for x in c]
    rows += [mod.PAUSE + json.dumps(pause), mod.RESUME + json.dumps(resume)]
    rows += [mod.TOTALS + 'queries=6 query_rows=12 observers=6 waits=6 costs=[2,2,0]']
    return '\n'.join(rows)


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

    def test_concurrent_requires_explicit_native_dx_and_scheduler_scope(self):
        mod.check_arm(CONFIG, 'Ds', '2', '4', defer=True, native_bank=True)
        for kwargs in ({'defer': False, 'native_bank': True}, {'defer': True, 'native_bank': False},
                       {'defer': True, 'native_bank': True, 'tbo': True}):
            with self.assertRaises(mod.Invalid): mod.check_arm(CONFIG, 'Ds', '2', '4', **kwargs)
        text = concurrent_log()
        mod.check_log(text, CONFIG, True, 4)
        with self.assertRaises(mod.Invalid): mod.check_log(text, CONFIG, True, 1)
        with self.assertRaises(mod.Invalid): mod.check_log(log(), CONFIG, True, 4)
        for key, value in [('dsolo', False), ('deferred', False), ('tp', 2), ('query_rows', 1),
                           ('tbo', True), ('native_bank', False), ('bank_hot_n', 33)]:
            scope = concurrent_fixture()[2]; scope[key] = value
            with self.subTest(key=key), self.assertRaises(mod.Invalid):
                mod.check_log(concurrent_log(scope=scope), CONFIG, True, 4)

    def test_concurrent_pause_resume_never_counts_busy_gap_or_stale_owner(self):
        a, c, _, pause, resume = concurrent_fixture()
        for marker in (mod.PAUSE, mod.RESUME):
            text = '\n'.join(x for x in concurrent_log().splitlines() if marker not in x)
            with self.assertRaises(mod.Invalid): mod.check_log(text, CONFIG, True, 4)
        for key, value in [('previous_epoch', 2), ('previous_seq', 8), ('ctx', 99), ('owner_slot', 3), ('ready', False)]:
            bad = dict(resume); bad[key] = value
            with self.subTest(key=key), self.assertRaises(mod.Invalid):
                mod.check_log(concurrent_log(resume=bad), CONFIG, True, 4)
        for key, value in [('frame', 10), ('step', 10), ('solo_span', 1), ('owner_slot', 3)]:
            bad = copy.deepcopy(a); bad[4][key] = value
            with self.subTest(key=key), self.assertRaises(mod.Invalid):
                mod.check_log(concurrent_log(a=bad), CONFIG, True, 4)
        bad = copy.deepcopy(c)
        bad[3].update(epoch=3, k=2, cycle_ns=1000000000, cost_valid=True, previous_span=1)
        with self.assertRaises(mod.Invalid): mod.check_log(concurrent_log(c=bad), CONFIG, True, 4)
        bad_pause = dict(pause, epoch=2)
        with self.assertRaises(mod.Invalid): mod.check_log(concurrent_log(pause=bad_pause), CONFIG, True, 4)

    def test_actual_mixed_resume_accepts_k0_without_a_verified_draft(self):
        text = concurrent_log()
        proof = mod.check_mixed_resume(text, 7)
        self.assertEqual(proof[0]['k'], 0)
        self.assertEqual(proof[0]['epoch'], 4)
        with self.assertRaises(mod.Invalid): mod.check_mixed_resume(text, 8)
        with self.assertRaises(mod.Invalid): mod.check_mixed_resume(text, 7, minimum_ctx=100)
        a, c, _, _, _ = concurrent_fixture()
        bad = copy.deepcopy(c); bad[3]['published'] = False
        with self.assertRaises(mod.Invalid): mod.check_mixed_resume(concurrent_log(c=bad), 7)


if __name__ == "__main__":
    unittest.main()
