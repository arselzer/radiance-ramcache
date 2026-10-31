#!/usr/bin/env python3
"""Synthetic ownership/capture/lender bindings only; no model or timing replay."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('mixed', Path(__file__).with_name('dspark_policy_mixed_gate.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def capture_log(seq=101):
    rows = []
    for epoch, pos in [(1, 20), (2, 600)]:
        proposals = []
        for depth in (1, 2):
            p = dict(v=1, event='proposal', seq=seq, epoch=epoch, anchor_pos=pos, pos=pos+depth,
                     depth=depth, token=70+depth, predecessor=55 if depth == 1 else 71,
                     candidate_count=8, candidate_p=.5, score=4., margin=1., margin_unbounded=False,
                     signal_valid=True, status='pending', verified=None, reached=None, accepted=None,
                     cancelled=False)
            proposals.append(p)
            rows.append(m.capture.MARKER + json.dumps(p))
        for p in proposals:
            out = dict(p, event='outcome', status='verify_cap', verified=False, reached=False, accepted=None)
            rows.append(m.capture.MARKER + json.dumps(out))
    return '\n'.join(rows)


class MixedPolicyGateTest(unittest.TestCase):
    def test_actual_query_capture_by_name_ctx_anchor_and_full_depth(self):
        r = dict(seq_by_name={'anchor': 101})
        queries, epochs = m.proposal_queries(capture_log(), r, 'fixed')
        self.assertEqual(set(queries), {('anchor', 20, 55), ('anchor', 600, 55)})
        self.assertEqual(epochs[(101, 2)], ('anchor', 600, 55))
        with self.assertRaises(ValueError): m.proposal_queries(capture_log(), dict(seq_by_name={'anchor': 102}), 'bad')
        with self.assertRaises(m.capture.Invalid): m.proposal_queries(capture_log() + '\n' + capture_log(), r, 'duplicate')
        missing = '\n'.join(x for x in capture_log().splitlines() if not ('"depth": 2' in x))
        with self.assertRaises(ValueError): m.proposal_queries(missing, r, 'short-query')

    def test_actions_bind_own_query_epoch_and_all_original_proposals(self):
        r = dict(seq_by_name={'anchor': 101})
        queries, epochs = m.proposal_queries(capture_log(), r, 'adaptive')
        actions = [dict(seq=101, epoch=epoch, ctx=ctx, anchor=55, ids=[71, 72]) for epoch, ctx in [(1, 20), (2, 600)]]
        text = lambda a: '\n'.join(m.policy.ACTION + json.dumps(x) for x in a)
        m.bind_actions(text(actions), r, queries, epochs)
        for field, value in [('seq', 102), ('epoch', 3), ('ctx', 601), ('anchor', 56), ('ids', [71, 73])]:
            bad = copy.deepcopy(actions); bad[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): m.bind_actions(text(bad), r, queries, epochs)

    def test_common_anchors_cover_both_solo_periods_and_compare_only_existing_queries(self):
        r = dict(seq_by_name={'anchor': 101})
        queries, _ = m.proposal_queries(capture_log(), r, 'fixed')
        self.assertEqual(m.compare_queries(queries, queries, 500)['anchor_after'], 1)
        bad = dict(queries); key = ('anchor', 600, 55); bad[key] = (71, 99, (), ())
        with self.assertRaises(ValueError): m.compare_queries(queries, bad, 500)
        with self.assertRaises(ValueError): m.compare_queries(queries, queries, 700)

    def test_no_action_while_lent_and_resume_matches_the_real_bank_query(self):
        before = 'DSBANKLEND transition epoch=1 action=lend phase=begin step=4 seq=0'
        restored = 'DSBANKLEND transition epoch=1 action=restore phase=ready step=80 seq=101'
        resumed = 'DSBANKLEND resume epoch=1 step=80 seq=101 query_rows=2 ready=1'
        action = m.policy.ACTION + json.dumps(dict(seq=101, epoch=3))
        text = '\n'.join([before, restored, resumed, action])
        m.bind_lender_policy(text, 101, [dict(step=80)])
        with self.assertRaises(ValueError): m.bind_lender_policy(text, 101, [dict(step=81)])
        with self.assertRaises(ValueError): m.bind_lender_policy('\n'.join([before, action, restored, resumed]), 101, [dict(step=80)])
        with self.assertRaises(ValueError): m.bind_lender_policy('\n'.join([before, resumed, restored]), 101, [dict(step=80)])

    def test_frozen_configuration_and_exact_five_roles_required(self):
        with self.assertRaises(ValueError): m.validate({}, {}, 'v1:0.25:0.75')
        with self.assertRaises(ValueError): m.validate({}, {}, m.FROZEN)

    def test_live_filtered_marker_prefix_is_bound_to_complete_log(self):
        bank = 'DSBANKLEND transition epoch=1 action=lend phase=begin'
        action = m.policy.ACTION + '{"seq":101}'
        receipt = dict(observed_lender_log=bank + '\n' + action)
        m.observed_prefix('ordinary startup\n' + bank + '\nunrelated\n' + action + '\nordinary end', receipt, True)
        with self.assertRaises(ValueError): m.observed_prefix(bank + '\n' + action.replace('101', '102'), receipt, True)
        with self.assertRaises(ValueError): m.observed_prefix(bank + '\n' + action, receipt, False)


if __name__ == '__main__': unittest.main()
