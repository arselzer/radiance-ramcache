#!/usr/bin/env python3
"""Offline mixed-policy byte/ownership/capture gate. No model calls, fitting or timing claims."""
import argparse
import json
from pathlib import Path
import sys

# Explicit trusted sibling imports also work when root invokes the offline checker with -I.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_requests
import ds41_bank_lend_probe as probe
import dspark_policy_gate as policy
import dspark_prob_analyze as capture

FROZEN = 'v1:0.937915564:0.97501117'
ROLES = ('base', 'fixed-off', 'adaptive-off', 'fixed-on', 'adaptive-on')


def need(ok, why):
    if not ok: raise ValueError(why)


def proposal_queries(text, receipt, role):
    records, missing = capture.pair_events(role, text.splitlines())
    need(not missing, role + ': capture has no terminal outcomes')
    names = {seq: name for name, seq in receipt['seq_by_name'].items()}
    groups = {}
    for record in records:
        p = record['proposal']
        need(p['seq'] in names, role + ': capture names a foreign request')
        key = (p['seq'], p['epoch'])
        depths = groups.setdefault(key, {})
        need(p['depth'] not in depths, role + ': duplicate query depth')
        depths[p['depth']] = p
    result, epochs = {}, {}
    signal_fields = ('candidate_count', 'candidate_p', 'score', 'margin', 'margin_unbounded', 'signal_valid')
    for (seq, epoch), depths in groups.items():
        need(set(depths) == {1, 2}, role + ': query was not the full unchanged Db2')
        first, second = depths[1], depths[2]
        need(second['predecessor'] == first['token'] and second['anchor_pos'] == first['anchor_pos'],
             role + ': proposal path changed anchor')
        key = (names[seq], first['anchor_pos'], first['predecessor'])
        need(key not in result, role + ': query anchor repeated ambiguously')
        result[key] = (first['token'], second['token'],
                       tuple(first[k] for k in signal_fields), tuple(second[k] for k in signal_fields))
        epochs[(seq, epoch)] = key
    return result, epochs


def bind_actions(text, receipt, queries, epochs):
    actions = [json.loads(x.split(policy.ACTION, 1)[1]) for x in text.splitlines() if policy.ACTION in x]
    need(actions, 'adaptive role has no actual queries')
    names = {seq: name for name, seq in receipt['seq_by_name'].items()}
    for a in actions:
        key = epochs.get((a['seq'], a['epoch']))
        need(key == (names.get(a['seq']), a['ctx'], a['anchor']), 'policy action rebound another capture anchor/epoch')
        need(tuple(a['ids']) == queries[key][:2], 'policy action IDs differ from actual full Db2 capture')
    need(len(actions) == len(epochs), 'capture/action query cohorts differ')


def compare_queries(reference, candidate, post_ctx):
    common = reference.keys() & candidate.keys()
    need(common, 'no actual common capture anchor')
    need(all(reference[k] == candidate[k] for k in common), 'same actual anchor proposed different IDs/signals')
    before = [k for k in common if k[0] == 'anchor' and k[1] < post_ctx]
    after = [k for k in common if k[0] == 'anchor' and k[1] >= post_ctx]
    need(before and after, 'common proposal anchors do not cover both initial solo and actual resumed tail')
    return dict(common=len(common), anchor_before=len(before), anchor_after=len(after))


def bind_lender_policy(text, anchor, resumes):
    borrowed = set()
    bank_resumes = set()
    for line in text.splitlines():
        if 'DSBANKLEND ' in line:
            row = probe.marker_rows(line)[0]
            if row.get('kind') == 'transition':
                if row.get('action') == 'lend' and row.get('phase') == 'begin': borrowed.add(row['epoch'])
                if row.get('action') == 'restore' and row.get('phase') == 'ready': borrowed.discard(row['epoch'])
            if row.get('kind') == 'resume' and row.get('seq') == str(anchor) and row.get('ready') == '1':
                need(not borrowed, 'bank query resumed before restored publication')
                bank_resumes.add(int(row['step']))
        if policy.ACTION in line:
            need(not borrowed, 'adaptive query/action ran while bank backing was borrowed')
    need(not borrowed, 'adaptive role ended with an open lender interval')
    need(any(r['step'] in bank_resumes for r in resumes), 'adaptive anchor resume is not the actual post-restore bank query')


def observed_prefix(text, receipt, adaptive):
    selected = [x for x in text.splitlines() if 'DSBANKLEND ' in x or adaptive and policy.MARKER in x]
    observed = receipt.get('observed_lender_log', '').splitlines()
    need(observed and observed == selected[:len(observed)], 'live marker prefix differs from final server log')


def validate(receipts, logs, config):
    need(config == FROZEN, 'this composition gate uses the frozen GLOBAL policy only; no threshold refit')
    need(set(receipts) == set(logs) == set(ROLES), 'five exact reference/fixed/adaptive/lender roles required')
    workload = None
    queries, resumed, timings = {}, {}, {}
    expected_role = {'base': 'no-bank', 'fixed-off': 'static', 'adaptive-off': 'static',
                     'fixed-on': 'lend', 'adaptive-on': 'lend'}
    for role in ROLES:
        r, text = receipts[role], logs[role]
        probe.check_receipt(r)
        need(r['role'] == expected_role[role], role + ': wrong actual lender role')
        adaptive = role.startswith('adaptive-')
        need(r.get('policy_config') == (config if adaptive else None), role + ': policy state/config differs')
        policy.check_log(text, config if adaptive else 'off', complete=True, nseq=4)
        need(workload is None or r['workload_sha256'] == workload, 'mixed workload changed across roles')
        workload = r['workload_sha256']
        if r['role'] == 'lend':
            observed_prefix(text, r, adaptive)
            probe.validate_markers(text, r)
        else:
            need(not probe.marker_rows(text), 'lender markers present in a lender-off reference')
        if adaptive:
            observed_prefix(text, r, True)
            proof = policy.check_mixed_resume(text, r['seq_by_name']['anchor'], probe.policy_resume_floor(r))
            if role == 'adaptive-on': bind_lender_policy(text, r['seq_by_name']['anchor'], proof)
            resumed[role] = min(x['ctx'] for x in proof)
        if role != 'base':
            q, epochs = proposal_queries(text, r, role)
            queries[role] = q
            if adaptive: bind_actions(text, r, q, epochs)
        else:
            need(capture.MARKER not in text, 'no-bank reference unexpectedly proposed')
        timings[role] = dict(load_workload_drain_s=r['load_workload_drain_s'],
                             workload_drain_s=r['workload_drain_s'])
    base = receipts['base']
    for role in ROLES[1:]:
        r = receipts[role]
        need([x['name'] for x in r['requests']] == [x['name'] for x in base['requests']], 'ordered reply names changed')
        need(r['workload'] == base['workload'], 'framed token IDs or request bodies changed')
        for a, b in zip(base['requests'], r['requests']):
            need(a['text'].encode('utf-8') == b['text'].encode('utf-8') and
                 a['timings']['predicted_n'] == b['timings']['predicted_n'] and
                 ab_requests.prompt_tokens(a['timings']) == ab_requests.prompt_tokens(b['timings']),
                 role + ': reply bytes or prompt/generated counts differ from no-bank base')
    comparisons = {}
    for fixed, adaptive in [('fixed-off', 'adaptive-off'), ('fixed-on', 'adaptive-on')]:
        r = receipts[fixed]
        fixed_post = len(r['workload']['requests'][0]['body']['prompt']) + r['solo_resumed_output'] - 1
        comparisons[fixed + ':' + adaptive] = compare_queries(queries[fixed], queries[adaptive], max(fixed_post, resumed[adaptive]))
    comparisons['adaptive-off:adaptive-on'] = compare_queries(queries['adaptive-off'], queries['adaptive-on'],
        max(resumed['adaptive-off'], resumed['adaptive-on']))
    return dict(schema='ds41-policy-mixed-binding-v1', valid=True, roles=5, replies=4,
                frozen_config=config, workload_sha256=workload, proposal_anchor_comparisons=comparisons,
                startup_workload_drain_timings=timings,
                note='No proposal equality versus no-bank. Process teardown/outer production restoration/cohort/card receipts remain mandatory in the root lifecycle gate; no performance claim here.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    for role in ROLES:
        p.add_argument('--' + role, required=True)
        p.add_argument('--' + role + '-log', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    receipts, logs = {}, {}
    for role in ROLES:
        receipt = Path(getattr(a, role.replace('-', '_')))
        log = Path(getattr(a, (role + '-log').replace('-', '_')))
        need(receipt.stat().st_size <= 64 << 20 and log.stat().st_size <= 64 << 20, 'unexpectedly large mixed artifact')
        receipts[role] = json.loads(receipt.read_text())
        logs[role] = log.read_text()
    result = validate(receipts, logs, a.config)
    out = Path(a.output)
    need(not out.exists(), 'never overwrite a mixed binding receipt')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print('DS41_POLICY_MIXED_BINDING_PASS roles=5 replies=4 anchor_resumed=1 actual_capture_binding=1')


if __name__ == '__main__': main()
