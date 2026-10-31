"""Run under build.lock/nice AFTER quiet window; creates provenance, never fits thresholds."""
from pathlib import Path
import hashlib
import json
import subprocess

W = Path('/home/as/ds41-claude-build/work')
D = W / 'autonomous-20261031/dme-quiet'
O = Path('/home/as/radiance-ds41-claude/tools/ds41-host-experts') / (D / 'window-relative.txt').read_text().strip()
assert not (W / 'WINDOW_PENDING').exists()
assert 'window exit 0 ' in (W / 'winDMEQ.log').read_text()
assert 'QFN up after' in (O / 'window.log').read_text()
assert (W / 'autonomous-20261031/dme-probe/summary.json').is_file()
c = json.loads((D / 'candidate.json').read_text())
provenance = json.loads((D / 'runtime-provenance.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
metadata_path = W / 'autonomous-20261031/dpc/metadata-frozen.json'
meta = json.loads(metadata_path.read_text())['requests']
arms = c['arms']
data = {}
for a in arms:
    assert not (O / (a + '.unmeasured.json')).exists(), a
    x = json.loads((O / (a + '.json')).read_text())
    assert x['concurrency'] == 1 and x['requests']
    assert all(not r['error'] and isinstance(r['text'], str) for r in x['requests'])
    data[a] = x['requests']
summary = {'window': O.name, 'plain_reply_counts': {}, 'rates': {}, 'fit_performed': False}
for key in ('codecost256', 'chatcost'):
    names = [a for a in arms if a.endswith('-U' + key)]
    plain = next(a for a in names if '-Ds' not in a)
    for a in names:
        assert [r['name'] for r in data[a]] == [r['name'] for r in data[plain]]
        for r, s in zip(data[a], data[plain]):
            assert r['text'].encode() == s['text'].encode(), (a, r['name'])
            for k in ('prompt_n', 'predicted_n'):
                assert r['timings'][k] == s['timings'][k], (a, r['name'], k)
        summary['rates'][a] = sum(r['timings']['predicted_n'] for r in data[a]) / sum(r['timings']['predicted_ms'] for r in data[a]) * 1000
    summary['plain_reply_counts'][key] = (len(names) - 1) * len(data[plain])

libs = {Path(name).name: digest for name, digest in c['libraries'].items()}
profile = {'id': O.name + '-common', 'hardware': '256GB DDR4-2666;4xRX9070XT16GB;192GiB expert cache;startup ledgers retained',
           'engine_commit': provenance['runtime_built_commit'], 'common_config_sha256': c['common_config_sha256'], 'libraries': libs}
profile['source_provenance'] = {'path': str(D / 'runtime-provenance.json'),
                                'sha256': sha(D / 'runtime-provenance.json'),
                                'reviewed_source_commit': c['engine_commit']}
fp = {}
for family, key in (('code', 'codecost256'), ('chat', 'chatcost')):
    cfg = c['family_profiles'][key]
    assert sha(Path(cfg['path'])) == cfg['sha256']
    fp[family] = {'id': family + '-original-db2', 'config_sha256': cfg['sha256'],
                  'hotset': {'path': cfg['hotset'], 'sha256': cfg['hotset_sha256']}}
manifest = {'schema': 'ds41-matched-cost-manifest-v1',
            'measurement_profile': O.name + ';native Db2 observer+early wait;quiet TRAIN only;no fitting',
            'same_drafter_observer_wait': True,
            'authority_assertion': 'Actual serving build3fe07391 is recorded separately from later fixture/harness source2167bd20; runtime-provenance.json retains the complete allowed diff. Operator retained the source/build snapshot and coherent plugin hashes,10 complete arms, original Db2 hotsets and192GiB profile. Existing verbose binding audit passed; no model warmup generation.',
            'profile': profile, 'family_profiles': fp,
            'frozen_split': {'metadata': str(metadata_path), 'sha256': sha(metadata_path), 'seed': 'ds41-prob-v1', 'heldout_fraction': 0.25},
            'runs': []}
for a in arms:
    if '-Dme' not in a:
        continue
    k = int(next(p[3:] for p in a.split('-') if p.startswith('Dme')))
    family = 'code' if a.endswith('-Ucodecost256') else 'chat'
    frozen_run = family + '2'
    source = {r['prompt_key']: r for r in meta if r['run'] == frozen_run}
    requests = [{'seq': seq, 'frozen_run': frozen_run, 'frozen_seq': source[row['name']]['seq']}
                for seq, row in enumerate(data[a], 1)]
    manifest['runs'].append({
        'run': family + '-C' + str(k), 'family': family, 'k': k, 'scope': 'cost_train',
        'logging': 'quiet', 'log_level': 'info', 'arm': a,
        'log': str(O / (a + '.server.log')), 'results': str(O / (a + '.json')),
        'profile_id': profile['id'], 'engine_commit': profile['engine_commit'],
        'common_config_sha256': profile['common_config_sha256'], 'libraries': libs,
        'family_profile_id': fp[family]['id'], 'config_sha256': fp[family]['config_sha256'], 'hotset': fp[family]['hotset'],
        'provenance_assertion': 'Exact retained launch/source/home cohort; shared query/observer/wait profile and family original Db2 hotset checked.',
        'request_mapping_assertion': 'Fresh server per arm, concurrency1;seq1..N in immutable TRAIN receipt order; no hidden generation or warmup requests.',
        'requests': requests,
    })
p = D / 'manifest.json'
assert not p.exists(), 'Never overwrite provenance'
p.write_text(json.dumps(manifest, indent=2) + '\n')
(D / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
subprocess.run(['python3', '-I', '/home/as/ds41-claude-build/wt-sync/tools/ds41-host-experts/dspark_cost_extract.py',
                '--manifest', str(p), '--out', str(D / 'matched-costs.json')], check=True)
print(json.dumps(summary, indent=2))
