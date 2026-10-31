from pathlib import Path
import hashlib
import json
import subprocess
W=Path('/home/as/ds41-claude-build/work')
D=W/'autonomous-20261031/dme-quiet'
T=W/'autonomous-20261031/dpc'
O=Path('/home/as/radiance-ds41-claude/tools/ds41-host-experts/build/ab-20261031T130014Z')
assert not (W/'WINDOW_PENDING').exists()
assert (D/'matched-costs.json').is_file()
out=D/'fit-once.json'
assert not out.exists(), 'Never refit against heldout feedback'
arms=(T/'arms.txt').read_text().splitlines()
traces=[]
for family in ('code','chat'):
 suffix='-Ucode256' if family=='code' else '-Uchat'
 a=next(a for a in arms if '-Ds2-' in a and '-Dpc-' in a and a.endswith(suffix))
 traces += ['--trace',family+'2='+str(O/(a+'.server.log'))]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan={'action':'Single TRAIN threshold search; global production candidate only; family results diagnostics',
 'traces':traces,'metadata':str(T/'metadata-frozen.json'),'metadata_sha256':sha(T/'metadata-frozen.json'),
 'costs':str(D/'matched-costs.json'),'costs_sha256':sha(D/'matched-costs.json'),
 'seed':'ds41-prob-v1','heldout_fraction':0.25,'grid_points':21,'bootstrap':200,
 'excluded':'Db1/verify-depth1 capture is not mixed into this Db2 policy replay. No iterative heldout selection.'}
assert not (D/'fit-plan.json').exists()
(D/'fit-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
subprocess.run(['python3','-I','/home/as/ds41-claude-build/wt-sync/tools/ds41-host-experts/dspark_prob_analyze.py','analyze',*traces,
 '--metadata',str(T/'metadata-frozen.json'),'--costs',str(D/'matched-costs.json'),
 '--seed','ds41-prob-v1','--heldout-fraction','0.25','--grid-points','21','--bootstrap','200','--out',str(out)],check=True)
x=json.loads(out.read_text())
print(json.dumps(x.get('candidates'),indent=2))
