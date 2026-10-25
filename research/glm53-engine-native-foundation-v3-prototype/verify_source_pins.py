#!/usr/bin/env python3
"""Hash/read only. No compiler, checker/model/device, network or service."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
for name,h in r['prototype_sources'].items():
    assert sha(HERE/name)==h,('prototype drift',name)
for name,h in r['reviewed_inputs'].items():
    assert sha(ROOT/name)==h,('input drift',name)
core=ROOT/'radiance-glm-engine-native-foundation-v2-src'
actual={str(p.relative_to(core)):sha(p) for p in sorted(core.rglob('*')) if p.is_file()}
assert actual==json.loads((HERE/'CORE_SOURCE_PINS.json').read_text()),'whole foundation drift'
original=ROOT/'radiance-glm-engine-native-foundation-src'
assert {str(p.relative_to(original)):sha(p) for p in sorted(original.rglob('*')) if p.is_file()}==r['frozen_foundation_v1_core'],'original Foundation Core changed'
base=ROOT/'radiance-glm-tiny-engine-native-src'
assert {str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*')) if p.is_file()}==r['frozen_base_core'],'base Core changed'
for copy,original in r['preserved_body_sources'].items():
    assert (HERE/copy).read_bytes()==(ROOT/original).read_bytes(),('preserved body drift',copy)
# V3 changes only the allocation-family hook block in the main checker.
v2=ROOT/'glm53-engine-native-foundation-v2-prototype'
v2receipt=json.loads((v2/'SOURCE_RECEIPT.json').read_text())
for name,h in v2receipt['prototype_sources'].items():
    assert sha(v2/name)==h,('frozen v2 prototype drift',name)
assert (HERE/'checker.cpp').read_text().replace(r['v3_nothrow_hook_block'],'',1)==(v2/'checker.cpp').read_text(),'v3 changed earlier checker case bodies'
s=(HERE/'contract/checker.cpp').read_text()
s=s.replace(r['contract_checker_additive_headers'],'',1)
a=s.index('      // BEGIN additive foundation private quotes;')
b=s.index('      // END additive foundation private quotes.\n',a)+len('      // END additive foundation private quotes.\n')
s=s[:a]+s[b:]
assert s.encode()==(ROOT/'glm53-tiny-engine-startup-contract-prototype/checker.cpp').read_bytes(),'old contract case bodies changed'
print('PASS hash-only foundation-v3',len(actual),'Core files;',len(r['prototype_sources']),'prototype files;',len(r['preserved_body_sources']),'preserved source bodies; inherited contract body exact')
