#!/usr/bin/env python3
"""Read/hash only: no compiler, checker, target import, device or model action."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=lambda p:{str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*')) if x.is_file()}
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
for rel,h in r['prototype_sources'].items():assert sha(HERE/rel)==h,('prototype drift',rel)
for rel,h in r['reviewed_inputs'].items():assert sha(ROOT/rel)==h,('frozen input drift',rel)
core=ROOT/r['new_core'];base=ROOT/r['base_core']
a=files(core);b=files(base)
assert a==json.loads((HERE/'CORE_SOURCE_PINS.json').read_text()),'whole new Core drift'
assert b==json.loads((HERE/'BASE_SOURCE_PINS.json').read_text()),'frozen v4 Core drift'
for rel,h in r['peer_owned_six_paths'].items():assert a[rel]==h,('peer-owned path drift',rel)
assert all(a[rel]==h for rel,h in b.items() if rel.startswith('abi/')),'original ABI27 changed'
for copied,original in r['preserved_source_bodies'].items():assert (HERE/copied).read_bytes()==(ROOT/original).read_bytes(),('inherited test body drift',copied)
assert (core/'core/runtime/cache_prepare4_coordinator.h').read_bytes()==(ROOT/r['coordinator_original']).read_bytes(),'passed coordinator body changed'
raw=(HERE/'mover_wait_fixture.h').read_text()
raw=raw.replace('bool model_allocation=false,bool raw_session=false):names','bool model_allocation=false):names')
raw=raw.replace('            // New fence-only fixture: actual source/Pools/Mover/bootstrap above; no Owner/Ctx is opened.\n            if(raw_session)return;\n','')
assert raw.encode()==(HERE/'regressions/compact/strict_p3/adapter_fixture.h').read_bytes(),'untracked raw-session fixture delta'
m=json.loads((HERE/'FACTORY_COPY_NAMESPACE_MAP.json').read_text())
for rel,pins in m['files'].items():
    source=ROOT/pins['original_source'];copy=core/'core/build/tiny_native_source'/rel
    assert sha(source)==pins['original'] and sha(copy)==pins['core_copy'],('factory copy drift',rel)
    s=copy.read_text()
    for old,new in m['namespace_only_mapping'].items():s=s.replace(new,old)
    assert s.encode()==source.read_bytes(),('non-namespace source copy mutation',rel)
print('PASS hash-only TP4 first slice',len(a),'Core files;',len(r['prototype_sources']),'prototype sources;',len(r['preserved_source_bodies']),'unchanged inherited bodies; closed full-source activation; no runtime assertion')
