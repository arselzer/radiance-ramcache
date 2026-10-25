#!/usr/bin/env python3
"""Read/hash only: no compiler, target import, checker/device/model or service."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
for rel,h in r['prototype_sources'].items():assert sha(HERE/rel)==h,('prototype drift',rel)
for rel,h in r['reviewed_inputs'].items():assert sha(ROOT/rel)==h,('frozen input drift',rel)
core=ROOT/'radiance-glm-engine-native-transport-v4-src'
actual={str(p.relative_to(core)):sha(p) for p in core.rglob('*') if p.is_file()}
assert actual==json.loads((HERE/'CORE_SOURCE_PINS.json').read_text()),'whole transport Core drift'
v1=ROOT/'radiance-glm-engine-native-transport-src'
assert {str(p.relative_to(v1)):sha(p) for p in v1.rglob('*') if p.is_file()}==json.loads((ROOT/'glm53-tiny-engine-native-transport-prototype/CORE_SOURCE_PINS.json').read_text()),'frozen v1 transport Core drift'
v2=ROOT/'radiance-glm-engine-native-transport-v2-src'
assert {str(p.relative_to(v2)):sha(p) for p in v2.rglob('*') if p.is_file()}==json.loads((ROOT/'glm53-tiny-engine-native-transport-v2-prototype/CORE_SOURCE_PINS.json').read_text()),'frozen v2 transport Core drift'
v3=ROOT/'radiance-glm-engine-native-transport-v3-src'
assert {str(p.relative_to(v3)):sha(p) for p in v3.rglob('*') if p.is_file()}==json.loads((ROOT/'glm53-tiny-engine-native-transport-v3-prototype/CORE_SOURCE_PINS.json').read_text()),'frozen v3 diagnostic Core drift'
base=ROOT/'radiance-glm-engine-native-foundation-v2-src'
assert {str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()}==r['frozen_base_core'],'frozen Foundation Core drift'
for copied,original in r['preserved_source_bodies'].items():assert (HERE/copied).read_bytes()==(ROOT/original).read_bytes(),('inherited test body drift',copied)
m=json.loads((HERE/'FACTORY_COPY_NAMESPACE_MAP.json').read_text())
for rel,pins in m['files'].items():
    source=ROOT/pins['original_source'];copy=core/'core/build/tiny_native_source'/rel
    assert sha(source)==pins['original'] and sha(copy)==pins['core_copy'],('factory source/copy drift',rel)
    s=copy.read_text()
    for old,new in m['namespace_only_mapping'].items():s=s.replace(new,old)
    assert s.encode()==source.read_bytes(),('non-namespace factory mutation',rel)
print('PASS hash-only transport',len(actual),'Core files;',len(r['prototype_sources']),'prototype sources;',len(r['preserved_source_bodies']),'unchanged fixture bodies;',len(m['files']),'reversible factory copies')
