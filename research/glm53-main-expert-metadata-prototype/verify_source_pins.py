#!/usr/bin/env python3
"""Read/hash only: no compiler, target import, model/device/API or service."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
for name,h in r['files'].items():assert sha(HERE/name)==h,('prototype source drift',name)
for name,h in r['reviewed_inputs'].items():assert sha(ROOT/name)==h,('frozen input drift',name)
pins=json.loads((HERE/'CORE_SOURCE_PINS.json').read_text())
for name,h in pins.items():assert sha(ROOT/'radiance-src'/name)==h,('original stock Core drift',name)
assert (HERE/'CORE_SOURCE_PINS.json').read_bytes()==(ROOT/'glm53-stock-full-graph-plan-v4-prototype/CORE_SOURCE_PINS.json').read_bytes()
print('PASS hash-only main metadata',len(r['files']),'prototype files;',len(pins),'matching original Core pins; frozen stock parent preserved')
