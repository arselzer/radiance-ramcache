#!/usr/bin/env python3
"""Read/hash only; no compiler, target import, payload/packing/device/model or services."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
for name,h in r['files'].items():assert sha(HERE/name)==h,('packer source drift',name)
for name,h in r['reviewed_inputs'].items():assert sha(ROOT/name)==h,('frozen input drift',name)
for copy,original in r['preserved_bodies'].items():assert (HERE/copy).read_bytes()==(ROOT/original).read_bytes(),('copied body drift',copy)
pins=json.loads((HERE/'CORE_SOURCE_PINS.json').read_text())
for name,h in pins.items():assert sha(ROOT/'radiance-src'/name)==h,('stock Core source drift',name)
print('PASS hash-only actual main packer',len(r['files']),'files;',len(pins),'matching original Core pins;',len(r['preserved_bodies']),'exact inherited bodies; no payload execution')
