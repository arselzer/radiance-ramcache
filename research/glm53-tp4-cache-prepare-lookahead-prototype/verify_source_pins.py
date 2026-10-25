#!/usr/bin/env python3
"""Source/hash only; no target import/compiler/runtime/device."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
for f,h in r['files'].items():assert sha(HERE/f)==h,('derivative drift',f)
for f,h in r['input_pins'].items():assert sha(ROOT/f)==h,('parent/source drift',f)
old=(ROOT/'glm53-tp4-cache-prepare-coordinator-prototype/cache_prepare4.h').read_text()
expected=old.replace('// Closed CPU executable specification. No Core/GPU/collective recovery support.','// NEW truthful-lookahead derivative; closed GLM main/no-PLE CPU specification.\n// No Core/GPU/collective recovery support; no actual pass overlap is admitted.')
expected=expected.replace('inline constexpr int ranks=4,experts=288,top_k=8,max_rows=128;','inline constexpr int ranks=4,experts=288,top_k=8,max_rows=128;\ninline constexpr int max_lookahead=max_rows; // staged tokens only; not model-executed rows')
expected=expected.replace('p.draft_pass||p.n_spec||p.n_ahead||p.total_rows<1','p.draft_pass||p.n_spec||p.n_ahead<0||p.n_ahead>max_lookahead||p.total_rows<1')
expected=expected.replace('(p.phase==Phase::decode&&p.total_rows!=1)||','(p.phase==Phase::decode&&(p.total_rows!=1||p.n_ahead!=0))||')
assert expected==(HERE/'cache_prepare4.h').read_text(),'undisclosed state-machine change'
print('PASS hash-only truthful-lookahead source',len(r['files']),'files;',len(r['input_pins']),'inputs; exact declared arithmetic delta only')
