#!/usr/bin/env python3
"""Read/hash only; no compiler, target import, checker/model/device/network."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
for rel,h in r['sources'].items():assert sha(HERE/rel)==h,('package drift',rel)
for rel,h in r['frozen_inputs'].items():assert sha(ROOT/rel)==h,('input drift',rel)
for rel,original in r['exact_copied_bodies'].items():assert (HERE/rel).read_bytes()==(ROOT/original).read_bytes(),('copied body drift',rel)
core=ROOT/r['matching_core'];actual={str(p.relative_to(core)):sha(p) for p in core.rglob('*') if p.is_file()}
assert actual==json.loads((HERE/'CORE_SOURCE_PINS.json').read_text()),'frozen whole Core drift'
s=(HERE/'storage/candidate_native_record_store.h').read_text()
s=s.replace('namespace glm53::verified_main { class VerifiedBundle; }\n','').replace('    friend class ::glm53::verified_main::VerifiedBundle; // local actual-reader-only admission; no public POD constructor\n','')
assert s.encode()==(ROOT/'glm53-sharded-native-provider-prototype/candidate_native_record_store.h').read_bytes(),'unchecked native catalog constructor/schema delta'
def body(text,fn):
 start=text.index(fn);at=text.index('{',start);depth=0
 for end in range(at,len(text)):
  if text[end]=='{':depth+=1
  elif text[end]=='}':
   depth-=1
   if not depth:return text[start:end+1]
 raise AssertionError('unclosed body')
old=(ROOT/'glm53-sharded-native-provider-prototype/rank_native_store.cpp').read_text();new=(HERE/'storage/rank_native_store.cpp').read_text()
assert body(old,'Error RankNativeStore::open(')==body(new,'Error RankNativeStore::open('),'old generic open widened'
print('PASS hash-only verified main source',len(r['sources']),'package sources;',len(actual),'unchanged Core files;',len(r['exact_copied_bodies']),'exact copied bodies; no runtime assertion')
