#!/usr/bin/env python3
"""Source/hash inspection only. No target import, compiler, model or runtime."""
from pathlib import Path
import hashlib,json,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=lambda d:{str(p.relative_to(d)):sha(p) for p in sorted(d.rglob('*')) if p.is_file()}
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text());core=ROOT/r['new_core'];base=ROOT/r['base_core']
a=files(core);b=files(base)
assert a==json.loads((HERE/'CORE_SOURCE_PINS.json').read_text()),'whole new Core drift'
assert b==json.loads((HERE/'BASE_SOURCE_PINS.json').read_text()),'frozen Core8 drift'
assert sorted(set(b)-set(a))==r['omitted_paths']==[]
assert {n for n,h in a.items() if b.get(n)!=h}==set(r['changed_paths'])
for n,h in b.items():
 if n.startswith(('abi/','arch/','libr4d/')):assert a[n]==h,('ABI/math/architecture drift',n)
for rel,value in r['prototype_sources'].items():assert sha(HERE/rel)==value,('fixture drift',rel)
for rel,value in r['reviewed_inputs'].items():assert sha(ROOT/rel)==value,('basis drift',rel)
for new,old in r['preserved_source_bodies'].items():assert (HERE/new).read_bytes()==(ROOT/old).read_bytes(),('preserved body drift',new)
for rel in ['regressions/device/CORE_SOURCE_PINS.json','regressions/compact/strict_p3/UNIFIED_CORE_PINS.json']:
 assert json.loads((HERE/rel).read_text())==a,('nested active map drift',rel)
m=json.loads((core/'core/build/glm_main_source_data/COMPOSITION.json').read_text())
for rel,v in m['files'].items():
 source=ROOT/'glm53-verified-main-source-prototype'/rel;copy=core/'core/build/glm_main_source_data'/rel
 assert sha(source)==v['parent_sha256'] and sha(copy)==v['private_sha256']
 text=source.read_text()
 for old,new in m['namespace_only_replacements']:text=text.replace(old,new)
 for old,new in m.get('namespace_reference_only_replacements',[]):text=re.sub(old,new,text)
 expected=hashlib.sha256(text.encode()).hexdigest()
 assert expected==v.get('prefill_profile_base_private_sha256',v['private_sha256']),('namespace-only BASE drift',rel)
 if 'prefill_profile_base_private_sha256' not in v:assert text.encode()==copy.read_bytes()
 else:assert (base/'core/build/glm_main_source_data'/rel).read_bytes()==text.encode(),('functional overlay base drift',rel)
def body(root,rel,sig):
 text=(root/rel).read_text();start=text.index(sig);at=text.index('{',start);depth=0;state='code';i=at
 while i<len(text):
  c=text[i];pair=text[i:i+2]
  if state=='line':
   if c=='\n':state='code'
  elif state=='block':
   if pair=='*/':state='code';i+=1
  elif state in ('"',"'"):
   if c=='\\':i+=1
   elif c==state:state='code'
  elif pair=='//':state='line';i+=1
  elif pair=='/*':state='block';i+=1
  elif c in ('"',"'"):state=c
  elif c=='{':depth+=1
  elif c=='}':
   depth-=1
   if depth==0:return text[start:i+1]
  i+=1
 raise AssertionError(('unclosed source body',rel,sig))
for rel,sig in r['unchanged_function_bodies']:assert body(core,rel,sig)==body(base,rel,sig),('unchanged function drift',rel,sig)
print('PASS hash/source only',len(a),'Core',len(r['prototype_sources']),'fixture',len(r['preserved_source_bodies']),'preserved',len(r['unchanged_function_bodies']),'unchanged function bodies; no runtime claim')
