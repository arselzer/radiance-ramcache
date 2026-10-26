#!/usr/bin/env python3
"""Read/hash only; no target import, compiler, device, payload or application."""
from pathlib import Path
import hashlib,json,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=lambda d:{str(p.relative_to(d)):sha(p) for p in sorted(d.rglob('*')) if p.is_file()}
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
core=ROOT/r['new_core'];base=ROOT/r['base_core']
a=files(core);b=files(base)
assert a==json.loads((HERE/'CORE_SOURCE_PINS.json').read_text()),'new whole-Core drift'
assert b==json.loads((HERE/'BASE_SOURCE_PINS.json').read_text()),'frozen parent drift'
for rel,value in r['prototype_sources'].items():assert sha(HERE/rel)==value,('prototype drift',rel)
for rel,value in r['reviewed_inputs'].items():assert sha(ROOT/rel)==value,('input drift',rel)
for new,old in r['preserved_source_bodies'].items():assert (HERE/new).read_bytes()==(ROOT/old).read_bytes(),('inherited body drift',new)
for rel,value in b.items():
 if rel.startswith('abi/'):assert a[rel]==value,('ABI drift',rel)
for rel,value in r['peer_owned_six_paths'].items():assert a[rel]==value,('peer body drift',rel)
m=json.loads((core/'core/build/glm_main_source_data/COMPOSITION.json').read_text())
for rel,pins in m['files'].items():
 source=ROOT/'glm53-verified-main-source-prototype'/rel
 copy=core/'core/build/glm_main_source_data'/rel
 assert sha(source)==pins['parent_sha256'] and sha(copy)==pins['private_sha256']
 text=source.read_text()
 for old,new in m['namespace_only_replacements']:text=text.replace(old,new)
 for old,new in m.get('namespace_reference_only_replacements',[]):text=re.sub(old,new,text)
 assert text.encode()==copy.read_bytes(),('non-namespace source delta',rel)
def function_body(root,rel,signature):
 text=(root/rel).read_text();start=text.index(signature);at=text.index('{',start);depth=0;state='normal';i=at
 while i<len(text):
  c=text[i];n=text[i:i+2]
  if state=='line':
   if c=='\n':state='normal'
  elif state=='block':
   if n=='*/':state='normal';i+=1
  elif state in ('"',"'"):
   if c=='\\':i+=1
   elif c==state:state='normal'
  else:
   if n=='//':state='line';i+=1
   elif n=='/*':state='block';i+=1
   elif c in ('"',"'"):state=c
   elif c=='{':depth+=1
   elif c=='}':
    depth-=1
    if not depth:return text[start:i+1]
  i+=1
 raise AssertionError(('unclosed function',rel,signature))
for rel,signature in r['unchanged_function_bodies']:
 assert function_body(core,rel,signature)==function_body(base,rel,signature),('old function drift',rel,signature)
print('PASS source/hash only:',len(a),'new Core files;',len(r['prototype_sources']),'prototype pins;',len(r['preserved_source_bodies']),'unchanged inherited source bodies; no runtime claim')
