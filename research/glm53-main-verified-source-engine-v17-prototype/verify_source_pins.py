#!/usr/bin/env python3
"""Source/hash-only inspection. No target import, compiler, model or runtime."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def files(d):
 assert not any(p.is_symlink() for p in d.rglob('*')),('symlink',d)
 return {str(p.relative_to(d)):sha(p) for p in sorted(d.rglob('*')) if p.is_file()}
r=json.loads((HERE/'SOURCE_RECEIPT.json').read_text());core=ROOT/r['new_core'];base=ROOT/r['base_core']
a=files(core);b=files(base)
assert a==json.loads((HERE/'CORE_SOURCE_PINS.json').read_text()),'whole Core12 drift'
assert b==json.loads((HERE/'BASE_SOURCE_PINS.json').read_text()),'frozen Core9 drift'
assert set(a)==set(b)
assert {n for n,h in a.items() if b[n]!=h}==set(r['changed_paths'])
for n,h in b.items():
 if n.startswith(('abi/','arch/','libr4d/')):assert a[n]==h,('ABI/math/architecture drift',n)
for rel,h in r['prototype_sources'].items():assert sha(HERE/rel)==h,('fixture drift',rel)
assert set(files(HERE))==set(r['prototype_sources'])|{'SOURCE_RECEIPT.json'},'closed fixture file inventory drift'
for rel,h in r['reviewed_inputs'].items():assert sha(ROOT/rel)==h,('input drift',rel)
for rel,source in r['preserved_source_bodies'].items():assert (HERE/rel).read_bytes()==(ROOT/source).read_bytes(),('old body drift',rel)
for rel in ('regressions/device/CORE_SOURCE_PINS.json','regressions/compact/strict_p3/UNIFIED_CORE_PINS.json'):
 assert json.loads((HERE/rel).read_text())==a,('nested private map drift',rel)
old=(base/'core/engine_glm_main.cpp').read_text();new=(core/'core/engine_glm_main.cpp').read_text()
o=old.index('int Engine::glm_main_body(){auto& s=*glm_main_;');oe=old.index('    size_t count=0;',o)
n=new.index('int Engine::glm_main_metadata_prepare(){auto& s=*glm_main_;');ne=new.index('    size_t count=0;',n)
assert new[:n]+old[o:oe]+new[ne:]==old.replace('build_ctx(r->index,1)','build_ctx(r->index,cfg_.max_tok)'), 'production extraction or one-line delta drift'
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
 raise AssertionError(('unclosed body',rel,sig))
for rel,sig in r['unchanged_function_bodies']:assert body(core,rel,sig)==body(base,rel,sig),('old function drift',rel,sig)
parent=ROOT/'glm53-main-verified-source-engine-v16-prototype'
old=(parent/'declaration_checker.cpp').read_text();new=(HERE/'declaration_checker.cpp').read_text()
assert old.count('snapshot.phase==GlmMainEnginePhase::Retired')==1
assert new.replace('snapshot.phase==GlmMainEnginePhase::Terminal','snapshot.phase==GlmMainEnginePhase::Retired')==old,'sole HOST terminal-phase expectation correction'
print('PASS hash/source only:',len(a),'Core;',len(r['prototype_sources']),'fixture;',len(r['preserved_source_bodies']),'preserved bodies;',len(r['unchanged_function_bodies']),'old functions; no runtime claim')
