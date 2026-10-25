#!/usr/bin/env python3
"""Execute the pinned official Engram forward bodies with a tiny NumPy CPU facade.

No PyTorch/device dependency or model weights. The AST extracts the original
function bodies unchanged; decorators/annotations are omitted. The facade covers
only operations used by these two reference functions. This is a small synthetic
math oracle, not a substitute for later PyTorch/GPU and full-model quality tests.

Usage: python3 THIS.py OFFICIAL_REFERENCE_DIR OUTPUT_JSON
OFFICIAL_REFERENCE_DIR contains inference/{engram,model}.py at
DeepSeek-AI/DeepSeek-V4.1-Flash-1017 commit
1bae0e1088993b9b82de08f59de5be5a34e42170 (MIT).
"""
import ast
import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def raw(x):
    return x.a if isinstance(x, Tensor) else x


def bf16(a):
    bits = np.asarray(a, np.float32).view(np.uint32)
    finite = (bits & 0x7f800000) != 0x7f800000
    rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
    return np.where(finite, rounded, bits).astype(np.uint32).__and__(np.uint32(0xffff0000)).view(np.float32)


class Tensor:
    def __init__(self, a, dtype=None):
        self.a = np.asarray(a)
        self.kind = dtype
    @property
    def shape(self):return self.a.shape
    @property
    def dtype(self):return self.kind or self.a.dtype
    @property
    def device(self):return 'cpu'
    def __getitem__(self, key):
        if isinstance(key, tuple):key=tuple(raw(k) for k in key)
        else:key=raw(key)
        return Tensor(self.a[key])
    def __setitem__(self, key, value):self.a[key]=raw(value)
    def size(self, axis):return self.a.shape[axis]
    def float(self):return Tensor(self.a.astype(np.float32))
    def to(self, dtype):return Tensor(bf16(self.a),'bf16') if dtype=='bf16' else Tensor(self.a.astype(dtype))
    def expand(self,*dims):return Tensor(np.broadcast_to(self.a,dims))
    def gather(self,dim,indices):return Tensor(np.take_along_axis(self.a,raw(indices),axis=dim))
    def clamp_min(self,value):return Tensor(np.maximum(self.a,value))
    def unsqueeze(self,axis):return Tensor(np.expand_dims(self.a,axis))
    def square(self):return Tensor(np.square(self.a))
    def mean(self,axis):return Tensor(np.mean(self.a,axis=axis))
    def sum(self,axis):return Tensor(np.sum(self.a,axis=axis))
    def abs(self):return Tensor(np.abs(self.a))
    def sqrt(self):return Tensor(np.sqrt(self.a))
    def flatten(self,start,end=-1):
        start%=self.a.ndim;end%=self.a.ndim
        return Tensor(self.a.reshape(self.a.shape[:start]+(-1,)+self.a.shape[end+1:]))
    def unflatten(self,axis,dims):
        axis%=self.a.ndim
        return Tensor(self.a.reshape(self.a.shape[:axis]+tuple(dims)+self.a.shape[axis+1:]))
    def split(self,sizes,dim=-1):return tuple(Tensor(a) for a in np.split(self.a,np.cumsum(sizes)[:-1],axis=dim))
    def masked_fill(self,mask,value):return Tensor(np.where(raw(mask),value,self.a))
    def __add__(self,x):return Tensor(self.a+raw(x))
    def __radd__(self,x):return self+x
    def __sub__(self,x):return Tensor(self.a-raw(x))
    def __rsub__(self,x):return Tensor(raw(x)-self.a)
    def __mul__(self,x):return Tensor(self.a*raw(x))
    def __rmul__(self,x):return self*x
    def __mod__(self,x):return Tensor(self.a%raw(x))
    def __or__(self,x):return Tensor(self.a|raw(x))
    def __lt__(self,x):return Tensor(self.a<raw(x))
    def __eq__(self,x):return Tensor(self.a==raw(x))
    def __invert__(self):return Tensor(~self.a)


class TorchFacade:
    bool=np.bool_
    last_sigmoid=None
    def where(self,condition,a,b):return Tensor(np.where(raw(condition),raw(a),raw(b)))
    def arange(self,*args,**kwargs):return Tensor(np.arange(*args))
    def zeros_like(self,x,dtype=None):return Tensor(np.zeros_like(raw(x),dtype=dtype))
    def stack(self,x,dim):return Tensor(np.stack([raw(a) for a in x],axis=dim))
    def cat(self,x,dim):return Tensor(np.concatenate([raw(a) for a in x],axis=dim))
    def bitwise_xor(self,a,b):return Tensor(np.bitwise_xor(raw(a),raw(b)))
    def rsqrt(self,x):return Tensor(np.float32(1)/np.sqrt(raw(x)))
    def copysign(self,a,b):return Tensor(np.copysign(raw(a),raw(b)))
    def sigmoid(self,x):
        self.last_sigmoid=Tensor(np.float32(1)/(np.float32(1)+np.exp(-raw(x))))
        return self.last_sigmoid


def official_function(path,class_name,name,torch):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    function.decorator_list=[];function.returns=None
    for argument in [*function.args.args,*function.args.kwonlyargs]:argument.annotation=None
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def main(reference,out_path):
    expected_sources={'inference/engram.py': '11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897', 'inference/model.py': 'e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d'}
    for name,expected in expected_sources.items():
        if hashlib.sha256((reference/name).read_bytes()).hexdigest()!=expected:
            raise ValueError("official source differs from pinned 1017 fixture reference: "+name)
    torch=TorchFacade()
    hash_forward=official_function(reference/'inference/engram.py','NgramHashState','forward',torch)
    gate_forward=official_function(reference/'inference/model.py','Engram','forward',torch)
    token_map=[0,1,3,1,2,5,4,0,2,3]
    primes=np.array([[11,13,17,19,23,29],[31,37,41,43,47,53]],np.int64)
    offsets=np.stack([np.cumsum([0,*row[:-1]]) for row in primes])
    multipliers=np.array([[3,7,11,13],[23,29,31,37]],np.int64)
    ids=np.array([[0,3,2,5,6,8,7,4,9,1,5,0]],np.int64)
    mask=np.array([[1,1,1,0,1,1,1,1,0,1,1,1]],np.bool_)
    controls={'asset_identity':'synthetic-official-1017-engram','compressed_vocab_size':6,
              'raw_pad_id':2,'max_ngram_size':4,'n_heads':2,'head_dim':3,'token_map':token_map,
              'layers':[{'layer_id':layer,'table_rows':int(p.sum()),'primes':p.tolist(),'offsets':o.tolist(),
                         'multipliers':m.tolist()} for layer,p,o,m in zip([1,14],primes,offsets,multipliers)]}
    state=SimpleNamespace(layout=SimpleNamespace(max_ngram_size=4),token_map=Tensor(np.array(token_map,np.int64)),
                          pad_id=token_map[2],cache=Tensor(np.full((1,32),-999,np.int64)),DEAD=-1,
                          primes=Tensor(primes.reshape(2,3,2)),offsets=Tensor(offsets),multipliers=Tensor(multipliers))
    hashes=hash_forward(state,Tensor(ids),0,Tensor(mask)).a
    state.cache=Tensor(np.full((1,32),-999,np.int64))
    pieces=[]
    for begin,end in [(0,1),(1,4),(4,6),(6,12)]:
        pieces.append(hash_forward(state,Tensor(ids[:,begin:end]),begin,Tensor(mask[:,begin:end])).a)
    assert np.array_equal(hashes,np.concatenate(pieces,axis=1))
    random=np.random.default_rng(1017)
    x=random.normal(0,.6,(1,3,2,4)).astype(np.float32)
    kv=random.normal(0,.5,(1,3,12)).astype(np.float32)
    q=np.array([[.25,.5,1,2],[1.5,.75,.5,.125]],np.float32)
    k=np.array([[1,.25,.75,.5],[.5,1.25,2,1]],np.float32)
    gate_mask=np.array([[1,0,1]],np.bool_)
    gate_cases=[]
    for output_dtype in ['float32','bf16']:
        h=x if output_dtype=='float32' else bf16(x)
        projected=kv if output_dtype=='float32' else bf16(kv)
        model=SimpleNamespace(wkv=lambda unused:Tensor(projected),embed=lambda unused:Tensor(np.zeros((1,3,6,3),np.float32)),
                              hc_mult=2,dim=4,q_weight=Tensor(q),k_weight=Tensor(k),eps=1e-6,clamp_value=1e-6)
        output=gate_forward(model,Tensor(h,output_dtype if output_dtype=='bf16' else None),Tensor(np.zeros((1,3,6),np.int64)),Tensor(gate_mask)).a
        gate=np.where(gate_mask[...,None],torch.last_sigmoid.a,np.float32(0))
        gate_cases.append({'dtype':output_dtype,'tokens':3,'copies':2,'dim':4,'eps':1e-6,
                           'x':h.flatten().tolist(),'projected_kv':projected.flatten().tolist(),
                           'q_weight':q.flatten().tolist(),'k_weight':k.flatten().tolist(),
                           'token_mask':gate_mask.astype(int).flatten().tolist(),
                           'gates':gate.flatten().tolist(),'output':output.flatten().tolist()})
    for zero in ['query','key']:
        h=np.zeros_like(x) if zero=='query' else x
        projected=kv.copy()
        if zero=='key':projected[...,:8]=0
        model=SimpleNamespace(wkv=lambda unused:Tensor(projected),embed=lambda unused:Tensor(np.zeros((1,3,6,3),np.float32)),
                              hc_mult=2,dim=4,q_weight=Tensor(q),k_weight=Tensor(k),eps=1e-6,clamp_value=1e-6)
        output=gate_forward(model,Tensor(h),Tensor(np.zeros((1,3,6),np.int64)),Tensor(gate_mask)).a
        gate=np.where(gate_mask[...,None],torch.last_sigmoid.a,np.float32(0))
        gate_cases.append({'dtype':'float32','boundary':'zero_'+zero,'tokens':3,'copies':2,'dim':4,'eps':1e-6,
                           'x':h.flatten().tolist(),'projected_kv':projected.flatten().tolist(),
                           'q_weight':q.flatten().tolist(),'k_weight':k.flatten().tolist(),
                           'token_mask':gate_mask.astype(int).flatten().tolist(),
                           'gates':gate.flatten().tolist(),'output':output.flatten().tolist()})
    result={'source':'deepseek-ai/DeepSeek-V4.1-Flash-1017 1bae0e1088993b9b82de08f59de5be5a34e42170',
            'numpy_version':np.__version__,'source_sha256':expected_sources,
            'method':'Unmodified official forward bodies executed via AST using a small NumPy float32 CPU facade; no model weights or GPU.',
            'controls':controls,'raw_ids':ids.flatten().tolist(),'token_mask':mask.astype(int).flatten().tolist(),
            'hashes':hashes.flatten().tolist(),'gate_cases':gate_cases}
    out_path.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
