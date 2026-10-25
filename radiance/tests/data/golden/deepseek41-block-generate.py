#!/usr/bin/env python3
"""Tiny serial block oracle: actual official Block/Gate/Expert/MoE bodies + NumPy.
Requires the sibling Engram CPU facade, NumPy and pinned official reference source.
No PyTorch/device/model dependency. See deepseek41-block.README.txt.
"""
import ast,copy,hashlib,importlib.util,json,sys,types
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('engram_facade',Path(__file__).with_name('deepseek41-engram-generate.py'))
facade=importlib.util.module_from_spec(spec);spec.loader.exec_module(facade)
T=facade.Tensor;raw=facade.raw;bf16=facade.bf16
T.size=lambda self,axis=None:self.a.shape if axis is None else self.a.shape[axis]
T.view=lambda self,*shape:T(self.a.reshape(shape[0] if len(shape)==1 and isinstance(shape[0],tuple) else shape),self.kind)
T.type_as=lambda self,other:self.to(other.dtype)
T.mean=lambda self,axis=None,keepdim=False:T(self.a.mean(axis=axis,keepdims=keepdim))
T.sum=lambda self,axis=None,keepdim=False,dim=None:T(self.a.sum(axis=dim if dim is not None else axis,keepdims=keepdim))
T.flatten=lambda self,start=0,end=-1:T(self.a.reshape(self.a.shape[:start%self.a.ndim]+(-1,)+self.a.shape[(end%self.a.ndim)+1:]),self.kind)
T.unsqueeze=lambda self,axis:T(np.expand_dims(self.a,axis),self.kind)
T.__truediv__=lambda self,value:T(self.a/raw(value))
T.tolist=lambda self:self.a.tolist()
def get(self,key):
    key=tuple(raw(v) for v in key) if isinstance(key,tuple) else raw(key)
    return T(self.a[key],self.kind)
def put(self,key,value):
    key=tuple(raw(v) for v in key) if isinstance(key,tuple) else raw(key)
    self.a[key]=raw(value)
T.__getitem__=get;T.__setitem__=put

def topk(self,k,dim=-1):
    index=np.argsort(-self.a,axis=dim,kind='stable');slices=[slice(None)]*self.a.ndim;slices[dim]=slice(0,k);index=index[tuple(slices)]
    return T(np.take_along_axis(self.a,index,axis=dim)),T(index)
T.topk=topk

def array(x):return np.asarray(x,np.float32)
def vector(x):return raw(x).reshape(-1).tolist()
class F:
    @staticmethod
    def linear(x,weight):return T(np.matmul(raw(x),raw(weight).T))
    @staticmethod
    def softplus(x):return T(np.where(raw(x)>20,raw(x),np.log1p(np.exp(raw(x)))))
    @staticmethod
    def silu(x):return T(raw(x)/(np.float32(1)+np.exp(-raw(x))))


def sinkhorn(mixes,scale,base,hc,iterations,eps):
    a=raw(mixes);s=raw(scale);b=raw(base)
    pre=1/(1+np.exp(-(a[...,:hc]*s[0]+b[:hc])))+eps
    post=2/(1+np.exp(-(a[...,hc:2*hc]*s[1]+b[hc:2*hc])))
    comb=(a[...,2*hc:]*s[2]+b[2*hc:]).reshape(*a.shape[:-1],hc,hc)
    comb=np.exp(comb-comb.max(axis=-1,keepdims=True));comb=comb/comb.sum(axis=-1,keepdims=True)+eps
    comb=comb/(comb.sum(axis=-2,keepdims=True)+eps)
    for _ in range(iterations-1):
        comb=comb/(comb.sum(axis=-1,keepdims=True)+eps);comb=comb/(comb.sum(axis=-2,keepdims=True)+eps)
    return T(pre),T(post),T(comb)


def main(reference,out):
    source=reference/'inference/model.py'
    pinned_sources={'model.py': 'e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d', 'kernel.py': '1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'}
    for name,digest in pinned_sources.items():
        if hashlib.sha256((reference/'inference'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('official source differs from pinned 1017: '+name)
    torch=facade.TorchFacade();torch.float32=np.float32
    torch.sum=lambda x,dim:T(raw(x).sum(axis=dim))
    torch.bincount=lambda x,minlength:T(np.bincount(raw(x),minlength=minlength))
    previous_where=torch.where
    torch.where=lambda condition,*args:tuple(T(a) for a in np.where(raw(condition))) if not args else previous_where(condition,*args)
    torch.clamp=lambda x,min=None,max=None:T(np.clip(raw(x),min,max))
    def method(cls,name):
        tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name);fn.decorator_list=[];fn.returns=None
        for arg in [*fn.args.args,*fn.args.kwonlyargs]:arg.annotation=None
        env={'torch':torch,'F':F,'hc_split_sinkhorn':sinkhorn,'linear':F.linear,'world_size':1}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env);return env[name]
    methods={cls:{name:method(cls,name) for name in names} for cls,names in {
       'Block':['hc_mixes','hc_pre','hc_post','forward'],'RMSNorm':['forward'],
       'Gate':['forward'],'Expert':['forward'],'MoE':['forward'],'Engram':['forward']}.items()}
    rng=np.random.default_rng(41017)
    def draw(shape,scale):return rng.normal(0,scale,shape).astype(np.float32)
    def hc():return {'fn':draw((24,16),.035).tolist(),'scale':[.4,.65,.25],
                     'base':np.linspace(-.4,.5,24,dtype=np.float32).tolist()}
    def expert():return {name:draw(shape,scale).tolist() for name,shape,scale in [('gate',(3,4),.8),('up',(3,4),.7),('down',(4,3),.35)]}
    base=[]
    for layer in range(2):
        w={'layer_id':layer,'dim':4,'copies':4,'sinkhorn_iterations':7,'norm_eps':1e-6,'hc_eps':1e-6,
           'attention_hc':hc(),'ffn_hc':hc(),'attention_norm':[1.1,.8,1.2,.9],'ffn_norm':[.9,1.25,.85,1.1],
           'attention_matrix':draw((4,4),.2).tolist(),
           'moe':{'dim':4,'intermediate':3,'topk':2,'temperature':1.15,'route_scale':1.5,'normalize':True,
                  'routed_clamp':.35,'shared_clamp':.27,'router':draw((3,4),.3).tolist(),'correction':[.3,-.4,.1],
                  'experts':[expert() for _ in range(3)],'shared':expert()}}
        if layer==1:w['engram']={'control_layer':0,'projection':draw((20,4),.09).tolist(),
            'q_weight':np.linspace(.3,1.4,16,dtype=np.float32).reshape(4,4).tolist(),
            'k_weight':np.linspace(1.3,.2,16,dtype=np.float32).reshape(4,4).tolist()}
        base.append(w)
    controls={'asset_identity':'tiny-block-engram','compressed_vocab_size':4,'raw_pad_id':2,'max_ngram_size':2,
              'n_heads':2,'head_dim':2,'token_map':[0,1,2,3],
              'layers':[{'layer_id':1,'table_rows':12,'primes':[5,7],'offsets':[0,5],'multipliers':[3,5]}]}
    row_ids=[3,8];embedding=draw((4,),.4);standalone_h=draw((1,1,4,4),.5);standalone_pre=array([[[.15,.55,.25,.35]]])
    head={'dim':4,'vocab':5,'norm_eps':1e-6,'norm':[1.2,.8,1.1,.9],'projection':draw((5,4),.3).tolist()}
    def norm(weight,x):
        obj=types.SimpleNamespace(weight=T(array(weight)),eps=1e-6)
        return methods['RMSNorm']['forward'](obj,x)
    cases=[]
    for precision in ['float32','bf16']:
        def cast(a):return bf16(array(a)) if precision=='bf16' else array(a)
        kind='bf16' if precision=='bf16' else None
        def lin(weight,x):
            value=raw(x)@array(weight).T
            return T(bf16(value),kind) if kind else T(value)
        weights=copy.deepcopy(base);hweights=copy.deepcopy(head)
        if kind:
            for w in weights:
                for name in ['attention_norm','ffn_norm','attention_matrix']:w[name]=cast(w[name]).tolist()
                w['moe']['router']=cast(w['moe']['router']).tolist()
                for ew in [*w['moe']['experts'],w['moe']['shared']]:
                    for name in ['gate','up','down']:ew[name]=cast(ew[name]).tolist()
                if 'engram' in w:
                    for name in ['projection','q_weight','k_weight']:w['engram'][name]=cast(w['engram'][name]).tolist()
            hweights['norm']=cast(hweights['norm']).tolist();hweights['projection']=cast(hweights['projection']).tolist()
        def make_expert(w,limit):
            obj=types.SimpleNamespace(w1=lambda x:lin(w['gate'],x),w2=lambda x:lin(w['down'],x),
                                      w3=lambda x:lin(w['up'],x),swiglu_limit=limit)
            return lambda x,weight=None:methods['Expert']['forward'](obj,x,weight)
        def make_block(w):
            m=w['moe'];gate=types.SimpleNamespace(weight=T(array(m['router'])),gate_temp=m['temperature'],
                score_func='sqrtsoftplus',bias=T(array(m['correction'])),bias_vl=None,topk=m['topk'],
                norm_topk_prob=m['normalize'],route_scale=m['route_scale'])
            moe=types.SimpleNamespace(dim=4,n_routed_experts=3,experts_start_idx=0,experts_end_idx=3,
                experts=[make_expert(e,m['routed_clamp']) for e in m['experts']],shared_experts=make_expert(m['shared'],m['shared_clamp']))
            def route(x,image=None):
                result=methods['Gate']['forward'](gate,x,image);moe.last_weights,moe.last_ids=result;return result
            moe.gate=route
            obj=types.SimpleNamespace(norm_eps=w['norm_eps'],hc_mult=4,hc_sinkhorn_iters=w['sinkhorn_iterations'],hc_eps=w['hc_eps'],
                 attn_norm=lambda x:norm(w['attention_norm'],x),ffn_norm=lambda x:norm(w['ffn_norm'],x),
                 attn=lambda x,pos,*args:lin(w['attention_matrix'],x),ffn=lambda x,image=None:methods['MoE']['forward'](moe,x,image))
            for prefix in ['attn','ffn']:
                for field in ['fn','scale','base']:setattr(obj,'hc_'+prefix+'_'+field,T(array(w[('attention' if prefix=='attn' else prefix)+'_hc'][field])))
            for name in ['hc_mixes','hc_pre','hc_post']:setattr(obj,name,types.MethodType(methods['Block'][name],obj))
            return obj,moe
        def add_engram(w,h):
            if 'engram' not in w:return h
            ew=w['engram'];rows=array([[np.sin((row*2+d)*.37)*.2 for d in range(2)] for row in row_ids]).reshape(1,1,2,2)
            rows=cast(rows)
            obj=types.SimpleNamespace(embed=lambda unused:T(rows,kind),wkv=lambda x:lin(ew['projection'],x),
                  hc_mult=4,dim=4,q_weight=T(array(ew['q_weight'])),k_weight=T(array(ew['k_weight'])),eps=1e-6,clamp_value=1e-6)
            return methods['Engram']['forward'](obj,h,T(np.array(row_ids).reshape(1,1,2)),None)
        # One complete block from asymmetric residual/pre state, including Engram first.
        w=weights[1];obj,moe=make_block(w);sh=T(cast(standalone_h),kind);sp=T(standalone_pre)
        after=add_engram(w,sh);result,pre=methods['Block']['forward'](obj,after,0,sp,None)
        standalone={'h':sh.a.reshape(-1).tolist(),'pre':sp.a.reshape(-1).tolist(),
                    'after_engram':vector(after),'output':vector(result),'next_pre':vector(pre),
                    'route_ids':vector(moe.last_ids),'route_weights':vector(moe.last_weights)}
        # Full two-block token plus final collapse/norm/head, as Transformer.forward does.
        initial=cast(embedding);h=T(np.tile(initial,(1,1,4,1)),kind);pre=T(array([[[1,0,0,0]]]))
        stages=[]
        for w in weights:
            h=add_engram(w,h);obj,moe=make_block(w);h,pre=methods['Block']['forward'](obj,h,0,pre,None)
            stages.append({'h':vector(h),'pre':vector(pre),'route_ids':vector(moe.last_ids),'route_weights':vector(moe.last_weights)})
        collapsed=obj.hc_pre(h,pre);normalized=norm(hweights['norm'],collapsed)
        logits=raw(normalized).astype(np.float32)@array(hweights['projection']).T
        cases.append({'precision':precision,'weights':weights,'head':hweights,'embedding':initial.tolist(),
            'standalone':standalone,'stages':stages,'head_collapsed':vector(collapsed),'head_normalized':vector(normalized),'logits':logits.reshape(-1).tolist()})
    result={'source':'official DeepSeek1017 model.py 1bae0e1088993b9b82de08f59de5be5a34e42170','source_sha256':pinned_sources,'numpy_version':np.__version__,
       'method':'Actual Block/Gate/Expert/MoE/Engram/RMSNorm forward bodies via AST+NumPy; Sinkhorn from kernel.py math; synthetic linear attention callback, not full attention.',
       'engram_controls':controls,'engram_hashes':row_ids,'cases':cases}
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
