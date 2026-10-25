#!/usr/bin/env python3
"""Tiny independent NumPy oracle, adapted from pinned official DeepSeek 1017 math.

This does not import C++ helpers, Torch, TileLang, DeepInfer, GPU libraries, or model
weights. It specifies the unquantized FP32 semantic path; optimized BF16/FP4 kernel
rounding is a distinct acceptance gate. See deepseek41-reference.README.txt.
"""
import argparse
import json
from pathlib import Path
import numpy as np

F = np.float32
rng = np.random.Generator(np.random.PCG64(1017))
def draw(shape, scale=.3):
    return (rng.standard_normal(shape) * scale).astype(np.float32)
def sigmoid(x):
    return (1 / (1 + np.exp(-x))).astype(np.float32)
def rms(x, w, eps):
    return ((x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + F(eps))) * w).astype(np.float32)
def mixes(h, fn, scale, base, iters=20, eps=1e-6, hc_eps=1e-6):
    c = h.shape[-2]
    flat = h.reshape(-1)
    coeff = (fn @ flat) / np.sqrt(np.mean(flat*flat) + F(eps))
    pre = sigmoid(coeff[:c] * scale[0] + base[:c]) + F(hc_eps)
    post = F(2) * sigmoid(coeff[c:2*c] * scale[1] + base[c:2*c])
    comb = (coeff[2*c:] * scale[2] + base[2*c:]).reshape(c,c)
    comb = np.exp(comb - comb.max(axis=-1, keepdims=True))
    comb = comb / comb.sum(axis=-1, keepdims=True) + F(hc_eps)
    comb = comb / (comb.sum(axis=0, keepdims=True) + F(hc_eps))
    for _ in range(iters-1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + F(hc_eps))
        comb = comb / (comb.sum(axis=0, keepdims=True) + F(hc_eps))
    return pre, post, comb

def swiglu(g, u, limit):
    if limit > 0:
        g = np.minimum(g,F(limit))
        u = np.clip(u,-F(limit),F(limit))
    return g * sigmoid(g) * u

def rotary(x, position, dim, theta, original, factor, fast, slow, inverse=False):
    x = x.copy()
    freqs = F(1) / np.power(F(theta), np.arange(0,dim,2,dtype=np.float32)/F(dim))
    if original:
        def corrected(rotations):
            return dim * np.log(original/(rotations*2*np.pi)) / (2*np.log(theta))
        low = max(np.floor(corrected(fast)),0)
        high = min(np.ceil(corrected(slow)),dim-1)
        ramp = np.clip((np.arange(dim//2,dtype=np.float32)-F(low))/F(max(high-low,1e-3)),0,1)
        smooth = F(1)-ramp
        freqs = (freqs/F(factor))*(F(1)-smooth) + freqs*smooth
    angle = F(position)*freqs
    if inverse:angle = -angle
    tail = x[-dim:].copy().reshape(-1,2)
    x[-dim::2] = tail[:,0]*np.cos(angle)-tail[:,1]*np.sin(angle)
    x[-dim+1::2] = tail[:,0]*np.sin(angle)+tail[:,1]*np.cos(angle)
    return x

def attention(q,kv,sinks,ids,scale):
    out = np.zeros_like(q)
    valid = [i for i in ids if i>=0]
    if not valid:return out
    logits = (q @ kv[valid].T)*F(scale)
    maximum = np.maximum(logits.max(axis=-1),sinks)
    probabilities = np.exp(logits-maximum[:,None])
    denominator = probabilities.sum(axis=-1) + np.exp(sinks-maximum)
    return ((probabilities @ kv[valid])/denominator[:,None]).astype(np.float32)

def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.integer,np.floating)):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value

def generate():
    result={"source":"deepseek-ai/DeepSeek-V4.1-Flash-1017", "revision":"1bae0e1088993b9b82de08f59de5be5a34e42170",
            "arithmetic":"FP32 semantic oracle before activation/cache/probability quantization", "seed":1017}
    # Two shifted sublayers, asymmetric source->destination comb, and final head collapse.
    h=draw((4,6)); previous=np.array([1,0,0,0],np.float32)
    mhc=[]
    for step in range(2):
        fn=draw((24,24),.15); scale=draw((3,),.8); base=draw((24,),.4)
        pre,post,comb=mixes(h,fn,scale,base)
        x=(previous[:,None]*h).sum(axis=0)
        norm=draw((6,),.2)+F(1); subweight=draw((6,6),.25)
        y=np.tanh(subweight@rms(x,norm,1e-6)).astype(np.float32)
        out=post[:,None]*y + np.einsum('sd,sn->dn',comb,h)
        mhc.append(dict(h=h, previous_pre=previous, fn=fn, scale=scale, base=base, norm=norm,
                        subweight=subweight, pre=pre, post=post, comb=comb, collapsed=x, output=out))
        h=out;previous=pre
    result['mhc']=mhc;result['final_head_collapse']=(previous[:,None]*h).sum(axis=0)
    routes=[]
    for topk in [1,6]:
        logits=draw((12,),3);bias=draw((12,),1.4);temperature=F(.75);route_scale=F(1.5)
        score=np.sqrt(np.logaddexp(F(0),logits/temperature))
        ids=np.argsort(-(score+bias),kind='stable')[:topk]
        weights=score[ids].copy()
        if topk>1:weights/=weights.sum()+F(1e-20)
        weights*=route_scale
        routes.append(dict(logits=logits,bias=bias,topk=topk,temperature=temperature,route_scale=route_scale,ids=ids,weights=weights))
    result['routing']=routes
    gate=np.array([-30,-11,-2,0,1,11,30],np.float32);up=np.array([15,-20,2,1,2,20,-20],np.float32)
    result['swiglu']=[dict(gate=gate,up=up,limit=limit,out=swiglu(gate,up,limit)) for limit in [0,10]]
    ropes=[]
    for original in [0,4096]:
        for position in [0,1,129,32769,262143]:
            x=draw((20,));cfg=dict(dim=16,theta=40000,original=original,factor=40,fast=32,slow=1)
            ropes.append(dict(x=x,position=position,**cfg,forward=rotary(x,position,**cfg),inverse=rotary(x,position,**cfg,inverse=True)))
    result['rope']=ropes
    result['attention']=[]
    for ids in [[0,1,3,6],[6,2,-1,4],[-1,-1,-1]]:
        q=draw((3,8));kv=draw((7,8));sinks=np.array([-3,0,2],np.float32);scale=F(8**-.5)
        result['attention'].append(dict(q=q,kv=kv,sinks=sinks,ids=ids,scale=scale,out=attention(q,kv,sinks,ids,scale)))
    q=draw((3,4));keys=draw((11,4));weights=draw((3,));global_heads=6
    scores=(np.maximum(q@keys.T,0)*(weights*F(4**-.5*global_heads**-.5))[:,None]).sum(axis=0)
    result['index']=dict(q=q,keys=keys,weights=weights,global_heads=global_heads,scores=scores)
    candidate=[]
    for visible in [0,1,5,8,11]:
        block_size=3;top_blocks=2;block_scores=np.full(4,-np.inf,np.float32)
        for i in range(visible):block_scores[i//block_size]=max(block_scores[i//block_size],scores[i])
        if visible:block_scores[(visible-1)//block_size]=np.inf
        blocks=np.argsort(-block_scores,kind='stable')[:top_blocks]
        keep=np.repeat(np.isin(np.arange(4),blocks[block_scores[blocks]>-np.inf]),block_size)[:11]
        ids=[i for i in range(visible) if keep[i]]
        ids=sorted(sorted(ids,key=lambda i:(-scores[i],i))[:4])
        ids+= [-1]*(min(4,len(scores))-len(ids))
        candidate.append(dict(visible=visible,block_size=block_size,top_blocks=top_blocks,keep=keep.astype(int),ids=ids))
    result['candidate']=candidate
    result['compression']=[]
    for ratio in [1,2]:
        kv=draw((7,5));gate=draw((7,5),15);norm=draw((5,),.2)+F(1);latents=[]
        for start in range(0,len(kv)-ratio+1,ratio):
            if ratio==1:latent=kv[start]
            else:
                exp=np.exp(gate[start:start+ratio]-gate[start:start+ratio].max(axis=0,keepdims=True))
                latent=(kv[start:start+ratio]*(exp/exp.sum(axis=0,keepdims=True))).sum(axis=0)
            latents.append(rms(latent,norm,1e-6))
        result['compression'].append(dict(ratio=ratio,kv=kv,scores=gate,norm=norm,latents=np.array(latents)))
    # Output groups must not mix heads before their independent low-rank projections.
    heads=draw((4,6));wa=draw((2,3,12));wb=draw((5,6));latent=np.einsum('grd,gd->gr',wa,heads.reshape(2,12))
    result['grouped_output']=dict(heads=heads,groups=2,rank=3,wa=wa,wb=wb,out=wb@latent.reshape(-1))
    return plain(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    args.output.write_text(json.dumps(generate(),indent=2,allow_nan=False)+'\n')
