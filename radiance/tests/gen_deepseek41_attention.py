#!/usr/bin/env python3
"""Small independent full-history NumPy attention oracle for official DeepSeek1017 math.

No C++ imports, GPU libraries, model weights, packed/FP4/FP8 quantization, or online calls.
The oracle precomputes whole-prefix projections and compression groups and slices causal
history by each query's position. The C++ implementation instead appends per-token rings,
partial compressors and per-source state. See the accompanying README for limitations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from gen_deepseek41_reference import rms,rotary,attention,plain
F=np.float32


def generate():
    rng=np.random.Generator(np.random.PCG64(411017))
    def draw(shape,scale=.35):return (rng.standard_normal(shape)*scale).astype(np.float32)
    c=dict(n_layers=6,n_embd=4,n_head=2,n_head_kv=1,head_dim=4,rope_head_dim=4,
           q_lora_rank=3,o_lora_rank=3,o_groups=2,n_ff=256,n_vocab=16,context_length=24,
           n_expert=8,n_expert_used=2,n_expert_shared=1,window_size=3,index_n_heads=2,
           index_head_dim=4,index_topk=2,norm_eps=1e-6,route_scale=1.5,gate_temp=1.0,
           norm_topk_prob=True,score_func='sqrtsoftplus',routed_clamp=[10.]*6,shared_clamp=[10.]*6,
           rope_type='yarn',rope_theta=25.,compress_rope_theta=100.,rope_factor=2.,original_seq_len=4,
           beta_fast=32.,beta_slow=1.,hc_mult=4,hc_sinkhorn_iters=20,hc_eps=1e-6,hc_shifted=True,
           candidate_source=4,candidate_topk_blocks=2,candidate_block_size=2,
           compress_ratios=[0,0,2,2,1,1],kv_source_layers=[2,4],index_k_source_layers=[2,4],
           index_source_layers=[2,4,5],draft_layers=0,engram_present=False)
    L,D,H,HD,QR,G,OR,IH,ID=[c[k] for k in ['n_layers','n_embd','n_head','head_dim','q_lora_rank',
                                                  'o_groups','o_lora_rank','index_n_heads','index_head_dim']]
    T=18
    def source(ids,l):return max((s for s in ids if s<=l),default=-1)
    def rotate(x,pos,l,inverse=False):
        compressed=c['compress_ratios'][l]!=0
        return rotary(x,pos,c['rope_head_dim'],c['compress_rope_theta'] if compressed else c['rope_theta'],
                      c['original_seq_len'] if compressed else 0,c['rope_factor'],c['beta_fast'],c['beta_slow'],inverse)
    weights=[]
    for l in range(L):
        w=dict(q_a=draw((QR,D)),q_norm=F(1)+draw((QR,),.12),q_b=draw((H*HD,QR)),
               kv=draw((HD,D)),kv_norm=F(1)+draw((HD,),.12),out_a=draw((G,OR,H*HD//G)),
               out_b=draw((D,G*OR)),sinks=np.array([-.5,.75],np.float32)+draw((H,),.1),
               compress_kv=[],compress_gate=[],compress_norm=[],index_q=[],index_head_weights=[],index_k=[],index_k_norm=[])
        if l in c['kv_source_layers']:
            w['compress_kv']=draw((HD,D));w['compress_norm']=F(1)+draw((HD,),.12)
            if c['compress_ratios'][l]==2:w['compress_gate']=draw((HD,D),1.3)
        if l in c['index_source_layers']:
            w['index_q']=draw((IH*ID,QR));w['index_head_weights']=draw((IH,D),.7)
        if l in c['index_k_source_layers']:
            w['index_k']=draw((ID,HD));w['index_k_norm']=F(1)+draw((ID,),.12)
        weights.append(w)
    inputs=draw((T,L,D),.8)
    qlow=[];queries=[];window=[];latents={};compressed={};indexkeys={}
    for l,w in enumerate(weights):
        low=rms(inputs[:,l]@w['q_a'].T,w['q_norm'],c['norm_eps'])
        q=(low@w['q_b'].T).reshape(T,H,HD)
        kv=rms(inputs[:,l]@w['kv'].T,w['kv_norm'],c['norm_eps'])
        qlow.append(low);queries.append(np.stack([np.stack([rotate(h,t,l) for h in q[t]]) for t in range(T)]))
        window.append(np.stack([rotate(kv[t],t,l) for t in range(T)]))
        if l in c['kv_source_layers']:
            ratio=c['compress_ratios'][l];kv=inputs[:,l]@w['compress_kv'].T
            projected=[]
            if ratio==1:projected=kv
            else:
                gates=inputs[:,l]@w['compress_gate'].T
                for first in range(0,T-T%ratio,ratio):
                    values=kv[first:first+ratio];scores=gates[first:first+ratio]
                    exp=np.exp(scores-scores.max(axis=0,keepdims=True))
                    projected.append((values*(exp/exp.sum(axis=0,keepdims=True))).sum(axis=0))
            latents[l]=rms(np.asarray(projected,np.float32),w['compress_norm'],c['norm_eps'])
            compressed[l]=np.stack([rotate(row,j*ratio,l) for j,row in enumerate(latents[l])])
    for l in c['index_k_source_layers']:
        w=weights[l];ks=source(c['kv_source_layers'],l);ratio=c['compress_ratios'][l]
        key=rms(latents[ks]@w['index_k'].T,w['index_k_norm'],c['norm_eps'])
        indexkeys[l]=np.stack([rotate(row,j*ratio,l) for j,row in enumerate(key)])
    traces=[]
    for t in range(T):
        token=[];indices={};candidates={}
        for l,w in enumerate(weights):
            ratio=c['compress_ratios'][l];first=max(0,t-c['window_size']+1)
            tr=dict(q_latent=qlow[l][t],query=queries[l][t],window_row=window[l][t],
                    compressed_latent=[],compressed_row=[],index_query=[],index_key=[],scores=[],
                    compressed_indices=[],candidates=[],compressed_first_position=-1,
                    window_indices=[p%c['window_size'] for p in range(first,t+1)])
            logical_kv=window[l][first:t+1]
            logical_selected=list(range(t-first+1))
            tr['combined_indices']=tr['window_indices'].copy()
            if ratio:
                visible=(t+1)//ratio;ks=source(c['kv_source_layers'],l);iks=source(c['index_k_source_layers'],l)
                produced=(t+1)%ratio==0;row=visible-1
                if l in c['kv_source_layers'] and produced:
                    tr['compressed_latent']=latents[l][row];tr['compressed_row']=compressed[l][row]
                    tr['compressed_first_position']=row*ratio
                if l in c['index_k_source_layers'] and produced:tr['index_key']=indexkeys[l][row]
                if l in c['index_source_layers']:
                    scores=np.empty(0,np.float32)
                    if visible:
                        qi=(qlow[l][t]@w['index_q'].T).reshape(IH,ID)
                        qi=np.stack([rotate(h,t,l) for h in qi]);tr['index_query']=qi
                        hw=inputs[t,l]@w['index_head_weights'].T
                        scores=(np.maximum(qi@indexkeys[iks][:visible].T,0)*(hw[:,None]*F(ID**-.5*IH**-.5))).sum(axis=0)
                    tr['scores']=scores
                    valid=list(range(visible))
                    if l==c['candidate_source']:
                        bs=c['candidate_block_size'];nblocks=(visible+bs-1)//bs
                        keep=np.zeros(visible,np.uint8)
                        if visible:
                            best=np.full(nblocks,-np.inf,np.float32)
                            for b in range(nblocks):best[b]=scores[b*bs:min((b+1)*bs,visible)].max()
                            best[(visible-1)//bs]=np.inf
                            for b in sorted(range(nblocks),key=lambda b:(-float(best[b]),b))[:c['candidate_topk_blocks']]:
                                keep[b*bs:min((b+1)*bs,visible)]=1
                        candidates[l]=keep
                    elif l>c['candidate_source']:
                        keep=candidates[c['candidate_source']]
                        valid=[p for p in valid if keep[p]]
                    ids=sorted(sorted(valid,key=lambda p:(-float(scores[p]),p))[:c['index_topk']])
                    ids += [-1]*(min(c['index_topk'],visible)-len(ids));indices[l]=ids
                ids=indices[source(c['index_source_layers'],l)];tr['compressed_indices']=ids
                if l>=c['candidate_source']:tr['candidates']=candidates[c['candidate_source']]
                tr['combined_indices'] += [c['window_size']+i if i>=0 else -1 for i in ids]
                logical_selected += [len(logical_kv)+i if i>=0 else -1 for i in ids]
                logical_kv=np.concatenate((logical_kv,compressed[ks][:visible]),axis=0)
            attended=attention(queries[l][t],logical_kv,w['sinks'],logical_selected,HD**-.5)
            unrotated=np.stack([rotate(h,t,l,True) for h in attended])
            low=np.einsum('grd,gd->gr',w['out_a'],unrotated.reshape(G,H*HD//G))
            tr['attended']=attended;tr['unrotated_heads']=unrotated;tr['output']=w['out_b']@low.reshape(-1)
            token.append(tr)
        traces.append(token)
    return plain(dict(source='DeepSeek-AI 1017 official model.py 1bae0e1088993b9b82de08f59de5be5a34e42170',
                      precision='FP32 semantic reference; no FP4/FP8 activation or cache quantizers',seed=411017,
                      config=c,weights=weights,inputs=inputs,traces=traces))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    args.output.write_text(json.dumps(generate(),indent=2,allow_nan=False)+'\n')
