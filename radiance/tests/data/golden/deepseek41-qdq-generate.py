#!/usr/bin/env python3
"""CPU Torch cast receipts plus an explicitly interpreted DS41 QDQ candidate.

No unchanged TileLang execution. FP4 quantization is an independent exhaustive
Fraction-distance codebook oracle grounded in the recorded AMD primary source.
Run with the existing CPU-only /home/as/venvs/embed/bin/python. No GPU queries.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

for key,value in {'ROCR_VISIBLE_DEVICES':'99','HIP_VISIBLE_DEVICES':'-1','CUDA_VISIBLE_DEVICES':'-1'}.items():
    os.environ[key]=value

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
KERNEL_SHA='1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455'
PRIMARY={
    'amd_hip_ocp_host.hpp':'560176aecd85b7760031a18121daeb83cba7362db1ecb5b10838409f946ba0e5',
    'amd_hip_fp4.h':'259ae8d7bad051b1b49df1b607e998668dda2b7c7b1928a9a546c869259f3d20',
    'amd_hip_bfloat16.h':'94c8f53a0a0a5d186b29e620d690f5ab013fac7e1c2eaded15a789fc0d7f8a60',
}

def u32(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def f32(value):return struct.unpack('<f',struct.pack('<I',value))[0]

def main():
    source=ROOT/'reference/inference/kernel.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=KERNEL_SHA:raise ValueError('pinned kernel source changed')
    for name,digest in PRIMARY.items():
        if hashlib.sha256((Path('/usr/include/hip/amd_detail')/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('primary vendor source changed: '+name)
    import torch
    if not torch.__version__.endswith('+cpu') or torch.version.hip is not None or torch.version.cuda is not None:
        raise RuntimeError('CPU-only Torch required')
    torch.set_num_threads(1);torch.set_default_device('cpu');torch.set_default_dtype(torch.float32)
    torch.set_flush_denormal(False)
    def tensor(values):return torch.tensor(values,dtype=torch.float32,device='cpu')
    def raw32(values):return [int(x)&0xffffffff for x in values.contiguous().view(torch.int32).reshape(-1).tolist()]
    def raw16(values):return [int(x)&0xffff for x in values.contiguous().view(torch.int16).reshape(-1).tolist()]
    def raw8(values):return values.contiguous().view(torch.uint8).reshape(-1).tolist()
    def from_raw(values):return tensor([f32(x) for x in values])
    def e4_cast(value):
        q=tensor([value]).to(torch.float8_e4m3fn);return raw8(q)[0],raw32(q.to(torch.float32))[0]
    positive=[0.,.5,1.,1.5,2.,3.,4.,6.]
    def fp4(value):
        magnitude=Fraction.from_float(abs(float(value)))
        code=min(range(8),key=lambda c:(abs(magnitude-Fraction.from_float(positive[c])),c&1))
        return code|((u32(value)>>28)&8)
    def fp4_value(code):return math.copysign(positive[code&7],-1. if code&8 else 1.)

    codes=torch.arange(256,dtype=torch.int32,device='cpu').to(torch.uint8)
    e4decoded=codes.view(torch.float8_e4m3fn).to(torch.float32)
    e4_codes=[{'code':code,'value_f32_bits':raw32(e4decoded)[code],
               'finite':math.isfinite(float(e4decoded[code]))} for code in range(256)]
    e4_inputs=[]
    for code in range(126):
        midpoint=(e4decoded[code]+e4decoded[code+1])/2
        for value in (torch.nextafter(midpoint,tensor(float('-inf'))),midpoint,torch.nextafter(midpoint,tensor(float('inf')))):
            e4_inputs.extend([float(value),-float(value)])
    for value in (448.,464.,f32(u32(464.)-1),f32(u32(464.)+1),465.,4096.,1e30,f32(0x7f7fffff),
                  float('inf'),float('nan'),f32(1),0.,-0.):e4_inputs.extend([value,-value])
    e4_boundary=[{'input_f32_bits':u32(x),'code':e4_cast(x)[0],
                  'decoded_f32_bits':e4_cast(x)[1],'finite':math.isfinite(f32(e4_cast(x)[1])),
                  'candidate_accepts':math.isfinite(x) and abs(x)<=464.} for x in e4_inputs]
    e8decoded=codes.view(torch.float8_e8m0fnu).to(torch.float32)
    e8_codes=[{'code':code,'value_f32_bits':raw32(e8decoded)[code],
               'finite':math.isfinite(float(e8decoded[code]))} for code in range(256)]
    fp4_inputs=[fp4_value(code) for code in range(16)]
    for lo,hi in zip(positive,positive[1:]):
        midpoint=tensor((lo+hi)/2)
        for value in (torch.nextafter(midpoint,tensor(float('-inf'))),midpoint,torch.nextafter(midpoint,tensor(float('inf')))):
            fp4_inputs.extend([float(value),-float(value)])
    fp4_inputs.extend([7.,-7.,f32(1),-f32(1),f32(0x7f7fffff),-f32(0x7f7fffff)])
    fp4_boundary=[{'input_f32_bits':u32(x),'code':fp4(x),'decoded_f32_bits':u32(fp4_value(fp4(x)))} for x in fp4_inputs]
    bf_inputs=[0,0x80000000,1,0x80000001,0x00008000,0x00008001,0x00018000,0x007fffff,
               0x00800000,0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x7f7f0000,
               0x7f7f7fff,0x7f7f8000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001]
    bf_cast=from_raw(bf_inputs).to(torch.bfloat16)
    bf_boundary=[{'input_f32_bits':x,'output_bf16_bits':y,
                  'candidate_accepts':math.isfinite(f32(x)) and (y&0x7f80)!=0x7f80}
                 for x,y in zip(bf_inputs,raw16(bf_cast))]

    def candidate(profile,input_values):
        width=16 if profile=='compressed_fp4' else 32
        x=tensor(input_values).to(torch.bfloat16)
        record={'profile':profile,'input_bf16_bits':raw16(x),'method':
                'interpreted pipeline; actual CPU Torch E4M3/BF16 casts; FP4 Fraction nearest-codebook oracle'}
        outputs=[];logical=[];scales=[];scale_codes=[]
        for start in range(0,len(input_values),width):
            group=x[start:start+width].to(torch.float32)
            maximum=group.abs().amax()
            if not math.isfinite(float(maximum)):
                record['candidate_status']='refuse';record['reason']='nonfinite input';return record
            if profile=='compressed_fp4':
                maximum=maximum.clamp_min(6*2**-9)
                unrounded=maximum/6
                # This CPU Torch build saturates E4M3 overflow. The candidate's
                # stronger policy REFUSES it; do not call that official behavior.
                if not math.isfinite(float(unrounded)) or float(unrounded)>464.:
                    record['candidate_status']='refuse';record['reason']='explicit candidate E4M3 overflow refusal';return record
                encoded=unrounded.to(torch.float8_e4m3fn)
                scale=encoded.to(torch.float32);scale_code=raw8(encoded)[0]
            else:
                maximum=maximum.clamp_min(1e-4 if profile=='window_fp8' else 6*2**-126)
                scaled=maximum*tensor(1/448 if profile=='window_fp8' else 1/6)
                raw=u32(float(scaled));exponent=((raw>>23)&255)-127+int((raw&0x7fffff)!=0)
                if not -126<=exponent<=127:
                    record['candidate_status']='refuse';record['reason']='scale exponent overflow';return record
                scale=tensor(f32((exponent+127)<<23));scale_code=raw8(scale.to(torch.float8_e8m0fnu))[0]
            if not math.isfinite(float(scale)) or float(scale)<=0:
                record['candidate_status']='refuse';record['reason']='nonfinite scale cast';return record
            normalized=(group/scale).clamp(-448 if profile=='window_fp8' else -6,448 if profile=='window_fp8' else 6)
            if profile=='window_fp8':
                q=normalized.to(torch.float8_e4m3fn);decoded=q.to(torch.float32);group_codes=raw8(q)
            else:
                group_codes=[fp4(value) for value in normalized.tolist()];decoded=tensor([fp4_value(c) for c in group_codes])
            product=decoded*scale
            y=product.to(torch.bfloat16)
            if not bool(torch.isfinite(product).all()) or not bool(torch.isfinite(y).all()):
                record['candidate_status']='refuse';record['reason']='dequant/output conversion overflow';return record
            outputs.extend(raw16(y));logical.extend(group_codes);scales.append(u32(float(scale)));scale_codes.append(scale_code)
        record.update(candidate_status='ok',output_bf16_bits=outputs,logical_codes=logical,
                      scale_f32_bits=scales,scale_codes=scale_codes)
        return record

    rows=[]
    for profile,width in [('window_fp8',32),('compressed_fp4',16),('index_fp4',32)]:
        patterns={'signed_zero':[0.,-0.]*(width),
                  'ordinary':[((i*37)%127-63)/16 for i in range(width*2)],
                  'tiny_floor':[.005,-.005,0.,-0.]*(width//2),
                  'subnormal_floor':[f32(0x00010000),-f32(0x00010000),0.,-0.]*(width//2),
                  'half_min_normal':[f32(0x00400000),-f32(0x00400000),0.,-0.]*(width//2),
                  'amax_six':([6.,-6.,1.2,-1.2]+[0.]*(width-4))*2,
                  'e4_scale_midpoint_464':[2784.]*(width*2),
                  'e4_scale_beyond_midpoint':[2800.]*(width*2),
                  'scale_overflow_or_large_finite':[4096.]*(width*2),
                  'output_overflow':[f32(0x7f7f0000)]*(width*2)}
        for name,values in patterns.items():
            record=candidate(profile,values);record['name']=name;rows.append(record)
    try:tensor([1.,2.]).to(torch.float4_e2m1fn_x2);fp4_error='unexpectedly supported'
    except Exception as error:fp4_error=type(error).__name__+': '+str(error)
    return {'schema':'deepseek41-cache-qdq-candidate-fixtures-v1',
            'official_revision':'1bae0e1088993b9b82de08f59de5be5a34e42170','kernel_sha256':KERNEL_SHA,
            'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'environment':{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
                           'torch_git':torch.version.git_version,'torch_hip':torch.version.hip,'torch_cuda':torch.version.cuda,
                           'flush_denormal':False,'num_threads':torch.get_num_threads(),
                           'device_masks':{k:os.environ[k] for k in ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']}},
            'primary_vendor_source_sha256':PRIMARY,
            'fp4_cpu_cast_boundary':fp4_error,
            'provenance':{'e4m3_codes_and_boundaries':'actual existing CPU Torch dtype execution',
                          'e8m0_codes':'actual existing CPU Torch dtype decode',
                          'bf16_boundaries':'actual existing CPU Torch dtype execution',
                          'fp4_boundaries':'independent exact Fraction-distance codebook oracle; AMD primary RNE evidence; NOT TileLang execution',
                          'rows':'new interpreted pipeline using actual Torch casts where supported; NOT unchanged official execution'},
            'stronger_candidate_refusals':'nonfinite, E4M3 magnitude>464, nonfinite dequant/output BF16; CPU Torch observed E4M3 overflow saturation is NOT adopted or assumed TileLang behavior',
            'e4m3_codes':e4_codes,'e4m3_boundaries':e4_boundary,'e8m0_codes':e8_codes,
            'fp4_boundaries':fp4_boundary,'bf16_boundaries':bf_boundary,'rows':rows,
            'gpu_execution':False,'official_tilelang_executed':False,'runtime_profile_changed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=HERE/'deepseek41-qdq.json');args=parser.parse_args()
    result=json.dumps(main(),indent=2,sort_keys=True,allow_nan=False)+'\n'
    if len(result.encode())>1_000_000:raise ValueError('fixture exceeds1MB cap')
    args.output.write_text(result);print('Wrote',args.output,'CPU format execution + interpreted candidate; no TileLang/GPU')
