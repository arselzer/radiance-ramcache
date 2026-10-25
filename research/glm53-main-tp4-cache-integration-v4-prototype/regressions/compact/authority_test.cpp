#include "allocation_hooks.h"
#include "baseline_metadata.h"
#include "mutations.h"
#include "place/native_weight_authority.h"
#include "place/planner.h"
#include <array>
#include <cstdio>
#include <iostream>
#include <stdexcept>
using namespace rad;
namespace {
uint64_t checks=0,cases=0,differential=0,oom_refusals=0;
void need(bool b,const std::string& why){++checks;if(!b)throw std::runtime_error(why);}
struct Input {std::vector<WeightInfo> weights;std::vector<int32_t> map;std::vector<MoveUnit> units;};
Input input(size_t units){
    need(units&&units<=16384,"bounded synthetic unit count");Input out;out.weights.resize(1+units*4);out.map.resize(out.weights.size(),-1);out.units.resize(units);
    for(size_t u=0;u<units;++u){auto& unit=out.units[u];unit.layer=int32_t(u/288);unit.expert=int32_t(u%288);
        for(size_t m=0;m<4;++m){const auto h=rad_weight(1+u*4+m);auto& w=out.weights[h];out.map[h]=int32_t(u);unit.weights.push_back(h);unit.offset.push_back(int64_t(m)*4096);
            w.name="weight."+std::to_string(h);w.source="logical."+std::to_string((h-1)/2);
            w.decl.group={unit.layer,unit.expert,int32_t(m)};w.decl.dtype=m%2?RAD_E8M0:RAD_FP4E2M1;w.decl.rank=2;
            w.decl.shape[0]=m<2?256:128;w.decl.shape[1]=m%2?4:128;
            w.decl.access=RAD_ACCESS_CONDITIONAL;w.decl.shard=RAD_SHARD_NONE;w.decl.n_row_parts=2;w.decl.row_parts[0]=w.decl.row_parts[1]=128;
            w.logical_bytes=int64_t(8192+4096*m);w.shard_lo=4;w.shard_hi=128;w.n_part_spans=2;
            w.part_lo[0]=1;w.part_hi[0]=3;w.part_total[0]=8;w.part_lo[1]=3;w.part_hi[1]=9;w.part_total[1]=16;
            w.enc=rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);w.enc_known=true;w.n_sel=1;w.sel[0]=int(m%2);w.sel_rows[0]=w.decl.shape[0];w.sel_cols[0]=w.decl.shape[1];
            w.layout_tag=std::string("tag\0role",8)+char('0'+m);w.stored_dtype=w.decl.dtype;w.stored_rank=2;
            w.stored_shape[0]=w.decl.shape[0];w.stored_shape[1]=w.decl.shape[1];w.stored_bytes=w.logical_bytes;w.align=4096;
        }
    }return out;
}
std::unique_ptr<NativeWeightAuthority> capture(const Input& f,uint64_t cap=32u<<20){std::unique_ptr<NativeWeightAuthority> out;std::string why;
    const int s=NativeWeightAuthority::capture(f.weights,f.map,f.units,cap,&out,&why);need(s==RAD_OK&&out,"capture: "+why);return out;}
void differential_fields(){++cases;auto f=input(4);auto& w=f.weights[1];
    w.name=std::string(500,'n');w.name[3]=0;w.source=std::string(480,'s');w.source[7]=0;
    w.layout_tag=std::string(512,'t');w.layout_tag[5]=0;
    auto authority=capture(f);const auto original=w;need(authority->matches(1,original),"baseline authority mismatch");
    const auto mutations=compact_tests::mutations();size_t guarded=0,ignored=0;
    for(const auto& test:mutations){auto changed=original;test.apply(changed);const bool old=compact_baseline::metadata(changed,original,true);
        need(old==!test.guarded,"baseline mutation did not exercise intended field: "+test.label);
        need(authority->matches(1,changed)==old,"compact predicate differs from frozen baseline: "+test.label);
        need(native_weight_metadata_equal(changed,original,true)==old,"shared predicate differs: "+test.label);
        need(native_weight_metadata_equal(changed,original,false)==compact_baseline::metadata(changed,original,false),"geometry equivalence differs: "+test.label);
        ++differential;if(test.guarded)++guarded;else ++ignored;
    }
    need(guarded>=600&&ignored==14,"full numeric/unused-array/encoding/string/ignored matrix missing");
    need(authority->stats().reference_size==32&&authority->stats().slice_size==8&&authority->stats().bucket_size==16&&authority->stats().geometry_size==sizeof(WeightInfo),"actual compact POD sizes");
    need(authority->stats().template_strings_empty&&!authority->stats().template_string_capacity_excess,"long ignored names/source/tag retained in templates");
    std::cout<<"DIFFERENTIAL guarded="<<guarded<<" ignored="<<ignored<<" sizeofWeightInfo="<<sizeof(WeightInfo)<<"\n";
}
void exact_strings_and_detach(){++cases;std::unique_ptr<NativeWeightAuthority> authority;WeightInfo saved;NativeAuthorityStats stats;
    {auto f=input(8);for(size_t i=1;i<f.weights.size();++i){auto& w=f.weights[i];w.name=std::string("same\0name",9)+char('A'+i%4);w.source=i%2?"":std::string("same\0source",11);w.decl.group.slot=int32_t(i%4);}
        f.weights.back().name.clear(); // Empty exact identity remains legal, no invented normalization.
        authority=capture(f);stats=authority->stats();saved=f.weights[1];
        need(stats.geometries==5,"names/group must not multiply geometry templates");
        need(stats.strings==6,"exact NUL-aware name/source dedup or empty string differs");
        for(size_t i=1;i<f.weights.size();++i)need(authority->matches(rad_weight(i),f.weights[i])&&authority->unit_for_weight(rad_weight(i))==f.map[i]&&authority->member_index(rad_weight(i))==int32_t((i-1)%4),"immutable exact identity/map/member");
    }
    need(authority->matches(1,saved)&&authority->unit_for_weight(1)==0&&authority->member_index(1)==0,"authority borrowed destroyed input strings/map");
    auto wrong=saved;wrong.name[6]^=1;need(!authority->matches(1,wrong),"embedded suffix compared using c_str");
    wrong=saved;wrong.layout_tag.back()^=1;need(!authority->matches(1,wrong),"same-length tag did not use owned exact arena bytes");
    for(int field=0;field<3;++field){wrong=saved;auto& text=field==0?wrong.name:field==1?wrong.source:wrong.layout_tag;
        text.push_back('\0');need(!authority->matches(1,wrong),"arena slice accepted added NUL/length extension");
        wrong=saved;auto& short_text=field==0?wrong.name:field==1?wrong.source:wrong.layout_tag;
        if(!short_text.empty()){short_text.pop_back();need(!authority->matches(1,wrong),"arena slice accepted proper byte prefix");}}
    need(!authority->matches(UINT32_MAX,saved)&&authority->unit_for_weight(UINT32_MAX)==-1&&authority->stored_bytes(UINT32_MAX)==-1,"outside handle indexing");
}
void budget_and_allocation_receipts(){++cases;auto f=input(32);NativeAuthorityStats quote;
    need(NativeWeightAuthority::inspect(f.weights,f.map,f.units,32u<<20,&quote)==RAD_OK&&quote.admissible,"metadata-only quote");
    std::unique_ptr<NativeWeightAuthority> authority;allocation_fault::begin();
    const int status=NativeWeightAuthority::capture(f.weights,f.map,f.units,32u<<20,&authority);
    allocation_fault::end();const auto peak=allocation_fault::peak,live=allocation_fault::live,calls=allocation_fault::calls;
    need(status==RAD_OK&&authority&&peak<=quote.construction_peak_bytes&&live<=quote.immutable_bytes,"requested allocation peak exceeds admitted immutable/scratch quote");
    const auto& actual=authority->stats();need(actual.weights==quote.weights&&actual.geometries==quote.geometries&&actual.arena_bytes==quote.arena_bytes&&actual.construction_peak_bytes==quote.construction_peak_bytes,"inspect/capture quote drift");
    std::cout<<"ALLOCATION W="<<actual.weights<<" G="<<actual.geometries<<" names="<<actual.strings<<" nameSourceBytes="<<actual.name_source_bytes<<" tagBytes="<<actual.layout_tag_bytes<<" immutable="<<actual.immutable_bytes<<" scratch="<<actual.scratch_bytes<<" upper="<<actual.construction_peak_bytes<<" observedRequestedPeak="<<peak<<" observedRetained="<<live<<" calls="<<calls<<"\n";
    const auto* old=authority.get();allocation_fault::begin();
    const int small=NativeWeightAuthority::capture(f.weights,f.map,f.units,1,&authority);allocation_fault::end();
    need(small==RAD_E_SCRATCH&&authority.get()==old&&allocation_fault::calls==0,"tiny budget allocated/replaced authority");
    auto bad=f;bad.weights[1].name.resize(513,'x');allocation_fault::begin();
    const int invalid=NativeWeightAuthority::capture(bad.weights,bad.map,bad.units,32u<<20,&authority);allocation_fault::end();
    need(invalid==RAD_E_SHAPE&&authority.get()==old&&allocation_fault::calls==0,"oversized string reached allocation/publication");
    for(int field=1;field<3;++field){bad=f;(field==1?bad.weights[1].source:bad.weights[1].layout_tag).resize(513,'x');allocation_fault::begin();
        const int too_long=NativeWeightAuthority::capture(bad.weights,bad.map,bad.units,32u<<20,&authority);allocation_fault::end();
        need(too_long==RAD_E_SHAPE&&authority.get()==old&&allocation_fault::calls==0,"oversized source/tag crossed pre-scratch boundary");}
    bad=f;bad.map[1]=int32_t(bad.units.size());need(NativeWeightAuthority::capture(bad.weights,bad.map,bad.units,32u<<20,&authority)==RAD_E_SHAPE&&authority.get()==old,"foreign unit map accepted");
    bad=f;bad.units[0].weights.push_back(1);need(NativeWeightAuthority::capture(bad.weights,bad.map,bad.units,32u<<20,&authority)==RAD_E_SHAPE&&authority.get()==old,"duplicate member mapping accepted");
}
void oom_atomicity(){++cases;auto f=input(8);auto kept=capture(f);const auto* prior=kept.get();bool succeeded=false;
    for(int64_t n=0;n<64;++n){allocation_fault::begin();allocation_fault::remaining=n;
        const int s=NativeWeightAuthority::capture(f.weights,f.map,f.units,32u<<20,&kept);allocation_fault::end();
        if(s==RAD_OK){need(kept&&kept.get()!=prior,"successful replacement did not publish");succeeded=true;break;}
        need(s==RAD_E_NOMEM&&kept.get()==prior&&allocation_fault::live==0,"allocator refusal leaked partial authority/replaced output");++oom_refusals;
        need(kept->matches(1,f.weights[1]),"prior authority damaged by allocation refusal");
    }need(succeeded&&oom_refusals>=8,"all exact-array/control allocation sites not reached");
}
void growth_and_honest_refusal(){++cases;
#ifndef RAD_NATIVE_AUTHORITY_TEST_COLLISIONS
    uint64_t previousRefs=0,previousArena=0;
    for(size_t units:{size_t(1),size_t(16),size_t(288),size_t(12384)}){auto f=input(units);auto authority=capture(f);const auto& s=authority->stats();
        need(s.weights==units*4+1&&s.geometries==5&&s.reference_bytes==s.weights*32&&s.name_source_bytes>=previousArena&&s.reference_bytes>previousRefs,"linear small-ref growth/fixed exact geometry lost");
        previousRefs=s.reference_bytes;previousArena=s.name_source_bytes;
        std::cout<<"SCALE syntheticUnits="<<units<<" W="<<s.weights<<" G="<<s.geometries<<" S="<<s.name_source_bytes<<" constructionUpper="<<s.construction_peak_bytes<<" originalInputWeightInfoBytes="<<f.weights.size()*sizeof(WeightInfo)<<"\n";
    }
    auto f=input(8192);for(size_t i=1;i<f.weights.size();++i)f.weights[i].stored_bytes+=int64_t(i)*32768;
    NativeAuthorityStats quote;need(NativeWeightAuthority::inspect(f.weights,f.map,f.units,16u<<20,&quote)==RAD_OK&&!quote.admissible&&quote.geometries==f.weights.size(),"all-unique geometry must quote honest over-budget need");
    auto tiny=input(1);auto kept=capture(tiny);const auto* old=kept.get();allocation_fault::begin();
    const int status=NativeWeightAuthority::capture(f.weights,f.map,f.units,16u<<20,&kept);allocation_fault::end();
    need(status==RAD_E_SCRATCH&&kept.get()==old&&allocation_fault::live==0&&allocation_fault::max_request<quote.geometry_bytes,"inadmissible templates were bulk allocated before refusal");
#else
    auto f=input(32);auto authority=capture(f);need(authority->stats().geometries==5,"forced collisions merged distinct geometry");
    for(size_t i=1;i<f.weights.size();++i)need(authority->matches(rad_weight(i),f.weights[i]),"forced string collisions lost exact identity");
#endif
}
void mixed_encodings(){++cases;auto f=input(2);
    for(size_t u=0;u<2;++u)for(size_t m=0;m<4;++m){auto& w=f.weights[1+u*4+m];const bool scale=m%2,down=m>=2;
        w.enc=u?rad_enc_affine(RAD_F8E4M3,RAD_F32,128,128):rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32);
        int64_t rows=0,cols=0;rad_enc_plane_dims(&w.enc.plane[scale?1:0],down?128:256,128,&rows,&cols);
        w.decl={};w.decl.dtype=w.enc.plane[scale?1:0].dtype;w.decl.rank=2;w.decl.shape[0]=rows;w.decl.shape[1]=cols;
        w.decl.group={int32_t(u),0,int32_t(m)};w.decl.access=RAD_ACCESS_CONDITIONAL;
        if(!down){w.decl.n_row_parts=2;w.decl.row_parts[0]=w.decl.row_parts[1]=rows/2;}
        w.n_part_spans=0;std::fill(std::begin(w.part_lo),std::end(w.part_lo),0);std::fill(std::begin(w.part_hi),std::end(w.part_hi),0);std::fill(std::begin(w.part_total),std::end(w.part_total),0);
        w.n_sel=1;std::fill(std::begin(w.sel),std::end(w.sel),0);std::fill(std::begin(w.sel_rows),std::end(w.sel_rows),0);std::fill(std::begin(w.sel_cols),std::end(w.sel_cols),0);
        w.sel[0]=scale?1:0;w.sel_rows[0]=rows;w.sel_cols[0]=cols;w.layout_tag.clear();w.stored_dtype=w.decl.dtype;w.stored_rank=2;
        std::fill(std::begin(w.stored_shape),std::end(w.stored_shape),0);w.stored_shape[0]=rows;w.stored_shape[1]=cols;
        w.stored_bytes=rad_enc_row_bytes(w.stored_dtype,cols)*rows;w.logical_bytes=w.stored_bytes;
    }
    auto authority=capture(f);need(authority->stats().geometries==9,"main/MTP selected-plane templates aliased");
    for(size_t h=1;h<f.weights.size();++h)need(authority->matches(rad_weight(h),f.weights[h]),"mixed main/MTP metadata lost");
    auto wrong=f.weights[1];wrong.enc=f.weights[5].enc;
    need(!authority->matches(1,wrong),"same-byte metadata accepted another canonical encoding");
    wrong=f.weights[6];wrong.stored_dtype=RAD_BF16;need(!authority->matches(6,wrong),"MTP F32 scale silently narrowed");
    f.weights[1].enc_known=false;authority=capture(f);wrong=f.weights[1];
    reinterpret_cast<unsigned char*>(&wrong.enc)[sizeof(RadEncoding)-1]^=1;
    need(!authority->matches(1,wrong),"enc_known=false bypassed old full encoding-byte authority");
}
}
int main(){try{differential_fields();exact_strings_and_detach();budget_and_allocation_receipts();oom_atomicity();growth_and_honest_refusal();mixed_encodings();need(cases==6,"all standalone authority groups executed");
#ifdef RAD_NATIVE_AUTHORITY_TEST_COLLISIONS
    const char* mode="forced-hash-collisions";
#else
    const char* mode="normal-hash";
#endif
    std::cout<<"PASS compact authority mode="<<mode<<" groups="<<cases<<" checks="<<checks<<" differential="<<differential<<" OOMrefusals="<<oom_refusals<<"; pure metadata, no Core archive/device/model payload\n";return 0;
}catch(const std::exception& e){allocation_fault::end();std::cerr<<"FAIL compact authority: "<<e.what()<<'\n';return 1;}}
