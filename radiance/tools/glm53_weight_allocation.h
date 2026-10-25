/* CPU metadata-only routed GLM probe. REAL Builder selected-plane/shard accounting and Planner;
 * config-derived synthetic sources, canonical identity layouts justified separately from source.
 * No registered GEMM row resolution, graph/layer execution, pool/device or payload access. */
#pragma once
#include "gguf_weight_allocation.h"
#include <array>

namespace rad::glm53_weight_probe {
using Json=nlohmann::json;
namespace wp=rad::weight_probe;
enum class Format {mxfp4,fp8_f32};
struct Layer {int id=0;Format format=Format::mxfp4;};
struct Inventory {
    std::string revision,expected_hf_prefix="model.language_model.layers";
    int main_layers=45,first_dense=3,mtp_layers=1,experts=288,topk=8;
    int64_t hidden=4096,nff=2048;
    std::vector<Layer> layers;
};
inline const char* name(Format f){return f==Format::mxfp4?"mxfp4_e8m0_g32":"fp8_e4m3_f32_b128";}
inline void validate(const Inventory& i) {
    if(i.main_layers<=0||i.main_layers>128||i.mtp_layers<0||i.mtp_layers>4||i.first_dense<0||i.first_dense>=i.main_layers||
       i.experts<=0||i.experts>4096||i.experts%2||i.topk<=0||i.topk>i.experts||
       i.hidden<=0||i.hidden>65536||i.hidden%128||i.nff<=0||i.nff>65536||i.nff%512||
       i.layers.empty()||i.layers.size()>132||i.layers.size()*size_t(i.experts)*4>100000)
        throw std::runtime_error("unreviewed GLM routed geometry");
    std::set<int> seen;
    for(const auto& l:i.layers)if(l.id<i.first_dense||l.id>=i.main_layers+i.mtp_layers||!seen.insert(l.id).second||
        (l.format!=Format::mxfp4&&l.format!=Format::fp8_f32))throw std::runtime_error("invalid GLM layer/format");
}
inline Inventory inventory(const Json& j) {
    if(j.at("schema")!="radiance-glm53-routed-inventory-v1"||j.at("checkpoint_tensor_headers_verified")!=false||
       j.at("checkpoint_payload_read")!=false)throw std::runtime_error("expected config-derived metadata fixture");
    Inventory i;i.revision=j.at("source_revision");i.expected_hf_prefix=j.at("expected_hf_prefix");
    i.main_layers=j.at("main_layers");i.first_dense=j.at("first_dense_layers");i.mtp_layers=j.at("mtp_layers");
    i.experts=j.at("experts");i.topk=j.at("top_k");i.hidden=j.at("hidden");i.nff=j.at("nff");
    if(!j.at("layers").is_array()||j.at("layers").size()>132)throw std::runtime_error("invalid bounded layers");
    for(const auto& l:j.at("layers")) {
        const std::string f=l.at("format");
        if(f!="mxfp4_e8m0_g32"&&f!="fp8_e4m3_f32_b128")throw std::runtime_error("unreviewed expert format");
        i.layers.push_back({l.at("layer").get<int>(),f=="mxfp4_e8m0_g32"?Format::mxfp4:Format::fp8_f32});
    }
    validate(i);
    if(i.revision!="b5688f25491202978c19c4d036eef579f61bbe07"||i.hidden!=4096||i.nff!=2048||i.experts!=288||
       i.main_layers!=45||i.first_dense!=3||i.mtp_layers!=1||i.topk!=8||i.layers.size()!=43||
       i.expected_hf_prefix!="model.language_model.layers")throw std::runtime_error("not the pinned GLM53 fixture");
    for(size_t x=0;x<i.layers.size();++x)if(i.layers[x].id!=int(x)+3||
        i.layers[x].format!=(x==42?Format::fp8_f32:Format::mxfp4))throw std::runtime_error("wrong main/MTP schedule");
    return i;
}
inline RadEncoding encoding(Format f) {
    return f==Format::mxfp4?rad_enc_affine(RAD_FP4E2M1,RAD_E8M0,1,32):
                            rad_enc_affine(RAD_F8E4M3,RAD_F32,128,128);
}
inline std::string logical(int layer,int expert,bool down) {
    return "blk."+std::to_string(layer)+".ffn_"+(down?"down":"gate_up")+"_exps."+std::to_string(expert)+".weight";
}
inline std::unique_ptr<Builder> build(Registry& registry,const Inventory& inv,int rank) {
    validate(inv);if(rank<0||rank>=4)throw std::runtime_error("TP4 rank outside0..3");
    RadModelMeta m{};m.arch_id="glm53_routed_weight_probe";m.name="config-derived not-serving";m.quant="mixed-MXFP4-FP8";
    m.n_layers=inv.main_layers+inv.mtp_layers;m.n_embd=inv.hidden;m.n_ff=inv.nff;m.n_expert=inv.experts;
    RadBuildCtx c{};c.rank=rank;c.world_size=4;c.max_tok=1;c.max_seqs=1;c.max_ctx=1;c.scope="";
    auto b=std::make_unique<Builder>(registry,m,c);b->set_instances(false);
    auto sources=std::make_shared<std::map<std::string,WeightSource>>();
    for(const auto& l:inv.layers)for(int e=0;e<inv.experts;++e)for(bool down:{false,true}) {
        WeightSource s{};s.enc=encoding(l.format);s.rank=2;s.shape[0]=down?inv.hidden:2*inv.nff;s.shape[1]=down?inv.nff:inv.hidden;
        sources->emplace(logical(l.id,e,down),s);
    }
    b->set_sources([sources](const std::string& name,const Program&,WeightSource* out) {
        const auto found=sources->find(name);if(found==sources->end())return RAD_E_NOTFOUND;
        *out=found->second;return RAD_OK;
    });
    const int64_t width=inv.nff/4;
    for(const auto& l:inv.layers) {
        // Same declaration order as actual MoeFP8: GU codes, GU scale, DN codes, DN scale,
        // each a consecutive parity-class run. Parts slice EACH fused gate/up projection.
        for(int member=0;member<4;++member)for(int parity=0;parity<2;++parity)for(int e=parity;e<inv.experts;e+=2) {
            const bool down=member>=2,scale=member%2;
            const int64_t sdiv=scale&&l.format==Format::fp8_f32?128:1;
            const int64_t sgk=scale?(l.format==Format::mxfp4?32:128):1;
            const auto src=logical(l.id,e,down);
            const auto weight_name=scale?src.substr(0,src.size()-6)+"scale":src;
            RadWeightDecl d{};d.dtype=scale?(l.format==Format::mxfp4?RAD_E8M0:RAD_F32):
                                                    (l.format==Format::mxfp4?RAD_FP4E2M1:RAD_F8E4M3);
            d.rank=2;d.shape[0]=down?inv.hidden/sdiv:2*width/sdiv;d.shape[1]=down?width/sgk:inv.hidden/sgk;
            d.access=RAD_ACCESS_CONDITIONAL;d.shard=down?RAD_SHARD_COL:RAD_SHARD_ROW;
            d.group.layer=l.id;d.group.expert=e;d.group.slot=down?1:0;
            if(!down){d.n_row_parts=2;d.row_parts[0]=width/sdiv;d.row_parts[1]=width/sdiv;}
            d.source=scale?intern(src):nullptr;d.planes=scale?"scale":"codes";
            const auto w=b->decl_weight(weight_name.c_str(),&d);if(!w)throw std::runtime_error("Builder refused "+weight_name);
            const int64_t divisor=down?sgk:sdiv;
            if(rad_weight_shard_span(b.get(),w,rank*width/divisor,(rank+1)*width/divisor)!=RAD_OK)
                throw std::runtime_error("Builder refused expert shard span");
        }
    }
    if(b->finish()!=RAD_OK) {
        std::string message="Builder refused GLM routed probe";for(const auto& e:b->errors())message+="\n"+e;
        throw std::runtime_error(message);
    }
    auto& p=b->program();
    for(const auto& l:inv.layers) {
        OpInfo op;op.op="synthetic_glm53_routed_readability";op.index=int32_t(p.ops.size());
        Band band;band.hi=1;band.dom[RAD_DOMAIN_DEVICE].row=wp::synthetic_device_row();op.bands.push_back(band);
        for(size_t h=1;h<p.weights.size();++h)if(p.weights[h].decl.group.layer==l.id) {
            op.weights.push_back(rad_weight(h));p.weights[h].first_use_op=op.index;p.weights[h].last_use_op=op.index;
            if(!p.weights[h].identity||!p.weights[h].layout_tag.empty()||p.weights[h].n_sel!=1)
                throw std::runtime_error("unexpected noncanonical selected-plane layout");
        }
        p.ops.push_back(std::move(op));
    }
    return b;
}
inline Json records(const Inventory& inv,const Program& p,const Plan& plan) {
    Json result=Json::array();std::set<std::pair<int64_t,int>> classes;
    for(const auto& mu:plan.units)if(mu.expert>=0) {
        const bool mtp=mu.layer>=inv.main_layers;
        if(!classes.insert({mu.slot_bytes,int(mtp)}).second)continue;
        Json item{{"format",name(mtp?Format::fp8_f32:Format::mxfp4)},
                  {"layer_example",mu.layer},{"expert_example",mu.expert},{"record_stride_bytes",mu.slot_bytes},
                  {"member_bytes_with_internal_alignment",mu.bytes},{"file_direct",mu.file_direct},{"members",Json::array()}};
        for(size_t k=0;k<mu.weights.size();++k) {
            const auto& w=p.weights[mu.weights[k]];
            item["members"].push_back({{"name",w.name},{"offset",mu.offset[k]},{"bytes",w.stored_bytes},
                {"align",w.align},{"dtype",rad_dtype_name(w.stored_dtype)},{"layout_tag",w.layout_tag},
                {"source",w.source},{"selected_role",w.enc.plane[w.sel[0]].role},
                {"declared_rows",w.decl.shape[0]},{"declared_cols",w.decl.shape[1]},
                {"shard",w.decl.shard},{"shard_lo",w.shard_lo},{"shard_hi",w.shard_hi}});
        }
        item["record_pack_requirements"]={"Losslessly concatenate the selected plane bytes at these ACTUAL Planner member offsets, zero/ignore padding explicitly.",
            "Gate/up ROW selection consists of a rank slice of gate followed by the SAME rank slice of up, not a contiguous half-stack.",
            "Down COL selection needs a row-by-row source-column gather; existing TP4 file_direct is false.",
            "Keep source/checkpoint revision, encoding, selected role, TP rank/span and stored-layout identity in record metadata; verify sizes/checksums when real payloads exist.",
            "A packed record is proposed SSD backing; this probe does not implement bounded RAM admission, sparse WTAB, demand I/O or lifetime protocol."};
        result.push_back(std::move(item));
    }
    return result;
}
inline Json report(const Inventory& inv,const Program& p,const Config& c) {
    Plan plan;PlannerInput input{&p,&c,p.ctx.rank,4,nullptr,0,true,false};
    const int status=Planner(input).plan(&plan);auto out=wp::report_plan(p,c,plan,status);
    out["proposed_record_classes"]=records(inv,p,plan);return out;
}
} // namespace rad::glm53_weight_probe
