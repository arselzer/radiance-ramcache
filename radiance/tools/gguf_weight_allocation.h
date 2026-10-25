/* Metadata-only allocation probe for a proposed native-GGUF weight graph.
 * No tensors/pools, device enumeration, kernel launch, or production configuration.
 * Builder assigns real stored layouts; Planner computes real movement-unit classes.
 * Synthetic device availability below is an explicit future-kernel assumption. */
#pragma once
#include "build/rad_build.h"
#include "place/planner.h"
#include "place/tp_plan.h"
#include "../arch/common/rad_gguf_relayout.h"
#include <nlohmann/json.hpp>
#include <cctype>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>

namespace rad::weight_probe {
using Json = nlohmann::json;
namespace gg = rad::gguf;
constexpr int64_t MiB = 1ll << 20;
inline int64_t aligned(int64_t n, int64_t a) { return (n + a - 1) / a * a; }
inline int64_t product(const std::vector<int64_t>& dims, size_t start = 0) {
    int64_t n = 1;
    for (size_t i = start; i < dims.size(); ++i) {
        if (dims[i] <= 0 || n > INT64_MAX / dims[i]) throw std::runtime_error("invalid inventory dimensions");
        n *= dims[i];
    }
    return n;
}
inline bool native_format(const std::string& type, gg::Format* out) {
    constexpr gg::Format formats[]{gg::Format::iq2_xxs, gg::Format::iq2_xs, gg::Format::iq2_s,
                                  gg::Format::iq3_xxs, gg::Format::q5_k, gg::Format::q8_0};
    auto lower=[](std::string s) { for(auto& c:s)c=char(std::tolower(static_cast<unsigned char>(c)));return s; };
    for (auto f : formats) if (lower(type) == lower(gg::geometry(f)->name)) { *out = f; return true; }
    return false;
}
inline uint32_t plain_type(const std::string& type) {
    if (type == "F32") return RAD_F32;
    if (type == "BF16") return RAD_BF16;
    if (type == "I32") return RAD_I32;
    if (type == "I64") return RAD_I64;
    throw std::runtime_error("unsupported inventory type: " + type);
}
struct Tensor {
    std::string name, type, role;
    std::vector<int64_t> dims; // GGUF order: columns first.
    int64_t bytes = 0;
    int layer = -1;
    bool expert = false;
};
struct Inventory {
    std::string revision;
    int layers = 0, experts = 0;
    int64_t hidden = 0, nff = 0;
    std::vector<Tensor> tensors;
};
enum class DensePolicy { replicated, recognized_row_shards };
inline const char* dense_policy_name(DensePolicy p) {
    return p==DensePolicy::replicated?"replicated_baseline":"reviewed_projection_row_shards";
}
struct DenseDecision {
    bool shard=false;
    std::string role="replicated", obligation="Unreviewed/control tensors stay replicated.";
};
inline DenseDecision dense_decision(const Tensor& t,const Inventory& inv,DensePolicy policy) {
    DenseDecision result;
    if(t.expert||policy==DensePolicy::replicated)return result;
    std::string role=t.name;
    if(t.layer>=0) {
        const std::string prefix="blk."+std::to_string(t.layer)+".";
        if(role.compare(0,prefix.size(),prefix)!=0)return result;
        role=role.substr(prefix.size());
    }
    if(t.layer<0&&role!="token_embd.weight"&&role!="output.weight")return result;
    std::vector<int64_t> want;
    std::string obligation;
    if(role=="token_embd.weight") {
        want={inv.hidden,129280};
        obligation="Vocabulary-row partition: exact owner lookup/broadcast or masked sum must reconstruct the full hidden embedding on every consuming rank.";
    } else if(role=="output.weight") {
        want={inv.hidden,129280};
        obligation="Gather complete vocabulary logits before existing sampling, or implement exact distributed normalization/selection with global vocabulary indices.";
    } else if(role=="attn_q_a.weight") {
        want={inv.hidden,1280};
        obligation="Gather the complete 1280-dimensional query latent before replicated RMSNorm and q_b; local RMSNorm changes the function.";
    } else if(role=="attn_q_b.weight") {
        want={1280,32768};
        obligation="Full 1280 query-latent input is required. Gather expanded Q or supply correctly owned whole local heads to head-local attention.";
    } else if(role=="attn_kv.weight") {
        want={inv.hidden,512};
        obligation="Gather complete 512-dimensional KV latent before replicated norm, RoPE, shared-KV cache writes and index consumers.";
    } else if(role=="attn_output_a.weight") {
        want={4096,1024,8};
        obligation="Grouped weights [8,1024,4096] retain two whole groups per rank; each owned group needs its matching 4096-channel attention input. Gather all 8192 output latents before whole-K output_b.";
    } else if(role=="attn_output_b.weight") {
        want={8192,inv.hidden};
        obligation="Requires full 8192-dimensional grouped latent input. Gather output hidden shards before residual/mHC consumers.";
    } else if(role=="ffn_gate_shexp.weight"||role=="ffn_up_shexp.weight") {
        want={inv.hidden,inv.nff};
        obligation="Gate/up each own matching 576-row slices; after elementwise activation, gather complete 2304-dimensional shared activation for row-sharded whole-K down.";
    } else if(role=="ffn_down_shexp.weight") {
        want={inv.nff,inv.hidden};
        obligation="Row shard N=5120 to 1280, preserving whole K=2304; gather full shared activation first and full hidden output before residual/mHC. Never substitute a K=576 slice.";
    } else if(role=="engram_wkv.weight") {
        want={6144,5*inv.hidden};
        obligation="Requires all 6144 Engram inputs. Gather complete 25600 outputs before [hc_mult+1,hidden]=[5,5120] reshape/gating; a 6400-row cut crosses hidden-vector boundaries.";
    } else if(role=="attn_compressor_gate.weight"||role=="attn_compressor_kv.weight") {
        want={inv.hidden,512};
        obligation="Gather complete paired score/KV projections before replicated compression, state update, full-latent normalization and cache publication, or prove equivalent channel-local state plus exact norm.";
    } else if(role=="indexer.attn_k.weight") {
        want={512,128};
        obligation="Requires full 512-dimensional KV input; gather full 128-dimensional index key before normalization/cache/scoring, unless exact distributed norm/dot is provided.";
    } else if(role=="indexer.attn_q_b.weight") {
        want={1280,4096};
        obligation="Requires full 1280 query latent; preserve whole 128-dimensional index heads. Gather Q or implement exact local-head scoring plus global score aggregation/top-k.";
    } else if(role=="indexer.proj.weight") {
        want={inv.hidden,32};
        obligation="Gather all 32 index-head weights or supply exact head-local weights with global index-score aggregation/top-k.";
    } else return result;
    gg::Format f;
    if(t.dims!=want||(!native_format(t.type,&f)&&t.type!="BF16"))
        throw std::runtime_error("reviewed dense projection has unreviewed shape/type: "+t.name);
    const int64_t rows=product(t.dims,1);
    if(rows<4||rows%4)throw std::runtime_error("reviewed dense rows do not divide TP4: "+t.name);
    result.shard=true;result.role=role;result.obligation=std::move(obligation);return result;
}
inline Inventory inventory(const Json& j) {
    if (j.at("schema") != "radiance-gguf-weight-inventory-v1")
        throw std::runtime_error("wrong bounded inventory schema");
    Inventory out;
    out.revision = j.at("source_revision"); out.layers = j.at("layers"); out.experts = j.at("experts");
    out.hidden = j.at("hidden"); out.nff = j.at("nff");
    if (out.layers <= 0 || out.layers > 1024 || out.experts <= 0 || out.experts > 4096 ||
        out.hidden <= 0 || out.nff < 1024 || out.nff % 256 || out.nff / 256 % 4 != 1)
        throw std::runtime_error("inventory does not support the proposed rotating 256-block TP4 partition");
    std::set<std::string> names;
    const std::regex expert_re(R"(blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight)");
    const std::regex layer_re(R"(blk\.(\d+)\..*)");
    std::map<int, std::set<std::string>> expert_roles; size_t expanded=0;
    if (!j.at("tensors").is_array() || j.at("tensors").size() > 16384)
        throw std::runtime_error("inventory tensor count exceeds metadata-only bound");
    for (const auto& row : j.at("tensors")) {
        Tensor t; t.name = row.at(0); t.type = row.at(1);
        t.dims = row.at(2).get<std::vector<int64_t>>(); t.bytes = row.at(3);
        if (t.name.empty() || t.name.size() > 256 || !names.insert(t.name).second ||
            t.dims.empty() || t.dims.size() > size_t(RAD_MAX_RANK) || t.bytes <= 0)
            throw std::runtime_error("invalid/duplicate inventory tensor");
        const int64_t count = product(t.dims);
        if(count>INT64_MAX/64)throw std::runtime_error("inventory extent exceeds safe byte arithmetic");
        std::smatch match;
        if (std::regex_match(t.name, match, expert_re)) {
            t.expert = true; t.layer = std::stoi(match[1]); t.role = match[2];
            const bool down = t.role == "down";
            const std::vector<int64_t> want{down ? out.nff : out.hidden,
                                            down ? out.hidden : out.nff, out.experts};
            if (t.dims != want || t.layer < 0 || t.layer >= out.layers)
                throw std::runtime_error("expert geometry mismatch: " + t.name);
            expert_roles[t.layer].insert(t.role);
        } else if (std::regex_match(t.name, match, layer_re)) {
            t.layer = std::stoi(match[1]);
            if (t.layer < 0 || t.layer >= out.layers) throw std::runtime_error("layer outside inventory");
        }
        expanded+=t.expert?size_t(out.experts):1u;
        if(expanded>100000)throw std::runtime_error("expanded weight count exceeds metadata-only bound");
        gg::Format f;
        int64_t expected = 0;
        if (native_format(t.type, &f)) {
            const auto* g = gg::geometry(f);
            if (t.dims[0] % g->values) throw std::runtime_error("non-native block width: " + t.name);
            const auto blocks = count / g->values;
            if (blocks > INT64_MAX / g->bytes) throw std::runtime_error("native byte size overflow");
            expected = blocks * g->bytes;
        } else {
            const auto dtype=plain_type(t.type);
            const int bits=rad_dtype_bits(dtype);
            // rad_dtype_bytes computes (count * bits + 7) / 8 in signed int64.
            // Check both the multiplication and its rounding addition first.
            if(bits<=0||count>(INT64_MAX-7)/bits)throw std::runtime_error("plain byte size overflow");
            expected = rad_dtype_bytes(dtype, count);
        }
        if (t.bytes != expected) throw std::runtime_error("payload extent disagrees with encoding: " + t.name);
        out.tensors.push_back(std::move(t));
    }
    for (int l = 0; l < out.layers; ++l)
        if (expert_roles[l] != std::set<std::string>{"gate", "up", "down"})
            throw std::runtime_error("missing expert projection in inventory");
    return out;
}
inline WeightSource source(const Tensor& t, Registry& importers) {
    WeightSource s{}; s.rank = 2; s.shape[0] = t.expert ? t.dims[1] : product(t.dims, 1);
    s.shape[1] = t.dims[0];
    gg::Format f;
    if (native_format(t.type, &f)) {
        const char* type=gg::geometry(f)->name;
        const auto* row = importers.checkpoint_importer("gguf", type);
        if (!row) throw std::runtime_error("missing real GGUF importer: " + t.type);
        const RadCheckpointImportWeight w{t.name.c_str(), type, s.shape[0], s.shape[1]};
        if (row->info->encoding(&w, &s.enc) != RAD_OK) throw std::runtime_error("importer encoding refused");
    } else s.enc = rad_enc_plain(plain_type(t.type));
    return s;
}
inline int cannot_launch(const RadArgs*, RadStream) { return RAD_E_STATE; }
inline const KernelRow* synthetic_device_row() {
    static const RadKernelInfo info{
        .name="allocation_probe_future_device_consumer", .op="allocation_probe", .family="misc",
        .computes="nothing; assumed device readability for allocation only", .shape="metadata only",
        .dtypes="native GGUF/plain", .domain=RAD_DOMAIN_DEVICE, .launch=cannot_launch};
    static const KernelRow row = [] { KernelRow r; r.info = &info; r.plugin = "allocation_probe_synthetic"; return r; }();
    return &row;
}
inline std::unique_ptr<Builder> build(Registry& kernels, Registry& importers,
                                      const Inventory& inv, int rank, bool rotate_by_expert=false,
                                      DensePolicy dense_policy=DensePolicy::replicated) {
    if (rank < 0 || rank >= 4) throw std::runtime_error("probe supports TP4 ranks 0..3");
    RadModelMeta m{}; m.arch_id="weight_allocation_probe"; m.name="not-a-serving-graph";
    m.quant="native-gguf"; m.n_layers=inv.layers; m.n_embd=inv.hidden; m.n_ff=inv.nff;
    RadBuildCtx c{}; c.rank=rank; c.world_size=4; c.max_tok=1; c.max_seqs=1; c.max_ctx=1; c.scope="";
    auto b = std::make_unique<Builder>(kernels, m, c); b->set_instances(false);
    auto inputs = std::make_shared<std::map<std::string, WeightSource>>();
    for (const auto& t : inv.tensors) inputs->emplace(t.name, source(t, importers));
    b->set_sources([inputs](const std::string& name, const Program&, WeightSource* out) {
        const auto base = name.substr(0, name.find('#'));
        const auto found = inputs->find(base); if (found == inputs->end()) return RAD_E_NOTFOUND;
        *out = found->second; return RAD_OK;
    });
    std::vector<rad_weight> plain;
    for (const auto& t : inv.tensors) {
        const auto& src = inputs->at(t.name);
        gg::Format f; const bool native = native_format(t.type, &f);
        const auto dense=dense_decision(t,inv,dense_policy);
        for (int e = 0, n = t.expert ? inv.experts : 1; e < n; ++e) {
            const int wide_rank = t.expert ? (t.layer + (rotate_by_expert?e:0)) % 4 : 0;
            const int64_t narrow = inv.nff / 256 / 4 * 256;
            const int64_t width = narrow + (rank == wide_rank ? 256 : 0);
            int64_t lo = 0;
            for (int r = 0; r < rank; ++r) lo += narrow + (r == wide_rank ? 256 : 0);
            const bool down = t.role == "down";
            RadWeightDecl d{}; d.dtype=src.enc.plane[0].dtype; d.rank=2;
            d.shape[0] = t.expert ? (down ? inv.hidden : width) : src.shape[0];
            d.shape[1] = t.expert ? (down ? width : inv.hidden) : src.shape[1];
            d.access = t.expert ? RAD_ACCESS_CONDITIONAL : RAD_ACCESS_PER_TOKEN;
            d.shard = t.expert ? (down ? RAD_SHARD_COL : RAD_SHARD_ROW) : RAD_SHARD_NONE;
            if(dense.shard){d.shape[0]=src.shape[0]/4;d.shard=RAD_SHARD_ROW;}
            d.group.layer=t.layer; d.group.expert=t.expert ? e : -1;
            d.group.slot=t.role=="up" ? 1 : t.role=="down" ? 2 : 0;
            const auto name = t.name + (t.expert ? "#" + std::to_string(e) : "");
            const auto w = b->decl_weight(name.c_str(), &d);
            if (!w) throw std::runtime_error("Builder refused weight: " + name);
            if (t.expert && rad_weight_shard_span(b.get(), w, lo, lo + width) != RAD_OK)
                throw std::runtime_error("Builder refused shard span: " + name);
            if(dense.shard&&rad_weight_shard_span(b.get(),w,rank*d.shape[0],(rank+1)*d.shape[0])!=RAD_OK)
                throw std::runtime_error("Builder refused dense row span: "+name);
            if (native) {
                if (!b->decl_op("test_repack_layout", nullptr, 0, &w, 1))
                    throw std::runtime_error("Builder refused native layout consumer: " + name);
            } else plain.push_back(w);
        }
    }
    if (b->finish() != RAD_OK) {
        std::string why="Builder refused the weight-only probe";
        for (const auto& e : b->errors()) why += "\n" + e;
        throw std::runtime_error(why);
    }
    auto& p = b->program();
    // The fixture assigned genuine host-layout metadata, but performs no inference. Model
    // future device availability explicitly, so pin_mapped does not misclassify its private
    // layouts. Keep no host fallback: a host read of these bytes has not been implemented.
    for (auto& op : p.ops) for (auto& band : op.bands) {
        band.dom[RAD_DOMAIN_DEVICE].row=synthetic_device_row();
        band.dom[RAD_DOMAIN_HOST]=Resolved{};
    }
    for (auto w : plain) {
        OpInfo op; op.op="allocation_probe_plain"; op.index=int32_t(p.ops.size()); op.weights={w};
        Band band; band.hi=1; band.dom[RAD_DOMAIN_DEVICE].row=synthetic_device_row(); op.bands.push_back(band);
        p.weights[w].first_use_op=op.index; p.weights[w].last_use_op=op.index; p.ops.push_back(std::move(op));
    }
    return b;
}
inline Json dense_partition_report(const Program& p,const Inventory& inv,DensePolicy policy) {
    Json out{{"policy",dense_policy_name(policy)},{"serving_graph_implemented",false},
             {"gathers_implemented",false},{"sharded",Json::array()},{"replicated_nonexpert_count",0}};
    std::map<std::string,const WeightInfo*> weights;
    for(size_t i=1;i<p.weights.size();++i)weights.emplace(p.weights[i].name,&p.weights[i]);
    int count=0;
    for(const auto& t:inv.tensors)if(!t.expert) {
        const auto decision=dense_decision(t,inv,policy);
        if(!decision.shard){++count;continue;}
        const auto found=weights.find(t.name);if(found==weights.end())throw std::runtime_error("dense report missing weight");
        const auto& w=*found->second;
        out["sharded"].push_back({{"name",t.name},{"source_dimensions_gguf_order",t.dims},
            {"rows",w.decl.shape[0]},{"cols",w.decl.shape[1]},{"shard",w.decl.shard},
            {"row_lo",w.shard_lo},{"row_hi",w.shard_hi},{"stored_bytes",w.stored_bytes},
            {"requires",decision.obligation}});
    }
    out["replicated_nonexpert_count"]=count;return out;
}
inline Json footprint(const Program& p) {
    WeightFootprint wf{}; weight_footprint(p, &wf, TablePlacement::Vram, TablePlacement::Vram, false, true);
    return {{"statics_bytes",wf.statics}, {"experts_bytes",wf.experts}, {"mapped_bytes",wf.mapped},
            {"expert_unit_max_bytes",wf.expert_unit_max}, {"expert_layer_max_bytes",wf.expert_layer_max},
            {"conservative_expert_spare_bytes",wf.expert_spare}, {"units",wf.n_units}};
}
inline Json expert_residency(const Plan& plan) {
    struct Layer { std::vector<uint8_t> declared,resident;int experts=0,residents=0; };
    std::map<int,Layer> layers;
    for(const auto& u:plan.units)if(u.layer>=0&&u.expert>=0) {
        auto& l=layers[u.layer];const size_t byte=size_t(u.expert)/8;
        if(l.declared.size()<=byte){l.declared.resize(byte+1);l.resident.resize(byte+1);}
        const auto bit=uint8_t(1u<<(u.expert%8));
        l.declared[byte]|=bit;l.experts=std::max(l.experts,u.expert+1);
        if(u.tier==Tier::VRAM){l.resident[byte]|=bit;++l.residents;}
    }
    auto hex=[](const std::vector<uint8_t>& bytes) {
        constexpr char digits[]="0123456789abcdef";std::string s;
        for(auto b:bytes){s+=digits[b>>4];s+=digits[b&15];}return s;
    };
    Json out=Json::object();
    for(const auto& [layer,l]:layers)
        out[std::to_string(layer)]={{"expert_id_extent",l.experts},{"resident_count",l.residents},
            {"declared_bits_hex",hex(l.declared)},{"vram_bits_hex",hex(l.resident)}};
    return out;
}
inline Json compare_expert_residency(const Json& ranks,const char* plan_key) {
    Json out{{"method","exact declared and VRAM bitmaps by layer and expert ID; no hashes"},
             {"all_rank_weight_plans_fit",true},{"resident_sets_agree",true},
             {"differences",Json::array()},{"tp_execution_validated",false}};
    if(ranks.size()!=4)throw std::runtime_error("TP residency comparison requires four rank reports");
    const auto& baseline=ranks.at(0).at(plan_key).at("expert_residency_by_layer");
    for(size_t rank=0;rank<ranks.size();++rank) {
        const auto& p=ranks.at(rank).at(plan_key);
        if(!p.at("fits_weight_only_assumptions").get<bool>())out["all_rank_weight_plans_fit"]=false;
        const auto& masks=p.at("expert_residency_by_layer");
        if(masks.size()!=baseline.size())out["resident_sets_agree"]=false;
        for(auto it=baseline.begin();it!=baseline.end();++it) {
            const auto found=masks.find(it.key());
            if(found==masks.end()||found->at("declared_bits_hex")!=it->at("declared_bits_hex")||
               found->at("vram_bits_hex")!=it->at("vram_bits_hex")) {
                out["resident_sets_agree"]=false;
                out["differences"].push_back({{"rank",rank},{"layer",it.key()},
                    {"rank0_bits_hex",it->at("vram_bits_hex")},
                    {"rank_bits_hex",found==masks.end()?Json(nullptr):found->at("vram_bits_hex")}});
            }
        }
    }
    const bool fit=out["all_rank_weight_plans_fit"],agree=out["resident_sets_agree"];
    out["common_resident_policy_satisfied"]=fit&&agree;
    out["status"]=!fit?"rank_planner_refused":!agree?"independent_local_residency_differs":"resident_masks_match";
    out["meaning"]="Ranks resolve canonical expert handles to valid local addresses. Different VRAM masks are diagnostic; a common initial set is an optional policy, not a proven universal correctness requirement. Numerical and collective execution remain unvalidated.";
    return out;
}
inline Json report_plan(const Program& prog, const Config& config,const Plan& p,int status) {
    Json out{{"planner_status",status},{"fits_weight_only_assumptions",status==RAD_OK&&p.fits()},
             {"weight_count",prog.weights.size()-1}, {"movement_unit_count",p.units.size()},
             {"vram_weight_budget_bytes",p.vram_budget}, {"configured_host_pool_backing_bytes",config.host_pool_total_mib()*MiB},
             {"host_weight_budget_bytes",p.host_budget}, {"file_stage_reserve_bytes",p.file_stage_reserve},
             {"file_stage_vram_bytes",p.file_stage_vram}, {"slab_overhead_bytes",p.slab_overhead},
             {"expert_residency_by_layer",expert_residency(p)},
             {"residency_bitmap_encoding","Hex pairs in increasing byte order; byte i bit b denotes expert ID 8*i+b. Refused-plan flags are diagnostic only."},
             {"notes",p.notes}, {"classes",Json::array()}, {"unfit",Json::array()}};
    for (const auto& u : p.unfit) out["unfit"].push_back({{"what",u.what},{"why",u.why},{"count",u.count},{"bytes",u.bytes}});
    int64_t payload=0, canonical=0, staging_max=0, slots=0, host=0, device=0, static_need=0;
    for (const auto& mu:p.units)
        if(mu.tier==Tier::VRAM&&mu.slab_class>=0&&mu.slab_slot<0)
            static_need=aligned(static_need,RAD_ALIGN_UNIT)+mu.bytes;
    for (size_t w = 1; w < prog.weights.size(); ++w) {
        const auto& wi = prog.weights[w]; payload += wi.stored_bytes;
        int64_t staged=0;
        for (int s = 0; s < wi.n_sel; ++s)
            staged += aligned(wi.sel_rows[s]*rad_enc_row_bytes(wi.enc.plane[wi.sel[s]].dtype,wi.sel_cols[s]),64);
        canonical += staged; staging_max=std::max(staging_max,staged);
    }
    for (size_t c = 0; c < p.classes.size(); ++c) {
        const auto& sc=p.classes[c]; Json j{{"name",sc.name},{"stride_bytes",sc.stride},{"unit_count",sc.n_units},
            {"vram_resident_slots",sc.n_slots},{"shadow_slots",sc.n_shadow},{"stage_slots",sc.n_stage}};
        int64_t hp=0, vr=0, raw=0, aligned_slots=0; bool example=false;
        Json widths=Json::array(); std::set<int64_t> seen_widths;
        for (const auto& mu : p.units) if (mu.slab_class==int32_t(c)) {
            raw+=mu.bytes; aligned_slots+=mu.slot_bytes;
            if (mu.tier==Tier::HostPinned) ++hp;
            if (mu.tier==Tier::VRAM) ++vr;
            if (!example) {
                j["example"]={{"layer",mu.layer},{"expert",mu.expert},{"unit_bytes",mu.bytes},{"file_direct",mu.file_direct},{"members",Json::array()}};
                for (size_t k=0;k<mu.weights.size();++k) {
                    const auto& w=prog.weights[mu.weights[k]];
                    j["example"]["members"].push_back({{"name",w.name},{"offset",mu.offset[k]},
                        {"align",w.align},{"stored_bytes",w.stored_bytes},{"layout_tag",w.layout_tag},
                        {"selected_planes",w.n_sel},{"rows",w.decl.shape[0]},{"cols",w.decl.shape[1]},
                        {"shard",w.decl.shard},{"shard_lo",w.shard_lo},{"shard_hi",w.shard_hi}});
                }
                example=true;
            }
            if (mu.expert>=0 && !mu.weights.empty()) {
                const auto& w=prog.weights[mu.weights[0]];
                const int64_t width=w.decl.shard==RAD_SHARD_COL?w.decl.shape[1]:w.decl.shape[0];
                if(seen_widths.insert(width).second)widths.push_back(width);
            }
        }
        const int64_t tail=(sc.n_slots+sc.n_shadow+sc.n_stage)>0?RAD_ALIGN_UNIT:0;
        const int64_t vram=(sc.n_slots+sc.n_shadow+sc.n_stage)*sc.stride+tail;
        j["host_slots"]=hp;j["host_slot_bytes"]=hp*sc.stride;j["vram_units"]=vr;j["vram_slab_reservation_bytes"]=vram;
        j["all_unit_slot_bytes"]=aligned_slots;j["unit_payload_plus_member_alignment_bytes"]=raw;j["expert_widths"]=widths;
        host+=hp*sc.stride;if(vram>0)device=aligned(device,RAD_ALIGN_UNIT)+vram;
        slots+=aligned_slots;out["classes"].push_back(std::move(j));
    }
    out["stored_weight_payload_bytes"]=payload;out["all_aligned_weight_slot_bytes"]=slots;
    out["canonical_selected_inputs_sum_64_aligned_bytes"]=canonical;
    out["largest_canonical_input_bytes"]=staging_max;out["host_slot_reservations_bytes"]=host;
    if(static_need>0)device=aligned(device,RAD_ALIGN_UNIT)+static_need;
    out["modeled_mover_static_vram_reservation_bytes"]=static_need;
    out["modeled_mover_vram_reservations_including_overhead_bytes"]=device;
    // On refusal, a default VRAM tier may remain on unfit units. Never present a refused
    // plan's slot counts as a successful allocation or reduce/rewrite its budgets.
    out["reservation_counts_valid_for_allocation"]=status==RAD_OK&&p.fits();
    return out;
}
inline Json report(const Program& prog, const Config& config) {
    Plan p;PlannerInput in{&prog,&config,prog.ctx.rank,4,nullptr,0,true,false};
    const int status=Planner(in).plan(&p);return report_plan(prog,config,p,status);
}
} // namespace rad::weight_probe
