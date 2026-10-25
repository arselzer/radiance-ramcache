/* Exercise the real layout/placement machinery without weights, GPU, or pools. */
#include "quant_fixture.h"
#include "../tools/gguf_weight_allocation.h"
#include <fstream>

using namespace rad;
namespace wp=rad::weight_probe;
namespace {
bool fixtures(Registry& kernels) {
    return kernels.load_dir(home()+"/testkernels/repack",{"radtest_repackkernel"})==RAD_OK &&
           qfix::reg().checkpoint_importer("gguf","iq2_xs");
}
wp::Inventory toy(int experts=4) {
    wp::Inventory inv;inv.revision="synthetic-bounded-test";inv.layers=4;inv.experts=experts;inv.hidden=5120;inv.nff=2304;
    const std::string types[]{"IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS"};
    for(int l=0;l<4;++l)for(const std::string role:{"gate","up","down"}) {
        wp::Tensor t;t.name="blk."+std::to_string(l)+".ffn_"+role+"_exps.weight";
        t.type=role=="down"?"IQ2_XS":types[l];t.layer=l;t.role=role;t.expert=true;
        t.dims={role=="down"?inv.nff:inv.hidden,role=="down"?inv.hidden:inv.nff,inv.experts};
        wp::gg::Format f;if(!wp::native_format(t.type,&f))throw std::runtime_error("bad synthetic type");
        const auto* g=wp::gg::geometry(f);t.bytes=wp::product(t.dims)/g->values*g->bytes;inv.tensors.push_back(t);
    }
    inv.tensors.push_back({"norm.weight","F32","",{513},2052,-1,false});
    inv.tensors.push_back({"hash.weight","I64","",{4,2},64,-1,false});
    inv.tensors.push_back({"dense5.weight","Q5_K","",{5120,2},7040,-1,false});
    inv.tensors.push_back({"dense8.weight","Q8_0","",{5120,2},10880,-1,false});
    return inv;
}
Config config(const char* placement,int64_t vram=0,int64_t host=0) {
    Config c;c.placement=placement;c.vram_weights_mib=vram;c.host_pool_mib=host;return c;
}
}
TEST(gguf_weight_allocation_pinned_inventory_is_complete_and_bounded) {
    std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/gguf-weight-inventory.json");
    REQUIRE(file.good());const auto j=wp::Json::parse(file);const auto inv=wp::inventory(j);
    CHECK_EQ(inv.revision,std::string("cb02422c900df489cea5833e8a92171224059b3c"));
    CHECK_EQ(inv.tensors.size(),1008u);CHECK_EQ(inv.layers,40);CHECK_EQ(inv.experts,384);
    CHECK_EQ(inv.hidden,5120);CHECK_EQ(inv.nff,2304);
    int experts=0;int64_t bytes=0;std::set<std::string> types;
    for(const auto& t:inv.tensors){experts+=t.expert;bytes+=t.bytes;types.insert(t.type);}
    CHECK_EQ(experts,120);CHECK_EQ(types.size(),10u);CHECK_EQ(bytes,int64_t(155220506880));
    // Duplicate entries, missing expert projections and broken block extents must fail closed.
    for(int kind=0;kind<5;++kind) {
        auto bad=j;
        if(kind==0)bad["tensors"].push_back(bad["tensors"][0]);
        if(kind==1) {
            auto& tensors=bad["tensors"];
            for(auto it=tensors.begin();it!=tensors.end();++it)
                if((*it)[0].get<std::string>()=="blk.0.ffn_gate_exps.weight"){tensors.erase(it);break;}
        }
        if(kind==2)bad["tensors"][0][3]=1;
        if(kind==3)bad["tensors"].push_back(wp::Json::array({"overflow64.weight","I64",wp::Json::array({INT64_MAX/64+1}),8}));
        if(kind==4)bad["tensors"].push_back(wp::Json::array({"overflowprod.weight","F32",wp::Json::array({INT64_MAX/2,4}),8}));
        bool refused=false;try{(void)wp::inventory(bad);}catch(const std::exception&){refused=true;}
        CHECK(refused);
    }
}
TEST(gguf_weight_allocation_uses_real_member_alignments_and_actual_tp4_classes) {
    Registry kernels;if(!fixtures(kernels))ARTIFACT_MISSING("native layout fixture and GGUF importers");
    const auto inv=toy();int64_t all_ranks_payload=0,source_bytes=0;
    for(const auto& t:inv.tensors)source_bytes+=t.bytes;
    for(int rank=0;rank<4;++rank) {
        auto builder=wp::build(kernels,qfix::reg(),inv,rank);const auto& prog=builder->program();
        const auto report=wp::report(prog,config("all_vram"));
        REQUIRE_EQ(report["planner_status"].get<int>(),RAD_OK);
        CHECK(report["reservation_counts_valid_for_allocation"].get<bool>());
        CHECK_EQ(report["movement_unit_count"].get<int>(),20); // 16 experts + four model weights.
        const int64_t payload=report["stored_weight_payload_bytes"];
        int64_t expected_payload=0;
        for(const auto& t:inv.tensors) {
            if(t.expert)expected_payload+=t.bytes/2304*(rank==t.layer%4?768:512);
            else expected_payload+=t.bytes;
        }
        CHECK_EQ(payload,expected_payload);all_ranks_payload+=payload;
        int expert_classes=0;std::set<int64_t> strides;
        for(const auto& c:report["classes"])if(c["example"]["expert"].get<int>()>=0) {
            ++expert_classes;strides.insert(c["stride_bytes"].get<int64_t>());
            REQUIRE_EQ(c["example"]["members"].size(),3u);
            int64_t end=0;
            for(const auto& w:c["example"]["members"]) {
                CHECK_EQ(w["align"].get<int64_t>(),256);
                CHECK_EQ(w["offset"].get<int64_t>(),wp::aligned(end,256));
                CHECK(!w["layout_tag"].get<std::string>().empty());
                end=w["offset"].get<int64_t>()+w["stored_bytes"].get<int64_t>();
            }
            CHECK_EQ(c["stride_bytes"].get<int64_t>(),wp::aligned(end,RAD_ALIGN_UNIT));
            CHECK(!c["example"]["file_direct"].get<bool>());
        }
        CHECK_EQ(expert_classes,4);CHECK_EQ(strides.size(),4u);
        CHECK_EQ(report["slab_overhead_bytes"].get<int64_t>(),0); // Uncapped all_vram is static.
        CHECK_EQ(report["host_slot_reservations_bytes"].get<int64_t>(),0);
        CHECK(report["canonical_selected_inputs_sum_64_aligned_bytes"].get<int64_t>()>payload);
        CHECK_EQ(wp::synthetic_device_row()->info->launch(nullptr,nullptr),RAD_E_STATE);
    }
    int64_t replicated=0;for(const auto& t:inv.tensors)if(!t.expert)replicated+=t.bytes;
    CHECK_EQ(all_ranks_payload,source_bytes+3*replicated);
}
TEST(gguf_weight_allocation_reports_real_shadow_slots_and_refusals) {
    Registry kernels;if(!fixtures(kernels))ARTIFACT_MISSING("native layout fixture and GGUF importers");
    // A meaningful resident set in every class keeps the first pass's spare estimate
    // below the budget. An 8 MiB/four-expert fixture lost every resident on pass two:
    // its one shadow per class cost more than all available expert room.
    auto builder=wp::build(kernels,qfix::reg(),toy(32),0);const auto& prog=builder->program();
    const auto candidate=wp::report(prog,config("expert_tiered",96,512));
    REQUIRE_EQ(candidate["planner_status"].get<int>(),RAD_OK);
    CHECK(candidate["host_slot_reservations_bytes"].get<int64_t>()>0);
    CHECK(candidate["slab_overhead_bytes"].get<int64_t>()>0);
    CHECK(candidate["modeled_mover_vram_reservations_including_overhead_bytes"].get<int64_t>()<=96*wp::MiB);
    CHECK_EQ(candidate["configured_host_pool_backing_bytes"].get<int64_t>(),512*wp::MiB);
    CHECK(candidate["host_slot_reservations_bytes"].get<int64_t>()<512*wp::MiB);
    int64_t shadow=0,resident=0;
    for(const auto& c:candidate["classes"])if(c["example"]["expert"].get<int>()>=0) {
        const int64_t n=c["vram_resident_slots"],s=c["shadow_slots"];
        CHECK_EQ(c["unit_count"].get<int64_t>(),32);
        CHECK(n>0);CHECK_EQ(s,std::min<int64_t>(RAD_PLACE_SHADOW_SLOTS,std::max<int64_t>(1,n/4)));
        shadow+=s;resident+=n;
    }
    CHECK(shadow>0);CHECK(resident>0);CHECK(resident<128);
    const auto no_host=wp::report(prog,config("expert_tiered",96,0));
    CHECK_EQ(no_host["planner_status"].get<int>(),RAD_E_NOMEM);
    CHECK(!no_host["reservation_counts_valid_for_allocation"].get<bool>());
    CHECK(!no_host["unfit"].empty());
    auto disk=config("expert_tiered",96,512);disk.weights_disk_tier=true;
    const auto on_disk=wp::report(prog,disk);
    // Allow the planner's conservative staging reservation, but it must still refuse
    // unsupported canonical-file reads if its host capacity is deliberately exhausted.
    disk.host_pool_mib=2;const auto disk_refused=wp::report(prog,disk);
    CHECK_EQ(disk_refused["planner_status"].get<int>(),RAD_E_NOMEM);
    CHECK(!disk_refused["reservation_counts_valid_for_allocation"].get<bool>());
    CHECK(on_disk["file_stage_reserve_bytes"].get<int64_t>()>0);
}
TEST(gguf_weight_allocation_optional_dense_rows_preserve_whole_k_and_controls) {
    Registry kernels;if(!fixtures(kernels))ARTIFACT_MISSING("native layout fixture and GGUF importers");
    std::ifstream file(std::string(RAD_GOLDEN_FIXTURES)+"/gguf-weight-inventory.json");
    REQUIRE(file.good());auto inv=wp::inventory(wp::Json::parse(file));
    inv.layers=4;inv.experts=4;
    inv.tensors.erase(std::remove_if(inv.tensors.begin(),inv.tensors.end(),[](const auto& t){return t.layer>=4;}),inv.tensors.end());
    for(auto& t:inv.tensors)if(t.expert){t.dims[2]=4;t.bytes=t.bytes/384*4;}
    inv.tensors.push_back({"blk.0.unknown_projection.weight","BF16","",{5120,128},1310720,0,false});
    constexpr auto policy=wp::DensePolicy::recognized_row_shards;
    int64_t source_bytes=0,replicated=0,all_rank_payload=0;
    for(const auto& t:inv.tensors) {
        source_bytes+=t.bytes;
        if(!t.expert&&!wp::dense_decision(t,inv,policy).shard)replicated+=t.bytes;
    }
    for(int rank=0;rank<4;++rank) {
        auto builder=wp::build(kernels,qfix::reg(),inv,rank,false,policy);const auto& p=builder->program();
        std::map<std::string,const WeightInfo*> weights;
        int64_t expected=0,payload=0;
        for(size_t i=1;i<p.weights.size();++i){weights.emplace(p.weights[i].name,&p.weights[i]);payload+=p.weights[i].stored_bytes;}
        for(const auto& t:inv.tensors) {
            if(t.expert){expected+=t.bytes/inv.nff*(rank==t.layer%4?768:512);continue;}
            const auto* w=weights.at(t.name);const auto dec=wp::dense_decision(t,inv,policy);
            const int64_t rows=wp::product(t.dims,1),cols=t.dims[0];
            CHECK_EQ(w->decl.shape[1],cols); // K is never split by this optional dense policy.
            if(dec.shard) {
                CHECK_EQ(w->decl.shard,int(RAD_SHARD_ROW));CHECK_EQ(w->decl.shape[0],rows/4);
                CHECK_EQ(w->shard_lo,rank*(rows/4));CHECK_EQ(w->shard_hi,(rank+1)*(rows/4));
                CHECK_EQ(w->stored_bytes,t.bytes/4);CHECK(!dec.obligation.empty());expected+=t.bytes/4;
            }else{CHECK_EQ(w->decl.shard,int(RAD_SHARD_NONE));CHECK_EQ(w->stored_bytes,t.bytes);expected+=t.bytes;}
        }
        CHECK_EQ(payload,expected);all_rank_payload+=payload;
        const auto* gate=weights.at("blk.0.ffn_gate_shexp.weight");
        const auto* down=weights.at("blk.0.ffn_down_shexp.weight");
        CHECK_EQ(gate->decl.shape[0],576);CHECK_EQ(gate->decl.shape[1],5120);
        CHECK_EQ(down->decl.shape[0],1280);CHECK_EQ(down->decl.shape[1],2304);
        const auto* oa=weights.at("blk.0.attn_output_a.weight");
        CHECK_EQ(oa->decl.shape[0],2048);CHECK_EQ(oa->decl.shape[1],4096);
        CHECK_EQ(oa->shard_lo%1024,0);CHECK_EQ((oa->shard_hi-oa->shard_lo)/1024,2);
        const auto* ew=weights.at("blk.1.engram_wkv.weight");
        CHECK_EQ(ew->decl.shape[0],6400);CHECK_EQ(ew->decl.shape[1],6144);
        for(const std::string name:{"blk.0.ffn_gate_inp.weight","blk.0.hc_attn_fn.weight",
                "blk.0.attn_q_a_norm.weight","blk.0.attn_sinks.weight","engram_primes.weight",
                "blk.1.engram_k.weight","blk.1.engram_q.weight","blk.0.unknown_projection.weight"})
            CHECK_EQ(weights.at(name)->decl.shard,int(RAD_SHARD_NONE));
        const auto report=wp::report(p,config("all_vram"));CHECK_EQ(report["planner_status"].get<int>(),RAD_OK);
        const auto dense=wp::dense_partition_report(p,inv,policy);
        CHECK(!dense["serving_graph_implemented"].get<bool>());CHECK(!dense["gathers_implemented"].get<bool>());
        CHECK(!dense["sharded"].empty());
    }
    CHECK_EQ(all_rank_payload,source_bytes+3*replicated);
    // A known role with a new, unreviewed geometry cannot silently acquire a TP layout.
    auto changed=inv.tensors.front();changed.name="blk.0.attn_output_a.weight";changed.layer=0;
    changed.expert=false;changed.type="Q5_K";changed.dims={4096,2048,4};
    bool refused=false;try{(void)wp::dense_decision(changed,inv,policy);}catch(const std::exception&){refused=true;}
    CHECK(refused);
}
TEST(gguf_weight_allocation_identical_rankings_with_unequal_costs_diverge) {
    Registry kernels;if(!fixtures(kernels))ARTIFACT_MISSING("native layout fixture and GGUF importers");
    auto inv=toy();
    // All layers have the same encoding and total source size. Only the rotating
    // 768/512 widths change a particular expert's cost on a particular rank.
    for(auto& t:inv.tensors)if(t.expert){t.type="IQ2_XS";t.bytes=wp::product(t.dims)/256*74;}
    auto c=config("expert_tiered",7,128);c.deterministic=true;
    // Determinism disables heat/shadows, isolating the initial greedy residency
    // choice. The ordering stays expert-major, then layer, on every rank.
    wp::Json ranks=wp::Json::array();
    for(int rank=0;rank<4;++rank) {
        auto b=wp::build(kernels,qfix::reg(),inv,rank);
        auto p=wp::report(b->program(),c);REQUIRE_EQ(p["planner_status"].get<int>(),RAD_OK);
        ranks.push_back({{"rank",rank},{"candidate",std::move(p)}});
    }
    const auto comparison=wp::compare_expert_residency(ranks,"candidate");
    CHECK(comparison["all_rank_weight_plans_fit"].get<bool>());
    CHECK(!comparison["resident_sets_agree"].get<bool>());
    CHECK(!comparison["common_resident_policy_satisfied"].get<bool>());
    CHECK(!comparison["tp_execution_validated"].get<bool>());CHECK(!comparison["differences"].empty());
    const auto& rank0=ranks[0]["candidate"]["expert_residency_by_layer"];
    const auto& rank2=ranks[2]["candidate"]["expert_residency_by_layer"];
    CHECK_EQ(rank0["3"]["vram_bits_hex"].get<std::string>(),std::string("00"));
    CHECK_EQ(rank2["3"]["vram_bits_hex"].get<std::string>(),std::string("01"));
    // Full exact bitmaps are retained, not only a count or a digest.
    CHECK_EQ(rank0["3"]["declared_bits_hex"].get<std::string>(),std::string("0f"));
    auto refused=ranks;refused[1]["candidate"]["fits_weight_only_assumptions"]=false;
    const auto failure=wp::compare_expert_residency(refused,"candidate");
    CHECK_EQ(failure["status"].get<std::string>(),std::string("rank_planner_refused"));
    CHECK(!failure["common_resident_policy_satisfied"].get<bool>());
}
RAD_TEST_MAIN()
