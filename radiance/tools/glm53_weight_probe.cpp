/* Host-only GLM53 routed weight accounting, config-derived fixture, no payloads/GPU/pools. */
#include "glm53_weight_allocation.h"
#include <filesystem>
#include <fstream>
#include <iostream>
namespace gp=rad::glm53_weight_probe;
int main(int argc,char** argv) {
    if(argc!=3&&argc!=5){std::cerr<<"usage: glm53-weight-probe inventory.json output.json [routed-weight-MiB host-MiB]\n";return 2;}
    try {
        if(std::filesystem::file_size(argv[1])>1024*1024)throw std::runtime_error("inventory exceeds1MiB metadata cap");
        std::ifstream f(argv[1]);const auto inv=gp::inventory(gp::Json::parse(f));rad::Registry registry;
        rad::Config all;all.placement="all_vram";all.vram_weights_mib=0;all.host_pool_mib=0;
        rad::Config candidate;candidate.placement="expert_tiered";
        if(argc==5) {
            candidate.vram_weights_mib=std::stoll(argv[3]);candidate.host_pool_mib=std::stoll(argv[4]);
            if(candidate.vram_weights_mib<=0||candidate.host_pool_mib<0||candidate.vram_weights_mib>1048576||candidate.host_pool_mib>1048576)
                throw std::runtime_error("explicit routed-only budgets must be bounded positive/nonnegative MiB");
        }
        gp::Json out{{"schema","radiance-glm53-routed-weight-allocation-v1"},{"source_revision",inv.revision},
            {"method","actual Builder canonical selected-plane/shard geometry + actual Planner; config-derived synthetic sources"},
            {"checkpoint_tensor_headers_verified",false},{"checkpoint_payload_read",false},{"synthetic_device_readability",true},
            {"actual_kernel_row_resolution",false},{"canonical_layout_evidence","source-reviewed actual r4d routed MXFP4/E8M0 and FP8/F32 hooks both preserve selected canonical planes"},
            {"tp_world",4},{"ranks",gp::Json::array()},{"candidate_all_local_plans_fit",true},
            {"budgets_scope","routed weights ONLY, explicit hypothetical inputs, no production setting recommendation"},
            {"excluded",{"shared/dense/nonexpert weights","KV/linear/prefix/checkpoint state","arena/scratch/logits","runtime/driver/modules/collectives","startup input/staging/peak"}},
            {"does_not_establish_model_fit_or_inference_speed",true},
            {"startup_caveat","Host pool is fully backed; loader has4GiB canonical-input target plus possible overshoot and other staging, not included."}};
        for(int rank=0;rank<4;++rank) {
            auto b=gp::build(registry,inv,rank);gp::Json r{{"rank",rank},{"uncapped_routed_only",gp::report(inv,b->program(),all)}};
            if(argc==5){r["candidate_routed_only"]=gp::report(inv,b->program(),candidate);
                if(!r["candidate_routed_only"]["fits_weight_only_assumptions"].get<bool>())out["candidate_all_local_plans_fit"]=false;}
            out["ranks"].push_back(std::move(r));
        }
        if(argc!=5)out["candidate_all_local_plans_fit"]=nullptr;
        std::ofstream destination(argv[2]);if(!destination)throw std::runtime_error("cannot open output");destination<<out.dump(2)<<'\n';
        if(!destination)throw std::runtime_error("cannot write output");
        return argc==5&&!out["candidate_all_local_plans_fit"].get<bool>()?1:0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
