/* Offline, host-only Planner report. This executable is an experimental research tool,
 * not installed serving support and not a model-fit check. */
#include "gguf_weight_allocation.h"
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace rad;
namespace wp=rad::weight_probe;
int main(int argc,char**argv) {
    if(argc<4||argc>9) {
        std::cerr<<"usage: gguf-weight-probe HOME inventory.json output.json [weight-MiB host-MiB] [--dense-row-shards] [--coordinate-residency] [--require-common-residency]\n"
                   "No weights are read or allocated. Optional budgets are hypothetical per-rank weights-only budgets.\n";
        return 2;
    }
    try {
        wp::DensePolicy dense_policy=wp::DensePolicy::replicated;
        bool coordinate=false,require_common=false;
        std::vector<std::string> budgets;
        for(int i=4;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--dense-row-shards")dense_policy=wp::DensePolicy::recognized_row_shards;
            else if(arg=="--coordinate-residency")coordinate=true;
            else if(arg=="--require-common-residency")require_common=true;
            else budgets.push_back(arg);
        }
        if(!budgets.empty()&&budgets.size()!=2)throw std::runtime_error("supply both optional budgets, or neither");
        if((coordinate||require_common)&&budgets.empty())throw std::runtime_error("an explicit common-residency policy requires weight and host budgets");
        if(std::filesystem::file_size(argv[2])>4*1024*1024)throw std::runtime_error("inventory exceeds 4 MiB bound");
        std::ifstream input(argv[2]);const auto inv=wp::inventory(wp::Json::parse(input));
        Registry kernels,importers;
        if(kernels.load_dir(std::string(argv[1])+"/testkernels/repack",{"radtest_repackkernel"})<0 ||
           importers.load_plugin(std::string(argv[1])+"/quantizers/libquant.so",0)<0)
            throw std::runtime_error("missing metadata-only repack fixture or real libquant importer");
        Config all;all.placement="all_vram";all.vram_weights_mib=0;all.host_pool_mib=0;
        Config candidate;candidate.placement="expert_tiered";
        if(!budgets.empty()) {
            candidate.vram_weights_mib=std::stoll(budgets[0]);candidate.host_pool_mib=std::stoll(budgets[1]);
            if(candidate.vram_weights_mib<=0||candidate.host_pool_mib<0||
               candidate.vram_weights_mib>1024*1024||candidate.host_pool_mib>1024*1024)
                throw std::runtime_error("budgets must be explicit bounded nonnegative MiB");
        }
        wp::Json report{{"schema","radiance-weight-allocation-probe-v1"},{"source_revision",inv.revision},
            {"method","real Registry importer + Builder layout hooks + Planner; metadata allocations only"},
            {"production_config_changed",false},{"gpu_execution",false},{"model_payload_read",false},
            {"synthetic_device_capability",true},{"tp_world",4},{"ranks",wp::Json::array()},
            {"dense_policy",wp::dense_policy_name(dense_policy)},
            {"residency_policy",coordinate?"coordinate_common_initial_mask":require_common?"require_common_initial_mask":"independent_local_pointers"},
            {"matching_masks_are_universal_correctness_requirement",false},
            {"assumptions",{
                "Native packed device consumers are assumed; metadata fixture launch refuses inference.",
                "Baseline expert wide rank is layer modulo 4; 768/512/512/512 at nff=2304, matching the existing TP4 helper.",
                "Each expert has separate gate/up/down packed members in one movement unit.",
                dense_policy==wp::DensePolicy::replicated?
                    "Every nonexpert tensor is replicated; future dense TP or shared-KV aliasing is not assumed.":
                    "OPTIONAL reviewed projection row sharding only; exact full-output/latent gathers or proven head/group-local consumption are graph obligations, not implemented serving support. Unknown/control/hash/norm/HC tensors stay replicated.",
                "Native IQ constants are implicit in future kernel code; kernel/module memory is not counted.",
                "Optional placement budgets are independent per-rank weight budgets, not a full hardware configuration."}},
            {"excluded",{"Engram sidecar and bounded row cache","KV and prefix caches","attention/indexer/state buffers",
                "activation arenas and scratch","collectives","kernel/driver allocations","load-time temporary memory"}},
            {"startup_caveat","Configured host pools are fully backed. Loader input target is 4 GiB plus permitted overshoot, sweep windows, and worker bounce/output temporaries; not a 1 GiB peak allowance."},
            {"does_not_establish_262k_fit",true}};
        bool candidate_fits=true;
        std::vector<std::unique_ptr<Builder>> builders;
        for(int rank=0;rank<4;++rank) {
            auto builder=wp::build(kernels,importers,inv,rank,false,dense_policy);
            wp::Json r{{"rank",rank},{"footprint",wp::footprint(builder->program())},
                       {"dense_partition",wp::dense_partition_report(builder->program(),inv,dense_policy)},
                       {"uncapped_weight_only_plan",wp::report(builder->program(),all)}};
            if(!budgets.empty()) {
                r["candidate_weight_only_plan"]=wp::report(builder->program(),candidate);
                candidate_fits=candidate_fits&&r["candidate_weight_only_plan"]["fits_weight_only_assumptions"].get<bool>();
            }
            report["ranks"].push_back(std::move(r));
            if(coordinate)builders.push_back(std::move(builder));
        }
        report["uncapped_tp_residency_check"]=wp::compare_expert_residency(report["ranks"],"uncapped_weight_only_plan");
        if(!budgets.empty()) {
            report["candidate_tp_residency_check"]=wp::compare_expert_residency(report["ranks"],"candidate_weight_only_plan");
            if(require_common&&!coordinate)candidate_fits=candidate_fits&&report["candidate_tp_residency_check"]["common_resident_policy_satisfied"].get<bool>();
        }
        if(coordinate) {
            std::vector<PlannerInput> inputs;
            for(int rank=0;rank<4;++rank)inputs.push_back({&builders[size_t(rank)]->program(),&candidate,rank,4,nullptr,0,true,false});
            std::vector<Plan> plans;TpInitialPlanDiagnostics d;
            const int status=coordinate_initial_expert_placement(inputs,&plans,&d);
            wp::Json history=wp::Json::array();
            for(const auto& round:d.rounds)history.push_back({{"resident_counts",round.resident_counts},{"intersection_count",round.intersection_count}});
            report["coordinated_initial_placement"]={{"status",status},{"why",d.why},{"rank_status",d.rank_status},
                {"planner_rounds_attempted",d.planner_rounds_attempted},{"completed_valid_rounds",history},
                {"common_resident_count",d.common_resident.size()},{"initial_placement_only",d.initial_placement_only},
                {"heat_safe",d.heat_safe},{"engine_wired",false},{"weight_padding_added",false},
                {"selection","conservative intersection; actual per-rank layout and Planner costs; no optimality/performance promise"}};
            candidate_fits=status==RAD_OK;
            if(status==RAD_OK) {
                for(int rank=0;rank<4;++rank)report["ranks"][rank]["coordinated_candidate_plan"]=
                    wp::report_plan(builders[size_t(rank)]->program(),candidate,plans[size_t(rank)],RAD_OK);
                report["coordinated_tp_residency_check"]=wp::compare_expert_residency(report["ranks"],"coordinated_candidate_plan");
                candidate_fits=report["coordinated_tp_residency_check"]["common_resident_policy_satisfied"].get<bool>();
            }
        }
        if(!candidate_fits)std::cerr<<"candidate rejected; see "<<(coordinate?"coordinated_initial_placement":"candidate_tp_residency_check")<<" in report\n";
        std::ofstream output(argv[3]);if(!output)throw std::runtime_error("cannot open report output");
        output<<report.dump(2)<<'\n';if(!output)throw std::runtime_error("cannot write report output");
        std::cerr<<"wrote host-only weight-allocation report; no inference/state capacity claim\n";
        return candidate_fits?0:3; // Refusal still writes the complete diagnostic report.
    }catch(const std::exception&e){std::cerr<<"gguf-weight-probe: "<<e.what()<<'\n';return 1;}
}
