#include "rad_test.h"
#include "place/ds41_scored_scatter.h"
using namespace rad;
namespace {
int launch(const RadArgs*,RadStream){return RAD_OK;}
int alien(const RadArgs*,RadStream){return RAD_OK;}
const int module1=0,module2=0;
const void*origin(RadLaunchFn f){return f==launch?&module1:f==alien?&module2:nullptr;}
bool identity(RadLaunchFn f,const char*s){return f==launch&&s&&!std::strcmp(s,rad_ds41_scored_scatter::kSymbol);}
struct F {
    Program p;RadKernelInfo info{};KernelRow row{};RadOpSchema schema{};
    RadOperandSpec operands[8]={{"logits",RAD_OPD_IN,0},{"bias",RAD_OPD_WEIGHT,0},
        {"expert_ids",RAD_OPD_OUT,0},{"weights",RAD_OPD_OUT,0},{"sorted_tok",RAD_OPD_OUT,0},
        {"expert_offset",RAD_OPD_OUT,0},{"expert_count",RAD_OPD_OUT,0},{"token2slot",RAD_OPD_OUT,1}};
    explicit F(int rank=0){
        schema.n_operands=8;schema.operands=operands;
        info.name=rad_ds41_scored_scatter::kRow;info.op=rad_ds41_scored_scatter::kOp;
        info.domain=RAD_DOMAIN_DEVICE;info.launch=launch;row.info=&info;row.plugin="libr4d";
        for(int i=0;i<40;++i){OpInfo o;o.op=info.op;o.schema=&schema;o.ranged_key="M";o.range_lo=o.range_hi=1;
            o.base.set_i("n_expert",384);o.base.set_i("n_local",96);o.base.set_i("top_k",6);o.base.set_i("expert_base",rank*96);
            o.base.set_s("score","sqrtsoftplus");o.base.set_s("dtype","f32");o.weights.push_back(i+1);
            Band b;b.hi=1;b.dom[RAD_DOMAIN_DEVICE].row=&row;o.bands.push_back(b);p.ops.push_back(o);}
    }
};
}
TEST(scored_scatter_flag_and_native_scope_are_strict_and_default_off){
    using namespace rad_ds41_scored_scatter;bool on=true;
    for(const char*s:{(const char*)nullptr,"","0"}){CHECK(parse(s,&on));CHECK(!on);}
    CHECK(parse("1",&on));CHECK(on);
    for(const char*s:{"2","01","true","-1","1 "," 1"})CHECK(!parse(s,&on));
    for(int r=0;r<4;++r)CHECK(scope(4,r,2048,384,96,6,r*96,true,true));
    CHECK(!scope(2,0,16,384,192,6,0,true,true));CHECK(!scope(4,0,0,384,96,6,0,true,true));
    CHECK(!scope(4,0,16,384,96,6,96,true,true));CHECK(!scope(4,0,16,384,96,6,0,false,true));
    CHECK(!scope(4,0,16,384,96,6,0,true,false));
}
TEST(scored_scatter_actual_all_rank_export_geometry_and_operand_roles_are_required){
    for(int rank=0;rank<4;++rank){F f(rank);int n=0;std::string why;
        CHECK(ds41_scored_scatter_admit(f.p,rank,&n,&why,origin,identity));CHECK_EQ(n,40);CHECK(why.empty());}
}
TEST(scored_scatter_startup_refuses_partial_unknown_tuned_alias_and_stale_contracts){
    const std::vector<std::function<void(F&)>>bad={
        [](F&f){f.p.ops.pop_back();},[](F&f){f.p.ops.push_back(f.p.ops[0]);},
        [](F&f){f.p.ops[1].weights=f.p.ops[0].weights;},[](F&f){f.p.ops[0].weights.clear();},
        [](F&f){f.p.ops[0].op="ds41_scored_topk_scatter_unknown";},
        [](F&f){f.p.ops[0].range_hi=2;},[](F&f){f.p.ops[0].ranged_key="N";},
        [](F&f){f.p.ops[0].base.set_i("top_k",5);},[](F&f){f.p.ops[0].base.set_i("expert_base",96);},
        [](F&f){f.p.ops[0].base.set_s("score","sigmoid");},[](F&f){f.p.ops[0].base.set_s("dtype","bf16");},
        [](F&f){f.operands[6].name="count";},[](F&f){f.operands[2].role=RAD_OPD_IN;},
        [](F&f){f.info.name="router_sigmoid_topk_scatter";},[](F&f){f.info.op="router_scored_topk";},
        [](F&f){f.info.launch=alien;},[](F&f){f.info.launch=nullptr;},[](F&f){f.row.plugin="override";},
        [](F&f){f.info.n_tunables=1;},[](F&f){f.p.ops[0].bands[0].dom[RAD_DOMAIN_HOST].row=&f.row;},
        [](F&f){f.p.ops[0].bands[0].dom[RAD_DOMAIN_DEVICE].choice="foo";},
        [](F&f){f.p.ops[0].bands.clear();},[](F&f){f.p.ops[0].bands[0].hi=2;}
    };
    for(const auto&mutate:bad){F f;mutate(f);int n=0;std::string why;
        CHECK(!ds41_scored_scatter_admit(f.p,0,&n,&why,origin,identity));CHECK(!why.empty());}
}
RAD_TEST_MAIN()
