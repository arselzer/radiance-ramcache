// Isolated research-only module: four unchanged libr4d operations + its actual layout/shape hooks.
// This is not the full libr4d module and is never installed/scanned for serving.
#include "r4d_plugin.h"
const void** r4d_dispatch_log=nullptr;
int r4d_dispatch_cap=0,r4d_dispatch_n=0;
namespace {
#define I(k) {k,RAD_P_INT,RAD_REQUIRED}
#define O(k) {k,RAD_P_INT,RAD_OPTIONAL}
#define S(k) {k,RAD_P_STR,RAD_REQUIRED}
const RadParamSpec params[]{I("M"),I("N"),I("K"),I("n_expert"),I("top_k"),I("group"),S("a_order"),S("dtype"),O("N_odd"),O("K_odd"),O("parts")};
#undef I
#undef O
#undef S
const RadOperandSpec operands[]{{"a",RAD_OPD_IN,0},{"a_scale",RAD_OPD_IN,1},{"w",RAD_OPD_WTAB,0},{"w_scale",RAD_OPD_WTAB,0},{"sorted_tok",RAD_OPD_IN,0},{"expert_offset",RAD_OPD_IN,0},{"w_odd",RAD_OPD_WTAB,0},{"w_odd_scale",RAD_OPD_WTAB,0},{"y",RAD_OPD_OUT,0}};
const RadOpSchema schema{"moe_gemm_q",params,11,operands,9,"Research-only registration of the unchanged canonical libr4d MXFP4 reader."};
const RadConstraint constraints[]{{"M",RAD_C_GE,1,nullptr},{"M",RAD_C_LE,32767,nullptr},{"N",RAD_C_GE,1,nullptr},{"K",RAD_C_DIV,128,nullptr},{"n_expert",RAD_C_GE,2,nullptr},{"n_expert",RAD_C_LE,512,nullptr},{"top_k",RAD_C_GE,1,nullptr},{"top_k",RAD_C_LE,32,nullptr},{"group",RAD_C_EQ,128,nullptr},{"dtype",RAD_C_IN,0,"mxfp4a8"}};
const RadKernelInfo row=[](){RadKernelInfo r{};r.name="moe_gemm_mxfp4a8";r.op="moe_gemm_q";r.family="moe";r.computes="unchanged libr4d canonical MXFP4 reader; isolated synthetic cache fixture";r.shape="K%128; even/odd canonical pointer tables";r.dtypes="E2M1/E8M0 weights,E4M3 activation,F32 scale,BF16 output";r.domain=RAD_DOMAIN_DEVICE;r.priority=R4D_PRIO_ONLY;r.constraints=constraints;r.n_constraints=10;r.launch=r4d_rad_moe_mxfp4;r.layout=r4d_rad_layout_moe_mxfp4;r.opd_shape=r4d_shape_moe_mxfp4;return r;}();
const RadParamSpec scatter_params[]{{"M",RAD_P_INT,RAD_REQUIRED},{"n_expert",RAD_P_INT,RAD_REQUIRED},{"top_k",RAD_P_INT,RAD_REQUIRED},{"expert_base",RAD_P_INT,RAD_OPTIONAL},{"protect",RAD_P_STR,RAD_OPTIONAL}};
const RadOperandSpec scatter_operands[]{{"expert_ids",RAD_OPD_IN,0},{"sorted_tok",RAD_OPD_OUT,0},{"expert_offset",RAD_OPD_OUT,0},{"expert_count",RAD_OPD_OUT,0}};
const RadOpSchema scatter_schema{"moe_scatter",scatter_params,5,scatter_operands,4,"Research-only registration of unchanged libr4d DEVICE route scatter."};
const RadConstraint scatter_constraints[]{{"M",RAD_C_GE,1,nullptr},{"n_expert",RAD_C_GE,1,nullptr},{"n_expert",RAD_C_LE,512,nullptr},{"top_k",RAD_C_GE,1,nullptr}};
const RadKernelInfo scatter_row=[](){RadKernelInfo r{};r.name="moe_scatter_i32";r.op="moe_scatter";r.family="moe";r.computes="unchanged libr4d DEVICE stable route scatter";r.shape="tiny one-launch profile";r.dtypes="I32";r.domain=RAD_DOMAIN_DEVICE;r.priority=R4D_PRIO_ONLY;r.constraints=scatter_constraints;r.n_constraints=4;r.launch=r4d_rad_moe_scatter;r.scratch=r4d_rad_moe_scatter_scratch;r.opd_shape=r4d_shape_moe_scatter;return r;}();
const RadParamSpec gated_params[]{{"M",RAD_P_INT,RAD_REQUIRED},{"n",RAD_P_INT,RAD_REQUIRED},{"limit",RAD_P_F64,RAD_REQUIRED},{"group",RAD_P_INT,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED}};
const RadOperandSpec gated_operands[]{{"gate_up",RAD_OPD_IN,0},{"q",RAD_OPD_OUT,0},{"scale",RAD_OPD_OUT,0},{"out_bf16",RAD_OPD_OUT,1}};
const RadOpSchema gated_schema{"swiglu_clamped_quant_fp8",gated_params,5,gated_operands,4,"Unchanged pinned GLM limit10 BF16 product then dynamic FP8 quantization."};
const RadConstraint gated_constraints[]{{"M",RAD_C_GE,1,nullptr},{"M",RAD_C_LE,65535,nullptr},{"n",RAD_C_DIV,4,nullptr},{"group",RAD_C_EQ,128,nullptr},{"dtype",RAD_C_IN,0,"bf16"}};
const RadKernelInfo gated_row=[](){RadKernelInfo r{};r.name="swiglu_clamped_quant_fp8";r.op="swiglu_clamped_quant_fp8";r.family="quant";r.computes="unchanged libr4d fused clamped SwiGLU and FP8 quantizer";r.shape="limit positive;group128";r.dtypes="BF16 input/product,E4M3 output,F32 scale";r.domain=RAD_DOMAIN_DEVICE;r.priority=R4D_PRIO_ONLY;r.constraints=gated_constraints;r.n_constraints=5;r.launch=r4d_rad_swiglu_clamped_quant_fp8;r.opd_shape=r4d_shape_gated_quant_fp8;return r;}();
const RadParamSpec gather_params[]{{"M",RAD_P_INT,RAD_REQUIRED},{"n",RAD_P_INT,RAD_REQUIRED},{"top_k",RAD_P_INT,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED},{"act",RAD_P_STR,RAD_OPTIONAL}};
const RadOperandSpec gather_operands[]{{"y_expert",RAD_OPD_IN,0},{"expert_w",RAD_OPD_IN,0},{"sorted_tok",RAD_OPD_IN,0},{"shared",RAD_OPD_IN,1},{"shared_gate",RAD_OPD_IN,1},{"y",RAD_OPD_OUT,0}};
const RadOpSchema gather_schema{"moe_gather",gather_params,5,gather_operands,6,"Unchanged native weighted routed-row gather, no shared arm."};
const RadConstraint gather_constraints[]{{"M",RAD_C_GE,1,nullptr},{"n",RAD_C_GE,1,nullptr},{"top_k",RAD_C_GE,1,nullptr},{"dtype",RAD_C_IN,0,"bf16"}};
const RadKernelInfo gather_row=[](){RadKernelInfo r{};r.name="moe_gather_bf16";r.op="moe_gather";r.family="moe";r.computes="unchanged libr4d native weighted route gather";r.shape="F32 route weights,ascending slot FMA";r.dtypes="BF16 expert/output,F32 route weights";r.domain=RAD_DOMAIN_DEVICE;r.priority=R4D_PRIO_ONLY;r.constraints=gather_constraints;r.n_constraints=4;r.launch=r4d_rad_moe_gather;r.scratch=r4d_rad_moe_gather_scratch;r.opd_shape=r4d_shape_moe_gather;return r;}();
const RadParamSpec project_params[]{{"M",RAD_P_INT,RAD_REQUIRED},{"N",RAD_P_INT,RAD_REQUIRED},{"K",RAD_P_INT,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED}};
const RadOperandSpec project_operands[]{{"a",RAD_OPD_IN,0},{"b",RAD_OPD_WEIGHT,0},{"y",RAD_OPD_OUT,0}};
const RadOpSchema project_schema{"gemm_nt_f32",project_params,4,project_operands,3,"Unchanged GLM F32 router projection with BF16 activation."};
const RadConstraint project_constraints[]{{"M",RAD_C_GE,1,nullptr},{"M",RAD_C_LE,65535,nullptr},{"N",RAD_C_GE,1,nullptr},{"N",RAD_C_LE,512,nullptr},{"K",RAD_C_GE,1,nullptr},{"K",RAD_C_LE,16384,nullptr},{"dtype",RAD_C_IN,0,"bf16"}};
const RadKernelInfo project_row=[](){RadKernelInfo r{};r.name="gemm_nt_f32_bf16";r.op="gemm_nt_f32";r.family="gemm";r.computes="unchanged GLM routing projection";r.shape="N<=512,K<=16384";r.dtypes="BF16,F32";r.domain=RAD_DOMAIN_DEVICE;r.priority=R4D_PRIO_ONLY;r.constraints=project_constraints;r.n_constraints=7;r.launch=r4d_rad_gemm_nt_f32;r.opd_shape=r4d_shape_gemm_nt_f32;return r;}();
const RadParamSpec route_params[]{{"M",RAD_P_INT,RAD_REQUIRED},{"n_expert",RAD_P_INT,RAD_REQUIRED},{"top_k",RAD_P_INT,RAD_REQUIRED},{"norm",RAD_P_INT,RAD_REQUIRED},{"route_scale",RAD_P_F64,RAD_REQUIRED},{"dtype",RAD_P_STR,RAD_REQUIRED}};
const RadOperandSpec route_operands[]{{"logits",RAD_OPD_IN,0},{"bias",RAD_OPD_WEIGHT,0},{"ids",RAD_OPD_OUT,0},{"weights",RAD_OPD_OUT,0}};
const RadOpSchema route_schema{"router_sigmoid_topk",route_params,6,route_operands,4,"Unchanged GLM sigmoid topk; correction affects selection only."};
const RadConstraint route_constraints[]{{"M",RAD_C_GE,1,nullptr},{"M",RAD_C_LE,65535,nullptr},{"n_expert",RAD_C_GE,1,nullptr},{"n_expert",RAD_C_LE,512,nullptr},{"top_k",RAD_C_GE,1,nullptr},{"top_k",RAD_C_LE,32,nullptr},{"dtype",RAD_C_IN,0,"f32"}};
const RadKernelInfo route_row=[](){RadKernelInfo r{};r.name="router_sigmoid_topk_f32";r.op="router_sigmoid_topk";r.family="moe";r.computes="unchanged GLM sigmoid correction routing";r.shape="experts<=512,topk<=32";r.dtypes="F32,I32";r.domain=RAD_DOMAIN_DEVICE;r.priority=R4D_PRIO_ONLY;r.constraints=route_constraints;r.n_constraints=7;r.launch=r4d_rad_router_sigmoid_topk;r.opd_shape=r4d_shape_router_sigmoid_topk;return r;}();
const RadPluginInfo info{RAD_PLUGIN_KERNEL,"glm53_native_device_rows","source-only-v1","Isolated synthetic one-card proof; unchanged libr4d scatter,MXFP4,SwiGLUquant andgather; no serving","gfx1201"};
}
extern "C" {
uint32_t rad_plugin_abi_version(){return RAD_ABI_VERSION;}
const RadPluginInfo* rad_plugin_info(){return &info;}
int rad_kernel_count(){return 6;}
const RadKernelInfo* rad_kernel_at(int i){return i==0?&row:i==1?&scatter_row:i==2?&gated_row:i==3?&gather_row:i==4?&project_row:i==5?&route_row:nullptr;}
int rad_kernel_schema_count(){return 6;}
const RadOpSchema* rad_kernel_schema_at(int i){return i==0?&schema:i==1?&scatter_schema:i==2?&gated_schema:i==3?&gather_schema:i==4?&project_schema:i==5?&route_schema:nullptr;}
int rad_kernel_concurrent(int){return 0;}
}
