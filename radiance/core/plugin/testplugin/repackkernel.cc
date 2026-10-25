/* Metadata-only native GGUF layout fixture. Built into testkernels/repack, outside the
 * serving kernel scan. It deliberately implements no inference and cannot launch. */
#include "../../../arch/common/rad_gguf_relayout.h"

namespace {
namespace gg = rad::gguf;
constexpr gg::Format formats[]{gg::Format::iq2_xxs,gg::Format::iq2_xs,gg::Format::iq2_s,
                              gg::Format::iq3_xxs,gg::Format::q5_k,gg::Format::q8_0};
int format_index(const RadEncoding* enc) {
    if(!enc||!rad_enc_valid(enc))return -1;
    for(unsigned i=0;i<sizeof(formats)/sizeof(formats[0]);++i) {
        const auto expected=gg::canonical_encoding(formats[i]);
        if(rad_enc_equal(enc,&expected))return int(i);
    }
    return -1;
}
int layout(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
           const RadTensor* planes,int n,RadLayout* out) {
    if(np!=0||operand!=0)return RAD_E_INVAL;
#define FMT_CASE(i,f) case i:return gg::abi_layout<gg::Format::f>(p,np,operand,enc,sel,planes,n,out)
    switch(format_index(enc)) {
        FMT_CASE(0,iq2_xxs);FMT_CASE(1,iq2_xs);FMT_CASE(2,iq2_s);
        FMT_CASE(3,iq3_xxs);FMT_CASE(4,q5_k);FMT_CASE(5,q8_0);
        default:return RAD_E_DTYPE;
    }
#undef FMT_CASE
}
int relayout(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
             const RadTensor* planes,int n,void* dst,int64_t bytes) {
    if(np!=0||operand!=0)return RAD_E_INVAL;
#define FMT_CASE(i,f) case i:return gg::abi_relayout<gg::Format::f>(p,np,operand,enc,sel,planes,n,dst,bytes)
    switch(format_index(enc)) {
        FMT_CASE(0,iq2_xxs);FMT_CASE(1,iq2_xs);FMT_CASE(2,iq2_s);
        FMT_CASE(3,iq3_xxs);FMT_CASE(4,q5_k);FMT_CASE(5,q8_0);
        default:return RAD_E_DTYPE;
    }
#undef FMT_CASE
}
int unrelayout(const RadParam* p,int np,int operand,const RadEncoding* enc,const int* sel,
              const void* src,int64_t bytes,const RadTensor* planes,int n) {
    if(np!=0||operand!=0)return RAD_E_INVAL;
#define FMT_CASE(i,f) case i:return gg::abi_unrelayout<gg::Format::f>(p,np,operand,enc,sel,src,bytes,planes,n)
    switch(format_index(enc)) {
        FMT_CASE(0,iq2_xxs);FMT_CASE(1,iq2_xs);FMT_CASE(2,iq2_s);
        FMT_CASE(3,iq3_xxs);FMT_CASE(4,q5_k);FMT_CASE(5,q8_0);
        default:return RAD_E_DTYPE;
    }
#undef FMT_CASE
}
int launch(const RadArgs*,RadStream) {return RAD_E_STATE;}
const RadOperandSpec operands[]{{"weight",RAD_OPD_WEIGHT,0}};
const RadOpSchema schema{"test_repack_layout",nullptr,0,operands,1,"native GGUF layout metadata fixture"};
const RadKernelInfo kernel{
    .name="test_native_gguf_layout",.op="test_repack_layout",.family="misc",
    .computes="nothing; metadata-only fixture",.shape="any",.dtypes="canonical GGUF planes",
    .domain=RAD_DOMAIN_HOST,.priority=1,.constraints=nullptr,.n_constraints=0,
    .tunables=nullptr,.n_tunables=0,.tune_valid=nullptr,.init=nullptr,.fini=nullptr,
    .launch=launch,.scratch=nullptr,.layout=layout,.relayout=relayout,
    .opd_shape=nullptr,.unrelayout=unrelayout,.replaces=nullptr
};
const RadPluginInfo info{RAD_PLUGIN_KERNEL,"radtest_repackkernel","0.1",
                         "metadata-only native GGUF layout fixture","host"};
}
extern "C" {
uint32_t rad_plugin_abi_version(void) {return RAD_ABI_VERSION;}
const RadPluginInfo* rad_plugin_info(void) {return &info;}
int rad_kernel_schema_count(void) {return 1;}
const RadOpSchema* rad_kernel_schema_at(int i) {return i==0?&schema:nullptr;}
int rad_kernel_count(void) {return 1;}
const RadKernelInfo* rad_kernel_at(int i) {return i==0?&kernel:nullptr;}
}
