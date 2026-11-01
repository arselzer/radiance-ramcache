/* Ownership fixture: HOST backend only. Root must run this exact target against the host build;
 * a skipped card build is not a host gate. No test may open a GPU outside its guarded window. */
#include "rad_test.h"
#include "runtime/ctx.h"
#include "build/rad_build.h"
#include "device/device.h"
#include "../arch/common/rad_ds41_fp4_runtime.h"
namespace f=rad_ds41_fp4_runtime;
namespace {
struct Fixture {
    rad::Program p;rad::Ctx ctx;
    explicit Fixture(bool device=true,bool host=true) {
        p.weights.resize(1);p.buffers.resize(1);p.ops.resize(1);
        for(int k=0;k<2;++k)if(k?host:device){rad::BufferInfo b;
            b.name=k?f::kHost:f::kDevice;b.decl.kind=RAD_BUF_PERSIST;b.decl.dtype=RAD_I64;
            b.decl.rank=1;b.decl.shape[0]=f::kWords;b.decl.domain=k?RAD_DOMAIN_HOST:RAD_DOMAIN_DEVICE;
            p.buffers.push_back(b);
        }
    }
    int start(){const int rc=rad::plan_buffers(p);if(rc<0)return rc;
        rad::CtxDesc d;d.program=&p;d.rank=2;d.world_size=4;d.record_passes=false;return ctx.init(d);}
};
size_t occurrences(const std::string&s,const char* needle){size_t n=0,p=0;while((p=s.find(needle,p))!=std::string::npos){++n;p+=std::strlen(needle);}return n;}
}
TEST(ds41_fp4_audit_owns_fixed_arenas_zeroes_once_and_reads_once_after_drain) {
    if(!rad::device_is_host())SKIP("requires HOST backend; root host gate must have zero skips");
    Fixture x;REQUIRE_EQ(x.start(),RAD_OK);CHECK_EQ(x.p.arena_fixed_bytes,int64_t(512));
    CHECK_EQ(x.p.host_arena_bytes,int64_t(512));
    auto* words=static_cast<uint64_t*>(x.ctx.buf_ptr(1));REQUIRE(words);
    for(int i=0;i<f::kWords;++i)CHECK_EQ(words[i],uint64_t(0));
    const int slot=f::slot(f::NormCapture,128);
    words[slot*f::FieldCount+f::Fast]=128;
    words[slot*f::FieldCount+f::CapturedFast]=128;
    words[slot*f::FieldCount+f::Blocks]=1;
    const auto out=stderr_of([&]{x.ctx.shutdown();});
    CHECK_EQ(occurrences(out,"ds41 fp4 classifier audit complete:"),size_t(1));
    CHECK(out.find("rank=2")!=std::string::npos);CHECK(out.find("fast=128 fallback=0 capture_fast=128")!=std::string::npos);
    CHECK(out.find("FAILED")==std::string::npos);
    const auto twice=stderr_of([&]{x.ctx.shutdown();});CHECK(twice.find("fp4 classifier audit")==std::string::npos);
}
TEST(ds41_fp4_audit_missing_malformed_or_duplicate_ownership_refuses) {
    if(!rad::device_is_host())SKIP("requires HOST backend; root host gate must have zero skips");
    {Fixture x(true,false);CHECK_EQ(x.start(),RAD_E_STATE);}
    {Fixture x(false,true);CHECK_EQ(x.start(),RAD_E_STATE);}
    {Fixture x;x.p.buffers[1].decl.shape[0]--;CHECK_EQ(x.start(),RAD_E_STATE);}
    {Fixture x;x.p.buffers[2].decl.domain=RAD_DOMAIN_DEVICE;CHECK_EQ(x.start(),RAD_E_STATE);}
    {Fixture x;x.p.buffers.push_back(x.p.buffers[1]);CHECK_EQ(x.start(),RAD_E_STATE);}
    {Fixture x(false,false);CHECK_EQ(x.start(),RAD_OK);const auto out=stderr_of([&]{x.ctx.shutdown();});
        CHECK(out.find("fp4 classifier audit")==std::string::npos);}
}
TEST(ds41_fp4_audit_overflow_or_capture_mismatch_never_emits_complete_authority) {
    if(!rad::device_is_host())SKIP("requires HOST backend; root host gate must have zero skips");
    for(bool overflow:{false,true}){Fixture x;REQUIRE_EQ(x.start(),RAD_OK);
        auto* w=static_cast<uint64_t*>(x.ctx.buf_ptr(1));REQUIRE(w);
        w[f::Fast]=1;w[f::Blocks]=1;w[overflow?f::Overflow:f::CapturedFast]=1;
        const auto out=stderr_of([&]{x.ctx.shutdown();});CHECK(out.find("audit FAILED:")!=std::string::npos);
        CHECK(out.find("audit complete:")==std::string::npos);}
}
RAD_TEST_MAIN()
