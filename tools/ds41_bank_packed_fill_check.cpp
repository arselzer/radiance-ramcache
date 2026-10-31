/* Guarded four-card raw-byte oracle. This invokes the production pair preparer
 * over actual published packed-cache leases and the production synchronous IO.
 * It is not a kernel-numeric or full Engine/model lifecycle oracle. */
#include "place/draft_bank_packed_fill.h"
#include "place/draft_bank_backup.h"
#include "../tools/ds41-host-experts/ds41_host_experts.h"
#include "../libquant/lq_ggml_planes.h"
#include "rad_device.h"
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
using namespace rad;
namespace pf=rad::draft_bank_packed_fill;
namespace d=rad::draft_bank_backup_detail;
namespace {
void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void checked(int rc,const std::string& why){need(rc>=0,why);}
struct Environment {
    bool present=false; std::string old;
    Environment(){const char* s=std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS");present=s;old=s?s:"";
        ::setenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS","1",1);}
    ~Environment(){if(present)::setenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS",old.c_str(),1);
        else ::unsetenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS");}
};
struct Cards {
    DraftBankBackupRanks ranks{};
    std::array<d::ImageRank,kDraftRanks> images{};
    std::array<void*,kDraftRanks> memory{};
    void* seed=nullptr;
    ~Cards(){for(int r=0;r<kDraftRanks;++r){if(ranks[r].device<0)continue;
        if(rad_dev_set(ranks[r].device)<0)continue;
        if(ranks[r].stream)(void)rad_stream_sync(ranks[r].stream);
        if(memory[r])rad_dev_free(memory[r],RAD_MEM_DEVICE);
        if(ranks[r].stream)rad_stream_destroy(ranks[r].stream);}
        if(seed&&rad_dev_set(0)>=0)rad_dev_free(seed,RAD_MEM_HOST_PINNED);}
    void init(){need(rad_dev_count()==4,"requires exactly four visible real cards");
        for(int r=0;r<4;++r){RadDeviceProps props{};checked(rad_dev_props(r,&props),"card properties");
            need(!props.is_host_backend&&!std::strcmp(props.arch,"gfx1201"),"requires four real gfx1201 cards");
            ranks[r].rank=r;ranks[r].device=r;checked(rad_dev_set(r),"select card");
            checked(rad_stream_create(&ranks[r].stream,0),"create stream");
            memory[r]=rad_dev_alloc(kDraftPairBytes+1536,RAD_MEM_DEVICE);need(memory[r],"allocate native pair and masks");
            images[r].scratch_bytes=kDraftPairBytes;images[r].spans[images[r].nspans++]={memory[r],kDraftPairBytes,true};
            for(int i=0;i<3;++i)images[r].spans[images[r].nspans++]={static_cast<char*>(memory[r])+kDraftPairBytes+i*512,512,false};}
        checked(rad_dev_set(0),"return rank0");}
};
struct Source {
    Environment env;
    Program p; Plan plan; NativeMappedFileCache cache; Mover::HostUnit h;
    std::vector<uint8_t> original;
    bool current=true;
    static bool epoch(void* p) noexcept{return static_cast<Source*>(p)->current;}
    explicit Source(int profile):original(size_t(kDraftPairBytes)){
        p.weights.resize(3);plan.units.resize(1);auto& m=plan.units[0];
        m.layer=9;m.expert=17;m.tier=Tier::SSD;m.slab_class=0;m.bytes=m.slot_bytes=kDraftPairBytes;
        m.weights={1,2};m.offset={0,12533760};
        const ds41_mapped::SlabCache::Sizes slots{ds41hx::packed_bytes(4608,5120,512),ds41hx::packed_bytes(5120,2304,320)};
        checked(cache.configure(22u<<20,4,0,0,NativeMappedFileCache::Layout::Packed,slots),"real packed cache configure");
        const auto pack=lq::ggml::pack_of(lq::ggml::GGML_TYPE_MXFP4);need(pack,"independent original relayout");
        for(int k=0;k<2;++k){auto& w=p.weights[size_t(k)+1];auto& decl=w.decl;
            const int n=k?5120:4608,cols=k?2304:5120,blocks=cols/32;
            decl.group.layer=9;decl.group.expert=17;decl.group.slot=k;decl.rank=2;
            decl.shape[0]=n;decl.shape[1]=cols;decl.access=RAD_ACCESS_CONDITIONAL;decl.shard=RAD_SHARD_NONE;
            w.layout_tag="ggml_mxfp4";w.stored_bytes=k?6266880:12533760;w.source_bank_generation=919;
            w.stored_dtype=RAD_U8;w.stored_rank=2;w.stored_shape[0]=n;w.stored_shape[1]=blocks*17;
            std::vector<uint8_t> codes(size_t(n)*size_t(cols/2)),scales(size_t(n)*size_t(blocks));
            for(size_t i=0;i<codes.size();++i)codes[i]=uint8_t(i*131+(i>>8)+profile*53+k*17);
            const int base=profile==0?0:profile==1?252:120;
            for(size_t i=0;i<scales.size();++i)scales[i]=uint8_t(base+i%4);
            // All256 original e8m0 bytes, including0/255, plus DN partial-round
            // tails and first/last payload bytes. These are byte copies, no FP interpretation.
            for(int i=0;i<256;++i)scales[size_t(i)]=uint8_t(i);
            scales.back()=255;scales[scales.size()-size_t(blocks)]=0;
            ds41hx::PackInfo info;need(ds41hx::pack_plan(scales.data(),n,cols,&info),"packed plan");
            NativeMappedFileCache::Write wr;
            need(cache.reserve_packed(size_t(k)+1,k,&wr)==NativeMappedFileCache::Status::Ok,"paired real packed reservation");
            need(ds41hx::pack_planes(codes.data(),scales.data(),n,cols,k?320:512,info,wr.host(),wr.bytes())==ds41hx::kPackOk,"native packed write");
            ds41hx::Mx view;need(ds41hx::packed_view(wr.host(),wr.bytes(),n,cols,&view)&&ds41hx::packed_scales_equal(view,scales.data()),"every original scale recovered before publish");
            need(wr.publish(&h.lease[k])==NativeMappedFileCache::Status::Ok,"publish real immutable lease");
            h.host[k]=h.lease[k].host();h.dev[k]=h.lease[k].device()?h.lease[k].device():h.host[k];
            h.bytes[k]=w.stored_bytes;h.slot[k]=int64_t(h.lease[k].bytes());
            for(int row=0;row<n;++row){const uint8_t* planes[2]={codes.data()+size_t(row)*size_t(cols/2),scales.data()+size_t(row)*size_t(blocks)};
                pack(planes,blocks,original.data()+m.offset[size_t(k)]+size_t(row)*size_t(blocks)*17);}
        }
        h.unit=0;h.n=2;h.packed=true;
    }
    ~Source(){for(auto& lease:h.lease)(void)lease.release();}
    int prepare(void* relay,pf::Workspace* work){return pf::prepare(p,plan,0,h,static_cast<uint8_t*>(relay),size_t(kDraftPairBytes),work,epoch,this);}
    void held(){need(h.lease[0].active()&&h.lease[1].active(),"source leases ended before copy completion");}
};
struct ShortIO final:DraftBankBackupIO {
    DraftBankBackupDeviceIO& io;void* destination=nullptr;bool short_copy=false;
    explicit ShortIO(DraftBankBackupDeviceIO& x):io(x){}
    int64_t allocation_charge(int64_t n)const noexcept override{return io.allocation_charge(n);}
    bool room(int64_t n,int64_t f,int64_t h,std::string* w)override{return io.room(n,f,h,w);}
    void* alloc_pinned(int64_t n)override{return io.alloc_pinned(n);}
    void free_pinned(void* p)noexcept override{io.free_pinned(p);}
    int copy_sync(int r,void* dst,const void* src,int64_t n)override{
        if(short_copy&&r==3&&dst==destination&&n==kDraftPairBytes){short_copy=false;return io.copy_sync(r,dst,src,n-1);}
        return io.copy_sync(r,dst,src,n);}
};
void run(int cap){
    need(cap>=512,"oracle needs explicit backup-mib >=512 including its test-only buffers");
    Cards cards;cards.init();DraftBankBackupDeviceIO real(cards.ranks);std::string why;
    constexpr int64_t headroom=2ll<<30;
    // Covers all concurrent oracle host allocations BEFORE any payload allocation:
    // 4*(pair snapshot+pair relay+1536 masks), one pair initializer, packed slab,
    // independent BLOCK pair and original input planes, rounding and metadata.
    need(real.room(512ll<<20,0,headroom,&why),why);
    cards.seed=real.alloc_pinned(kDraftPairBytes);need(cards.seed,"one pinned initializer");
    std::memset(cards.seed,0x3c,size_t(kDraftPairBytes));
    for(int r=0;r<4;++r)for(int i=0;i<cards.images[r].nspans;++i)
        checked(real.copy_sync(r,cards.images[r].spans[i].base,cards.seed,cards.images[r].spans[i].bytes),"seed complete bank/masks");
    DraftBankBackupBudget budget;budget.backup_mib=cap;budget.system_headroom_bytes=headroom;
    // Charge the entire conservative oracle-only allowance, less the Image's own
    // pinned storage which capture independently charges; it remains below512MiB.
    budget.controller_metadata_bytes=192ll<<20;
    ShortIO io(real);d::Image image;checked(image.capture(cards.images,budget,sizeof(cards)+sizeof(image)+sizeof(io),&io,&why),why);
    uint64_t compared=0;
    for(int profile=0;profile<3;++profile){Source source(profile);image.mark_borrowed();
        for(int r=0;r<4;++r){pf::Workspace work;void* relay=image.scratch(r);source.held();
            checked(source.prepare(relay,&work),"production packed pair preparation");
            need(work.raw_bytes==uint64_t(kDraftPairBytes),"both inverses did not complete");
            checked(real.copy_sync(r,cards.memory[r],relay,kDraftPairBytes),"whole-pair H2D fill");source.held();
            // Complement expected bytes first: a short D2H success must be visible,
            // including zero/ff final bytes. Compare EVERY native raw byte.
            for(size_t j=0;j<size_t(kDraftPairBytes);++j)static_cast<uint8_t*>(relay)[j]=uint8_t(~source.original[j]);
            checked(real.copy_sync(r,relay,cards.memory[r],kDraftPairBytes),"strict full-pair readback");source.held();
            need(!std::memcmp(relay,source.original.data(),size_t(kDraftPairBytes)),"packed fill differs from original raw BLOCK bytes");
            compared+=uint64_t(kDraftPairBytes);
        }
        checked(image.restore_verify_all(&why),why);need(image.ready(),"all-rank restore did not complete");
    }
    // Stale epoch and malformed second header must produce no usable prepared
    // pair; then a partially overwritten cohort must still strictly restore.
    {Source source(2);pf::Workspace work;source.current=false;
        need(source.prepare(image.scratch(0),&work)<0&&work.refusal==pf::Refusal::Epoch,"stale source admitted");source.current=true;
        auto* dn=const_cast<uint8_t*>(source.h.host[1]);const auto saved=dn[0];dn[0]^=1;
        need(source.prepare(image.scratch(0),&work)<0&&work.refusal==pf::Refusal::Inverse,"bad second header admitted");dn[0]=saved;
        image.mark_borrowed();checked(source.prepare(image.scratch(0),&work),"valid retry after refusal");
        checked(real.copy_sync(0,cards.memory[0],image.scratch(0),kDraftPairBytes/2+31),"partial target fill");source.held();
        checked(image.restore_verify_all(&why),why);need(image.ready(),"partial-fill rollback not verified");}
    image.mark_borrowed();io.destination=cards.memory[3];io.short_copy=true;
    checked(rad_dev_set(3),"select final card");checked(rad_memset_async(cards.memory[3],0xa5,kDraftPairBytes,cards.ranks[3].stream),"poison last card");
    checked(rad_stream_sync(cards.ranks[3].stream),"poison drain");
    need(image.restore_verify_all(&why)<0&&image.failed()&&!image.ready(),"short restore escaped strict validation");
    std::printf("PASS: DSBANK_PACKED_FILL_CARD ranks=4 profiles=3 pair_bytes=%lld compared_bytes=%llu leases=through_sync_copy partial_fill_rollback=1 short_restore_fatal=1 masks=1\n",
                (long long)kDraftPairBytes,(unsigned long long)compared);
}
} // namespace
int main(int argc,char** argv){try{bool inside=false;int cap=0;
    for(int i=1;i<argc;++i){if(!std::strcmp(argv[i],"--inside-guarded-window")&&!inside)inside=true;
        else if(!std::strcmp(argv[i],"--backup-mib")&&!cap&&i+1<argc){char* end=nullptr;errno=0;const char* value=argv[++i];
            long n=std::strtol(value,&end,10);need(!errno&&end!=value&&!*end&&n>=512&&n<=1048576,"invalid explicit backup cap");cap=int(n);}
        else throw std::runtime_error("unknown/duplicate/incomplete argument");}
    need(inside&&cap,"requires --inside-guarded-window --backup-mib 512 before device access");run(cap);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL DSBANK_PACKED_FILL_CARD: %s\n",e.what());return 1;}}
