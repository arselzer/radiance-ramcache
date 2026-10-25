#include "source_factory.h"
#include "contract/fixture_writer.h"
#include "contract/graph_quote.h"
#include "storage/physical_source_bridge.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <mutex>
#include <new>
#include <sys/stat.h>
#include <unistd.h>

namespace glm53::engine_source {
namespace cold=glm53::engine_core_contract;
namespace bridge=glm53::engine_core_physical;
namespace native=rad::engine_core_expert_store;
namespace shard=glm53::engine_core_sharded;
namespace fs=std::filesystem;
namespace {
constexpr uint64_t seed=14695981039346656037ull;
struct Failure {int status;const char* why;};
void need(bool b,int status,const char* why){if(!b)throw Failure{status,why};}
void explain(std::string* out,const char* s)noexcept{if(out)try{*out=s;}catch(...){}}
void success(int rc,const char* why){need(rc==RAD_OK,rc,why);}
int convert(native::Status s)noexcept{
    switch(s){case native::Status::ok:return RAD_OK;case native::Status::no_memory:return RAD_E_NOMEM;
    case native::Status::budget:return RAD_E_SCRATCH;case native::Status::io:case native::Status::checksum:return RAD_E_IO;
    case native::Status::unsupported:return RAD_E_UNSUPPORTED;default:return RAD_E_INVAL;}
}
int convert(bridge::Status s)noexcept{
    switch(s){case bridge::Status::ok:return RAD_OK;case bridge::Status::no_memory:return RAD_E_NOMEM;
    case bridge::Status::io:case bridge::Status::checksum:return RAD_E_IO;
    case bridge::Status::unsupported:return RAD_E_UNSUPPORTED;default:return RAD_E_INVAL;}
}
struct Fd {
    int n=-1;explicit Fd(int fd=-1):n(fd){}~Fd(){if(n>=0)::close(n);}
    Fd(const Fd&)=delete;
};
struct Stamp {
    dev_t device;ino_t inode;off_t bytes;timespec mt,ct;
    bool operator==(const Stamp& b)const noexcept{return device==b.device&&inode==b.inode&&bytes==b.bytes&&
        mt.tv_sec==b.mt.tv_sec&&mt.tv_nsec==b.mt.tv_nsec&&ct.tv_sec==b.ct.tv_sec&&ct.tv_nsec==b.ct.tv_nsec;}
};
Stamp stamp(int fd){struct stat s{};need(!::fstat(fd,&s)&&S_ISREG(s.st_mode)&&s.st_size>=0,RAD_E_IO,"source descriptor stamp");
    return {s.st_dev,s.st_ino,s.st_size,s.st_mtim,s.st_ctim};}
void read_exact(int fd,void* dst,size_t n,uint64_t offset){
    need(offset<=uint64_t(INT64_MAX)&&n<=uint64_t(INT64_MAX)-offset,RAD_E_INVAL,"bounded physical range");
    auto* out=static_cast<std::byte*>(dst);
    for(size_t at=0;at<n;){const auto got=::pread(fd,out+at,n-at,off_t(offset+at));
        if(got<0&&errno==EINTR)continue;
        need(got>0,RAD_E_IO,"exact source range read");at+=size_t(got);}
}
uint64_t hash(uint64_t h,const std::byte* p,size_t n)noexcept{
    for(size_t i=0;i<n;++i){h^=std::to_integer<uint8_t>(p[i]);h*=1099511628211ull;}return h;
}
// All private output names are closed. Cleanup allocates no strings, opens no
// directory enumeration, and never traverses a caller-supplied subtree.
struct WorkGeneration {
    std::string path;Fd root;
    explicit WorkGeneration(const std::string& parent){
        need(parent.size()<=320&&parent.find('\0')==std::string::npos&&fs::path(parent).is_absolute(),RAD_E_INVAL,"bounded absolute work parent");
        struct stat s{};need(!::lstat(parent.c_str(),&s)&&S_ISDIR(s.st_mode),RAD_E_INVAL,"ordinary work parent");
        auto canonical=fs::canonical(parent).string();need(canonical==parent,RAD_E_INVAL,"canonical work parent required");
        auto text=parent+"/tiny-engine-source-XXXXXX";std::array<char,384> buffer{};
        need(text.size()+1<=buffer.size(),RAD_E_INVAL,"private work path cap");std::memcpy(buffer.data(),text.c_str(),text.size()+1);
        path.reserve(text.size());char* made=::mkdtemp(buffer.data());need(made,RAD_E_IO,"private source generation");
        try{path=made;root.n=::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);need(root.n>=0,RAD_E_IO,"private generation fd");}
        catch(...){::rmdir(made);throw;}
    }
    ~WorkGeneration()noexcept{
        if(root.n<0)return;
        for(int directory=0;directory<4;++directory){
            char leaf[40];if(directory==0)std::snprintf(leaf,sizeof leaf,"canonical");
            else std::snprintf(leaf,sizeof leaf,"layer_shard_%d",directory-1);
            const int fd=::openat(root.n,leaf,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
            if(fd>=0){for(size_t i=0;i<(directory?32u:128u);++i){char file[48];
                    if(directory)std::snprintf(file,sizeof file,"segment-%08zu.bin",i);
                    else std::snprintf(file,sizeof file,"plane-%04zu.bin",i);
                    (void)::unlinkat(fd,file,0);}
                (void)::unlinkat(fd,"manifest.json",0);::close(fd);}
            (void)::unlinkat(root.n,leaf,AT_REMOVEDIR);
        }
        (void)::rmdir(path.c_str());
    }
};

struct HeaderRow {cold::TensorSpec spec;uint64_t offset=0;const rad::SrcTensor* raw=nullptr;};
std::vector<HeaderRow> physical_rows(const cold::SourceCheckpoint& owner){
    success(owner.audit(),"same Checkpoint identity before range derivation");
    const auto path=owner.root()+"/model.safetensors";Fd fd(::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));
    need(fd.n>=0,RAD_E_IO,"physical header descriptor");const auto before=stamp(fd.n);
    uint64_t header_bytes=0;read_exact(fd.n,&header_bytes,8,0);
    need(std::endian::native==std::endian::little&&header_bytes>0&&header_bytes<=65536&&
         before.bytes==int64_t(8+header_bytes+1020544),RAD_E_FORMAT,"physical header extent");
    std::string text(size_t(header_bytes),' ');read_exact(fd.n,text.data(),text.size(),8);
    const auto json=nlohmann::ordered_json::parse(text);const auto specs=cold::tensor_specs();
    need(specs.size()==219&&json.is_object()&&json.size()==220,RAD_E_FORMAT,"physical header count");
    std::vector<HeaderRow> out;out.reserve(219);uint64_t at=0;
    std::array<std::byte,4096> buffer{};
    for(const auto& spec:specs){
        need(spec.name.size()<=512&&spec.bytes<=131072&&json.contains(spec.name),RAD_E_FORMAT,"named physical range");
        const auto& a=json.at(spec.name);need(a.is_object()&&a.at("dtype")==spec.dtype&&a.at("shape")==spec.shape&&
             a.at("data_offsets")==nlohmann::ordered_json::array({at,at+spec.bytes}),RAD_E_FORMAT,"actual named physical offset/type/shape");
        const auto* raw=owner.checkpoint().find(spec.name);const uint32_t dtype=spec.dtype=="U8"?RAD_U8:spec.dtype=="F32"?RAD_F32:RAD_BF16;
        need(raw&&raw->data&&raw->name==spec.name&&raw->rad_dtype==dtype&&raw->shape==spec.shape&&raw->bytes==int64_t(spec.bytes),RAD_E_SHAPE,"same opened raw tensor identity");
        const uint64_t offset=8+header_bytes+at;const auto* mapped=static_cast<const std::byte*>(raw->data);
        for(uint64_t done=0;done<spec.bytes;){const auto n=size_t(std::min<uint64_t>(buffer.size(),spec.bytes-done));read_exact(fd.n,buffer.data(),n,offset+done);
            need(!std::memcmp(mapped+done,buffer.data(),n),RAD_E_STATE,"current physical range differs from same opened mapping");
            for(size_t i=0;i<n;++i)need(std::to_integer<uint8_t>(buffer[i])==cold::expected_byte(spec,done+i),RAD_E_FORMAT,"closed synthetic physical recipe");
            done+=n;}
        out.push_back({spec,offset,raw});at+=spec.bytes;
    }
    need(at==1020544&&stamp(fd.n)==before,RAD_E_STATE,"physical source changed during range binding");
    success(owner.audit(),"same Checkpoint identity after range derivation");return out;
}
const HeaderRow& row_named(const std::vector<HeaderRow>& rows,const std::string& name){
    const auto found=std::find_if(rows.begin(),rows.end(),[&](const auto& r){return r.spec.name==name;});
    need(found!=rows.end(),RAD_E_SHAPE,"physical range not found");return *found;
}
bridge::Catalog make_catalog(const cold::SourceCheckpoint& owner,const cold::BoundSources& sources,
                             const rad::Program& program,const rad::Plan& plan){
    const auto rows=physical_rows(owner);bridge::Catalog catalog;
    catalog.source_id="same-checkpoint-synthetic-tiny-tp1";catalog.revision="closed-219-quark-ranges-v1";catalog.tensors.reserve(64);
    const auto path=owner.root()+"/model.safetensors";Fd fd(::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW));need(fd.n>=0,RAD_E_IO,"range checksum fd");
    const auto before=stamp(fd.n);std::array<std::byte,4096> buffer{};size_t count=0;
    for(const auto& unit:plan.units){if(unit.expert<0)continue;
        need(unit.weights.size()==4&&unit.layer>=0&&unit.layer<8&&unit.expert<4,RAD_E_SHAPE,"actual four-member source unit");
        for(size_t projection=0;projection<2;++projection){
            const auto handle=unit.weights[2*projection];need(handle>0&&handle<program.weights.size(),RAD_E_SHAPE,"actual source handle");const auto& weight=program.weights[handle];
            const auto* logical=sources.find(weight.source);need(logical&&logical->trivial&&!logical->imported&&logical->rank==2&&
                 logical->pieces.size()==(projection?1u:2u)&&logical->scales.size()==logical->pieces.size(),RAD_E_SHAPE,"same cached logical planes");
            bridge::Tensor tensor;tensor.layer=unit.layer;tensor.expert=unit.expert;auto& expected=tensor.expected;
            expected.logical=weight.source;expected.encoding=logical->enc;expected.shape={logical->shape[0],logical->shape[1]};
            expected.projection=projection?native::Projection::down:native::Projection::gate_up;
            for(size_t plane=0;plane<2;++plane){uint64_t total=0,digest=seed;
                for(size_t part=0;part<logical->pieces.size();++part){const auto* imported=plane?logical->scales[part]:logical->pieces[part];
                    need(imported&&imported->data&&imported->bytes>0,RAD_E_SHAPE,"actual imported plane span");const auto& physical=row_named(rows,imported->name);
                    const auto stem="tiny.layers."+std::to_string(unit.layer)+".experts."+std::to_string(unit.expert)+"."+(projection?"down":part?"up":"gate")+"_proj.weight";
                    const auto name=stem+(plane?"_scale":"");const rad::SrcTensor *code=nullptr,*scale=nullptr;
                    success(owner.checkpoint().import_piece(stem,&code,&scale,nullptr),"same opened import_piece");
                    need(imported==(plane?scale:code)&&imported->name==name&&physical.raw==owner.checkpoint().find(name)&&
                         imported->data==physical.raw->data&&imported->bytes==physical.raw->bytes&&
                         imported->rad_dtype==uint32_t(plane?RAD_E8M0:RAD_FP4E2M1)&&
                         imported->shape==std::vector<int64_t>({128,plane?4:128}),RAD_E_STATE,"ordered actual code/scale object and raw range identity");
                    bridge::Piece piece;piece.part=projection?bridge::Part::down:part?bridge::Part::up:bridge::Part::gate;
                    piece.tensor=expected.logical+"/"+(projection?"down":part?"up":"gate")+"/"+(plane?"scale":"codes");
                    piece.path=path;piece.offset=physical.offset;piece.rows=128;piece.row_bytes=plane?4:64;piece.bytes=physical.spec.bytes;
                    piece.dtype=plane?RAD_E8M0:RAD_FP4E2M1;piece.complete=true;uint64_t piece_digest=seed;
                    for(uint64_t at=0;at<piece.bytes;){const auto n=size_t(std::min<uint64_t>(buffer.size(),piece.bytes-at));read_exact(fd.n,buffer.data(),n,piece.offset+at);
                        need(!std::memcmp(buffer.data(),static_cast<const std::byte*>(imported->data)+at,n),RAD_E_STATE,"checksum source differs from mapped imported plane");
                        digest=hash(digest,buffer.data(),n);piece_digest=hash(piece_digest,buffer.data(),n);at+=n;}
                    piece.checksum=piece_digest;total+=piece.bytes;tensor.planes[plane].push_back(std::move(piece));++count;
                }
                expected.planes[plane].bytes=total;expected.planes[plane].checksum=digest;
            }
            catalog.tensors.push_back(std::move(tensor));
        }
    }
    need(count==192&&catalog.tensors.size()==64&&stamp(fd.n)==before,RAD_E_STATE,"complete immutable physical source coverage");
    success(owner.audit(),"same source final range audit");return catalog;
}
struct RetentionQuota {std::mutex mutex;size_t holders=0;};
struct PreparedState {
    // Tickets first => last destruction, after provider, workdir and binding.
    rad::TinyTicket provider_ticket,cache_ticket,adapter_ticket;
    std::shared_ptr<CheckpointBinding> binding;
    std::unique_ptr<WorkGeneration> work;
    const rad::Program* program=nullptr;const rad::Plan* plan=nullptr;
    std::shared_ptr<glm53::engine_core_p3::ShardedNativeSource> provider;
    std::vector<rad::RecordOwnerBinding> owned;
    std::vector<int32_t> resident;
    std::shared_ptr<RetentionQuota> quota=std::make_shared<RetentionQuota>();
    PreparedReceipt receipt;
    std::mutex operation;bool closed=false,finished=false;
    ~PreparedState(){if(provider)(void)provider->close();}
};
struct RetainedBatch {
    std::shared_ptr<PreparedState> state;
    std::shared_ptr<RetentionQuota> quota;
    rad::NativeRecordBatch inner;
    RetainedBatch(std::shared_ptr<PreparedState> s,std::shared_ptr<RetentionQuota> q):state(std::move(s)),quota(std::move(q)){}
    ~RetainedBatch(){inner={};std::lock_guard lock(quota->mutex);--quota->holders;}
};
thread_local bool in_factory=false;
struct Active {Active(){in_factory=true;}~Active(){in_factory=false;}};
class RetainedSource final:public rad::NativeRecordSource {
public:
    explicit RetainedSource(std::shared_ptr<PreparedState> s):state_(std::move(s)){}
    std::span<const rad::NativeRecordDescriptor> catalog()const noexcept override{return state_->provider->catalog();}
    int acquire(std::span<const uint32_t> ids,rad::NativeRecordBatch* out)override{
        if(in_factory)return RAD_E_STATE;
        Active active;std::lock_guard lock(state_->operation);
        if(!out||state_->closed||!state_->finished||ids.empty()||ids.size()>3)return RAD_E_STATE;
        for(const auto id:ids)if(id>=catalog().size())return RAD_E_INVAL;
        const int identity=state_->binding->audit();if(identity<0)return identity;
        bool charged=false;
        try{
            {std::lock_guard quota_lock(state_->quota->mutex);if(state_->quota->holders>=16)return RAD_E_SCRATCH;++state_->quota->holders;charged=true;}
            auto holder=std::make_shared<RetainedBatch>(state_,state_->quota);charged=false;
            rad::NativeRecordBatch ready;ready.leases.reserve(ids.size());
            const int rc=state_->provider->acquire(ids,&holder->inner);if(rc<0)return rc;
            // ALL fallible outer preparation preceded provider's transactional
            // ready publication. Remaining copies fit reserved capacity.
            for(const auto& item:holder->inner.leases)ready.leases.push_back({item.record,item.bytes,holder});
            out->leases.swap(ready.leases);return RAD_OK;
        }catch(const std::bad_alloc&){if(charged){std::lock_guard quota_lock(state_->quota->mutex);--state_->quota->holders;}return RAD_E_NOMEM;}
        catch(...){if(charged){std::lock_guard quota_lock(state_->quota->mutex);--state_->quota->holders;}return RAD_E_STATE;}
    }
private:std::shared_ptr<PreparedState> state_;
};
} // namespace

struct CheckpointBinding::Impl {
    rad::TinyTicket source_ticket,factory_ticket;
    rad::TinyBudget* budget=nullptr;
    std::shared_ptr<const cold::SourceCheckpoint> owner;
    std::unique_ptr<const cold::ColdSourceSeal> seal;
    std::unique_ptr<cold::BoundSources> sources;
    BindingReceipt receipt;
};
CheckpointBinding::CheckpointBinding(std::unique_ptr<Impl> p):impl_(std::move(p)){}
CheckpointBinding::~CheckpointBinding()=default;
int CheckpointBinding::open(const std::string& path,rad::TinyBudget& budget,std::shared_ptr<CheckpointBinding>* out,std::string* why){
    try{
        need(out&&!*out&&budget.admitted(),RAD_E_INVAL,"empty binding output and admitted budget required");
        // Stack tickets charge BEFORE all owner/cache/control allocation.
        rad::TinyTicket source,factory;success(budget.take(rad::TinyCategory::Source,12u<<20,&source),"source precharge");
        success(budget.take(rad::TinyCategory::Factory,8u<<20,&factory),"factory precharge");
        auto p=std::make_unique<Impl>();p->source_ticket=std::move(source);p->factory_ticket=std::move(factory);p->budget=&budget;
        success(cold::open_source_checkpoint(path,&p->owner,why),"bounded actual Checkpoint open");
        std::unique_ptr<const cold::InspectedSource> inspected;
        success(cold::inspect_checkpoint_source(p->owner,path,&inspected,why),"actual source inspection");
        success(cold::seal_source(*inspected,&p->seal,why),"private source seal");
        p->sources=std::make_unique<cold::BoundSources>(*p->seal);const auto& q=p->seal->source();
        p->receipt={q.fingerprint,q.file_bytes,q.payload_bytes,q.physical_tensors,q.artifact_files};
        auto ready=std::shared_ptr<CheckpointBinding>(new CheckpointBinding(std::move(p)));
        success(ready->audit(why),"binding final source audit");*out=std::move(ready);return RAD_OK;
    }catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
rad::Checkpoint& CheckpointBinding::checkpoint()const noexcept{return impl_->owner->checkpoint();}
rad::WeightSourceFn CheckpointBinding::source_fn(){auto owner=shared_from_this();auto fn=impl_->sources->fn();
    return [owner=std::move(owner),fn=std::move(fn)](const std::string& n,const rad::Program& p,rad::WeightSource* w){return fn(n,p,w);};}
const rad::CkptWeight* CheckpointBinding::find(const std::string& n)const{return impl_->sources->find(n);}
std::string CheckpointBinding::missing(const std::string& n)const{return impl_->sources->missing(n);}
std::vector<std::string> CheckpointBinding::refused()const{return impl_->sources->refused();}
int CheckpointBinding::audit(std::string* why)const{return impl_->owner->audit(why);}
bool CheckpointBinding::permits_zero_head(const RadModelMeta& m,const RadBuildCtx& c)const noexcept{return impl_->seal->permits_zero_head_metadata(m,c);}
const BindingReceipt& CheckpointBinding::receipt()const noexcept{return impl_->receipt;}

struct PreparedRecords::Impl {std::shared_ptr<PreparedState> state;std::shared_ptr<RetainedSource> source;};
PreparedRecords::PreparedRecords(std::unique_ptr<Impl> p):impl_(std::move(p)){}
PreparedRecords::~PreparedRecords(){(void)close();}
int PreparedRecords::open(std::shared_ptr<CheckpointBinding> binding,const rad::Program& program,const rad::Plan& plan,
                          const std::string& parent,rad::TinyBudget& budget,std::unique_ptr<PreparedRecords>* out,std::string* why){
    if(in_factory)return RAD_E_STATE;
    Active active;
    try{
        need(out&&!*out&&binding&&binding->impl_->budget==&budget&&budget.admitted(),RAD_E_INVAL,"same admitted budget, binding and empty prepared output");
        rad::TinyTicket provider,cache,adapter;
        success(budget.take(rad::TinyCategory::Provider,24u<<20,&provider),"provider precharge");
        success(budget.take(rad::TinyCategory::Cache,98304+65536,&cache),"whole cache reservation");
        success(budget.take(rad::TinyCategory::Adapter,65536,&adapter),"adapter precharge");
        auto state=std::make_shared<PreparedState>();state->provider_ticket=std::move(provider);state->cache_ticket=std::move(cache);state->adapter_ticket=std::move(adapter);
        state->binding=std::move(binding);state->program=&program;state->plan=&plan;
        auto& original=*state->binding->impl_;cold::GraphQuote quote;
        success(cold::quote_declared_graph(*original.seal,program,plan,*original.sources,&quote,why),"same opened Checkpoint graph/source quote");
        auto physical=make_catalog(*original.owner,*original.sources,program,plan);
        state->work=std::make_unique<WorkGeneration>(parent);bridge::Result canonical;
        success(convert(bridge::materialize(program,plan,physical,state->work->path+"/canonical",&canonical,why)),"physical range bundle");
        need(canonical.source_bytes==835584&&canonical.output_bytes==835584&&canonical.physical_pieces==192,RAD_E_SHAPE,"closed materialized source receipt");
        std::array<std::vector<native::RecordSpec>,3> specs;for(auto& group:specs)group.reserve(12);
        size_t captured=0;for(size_t u=0;u<plan.units.size();++u){if(plan.units[u].expert<0)continue;
            native::RecordSpec spec;success(convert(native::capture_unit(program,plan,u,canonical.sources,&spec,why)),"actual Program/Plan record capture");
            need(spec.key().layer>=0&&spec.key().layer<8,RAD_E_SHAPE,"captured layer range");specs[size_t(spec.key().layer%3)].push_back(std::move(spec));++captured;}
        need(captured==32,RAD_E_SHAPE,"complete actual expert record capture");
        std::array<shard::ShardInput,3> inputs;uint64_t total=0;
        for(size_t i=0;i<inputs.size();++i){auto& input=inputs[i];input.name="layer_shard_"+std::to_string(i);input.specs=specs[i];input.sources=&canonical.sources;
            input.limits.max_segment_bytes=65536;native::Receipt receipt;
            success(convert(native::pack(input.specs,canonical.sources,state->work->path+"/"+input.name,input.limits,&receipt,why)),"bounded native shard publication");
            input.manifest_checksum=receipt.manifest_checksum;input.limits.max_segments=receipt.segments;total+=receipt.total_bytes;}
        success(glm53::engine_core_p3::ShardedNativeSource::open(program,plan,state->work->path,inputs,98304,65536,&state->provider,why),"verified P3 source open");
        need(state->provider->catalog().size()==24&&state->provider->bootstrap_count()==8&&total==1048576,RAD_E_SHAPE,"owned24/bootstrap8/32strides");
        state->owned.reserve(24);state->resident.reserve(8);
        for(size_t u=0;u<plan.units.size();++u){const auto& unit=plan.units[u];if(unit.expert<0)continue;
            if(unit.expert==0)state->resident.push_back(int32_t(u));
            else{const auto all=state->provider->catalog();const auto found=std::find_if(all.begin(),all.end(),[&](const auto& d){return d.layer==unit.layer&&d.expert==unit.expert;});
                need(found!=all.end(),RAD_E_STATE,"actual owned unit to typed descriptor identity");state->owned.push_back({int32_t(u),uint32_t(found-all.begin())});}}
        need(state->owned.size()==24&&state->resident.size()==8,RAD_E_STATE,"complete owned/resident metadata derivation");
        state->receipt={canonical.physical_fingerprint,canonical.source_bytes,total,32,24,8,3};
        // The normalized P3 generation owns descriptors/verification. Temporary
        // specs/catalogs die here; no full source payload vector was created.
        success(state->binding->audit(why),"factory final same source identity");
        success(glm53::engine_core_p3::status(state->provider->store_for_test().audit_all()),"factory final native generation audit");
        auto p=std::make_unique<Impl>();p->state=std::move(state);p->source=std::make_shared<RetainedSource>(p->state);
        auto ready=std::unique_ptr<PreparedRecords>(new PreparedRecords(std::move(p)));*out=std::move(ready);return RAD_OK;
    }catch(const Failure& e){explain(why,e.why);return e.status;}catch(const std::bad_alloc&){return RAD_E_NOMEM;}catch(...){return RAD_E_STATE;}
}
std::shared_ptr<rad::NativeRecordSource> PreparedRecords::source()const noexcept{return impl_->source;}
std::span<const rad::NativeRecordDescriptor> PreparedRecords::catalog()const noexcept{return impl_->source->catalog();}
std::span<const rad::RecordOwnerBinding> PreparedRecords::owned()const noexcept{return impl_->state->owned;}
std::span<const int32_t> PreparedRecords::resident_units()const noexcept{return impl_->state->resident;}
int PreparedRecords::bootstrap(int32_t unit,std::span<std::byte> target){
    if(in_factory)return RAD_E_STATE;
    Active active;auto& state=*impl_->state;std::lock_guard lock(state.operation);
    if(state.closed||state.finished)return RAD_E_STATE;
    const int rc=state.binding->audit();if(rc<0)return rc;
    return state.provider->bootstrap(unit,target);
}
int PreparedRecords::finish_bootstrap(){
    if(in_factory)return RAD_E_STATE;
    Active active;auto& state=*impl_->state;std::lock_guard lock(state.operation);
    if(state.closed||state.finished)return RAD_E_STATE;
    const int rc=state.binding->audit();if(rc<0)return rc;
    const int finish=state.provider->finish_bootstrap();
    if(finish==RAD_OK)state.finished=true;
    return finish;
}
int PreparedRecords::audit(std::string* why)const{
    if(in_factory)return RAD_E_STATE;
    Active active;auto& state=*impl_->state;std::lock_guard lock(state.operation);
    if(state.closed)return RAD_E_STATE;
    auto& original=*state.binding->impl_;cold::GraphQuote quote;
    const int graph=cold::quote_declared_graph(*original.seal,*state.program,*state.plan,*original.sources,&quote,why);if(graph<0)return graph;
    return glm53::engine_core_p3::status(state.provider->store_for_test().audit_all());
}
int PreparedRecords::close(){
    if(in_factory)return RAD_E_STATE;
    Active active;auto& state=*impl_->state;std::lock_guard lock(state.operation);
    if(state.closed)return RAD_OK;
    state.closed=true;return state.provider?state.provider->close():RAD_OK;
}
const PreparedReceipt& PreparedRecords::receipt()const noexcept{return impl_->state->receipt;}
} // namespace glm53::engine_source
