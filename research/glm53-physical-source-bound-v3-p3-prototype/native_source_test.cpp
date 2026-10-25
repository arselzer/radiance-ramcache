#include "adapter_fixture.h"
#include "foreign_fixture.h"
#include <atomic>
#include <new>
namespace allocation_fault {
thread_local int64_t remaining=-1;
thread_local uint64_t triggers=0;
thread_local size_t peak_request=0;
void fail_if_armed(){if(remaining>=0&&remaining--==0){remaining=-1;++triggers;throw std::bad_alloc();}}
void* allocate(size_t bytes){peak_request=std::max(peak_request,bytes);fail_if_armed();if(void* p=std::malloc(bytes?bytes:1))return p;throw std::bad_alloc();}
void* aligned(size_t bytes,size_t alignment){fail_if_armed();
    peak_request=std::max(peak_request,bytes);
    if(!alignment||(alignment&(alignment-1)))throw std::bad_alloc();
    alignment=std::max(alignment,sizeof(void*));void* result=nullptr;
    if(::posix_memalign(&result,alignment,bytes?bytes:1)!=0)throw std::bad_alloc();
    return result;
}
}
void* operator new(size_t n){return allocation_fault::allocate(n);}
void* operator new[](size_t n){return allocation_fault::allocate(n);}
void* operator new(size_t n,const std::nothrow_t&)noexcept{try{return allocation_fault::allocate(n);}catch(...){return nullptr;}}
void* operator new[](size_t n,const std::nothrow_t&)noexcept{try{return allocation_fault::allocate(n);}catch(...){return nullptr;}}
void* operator new(size_t n,std::align_val_t alignment){return allocation_fault::aligned(n,size_t(alignment));}
void* operator new[](size_t n,std::align_val_t alignment){return allocation_fault::aligned(n,size_t(alignment));}
void* operator new(size_t n,std::align_val_t alignment,const std::nothrow_t&)noexcept{try{return allocation_fault::aligned(n,size_t(alignment));}catch(...){return nullptr;}}
void* operator new[](size_t n,std::align_val_t alignment,const std::nothrow_t&)noexcept{try{return allocation_fault::aligned(n,size_t(alignment));}catch(...){return nullptr;}}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,size_t)noexcept{std::free(p);}
void operator delete[](void* p,size_t)noexcept{std::free(p);}
void operator delete(void* p,const std::nothrow_t&)noexcept{std::free(p);}
void operator delete[](void* p,const std::nothrow_t&)noexcept{std::free(p);}
void operator delete(void* p,std::align_val_t)noexcept{std::free(p);}
void operator delete[](void* p,std::align_val_t)noexcept{std::free(p);}
void operator delete(void* p,size_t,std::align_val_t)noexcept{std::free(p);}
void operator delete[](void* p,size_t,std::align_val_t)noexcept{std::free(p);}
void operator delete(void* p,std::align_val_t,const std::nothrow_t&)noexcept{std::free(p);}
void operator delete[](void* p,std::align_val_t,const std::nothrow_t&)noexcept{std::free(p);}

namespace io_fault {
std::atomic<int> remaining{-1};int after_count=0;void(*after)()=nullptr;
std::atomic<uint64_t> calls{0},triggers{0};
}
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __wrap_pread(int fd,void* data,size_t bytes,off_t offset){
    ++io_fault::calls;const int n=io_fault::remaining.load();if(n>=0&&io_fault::remaining.fetch_sub(1)==0){io_fault::remaining=-1;++io_fault::triggers;errno=EIO;return -1;}
    const auto count=__real_pread(fd,data,bytes,offset);
    if(count>0&&io_fault::after&&io_fault::after_count--==0){const auto f=io_fault::after;io_fault::after=nullptr;f();}
    return count;
}
namespace write_fault {
int remaining=-1;uint64_t calls=0,triggers=0,collisions=0;const char* collide=nullptr;
}
extern "C" ssize_t __real_pwrite(int,const void*,size_t,off_t);
extern "C" ssize_t __wrap_pwrite(int fd,const void* data,size_t bytes,off_t offset){
    ++write_fault::calls;
    if(write_fault::remaining>=0&&write_fault::remaining--==0){write_fault::remaining=-1;++write_fault::triggers;errno=EIO;return -1;}
    const auto count=__real_pwrite(fd,data,bytes,offset);
    if(count>0&&write_fault::collide){const char* path=write_fault::collide;write_fault::collide=nullptr;
        if(::mkdir(path,0700)==0)++write_fault::collisions;}
    return count;
}
namespace glm53::physical_fixture {
void test_authority(const rad::Program& program,const rad::Plan& plan,const bridge::Catalog& input,const std::filesystem::path& root,bridge::Result* result){
    const auto kept=*result;auto& mutable_program=const_cast<rad::Program&>(program);auto& mutable_plan=const_cast<rad::Plan&>(plan);
    const auto it=std::find_if(plan.units.begin(),plan.units.end(),[](const auto& u){return u.expert>=0;});need(it!=plan.units.end(),"authority test expert unit");
    const size_t unit=size_t(it-plan.units.begin());const auto handle=it->weights[1];
    for(int k=0;k<8;++k){
        auto& weight=mutable_program.weights[handle];auto& group=mutable_plan.units[unit];
        const auto old_group=weight.decl.group;const auto old_map=mutable_plan.unit_of_weight[handle];const auto old_offset=group.offset[1];
        const auto old_shape=weight.stored_shape[1];const auto old_decl_shape=weight.decl.shape[1];const auto old_dtype=weight.stored_dtype;
        const auto old_source=weight.source;const auto old_span=weight.shard_hi;
        const auto destination=(root/("physical-authority-"+std::to_string(k))).string();std::string why;
        if(k==0)weight.decl.group.expert=(weight.decl.group.expert+1)%4;
        if(k==1)mutable_plan.unit_of_weight[handle]=-1;
        if(k==2)group.offset[1]+=4096;
        if(k==3)weight.stored_shape[1]*=2;
        if(k==4)weight.decl.shape[1]*=2;
        if(k==5)weight.stored_dtype=RAD_F32;
        if(k==6)weight.source.assign(513,'x');
        if(k==7)--weight.shard_hi;
        const auto reads=io_fault::calls.load(),writes=write_fault::calls;allocation_fault::peak_request=0;
        const auto status=bridge::materialize(program,plan,input,destination,result,&why);const auto peak=allocation_fault::peak_request;
        weight.decl.group=old_group;mutable_plan.unit_of_weight[handle]=old_map;group.offset[1]=old_offset;weight.stored_shape[1]=old_shape;weight.decl.shape[1]=old_decl_shape;weight.stored_dtype=old_dtype;weight.source=old_source;weight.shard_hi=old_span;
        need(status==(k==6?bridge::Status::invalid:bridge::Status::identity)&&bridge::same_catalog(kept.sources,result->sources)&&!std::filesystem::exists(destination),"forged actual authority published physical source");
        need(io_fault::calls==reads&&write_fault::calls==writes,"actual authority refusal performed payload I/O");
        if(k==6)need(peak<513,"overlong actual source allocated before its bound");
        ++authority_refusals;
    }
    auto ignored=input;ignored.tensors[0].expected.planes[0].path.assign(1u<<20,'z');
    const auto destination=(root/"physical-ignored-path").string();bridge::Result valid;std::string why;allocation_fault::peak_request=0;
    const auto status=bridge::materialize(program,plan,ignored,destination,&valid,&why);
    need(status==bridge::Status::ok&&allocation_fault::peak_request<(1u<<20),"ignored expected plane path was copied or treated as physical authority");
    need(valid.source_bytes==835584&&valid.output_bytes==835584&&valid.physical_fingerprint==kept.physical_fingerprint,"ignored path changed canonical physical authority");
}
void test_failures(const rad::Program& program,const rad::Plan& plan,const bridge::Catalog& input,const std::filesystem::path& root,bridge::Result* result){
    const auto kept=*result;
    auto unchanged=[&]{return bridge::same_catalog(kept.sources,result->sources)&&kept.physical_fingerprint==result->physical_fingerprint&&
        kept.manifest_checksum==result->manifest_checksum&&kept.source_bytes==result->source_bytes&&kept.output_bytes==result->output_bytes&&
        kept.source_read_bytes==result->source_read_bytes&&kept.physical_pieces==result->physical_pieces&&kept.window_bytes==result->window_bytes&&kept.metadata_reservation==result->metadata_reservation;};
    size_t first_pass_reads=0;for(const auto& tensor:input.tensors)for(const auto& plane:tensor.planes)for(const auto& piece:plane)first_pass_reads+=(piece.bytes+4095)/4096;
    need(first_pass_reads==288,"closed source verification read count");
    for(int k=0;k<4;++k){
        const auto destination=(root/("physical-fault-"+std::to_string(k))).string();std::string why;
        const auto reads=io_fault::calls.load(),read_triggers=io_fault::triggers.load();
        const auto writes=write_fault::calls,write_triggers=write_fault::triggers,collisions=write_fault::collisions,allocations=allocation_fault::triggers;
        if(k==0)io_fault::remaining=int(first_pass_reads+1); // one copied window before late source failure
        if(k==1)write_fault::remaining=0;
        if(k==2)write_fault::collide=destination.c_str();
        if(k==3)allocation_fault::remaining=0;
        const auto status=bridge::materialize(program,plan,input,destination,result,&why);
        io_fault::remaining=-1;write_fault::remaining=-1;write_fault::collide=nullptr;allocation_fault::remaining=-1;
        need(unchanged(),"physical fault replaced caller result or accounting");
        if(k==0)need(status==bridge::Status::io&&io_fault::triggers==read_triggers+1&&write_fault::calls>writes&&write_fault::triggers==write_triggers,"late physical read failure did not fire after output began");
        if(k==1)need(status==bridge::Status::io&&write_fault::triggers==write_triggers+1&&io_fault::triggers==read_triggers,"physical output write failure did not fire");
        if(k==2)need(status==bridge::Status::exists&&write_fault::collisions==collisions+1&&std::filesystem::is_empty(destination)&&io_fault::triggers==read_triggers&&write_fault::triggers==write_triggers,"late NOREPLACE collision not preserved");
        if(k==3)need(status==bridge::Status::no_memory&&allocation_fault::triggers==allocations+1&&io_fault::calls==reads&&write_fault::calls==writes,"physical metadata OOM performed payload I/O");
        need(k==2||!std::filesystem::exists(destination),"failed physical output published");
        const auto prefix=std::filesystem::path(destination).filename().string()+".staging.";
        for(const auto& entry:std::filesystem::directory_iterator(root))need(!entry.path().filename().string().starts_with(prefix),"physical fault left owned staging directory");
        if(k==2)need(std::filesystem::remove(destination),"remove test-owned collision directory");
        ++fault_groups;
    }
}
}
namespace wait_fault {RadStream stream=nullptr;int remaining=-1;size_t failures=0;}
extern "C" int __real_rad_event_wait(RadStream,RadEvent);
extern "C" int __wrap_rad_event_wait(RadStream stream,RadEvent event){
    if(stream==wait_fault::stream&&wait_fault::remaining>=0&&wait_fault::remaining--==0){wait_fault::remaining=-1;++wait_fault::failures;return RAD_E_IO;}
    return __real_rad_event_wait(stream,event);
}
namespace {
std::array<uint64_t,26> source_counts(const sc::Stats& x){return {x.acquires,x.refused,x.hot_hits,x.cold_hits,x.misses,x.duplicates,x.payload_opens,x.payload_reads,x.payload_read_bytes,x.fd_evictions,x.cell_lookups,x.dedup_touches,x.metadata_opens,x.stat_probes,x.global_audits,x.evictions,x.admission_bypasses,x.usage.hot_bytes,x.usage.cold_bytes,x.usage.live_records,x.usage.live_leases,x.usage.lease_vector_slots,x.cached_payload_fds,x.setup_verified_bytes,x.setup_manifest_bytes,x.setup_payload_io_bytes};}
std::array<uint64_t,8> owner_counts(const NativeBackingStats& x){return {x.prepares,x.source_batches,x.source_bytes,x.copy_bytes,x.published_units,x.retirements,x.held_leases,x.peak_held_leases};}
void terminal_barriers(Fixture& f,int first){
    need(first<0&&f.ctx.terminal_status()==first,"actual Core did not latch permanent model failure");
    const std::string diagnostic=f.ctx.step_error();need(!diagnostic.empty()&&diagnostic.size()<2048,"first bounded diagnostic missing");
    const auto caller=f.caller;const auto version=f.caller_version;const auto source=source_counts(f.source->stats());
    const auto owner=owner_counts(f.owner->stats());const auto epoch=f.mover.residency()->epoch();
    const auto alloc=payload_totals();const auto copies=f.ctx.expert_prepare_copies();const auto waits=f.ctx.stream_waits();
    const auto entries=f.step_entries;const auto callbacks=f.model_allocation_source?f.model_allocation_source->callbacks:0;
    std::array<uint16_t,T*H> input{};std::copy_n(static_cast<const uint16_t*>(f.ctx.buf_ptr(f.selected().w.h.x)),input.size(),input.begin());
    const auto weight=f.plan.units[size_t(residents(f).front())].weights.front();
    RadBatch b{};b.n_tok=T;b.n_seq=1;b.phase=RAD_PHASE_PREFILL;b.step=100;
    active=&f;const int direct=f.ctx.run_step(&b);active=nullptr;
    need(direct==first&&f.run()==first&&f.ctx.expert_prepare(nullptr)==first&&
         f.ctx.set_lane(1)==first&&f.ctx.weight_ptr(weight)==nullptr,"terminal direct Core/helper guard did not preserve first refusal");
    need(f.caller==caller&&f.caller_version==version&&f.step_entries==entries&&
         f.ctx.expert_prepare_copies()==copies&&f.ctx.stream_waits()==waits&&payload_totals()==alloc&&
         source_counts(f.source->stats())==source&&owner_counts(f.owner->stats())==owner&&
         f.mover.residency()->epoch()==epoch&&std::string(f.ctx.step_error())==diagnostic&&
         (!f.model_allocation_source||f.model_allocation_source->callbacks==callbacks)&&
         std::equal(input.begin(),input.end(),static_cast<const uint16_t*>(f.ctx.buf_ptr(f.selected().w.h.x))),
         "terminal retry changed input/caller/version/source/I/O/owner/wait/allocation/publication");
    auto* program=&f.builder->program();f.ctx.shutdown();const auto retired=payload_totals();
    CtxDesc desc{};desc.program=program;desc.stream=f.compute.value;desc.residency=f.mover.residency();
    desc.arch_step=step;desc.record_passes=false;desc.expert_prepare_owner=f.owner.get();
    need(f.ctx.init(desc)==first&&payload_totals()==retired&&source_counts(f.source->stats())==source&&
         std::string(f.ctx.step_error())==diagnostic&&f.caller==caller&&f.caller_version==version,
         "terminal same-Ctx reinit allocated/rebound/read source or lost diagnostic/caller");
}
uint32_t record(const Source& source,int layer,int expert){const auto catalog=source.catalog();const auto found=std::find_if(catalog.begin(),catalog.end(),[&](const auto& d){return d.layer==layer&&d.expert==expert;});need(found!=catalog.end(),"owned record lookup");return uint32_t(found-catalog.begin());}
bool untouched_gu(Fixture& f){const auto* p=static_cast<const uint8_t*>(f.ctx.buf_ptr(f.selected().w.egu));return std::all_of(p,p+size_t(R*K*2*H*2),[](uint8_t x){return x==0x5a;});}
void connected(const std::string& home){++cases;Fixture reference(home,false),f(home,true);
    need(f.original_expert_reads==8&&reference.original_expert_reads==32,"old owned expert payload loaded into Mover");
    const auto setup=f.source->stats();need(setup.catalog_records==32&&setup.shards==3&&setup.payload_reads==8&&setup.usage.hot_bytes==0&&setup.usage.cold_bytes==0,"bootstrap/setup/demand accounting mixed");
    size_t pass=0;
    for(const auto [layer,mode]:std::array<std::pair<int,int>,8>{{{0,0},{0,0},{7,0},{0,0},{7,0},{7,2},{7,2},{3,3}}}){
        reference.active_layer=f.active_layer=layer;reference.mode=f.mode=mode;
        good(reference.run(),"same-byte all-resident actual HOST baseline");const auto before=f.source->stats();
        good(f.run(),"actual current routes/sharded source/Mover/FFN");need(f.caller==reference.caller,"sharded source changed connected HOST outputs");
        const auto after=f.source->stats();
        if(pass==1||pass==3||pass==4||pass==6)need(after.payload_reads==before.payload_reads&&after.payload_opens==before.payload_opens,"HOT hit reread/reopened payload");
        for(const auto id:f.source->last_request())need(f.source->catalog()[id].layer==layer&&f.source->catalog()[id].expert>0,"routed source included another layer/static expert0");
        need(f.owner->stats().held_leases==0&&f.source->adapter_usage().live_holders==0,"true last reader retained source Holder");
        good(f.mover.native_record_guards_host(),"connected guard bytes");++pass;
    }
    const auto stats=f.source->stats();need(stats.hot_hits&&stats.fd_evictions&&stats.evictions&&stats.usage.peak_hot_bytes<=65536&&stats.usage.peak_cold_bytes==98304,"global cache/FD/whole-union path not exercised");
    need(f.owner->stats().peak_held_leases==3,"three-record routed union missing");
}
void backing_and_budget(const std::string& home){++cases;Fixture f(home,true);const auto& bytes=f.inventory->footprint();
    need(bytes.owned_units==24&&bytes.consumer_slots==3&&bytes.regions==4&&bytes.native_device_bytes==196608&&bytes.landing_bytes==196608,"native reusable consumers undercharged");
    need(f.producer_retained==0&&f.initial.capacity()==0&&f.specs.capacity()==0&&f.source_catalog.tensors.empty(),"producer/original record payload retained before forward");
    need(f.before_mover==f.after_mover&&f.mover.record_base_allocation_bytes()==bytes.original_static_bytes,"hidden old allocation outside actual reduced Pools");
    need(f.pools.weights().used()==int64_t(bytes.weights_minimum())&&f.pools.host_pinned().used()==196608&&f.pool_config.host_pool_mib==0,"old host/full pool retained");
    for(size_t u=0;u<f.plan.units.size();++u)if(f.plan.units[u].expert>0)need(!f.mover.unit_base(int32_t(u))&&f.plan.units[u].tier==Tier::NativeStore,"native unit has old fallback");
    const auto fixed=residents(f);std::vector<std::pair<int32_t,uint64_t>> resident_hashes;
    for(const auto u:fixed)resident_hashes.push_back({u,es::checksum({static_cast<const std::byte*>(f.mover.unit_base(u)),size_t(f.plan.units[size_t(u)].bytes)})});
    std::vector<std::pair<rad_weight,WeightSlot>> slots;for(const auto u:fixed)for(const auto w:f.plan.units[size_t(u)].weights)slots.push_back({w,*f.mover.residency()->lookup(w)});
    need(f.source->bootstrap(fixed.front(),{})<0,"static bootstrap reopened after binding");
    const auto process=f.source->process_usage();const auto stats=f.source->stats();
    uint64_t ctx_bytes=0;for(size_t n=0;n<f.before_ctx.size();++n)ctx_bytes+=uint64_t(f.after_ctx[n].second-f.before_ctx[n].second);
    const auto component=Fixture::strict_extra+f.device_allocator.physical+f.host_allocator.physical+process.metadata_reserved+process.cache_reserved+bytes.metadata_reservation+65536+ctx_bytes+uint64_t(f.pools.kv().committed());
    need(component<Fixture::admitted&&ctx_bytes<4u<<20&&process.metadata_reserved<=24u<<20&&bytes.metadata_reservation<=4u<<20,"composite admission exceeded");
    good(f.run(),"budgeted actual HOST forward");for(const auto& [w,old]:slots){const auto* current=f.mover.residency()->lookup(w);need(current->ptr==old.ptr&&current->ready==old.ready&&current->generation==old.generation,"static resident generation changed");}
    for(const auto& [u,hash]:resident_hashes)need(es::checksum({static_cast<const std::byte*>(f.mover.unit_base(u)),size_t(f.plan.units[size_t(u)].bytes)})==hash,"static resident bytes changed");
    need(f.pools.kv().used()==0&&f.pools.kv().bytes()==1<<20&&f.pools.kv().committed()==0,"expert source consumed KV budget");
    std::cout<<"RECEIPT admitted="<<Fixture::admitted<<" declared-upper="<<Fixture::declared_upper<<" component-observed="<<component
             <<" pool-requested="<<f.device_allocator.requested+f.host_allocator.requested<<" pool-rounded="<<f.device_allocator.physical+f.host_allocator.physical
             <<" native="<<bytes.native_device_bytes<<" landing="<<bytes.landing_bytes<<" native-meta="<<bytes.metadata_reservation
             <<" provider-steady-meta="<<process.metadata_reserved<<" provider-setup-peak="<<process.peak_metadata_reserved<<" HOT+COLD="<<process.cache_reserved
             <<" adapter-reserved=65536 strict-helper-reserved="<<Fixture::strict_extra<<" ctx-raddev-delta="<<ctx_bytes<<" setup-native-verified="<<stats.setup_verified_bytes<<" setup-payload-IO="<<stats.setup_payload_io_bytes
             <<" bootstrap-reads="<<stats.payload_reads<<" bootstrap-bytes="<<stats.payload_read_bytes<<" KV-logical="<<f.pools.kv().bytes()<<" KV-committed="<<f.pools.kv().committed()<<" buffered-OS-cache-excluded\n";
}
void pressure(const std::string& home){++cases;Fixture f(home,true,98304,8,0);const std::array held_id{record(*f.source,7,1)};NativeRecordBatch held;good(f.source->acquire(held_id,&held),"retain external unselected COLD lease");
    const auto before=f.source->stats();const auto caller=f.caller;const auto epoch=f.mover.residency()->epoch();f.mode=3;
    need(f.run()<0&&f.caller==caller&&untouched_gu(f)&&f.mover.residency()->epoch()==epoch,"whole-union pressure published partial expert/launch/output");
    need(f.source->stats().payload_reads==before.payload_reads&&f.owner->stats().published_units==0&&f.owner->stats().held_leases==0,"pressure performed partial I/O/publication");
    need(f.run()<0,"failed pressure pass reused");held.leases.clear();need(f.source->adapter_usage().live_holders==0&&f.source->accounting().usage().cold_bytes==0,"retained lease pressure leaked");
    terminal_barriers(f,f.ctx.terminal_status());
}
fs::path touched_file;
void touch_file(){struct stat s{};need(!::stat(touched_file.c_str(),&s),"late stat witness");timespec times[2]{s.st_atim,s.st_mtim};++times[1].tv_sec;need(!::utimensat(AT_FDCWD,touched_file.c_str(),times,0),"late selected stamp change");}
void source_errors(const std::string& home){++cases;
    for(int kind=0;kind<2;++kind){Fixture f(home,true);f.mode=3;const auto caller=f.caller;const auto epoch=f.mover.residency()->epoch();const auto before=f.source->stats();
        if(!kind)io_fault::remaining=1;
        else{touched_file=f.temp.path/"generation/layer_shard_0/segment-00000001.bin";io_fault::after_count=2;io_fault::after=touch_file;}
        const int e=f.run();io_fault::remaining=-1;io_fault::after=nullptr;
        need(e<0&&f.caller==caller&&untouched_gu(f)&&f.mover.residency()->epoch()==epoch,"late source failure reached native readers/publication");
        const auto after=f.source->stats();need(after.payload_reads>=before.payload_reads+2&&after.usage.hot_bytes==0&&after.usage.cold_bytes==0&&f.owner->stats().held_leases==0&&f.source->adapter_usage().live_holders==0,"failed source transaction retained private/ready payload");
        need(f.run()<0,"late source failed pass reused");
        terminal_barriers(f,e);
    }
}
void describe_authority(const std::string& home){++cases;Fixture f(home,true);auto& store=f.source->store_for_test();const auto local=f.source->handle_for_test(0);sc::BoundInfo exact;
    need(store.describe(local,&exact)==sc::Error::ok&&exact.program==&f.builder->program(),"immutable describe identity");const auto kept=exact;
    need(store.describe({},&exact)==sc::Error::identity&&exact==kept,"empty describe changed output");
    // Negative-only independent rank metadata fixture: no kernel or TP execution.
    foreign_test::Fixture foreign(3,1);const auto root=foreign.temp.path/"negative_generation";need(fs::create_directory(root),"foreign store root");
    std::array<std::vector<es::RecordSpec>,3> groups;std::array<sc::ShardInput,3> inputs;
    for(const auto& spec:foreign.specs)groups[size_t(spec.key().layer%3)].push_back(spec);
    for(size_t n=0;n<3;++n){auto& input=inputs[n];input.name="negative_"+std::to_string(n);input.specs=groups[n];input.sources=&foreign.catalog;input.limits=foreign.limits;
        es::Receipt receipt;good(es::pack(input.specs,foreign.catalog,(root/input.name).string(),input.limits,&receipt),"negative-only native source");input.manifest_checksum=receipt.manifest_checksum;input.limits.max_segments=receipt.segments;}
    sc::Limits limits;limits.hot_bytes=limits.cold_bytes=0;limits.max_live_records=32;limits.max_live_leases=4;limits.max_records=64;limits.max_batch=3;
    std::unique_ptr<sc::RankNativeStore> other;need(sc::RankNativeStore::open(f.source->process_for_test(),foreign.builder->program(),foreign.plan,root.string(),9,inputs,limits,&other)==sc::Error::ok,"foreign immutable metadata owner");
    sc::BoundRecord foreign_handle;need(other->bind(foreign.specs.front().key(),&foreign_handle)==sc::Error::ok,"foreign bound handle");
    const auto reads=f.source->stats().payload_reads;need(store.describe(foreign_handle,&exact)==sc::Error::identity&&exact==kept&&f.source->stats().payload_reads==reads,"foreign describe accepted/mutated/I/O");
    Fixture same_shape(home,false);Ctx failed;CtxDesc desc{};desc.program=&same_shape.builder->program();desc.stream=f.compute.value;desc.residency=f.mover.residency();desc.arch_step=step;desc.record_passes=false;desc.expert_prepare_owner=f.owner.get();
    need(failed.init(desc)<0&&f.source->stats().payload_reads==reads,"foreign same-shaped Program accepted native owner");
}
void late_issue_and_wait(const std::string& home){++cases;
    for(int kind=0;kind<2;++kind){Fixture f(home,true);const auto caller=f.caller;
        if(!kind)f.fail_dn=true;else{wait_fault::stream=f.compute.value;wait_fault::remaining=0;}
        const auto failures=wait_fault::failures;const int e=f.run();wait_fault::stream=nullptr;wait_fault::remaining=-1;
        const auto owner=f.owner->stats();const auto adapter=f.source->adapter_usage();
        const std::string context="late issue/wait kind="+std::to_string(kind)+" status="+std::to_string(e)+
            " caller_unchanged="+std::to_string(f.caller==caller)+" published="+std::to_string(owner.published_units)+
            " held_leases="+std::to_string(owner.held_leases)+" live_holders="+std::to_string(adapter.live_holders)+
            " retirements="+std::to_string(owner.retirements)+" wait_faults="+std::to_string(wait_fault::failures-failures);
        need(e<0&&f.caller==caller&&owner.published_units>0&&owner.held_leases==0&&adapter.live_holders==0,context.c_str());
        if(kind)need(wait_fault::failures==failures+1,"actual checked event wait not injected");
        need(f.run()<0,"late issue/checked-wait fatal pass reused");good(f.mover.native_record_guards_host(),"late issue/wait pool guards");
        terminal_barriers(f,e);
    }
}
void allocation_family_smoke(){
    void* p=::operator new(7);::operator delete(p);p=::operator new[](9);::operator delete[](p,size_t(9));
    p=::operator new(11,std::nothrow);const bool scalar=p!=nullptr;::operator delete(p,size_t(11));need(scalar,"nothrow scalar allocation family");
    p=::operator new[](13,std::nothrow);const bool array=p!=nullptr;::operator delete[](p,std::nothrow);need(array,"nothrow array allocation family");
    const auto alignment=std::align_val_t(64);
    p=::operator new(15,alignment);const bool aligned=uintptr_t(p)%64==0;::operator delete(p,size_t(15),alignment);need(aligned,"aligned scalar allocation family");
    p=::operator new[](17,alignment);const bool aligned_array=uintptr_t(p)%64==0;::operator delete[](p,alignment);need(aligned_array,"aligned array allocation family");
    p=::operator new(19,alignment,std::nothrow);const bool nt_aligned=p&&uintptr_t(p)%64==0;::operator delete(p,alignment,std::nothrow);need(nt_aligned,"aligned nothrow scalar allocation family");
    p=::operator new[](21,alignment,std::nothrow);const bool nt_array=p&&uintptr_t(p)%64==0;::operator delete[](p,size_t(21),alignment);need(nt_array,"aligned nothrow array allocation family");
}
void facade_contract(const std::string& home){++cases;allocation_family_smoke();Fixture f(home,true);const auto one=record(*f.source,7,1),two=record(*f.source,7,2);const std::array duplicate{two,one,two};NativeRecordBatch out;
    good(f.source->acquire(duplicate,&out),"facade first-occurrence dedup");need(out.leases.size()==2&&out.leases[0].record==two&&out.leases[1].record==one&&out.leases[0].lifetime&&out.leases[1].lifetime,"native source result order/lifetime");
    const auto* first=out.leases[0].bytes.data();const auto reads=f.source->stats().payload_reads;const std::array bad{one,uint32_t(f.source->catalog().size())};
    need(f.source->acquire(bad,&out)<0&&out.leases.size()==2&&out.leases[0].bytes.data()==first&&f.source->stats().payload_reads==reads,"late bad facade ID changed caller/read payload");
    const std::array missing{record(*f.source,0,1),record(*f.source,0,2),record(*f.source,0,3)};
    const auto old_count=out.leases.size();const auto old_pointer=out.leases[0].bytes.data();
    for(int64_t position=0;position<2;++position){
        const auto before=f.source->stats();const auto holders=f.source->adapter_usage().live_holders;
        // No allocating call arguments/assertion strings are evaluated while the fault is armed.
        allocation_fault::remaining=position;const int refused=f.source->acquire(missing,&out);allocation_fault::remaining=-1;
        const auto after=f.source->stats();
        need(refused==RAD_E_NOMEM&&out.leases.size()==old_count&&out.leases[0].bytes.data()==old_pointer,
             "facade precommit allocation changed caller result");
        need(after.acquires==before.acquires&&after.misses==before.misses&&after.hot_hits==before.hot_hits&&
             after.cold_hits==before.cold_hits&&after.payload_reads==before.payload_reads&&after.payload_opens==before.payload_opens&&
             after.usage.hot_bytes==before.usage.hot_bytes&&after.usage.cold_bytes==before.usage.cold_bytes&&
             after.usage.live_records==before.usage.live_records&&f.source->adapter_usage().live_holders==holders,
             "facade Holder/result allocation failure reached Store ready/miss/HOT/I/O or leaked credit");
    }
    good(f.source->acquire(missing,&out),"source union succeeds after both facade allocation faults");
    need(out.leases.size()==3,"facade recovery lost a selected expert");
    const auto generation=f.source->catalog()[0].source_generation;for(const auto& d:f.source->catalog())need(d.expert>0&&d.source_generation==generation,"shards became separate native generations/static catalog entries");
    out.leases.clear();need(f.source->adapter_usage().live_holders==0,"facade shared Holder credit leak");
    f.active_layer=7;f.mode=2;good(f.run(),"actual layer uses prebound source after caller metadata release");need(f.producer_retained==0&&f.specs.empty()&&f.source_catalog.tensors.empty(),"fresh metadata/payload retained across runtime");
    const auto unit=std::find_if(f.plan.units.begin(),f.plan.units.end(),[](const auto& u){return u.layer==7&&u.expert==2;});need(unit!=f.plan.units.end(),"selected mutable Program witness");
    auto& weight=f.builder->program().weights[unit->weights[0]];const auto rows=weight.stored_shape[0],cols=weight.stored_shape[1];
    weight.stored_shape[0]*=2;weight.stored_shape[1]/=2; // same bytes, different native interpretation
    const auto old_caller=f.caller;const auto io=f.source->stats().payload_reads;const int refused=f.run();
    weight.stored_shape[0]=rows;weight.stored_shape[1]=cols;
    need(refused<0&&f.caller==old_caller&&f.source->stats().payload_reads==io,"same-size live Program mutation refreshed source authority");
    terminal_barriers(f,refused);

}
void model_allocation_failures(const std::string& home){++cases;
    for(int64_t site=0;site<2;++site){Fixture f(home,true,98304,8,65536,true);
        good(f.run(),"actual successful publication before model allocation failure");
        need(f.caller_version==1&&f.model_allocation_source,"positive model/decorator setup");
        const auto caller=f.caller;const auto before=source_counts(f.source->stats());
        const auto published=f.owner->stats().published_units;const auto callbacks=f.model_allocation_source->callbacks;
        f.model_allocation_source->armed=site;f.active_layer=7;f.mode=2;
        const int failure=f.run();
        need(failure==RAD_E_NOMEM&&f.ctx.terminal_status()==failure&&f.caller==caller&&f.caller_version==1&&
             source_counts(f.source->stats())==before&&f.owner->stats().published_units==published&&
             f.owner->stats().held_leases==0&&f.source->adapter_usage().live_holders==0&&
             f.model_allocation_source->callbacks==callbacks+1&&allocation_fault::remaining==-1,
             "model facade allocation failure committed source/Native/caller or failed to latch");
        terminal_barriers(f,failure);
    }
}
void model_io_after_publication(const std::string& home){++cases;Fixture f(home,true);
    good(f.run(),"actual successful publication before late source I/O failure");
    need(f.caller_version==1,"prior caller version absent");const auto caller=f.caller;
    f.active_layer=7;f.mode=3;io_fault::remaining=1;
    const int failure=f.run();io_fault::remaining=-1;
    need(failure==RAD_E_IO&&f.ctx.terminal_status()==failure&&f.caller==caller&&f.caller_version==1&&
         f.owner->stats().held_leases==0&&f.source->adapter_usage().live_holders==0,
         "source I/O after success replaced caller publication or escaped actual permanent policy");
    terminal_barriers(f,failure);
}
void close_leases(const std::string& home){++cases;Fixture f(home,true);good(f.run(),"positive forward before final retirement");
    const std::array id{record(*f.source,7,1)};NativeRecordBatch held;good(f.source->acquire(id,&held),"retain source lease beyond facade");const auto bytes=std::vector<std::byte>(held.leases[0].bytes.begin(),held.leases[0].bytes.end());
    good(f.owner->quiesce(),"last actual readers complete");f.ctx.shutdown();good(f.owner->close_after_context_shutdown(),"native withdrawal after Ctx");f.owner.reset();
    auto handle=f.source->handle_for_test(0);sc::BoundInfo original;need(f.source->store_for_test().describe(handle,&original)==sc::Error::ok,"describe before seal");const auto kept=original;
    good(f.source->close(),"source close after Core retirement");need(f.source->store_for_test().describe(handle,&original)==sc::Error::closed&&original==kept,"sealed/stale describe changed output");
    std::weak_ptr<Source> weak=f.source;handle={};f.source.reset();need(weak.expired(),"source/Holder ownership cycle");
    need(std::equal(bytes.begin(),bytes.end(),held.leases[0].bytes.begin(),held.leases[0].bytes.end()),"lease bytes invalid after source destruction");
    std::shared_ptr<sc::ProcessBudget> replacement;need(sc::ProcessBudget::create({},&replacement)==sc::Error::budget&&!replacement,"retained NativeRecordLease evaded process budget token");
    held.leases.clear();need(sc::ProcessBudget::create({},&replacement)==sc::Error::ok,"last native Holder/token did not release global budget");
    good(f.mover.native_record_guards_host(),"final source close consumer guards");
}
}
int main(int argc,char** argv){const char* current="arguments";try{
    need(argc==3&&!std::strcmp(argv[1],"--radiance-home"),"usage: glm53_native_source_strict_policy_test --radiance-home MATCHING_UNIFIED_HOST_HOME");need(device_is_host(),"HOST only");
    need(rad_core_model_failure_policy_version()==RAD_MODEL_FAILURE_POLICY_VERSION,"matching unified Core+Engine PermanentV1 capability required");
    using Test=void(*)(const std::string&);const std::array<std::pair<const char*,Test>,10> tests{{{"connected",connected},{"backing_and_budget",backing_and_budget},{"pressure",pressure},{"source_errors",source_errors},{"describe_authority",describe_authority},{"late_issue_and_wait",late_issue_and_wait},{"facade_contract",facade_contract},{"close_leases",close_leases},{"model_allocation_failures",model_allocation_failures},{"model_io_after_publication",model_io_after_publication}}};
    for(const auto& [name,test]:tests){current=name;test(argv[2]);}
    need(cases==10,"eight original direct-source groups plus two strict model-failure groups");
    need(glm53::physical_fixture::negative_cases==10&&glm53::physical_fixture::fault_groups==4&&glm53::physical_fixture::authority_refusals==8&&glm53::physical_fixture::materializations>0&&glm53::physical_fixture::verified_bytes==835584*glm53::physical_fixture::materializations,"complete physical-source bridge coverage");
    std::cout<<"PHYSICAL_BRIDGE negative_cases=10 fault_groups=4 authority_refusals=8 bytes_per_fixture=835584 materializations="<<glm53::physical_fixture::materializations<<" verified_bytes="<<glm53::physical_fixture::verified_bytes<<"\n";
    std::cout<<"PASS "<<cases<<" actual HOST strict-policy sharded source/native-owner/connected-FFN cases, "<<checks<<" checks; synthetic main TP1, no Engine/GPU/checkpoint/TP serving claim\n";return 0;
}catch(const std::exception& e){allocation_fault::remaining=-1;io_fault::remaining=-1;io_fault::after=nullptr;wait_fault::stream=nullptr;std::cerr<<"FAIL case="<<current<<" "<<e.what()<<'\n';return 1;}}
