#include "rank_native_store.h"
#include "sharded_fixture.h"
#include <condition_variable>
#include <latch>
#include <new>
#include <optional>
#include <stop_token>
#include <thread>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
namespace sc=glm53::sharded;
namespace ft=sharded_test;
using ft::need;
namespace faults {
thread_local int64_t alloc=-1;
std::atomic<int> read_after{-1};
std::atomic<bool> corrupt{false};
void(*after_read)()=nullptr;
int after_read_count=0;
thread_local bool measure_fds=false;
thread_local size_t peak_fds=0;
size_t fd_count(){DIR* directory=::opendir("/proc/self/fd");if(!directory)throw std::runtime_error("/proc/self/fd observer unavailable");size_t count=0;
    while(const auto* entry=::readdir(directory))if(std::strcmp(entry->d_name,".")&&std::strcmp(entry->d_name,".."))++count;
    ::closedir(directory);return count-1; // exclude the observer's own descriptor
}
void* allocate(size_t n){if(alloc>=0&&alloc--==0){alloc=-1;throw std::bad_alloc();}if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
}
void* operator new(size_t n){return faults::allocate(n);}
void* operator new[](size_t n){return faults::allocate(n);}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,size_t)noexcept{std::free(p);}
void operator delete[](void* p,size_t)noexcept{std::free(p);}
extern "C" ssize_t __real_pread(int,void*,size_t,off_t);
extern "C" ssize_t __wrap_pread(int fd,void* dst,size_t n,off_t at){
    if(faults::measure_fds)faults::peak_fds=std::max(faults::peak_fds,faults::fd_count());
    const int countdown=faults::read_after.load();
    if(countdown>=0&&faults::read_after.fetch_sub(1)==0){faults::read_after=-1;errno=EIO;return -1;}
    const auto result=__real_pread(fd,dst,n,at);
    if(result>0&&faults::corrupt.exchange(false))static_cast<std::byte*>(dst)[size_t(result)-1]^=std::byte{1};
    if(result>0&&faults::after_read&&faults::after_read_count--==0){const auto callback=faults::after_read;faults::after_read=nullptr;callback();}
    return result;
}
namespace {
void ok(sc::Error e,const char* why){need(e==sc::Error::ok,std::string(why)+": "+sc::name(e));}
std::shared_ptr<sc::ProcessBudget> process(){std::shared_ptr<sc::ProcessBudget> p;ok(sc::ProcessBudget::create({},&p),"process budget");return p;}
struct Stores {
    ft::Fixture fixture;fs::path root;
    std::array<std::vector<es::RecordSpec>,3> specs;
    std::array<es::Receipt,3> receipts;
    std::array<sc::ShardInput,3> inputs;
    explicit Stores(int layers=3,int rank=0):fixture(layers,rank),root(fixture.temp.path/"generation"){
        need(fs::create_directory(root),"generation directory");uint64_t bytes=0;
        for(const auto& spec:fixture.specs)specs[size_t(spec.key().layer%3)].push_back(spec);
        for(size_t i=0;i<3;++i){auto& input=inputs[i];input.name="layer_shard_"+std::to_string(i);input.specs=specs[i];input.sources=&fixture.catalog;input.limits=fixture.limits;
            std::string why;ft::good(es::pack(specs[i],fixture.catalog,(root/input.name).string(),input.limits,&receipts[i],&why),"pack native shard "+why);
            input.manifest_checksum=receipts[i].manifest_checksum;input.limits.max_segments=receipts[i].segments;bytes+=receipts[i].total_bytes;
        }
        need(bytes<(8u<<20),"actual native fixture exceeds tiny cap");
    }
    std::unique_ptr<sc::RankNativeStore> open(std::shared_ptr<sc::ProcessBudget> p,sc::Limits l={}){
        std::unique_ptr<sc::RankNativeStore> out;std::string why;const auto e=sc::RankNativeStore::open(std::move(p),fixture.builder->program(),fixture.plan,root.string(),7,inputs,l,&out,&why);
        need(e==sc::Error::ok,"once-bound store: "+std::string(sc::name(e))+" "+why);return out;
    }
    sc::BoundRecord bind(sc::RankNativeStore& s,int layer,int expert=0){sc::BoundRecord out;const es::Key key{4,uint32_t(fixture.rank),layer,expert,layer%3==1?es::RecordClass::mtp_fp8_f32:es::RecordClass::main_mxfp4_e8m0};ok(s.bind(key,&out),"bound record handle");return out;}
    void exact(const sc::Batch& batch){for(const auto& lease:batch.leases){const auto it=std::find_if(fixture.specs.begin(),fixture.specs.end(),[&](const auto& x){return x.key()==lease.key();});need(it!=fixture.specs.end(),"lease key");const auto expected=fixture.expected(*it);
            need(std::equal(expected.begin(),expected.end(),lease.bytes().begin(),lease.bytes().end()),"independent native byte oracle including padding");ft::compared_bytes+=expected.size();}}
};
struct Work {uint64_t cells,dedup,metadata,stats,reads,opens;bool operator==(const Work&)const=default;};
Work work(const sc::Stats& after,const sc::Stats& before){return {after.cell_lookups-before.cell_lookups,after.dedup_touches-before.dedup_touches,after.metadata_opens-before.metadata_opens,after.stat_probes-before.stat_probes,after.payload_reads-before.payload_reads,after.payload_opens-before.payload_opens};}
void same(const sc::Batch& out,const std::vector<const std::byte*>& pointers){need(out.leases.size()==pointers.size(),"failure changed caller lease count");for(size_t i=0;i<pointers.size();++i)need(out.leases[i].bytes().data()==pointers[i],"failure changed caller lease identity");}
std::vector<const std::byte*> pointers(const sc::Batch& b){std::vector<const std::byte*> out;for(const auto& l:b.leases)out.push_back(l.bytes().data());return out;}
void catalog_and_hot_growth(){++ft::cases;std::optional<Work> expected;
    for(const int layers:{3,18}){Stores f(layers);auto p=process();const auto fd_baseline=faults::fd_count();faults::peak_fds=fd_baseline;faults::measure_fds=true;sc::Limits profile;profile.max_rank_fd_credits=20;auto s=f.open(p,profile);faults::measure_fds=false;const auto initial=s->stats();
        need(faults::peak_fds-fd_baseline==initial.startup_actual_fd_peak&&faults::fd_count()==fd_baseline+1,"observed startup descriptor peak/root count disagrees with receipt");need(initial.catalog_records==size_t(layers*2)&&initial.shards==3&&initial.cached_payload_fds==0&&p->usage().open_fd_credits==1,"startup retained per-shard Reader FDs");
        need(initial.startup_actual_fd_peak<=initial.startup_reserved_fd_peak&&initial.startup_reserved_fd_peak<=20,"startup Reader FD receipt");
        const std::array request{f.bind(*s,0),f.bind(*s,1),f.bind(*s,0)};sc::Batch first;ok(s->acquire(request,&first),"whole dedup union");need(first.leases.size()==2,"first occurrence dedup");f.exact(first);
        for(int layer=2;layer<layers;++layer){const std::array one{f.bind(*s,layer)};sc::Batch temp;ok(s->acquire(one,&temp),"bounded broker eviction");f.exact(temp);}
        need(s->stats().cached_payload_fds<=2&&p->usage().open_fd_credits<=3,"payload FD cap");
        const auto before=s->stats();sc::Batch hot;ok(s->acquire(request,&hot),"HOT after FD eviction");f.exact(hot);const auto delta=work(s->stats(),before);
        need(delta.reads==0&&delta.opens==0&&delta.cells==2&&delta.dedup==3,"HOT read or global cell scan");
        if(expected)need(delta==*expected,"unused catalog growth increased fixed HOT metadata work");else expected=delta;
        for(size_t i=0;i<2;++i)need(first.leases[i].bytes().data()==hot.leases[i].bytes().data(),"HOT byte identity not reused");
        need(s->stats().global_audits==0&&s->reload()==sc::Error::unsupported,"hidden global audit/reload");
    }
}
void shared_budget_and_lifetime(){++ft::cases;Stores f;auto p=process();sc::Limits l;l.hot_bytes=90112;l.cold_bytes=180224;auto s=f.open(p,l);
    const std::array first{f.bind(*s,0),f.bind(*s,1)};sc::Batch held;ok(s->acquire(first,&held),"main/MTP shared HOT");f.exact(held);
    const std::array next{f.bind(*s,0,1),f.bind(*s,1,1)};sc::Batch other;ok(s->acquire(next,&other),"live HOT protects shared cross-shard budget");f.exact(other);
    auto use=s->accounting().usage();need(use.hot_bytes==90112&&use.cold_bytes==90112&&use.live_records==4,"per-shard quota or live lease eviction");
    auto accounting=std::optional<sc::Accounting>(s->accounting());ok(s->close(),"close with live leases");s.reset();
    need(p->usage().open_fd_credits==0&&p->usage().rank_owners==1,"close lost token or leaked FD");f.exact(held);f.exact(other);
    std::unique_ptr<sc::RankNativeStore> refused;need(sc::RankNativeStore::open(p,f.fixture.builder->program(),f.fixture.plan,f.root.string(),8,f.inputs,l,&refused)==sc::Error::budget&&!refused,"old leases permitted hidden new rank quota");
    std::shared_ptr<sc::ProcessBudget> second;need(sc::ProcessBudget::create({},&second)==sc::Error::budget&&!second,"second process ledger bypassed retained token");
    held.leases.clear();other.leases.clear();need(accounting->usage().hot_bytes==0&&accounting->usage().cold_bytes==0&&accounting->usage().live_leases==0,"lease payload/accounting not refunded");
    // Handles and accounting also own bounded generation/ledger metadata; release both.
    auto mutable_first=first;auto mutable_next=next;mutable_first={};mutable_next={};
    accounting.reset();
    need(p->usage().cache_reserved==l.hot_bytes+l.cold_bytes,"immutable bound handles lost budget ownership");
}
void route_identity_and_caps(){++ft::cases;Stores f(3,0),other(3,1);auto p=process();auto s=f.open(p),t=other.open(p);const std::array good_request{f.bind(*s,0)};
    sc::Batch out;ok(s->acquire(good_request,&out),"identity baseline");const auto kept=pointers(out);const auto before=s->stats();
    const std::array wrong{good_request[0],other.bind(*t,1)};need(s->acquire(wrong,&out)==sc::Error::identity,"foreign rank handle accepted");same(out,kept);
    const std::array invalid{sc::BoundRecord{}};need(s->acquire(invalid,&out)==sc::Error::identity,"empty forged handle accepted");same(out,kept);
    need(s->stats().payload_reads==before.payload_reads,"identity error read payload");ok(s->acquire(good_request,&out),"caller handle error must not seal valid generation");
    auto unknown=es::Key{4,0,0,0,es::RecordClass::mtp_fp8_f32};sc::BoundRecord untouched=good_request[0];need(s->bind(unknown,&untouched)==sc::Error::identity,"wrong class bound");
    std::stop_source stop;stop.request_stop();need(s->acquire(good_request,&out,stop.get_token())==sc::Error::cancelled,"cancelled request acquired");same(out,kept);
    const auto baseline=p->usage();
    for(int kind=0;kind<5;++kind){sc::Limits l;if(kind==0)l.max_records=1;if(kind==1)l.max_metadata_bytes=1;if(kind==2)l.max_rank_fd_credits=3;if(kind==3)l.buffered_io=false;if(kind==4)l.max_segments=1;
        std::unique_ptr<sc::RankNativeStore> fail;const auto e=sc::RankNativeStore::open(p,f.fixture.builder->program(),f.fixture.plan,"/absent",9,f.inputs,l,&fail);
        need(e!=sc::Error::ok&&e!=sc::Error::io&&!fail,"cap refusal reached missing-store I/O");
        need(p->usage().open_fd_credits==baseline.open_fd_credits&&p->usage().rank_owners==baseline.rank_owners,"cap refusal leaked owner/FD");}
}
void union_failure_and_retry(){++ft::cases;Stores f;auto p=process();sc::Limits l;l.hot_bytes=0;l.cold_bytes=131072;auto s=f.open(p,l);
    const std::array base{f.bind(*s,2,1)};sc::Batch caller;ok(s->acquire(base,&caller),"failure caller baseline");const auto kept=pointers(caller);
    const std::array request{f.bind(*s,0),f.bind(*s,1)};const auto before=s->stats();faults::read_after=1;
    need(s->acquire(request,&caller)==sc::Error::io,"late real pread failure absent");faults::read_after=-1;same(caller,kept);
    auto after=s->stats();need(after.usage.cold_bytes==32768&&after.usage.live_records==1&&after.payload_reads==before.payload_reads+2,"failed private union became ready/leaked reservation");
    const auto reads=after.payload_reads;sc::Batch retry;ok(s->acquire(request,&retry),"good after late read refusal");f.exact(retry);need(s->stats().payload_reads==reads+2,"earlier private fill published on late failure");
    retry.leases.clear();const auto reads2=s->stats().payload_reads;
    std::vector<sc::BoundRecord> all;for(int layer=0;layer<3;++layer)for(int e=0;e<2;++e)all.push_back(f.bind(*s,layer,e));
    need(s->acquire(all,&caller)==sc::Error::budget&&s->stats().payload_reads==reads2,"whole union pressure performed partial record I/O");same(caller,kept);
}
fs::path touched;
void touch_selected(){struct stat state{};need(!::stat(touched.c_str(),&state),"touch fixture stat");timespec times[2]{state.st_atim,state.st_mtim};times[1].tv_sec+=1;need(!::utimensat(AT_FDCWD,touched.c_str(),times,0),"touch selected while read completes");}
void audit_transaction_and_policy(){++ft::cases;
    {Stores f;auto p=process();auto s=f.open(p);const std::array a{f.bind(*s,0)};sc::Batch held;ok(s->acquire(a,&held),"unselected policy baseline");const auto kept=pointers(held);const auto before=s->stats();
        auto path=f.root/f.inputs[2].name/"segment-00000000.bin";auto bytes=ft::read(path);need(!bytes.empty(),"unused corruption target");bytes.back()^=std::byte{1};ft::write(path,bytes);
        ok(s->acquire(a,&held),"explicit selected-audit allows separate unused corruption until audit");same(held,kept);need(s->stats().payload_reads==before.payload_reads,"HOT read under selected audit");
        need(s->audit_all()==sc::Error::identity,"explicit all-files audit missed unused corruption");need(s->acquire(a,&held)==sc::Error::closed,"audit mutation did not seal generation");f.exact(held);
    }
    {Stores f;auto p=process();auto s=f.open(p);const std::array a{f.bind(*s,0),f.bind(*s,1)};sc::Batch out;touched=f.root/f.inputs[1].name/"segment-00000000.bin";
        faults::after_read_count=1;faults::after_read=touch_selected;
        need(s->acquire(a,&out)==sc::Error::identity&&out.leases.empty(),"post-fill selected audit did not precede ready commit");
        const auto stats=s->stats();need(stats.payload_reads==2&&stats.usage.hot_bytes==0&&stats.usage.cold_bytes==0&&stats.usage.live_records==0,"final audit failure published private fill or ran before all reads");
    }
}
void lease_and_stamp_limits(){++ft::cases;Stores f;auto p=process();sc::Limits l;l.max_live_leases=2;l.max_transactions=4;auto s=f.open(p,l);const std::array request{f.bind(*s,0)};
    sc::Batch a,b,c;ok(s->acquire(request,&a),"lease one");ok(s->acquire(request,&b),"lease two");need(s->acquire(request,&c)==sc::Error::budget&&c.leases.empty(),"unbounded lease metadata");
    a=sc::Batch{};ok(s->acquire(request,&c),"released lease credit reused");const auto reads=s->stats().payload_reads;
    need(s->acquire(request,&a)==sc::Error::exhausted&&s->stats().payload_reads==reads,"stamp limit silently wrapped/reset");
}
void allocation_and_startup(){++ft::cases;Stores f;auto p=process();const auto baseline=p->usage();
    const std::string prepared_root=f.root.string(); // prepare allocating call arguments BEFORE arming new

    for(int n=0;n<6;++n){std::unique_ptr<sc::RankNativeStore> out;faults::alloc=n;const auto status=sc::RankNativeStore::open(p,f.fixture.builder->program(),f.fixture.plan,prepared_root,4,f.inputs,{},&out);faults::alloc=-1;
        need(status==sc::Error::no_memory&&!out,"setup allocation refusal not atomic");const auto after=p->usage();need(after.rank_owners==baseline.rank_owners&&after.open_fd_credits==baseline.open_fd_credits&&after.metadata_reserved==baseline.metadata_reserved&&after.cache_reserved==baseline.cache_reserved,"setup OOM leaked quota/FD/metadata");}
    auto s=f.open(p);const std::array request{f.bind(*s,0),f.bind(*s,1)};
    for(int n=0;n<6;++n){sc::Batch out;faults::alloc=n;const auto status=s->acquire(request,&out);faults::alloc=-1;
        need(status==sc::Error::no_memory&&out.leases.empty(),"private fill allocation refusal");const auto u=s->accounting().usage();need(u.hot_bytes==0&&u.cold_bytes==0&&u.live_records==0&&u.live_leases==0&&u.lease_vector_slots==0,"miss OOM leaked ready bytes/lease credits");}
    sc::Batch good;ok(s->acquire(request,&good),"recovery after payload OOM");f.exact(good);
}
void concurrent_hits(){++ft::cases;Stores f;auto p=process();auto s=f.open(p);const std::array request{f.bind(*s,0),f.bind(*s,1)};sc::Batch seed;ok(s->acquire(request,&seed),"concurrent seed");
    std::array<std::vector<std::byte>,2> expected;for(size_t i=0;i<2;++i)expected[i].assign(seed.leases[i].bytes().begin(),seed.leases[i].bytes().end());
    std::array<sc::Batch,6> batches;std::latch go(1);std::atomic<bool> okay{true};std::vector<std::thread> threads;
    try{for(size_t n=0;n<batches.size();++n)threads.emplace_back([&,n]{go.wait();for(int j=0;j<30;++j){sc::Batch out;if(s->acquire(request,&out)!=sc::Error::ok||out.leases.size()!=2){okay=false;return;}
                    for(size_t k=0;k<2;++k){if(!std::equal(expected[k].begin(),expected[k].end(),out.leases[k].bytes().begin(),out.leases[k].bytes().end()))okay=false;}batches[n]=std::move(out);}});}
    catch(...){go.count_down();for(auto& t:threads)t.join();throw;}go.count_down();for(auto& t:threads)t.join();
    need(okay&&s->stats().payload_reads==2,"serialized concurrent overlap reread/corrupt payload");
    for(const auto& b:batches)for(size_t i=0;i<2;++i)need(b.leases[i].bytes().data()==seed.leases[i].bytes().data(),"concurrent shared immutable identity");
}
sc::RankNativeStore* callback_store=nullptr;sc::BoundRecord callback_record;sc::Error callback_status=sc::Error::ok;
void reenter(){sc::Batch out;callback_status=callback_store->acquire({&callback_record,1},&out);}
void reentry_and_checksum(){++ft::cases;
    {Stores f;auto p=process();auto s=f.open(p);callback_store=s.get();callback_record=f.bind(*s,0);faults::after_read_count=0;faults::after_read=reenter;sc::Batch out;ok(s->acquire({&callback_record,1},&out),"outer read with test-only reentry");need(callback_status==sc::Error::reentrant,"same-thread reentry deadlocked/entered mutation");callback_record={};callback_store=nullptr;}
    {Stores f;auto p=process();auto s=f.open(p);const std::array request{f.bind(*s,0)};sc::Batch out;faults::corrupt=true;need(s->acquire(request,&out)==sc::Error::checksum&&out.leases.empty(),"actual selected read checksum not checked");need(s->accounting().usage().live_records==0,"checksum failure became ready");}
}

void startup_parser_and_capture(){++ft::cases;
    {Stores f(18);es::RecordSpec kept=f.fixture.specs.front();const auto original=kept.metadata_json();std::string why;
        need(es::capture_unit(f.fixture.builder->program(),f.fixture.plan,0,f.fixture.catalog,&kept,&why)==es::Status::budget&&kept.metadata_json()==original,
             "default capture source/count cap silently enlarged");
        ft::good(es::capture_unit(f.fixture.builder->program(),f.fixture.plan,0,f.fixture.catalog,f.fixture.limits,&kept,&why),"explicit capture caps");
    }
    for(int kind=0;kind<3;++kind){Stores f;auto p=process();const auto baseline=p->usage();
        if(kind==0){const auto path=f.root/f.inputs[2].name/"manifest.json";const auto bytes=ft::read(path);std::string text(reinterpret_cast<const char*>(bytes.data()),bytes.size());
            text.insert(1,"\"origin\":\"synthetic_fixture\",");const auto mutated=std::span<const std::byte>(reinterpret_cast<const std::byte*>(text.data()),text.size());ft::write(path,mutated);f.inputs[2].manifest_checksum=es::checksum(mutated);
        }
        if(kind==1){const auto path=f.root/f.inputs[2].name/"segment-00000000.bin";fs::resize_file(path,fs::file_size(path)-1);}
        if(kind==2){const auto path=f.root/f.inputs[2].name/"segment-00000000.bin";auto bytes=ft::read(path);need(!bytes.empty(),"startup checksum target");bytes.back()^=std::byte{1};ft::write(path,bytes);}
        std::unique_ptr<sc::RankNativeStore> out;need(sc::RankNativeStore::open(p,f.fixture.builder->program(),f.fixture.plan,f.root.string(),1,f.inputs,{},&out)!=sc::Error::ok&&!out,"late malformed store returned ready catalog/cache");
        const auto after=p->usage();need(after.rank_owners==0&&after.open_fd_credits==0&&after.metadata_reserved==baseline.metadata_reserved&&after.cache_reserved==0,"late startup error leaked catalogs/Reader FDs/quota");
    }
}
void selected_mutation(){++ft::cases;
    for(int kind=0;kind<3;++kind){Stores f;auto p=process();auto s=f.open(p);const std::array request{f.bind(*s,0)};sc::Batch out;ok(s->acquire(request,&out),"selected mutation HOT baseline");const auto before=s->stats();const auto kept=pointers(out);const auto path=f.root/f.inputs[0].name/"segment-00000000.bin";
        if(kind==0){auto bytes=ft::read(path);need(!bytes.empty(),"selected corruption target");bytes[0]^=std::byte{1};ft::write(path,bytes);}
        if(kind==1)fs::resize_file(path,fs::file_size(path)-1);
        if(kind==2){const auto old=path.string()+"-old";fs::rename(path,old);fs::create_symlink(old,path);}
        need(s->acquire(request,&out)==sc::Error::identity&&s->stats().payload_reads==before.payload_reads,"selected HOT mutation/symlink read or accepted");same(out,kept);f.exact(out);
        need(s->acquire(request,&out)==sc::Error::closed,"selected mutation did not seal generation");
    }
}
std::mutex read_mutex;std::condition_variable read_cv;bool read_entered=false,read_release=false;
void block_read(){std::unique_lock lock(read_mutex);read_entered=true;read_cv.notify_all();read_cv.wait(lock,[]{return read_release;});}
void close_during_read(){++ft::cases;Stores f;auto p=process();auto s=f.open(p);const std::array request{f.bind(*s,0),f.bind(*s,1)};
    read_entered=false;read_release=false;faults::after_read_count=0;faults::after_read=block_read;
    std::stop_source stop;sc::Batch out;sc::Error worker_status=sc::Error::ok,close_status=sc::Error::invalid;
    std::thread reader([&]{worker_status=s->acquire(request,&out,stop.get_token());});
    {std::unique_lock lock(read_mutex);read_cv.wait(lock,[]{return read_entered;});}
    std::latch close_started(1);std::atomic<bool> close_done{false};
    std::thread closer;
    try{closer=std::thread([&]{close_started.count_down();close_status=s->close();close_done=true;});}
    catch(...){stop.request_stop();{std::lock_guard lock(read_mutex);read_release=true;}read_cv.notify_all();reader.join();throw;}
    close_started.wait();while(!s->is_closing())std::this_thread::yield();const bool was_blocked=!close_done;
    {std::lock_guard lock(read_mutex);read_release=true;}read_cv.notify_all();reader.join();closer.join();
    need(was_blocked&&worker_status==sc::Error::cancelled&&close_status==sc::Error::ok&&out.leases.empty(),"active read/close did not refuse transaction");
    need(p->usage().open_fd_credits==0&&s->accounting().usage().live_records==0,"active close leaked FD/private fill");
}

void cold_release_race(){++ft::cases;Stores f;auto p=process();sc::Limits l;l.hot_bytes=0;l.cold_bytes=65536;auto s=f.open(p,l);const std::array request{f.bind(*s,0)};
    sc::Batch current;ok(s->acquire(request,&current),"cold race seed");const auto expected=std::vector<std::byte>(current.leases[0].bytes().begin(),current.leases[0].bytes().end());
    for(int iteration=0;iteration<200;++iteration){sc::Batch retiring=std::move(current);std::latch ready(1);std::atomic<bool> go{false};
        std::thread releaser([retired=std::move(retiring),&ready,&go,iteration]()mutable{ready.count_down();while(!go.load())std::this_thread::yield();for(int n=0;n<iteration%4;++n)std::this_thread::yield();retired=sc::Batch{};});
        ready.wait();go=true;const auto e=s->acquire(request,&current);releaser.join();
        need(e==sc::Error::ok&&current.leases.size()==1&&std::equal(expected.begin(),expected.end(),current.leases[0].bytes().begin(),current.leases[0].bytes().end()),"last cold lease race caused false I/O/corrupt payload");
    }
    current=sc::Batch{};const auto usage=s->accounting().usage();need(usage.hot_bytes==0&&usage.cold_bytes==0&&usage.live_records==0&&usage.live_leases==0&&usage.lease_vector_slots==0,"cold race left payload/metadata charges");
}

void owned_snapshot(){++ft::cases;Stores f;auto p=process();auto s=f.open(p);const std::array request{f.bind(*s,0),f.bind(*s,1)};
    // Expected input wrappers and source strings are bind-time data, never mutable authority.
    // Program/native published files remain unchanged; a later Core adapter must keep its
    // independently checked Program/Plan lifetime and immutable inventory contract.
    for(auto& input:f.inputs){input.sources=nullptr;input.specs={};input.name="changed-caller-wrapper";}
    for(auto& group:f.specs)group.clear();f.fixture.catalog={};
    sc::Batch out;ok(s->acquire(request,&out),"owned normalized snapshot after caller metadata release");f.exact(out);
    const auto before=s->stats();sc::Batch hot;ok(s->acquire(request,&hot),"owned snapshot HOT");
    need(s->stats().payload_reads==before.payload_reads,"owned snapshot HOT reread");f.exact(hot);
}
}
int main(){const char* current_case="startup";try{
    using Test=void(*)();
    const std::array<std::pair<const char*,Test>,14> tests{{
        {"catalog_and_hot_growth",catalog_and_hot_growth},{"shared_budget_and_lifetime",shared_budget_and_lifetime},
        {"route_identity_and_caps",route_identity_and_caps},{"union_failure_and_retry",union_failure_and_retry},
        {"audit_transaction_and_policy",audit_transaction_and_policy},{"lease_and_stamp_limits",lease_and_stamp_limits},
        {"allocation_and_startup",allocation_and_startup},{"concurrent_hits",concurrent_hits},
        {"reentry_and_checksum",reentry_and_checksum},{"startup_parser_and_capture",startup_parser_and_capture},
        {"selected_mutation",selected_mutation},{"close_during_read",close_during_read},
        {"cold_release_race",cold_release_race},{"owned_snapshot",owned_snapshot}
    }};
    for(const auto& test:tests){current_case=test.first;test.second();}
    std::cout<<"PASS "<<ft::cases<<" synthetic once-bound sharded-provider cases, "<<ft::checks.load()<<" checks, independent_native_bytes="<<ft::compared_bytes<<"; selected-audit/buffered CPU profile, no Core adapter/GPU/checkpoint claim\n";return 0;
}catch(const std::exception& e){faults::alloc=-1;faults::read_after=-1;faults::after_read=nullptr;std::cerr<<"FAIL case="<<current_case<<" "<<e.what()<<'\n';return 1;}}
