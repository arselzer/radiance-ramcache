#include "record_cache_bridge.h"
#include "bridge_fixture.h"
#include <latch>
#include <cstdlib>
#include <new>
#include <thread>
#include <stop_token>
// Isolated test-only ordinary allocation failures. No runtime/store/cache test hooks.
namespace allocation_fault {
thread_local int64_t remaining=-1;
thread_local bool measure=false;
thread_local size_t seen=0;
void* allocate(size_t n){
    if(measure)++seen;
    if(remaining>=0&&remaining--==0){remaining=-1;throw std::bad_alloc();}
    if(void* p=std::malloc(n?n:1))return p;
    throw std::bad_alloc();
}
}
void* operator new(size_t n){return allocation_fault::allocate(n);}
void* operator new[](size_t n){return allocation_fault::allocate(n);}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,size_t)noexcept{std::free(p);}
void operator delete[](void* p,size_t)noexcept{std::free(p);}
namespace br=glm53::record_bridge;
namespace ec=glm53::cache;
namespace bt=bridge_test;
namespace {
using bt::need;using bt::Fixture;using bt::good;using bt::read;using bt::write;
br::Fresh fresh(Fixture& f,const es::Receipt& r){return {f.specs,f.catalog,r.manifest_checksum};}
std::unique_ptr<br::RecordCache> open(Fixture& f,const fs::path& path,const es::Receipt& r,br::Limits limits={}){
    std::unique_ptr<br::RecordCache> result;std::string why;const auto e=br::RecordCache::open(path.string(),fresh(f,r),limits,&result,&why);need(e==ec::Error::ok,"bridge open: "+std::string(ec::to_string(e))+" "+why);return result;
}
const es::RecordSpec& spec(Fixture& f,const ec::RecordRef& ref){const auto i=std::find_if(f.specs.begin(),f.specs.end(),[&](const auto& s){const auto k=s.key();return k.world==ref.key.tp&&k.rank==ref.key.rank&&k.layer==ref.key.layer&&k.expert==ref.key.expert;});need(i!=f.specs.end(),"native spec for cache ref");return *i;}
std::array<ec::RecordRef,2> pair(const br::RecordCache& c,int expert){std::array<ec::RecordRef,2> result;for(const auto& r:c.records())if(r.key.expert==expert){need(r.key.layer==0||r.key.layer==1,"main/MTP layers");result[size_t(r.key.layer)]=r;}need(result[0].length==32768&&result[1].length==57344&&result[0].key.record_class!=result[1].key.record_class,"distinct main/MTP class record identities");return result;}
void bytes(Fixture& f,const ec::Batch& batch){for(const auto& lease:batch.leases){const auto expected=f.expected(spec(f,lease.ref()));need(std::equal(expected.begin(),expected.end(),lease.bytes().begin(),lease.bytes().end()),"HOT lease exact independent native bytes including all padding");bt::compared_bytes+=expected.size();}}
void same(const ec::Batch& batch,const ec::Batch& before){need(batch.leases.size()==before.leases.size(),"failure changed caller lease count");for(size_t i=0;i<before.leases.size();++i)need(batch.leases[i].bytes().data()==before.leases[i].bytes().data()&&batch.leases[i].ref()==before.leases[i].ref(),"failure replaced caller live lease");}
void hot_and_ownership(){
    ++bt::cases;Fixture f;const auto path=f.temp.path/"ready";const auto r=f.pack(path);auto c=open(f,path,r);const auto p0=pair(*c,0),p1=pair(*c,1);const auto initial=c->stats();need(initial.setup_reads==16&&initial.setup_success_bytes==720896&&initial.peak_staging_bytes==57344&&initial.staging_bytes==0&&initial.watched_files==13,"bounded explicit startup reader/checksum receipt");
    std::array<ec::RecordRef,3> repeated{p0[0],p0[1],p0[0]};ec::Batch held;need(c->acquire(repeated,fresh(f,r),&held)==ec::Error::ok&&held.leases.size()==2,"deduplicate whole current union");bytes(f,held);const auto first=c->stats();need(first.provider_reads==2&&first.copy_bytes==90112&&first.provider_success_bytes==90112&&first.cache.duplicate_demands==1,"explicit extra Reader vector/copy accounting");
    ec::Batch hot;need(c->acquire(p0,fresh(f,r),&hot)==ec::Error::ok,"HOT same main/MTP pair");bytes(f,hot);const auto second=c->stats();need(second.provider_reads==first.provider_reads&&second.copy_bytes==first.copy_bytes&&second.stat_checks>first.stat_checks&&second.cache.hot_hits==first.cache.hot_hits+2,"HOT hit reread payload or skipped fresh stat check");for(size_t i=0;i<2;++i)need(held.leases[i].bytes().data()==hot.leases[i].bytes().data(),"HOT immutable bytes not reused");
    ec::Batch cold;need(c->acquire(p1,fresh(f,r),&cold)==ec::Error::ok,"new pair while old pair is leased");bytes(f,cold);need(c->trim()==ec::Error::ok,"trim live records");const auto used=c->accounting().usage();need(used.hot_bytes==90112&&used.cold_bytes==90112&&used.live_records==4,"live main/MTP leases lost charges/evicted");
    const auto ledger=c->accounting();c.reset();need(ledger.usage().hot_bytes==90112&&ledger.usage().cold_bytes==90112,"bridge destruction lost outstanding lease accounting");bytes(f,held);bytes(f,hot);bytes(f,cold);held.leases.clear();hot.leases.clear();cold.leases.clear();need(ledger.usage().hot_bytes==0&&ledger.usage().cold_bytes==0&&ledger.usage().live_records==0,"last Reader/cache bridge lease did not refund");
}
void identity_before_io(){
    ++bt::cases;Fixture f;const auto path=f.temp.path/"ready";const auto r=f.pack(path);auto c=open(f,path,r);const auto p=pair(*c,0);ec::Batch out;need(c->acquire(p,fresh(f,r),&out)==ec::Error::ok,"baseline identity leases");const auto before=out;const auto stats=c->stats();
    for(int mutation=0;mutation<8;++mutation){auto wrong=p;if(mutation==0)wrong[1].key.rank=1;if(mutation==1)wrong[1].key.tp=1;if(mutation==2)wrong[1].key.record_class=wrong[0].key.record_class;if(mutation==3)wrong[1].key.layout_id+="x";if(mutation==4)wrong[1].key.source_id+="x";if(mutation==5)++wrong[1].offset;if(mutation==6)--wrong[1].length;if(mutation==7)wrong[1].checksum^=1;need(c->acquire(wrong,fresh(f,r),&out)==ec::Error::identity_mismatch,"cross-rank/class/layout/record ref accepted");same(out,before);}
    auto catalog=f.catalog;catalog.revision+="-changed";need(c->acquire(p,{f.specs,catalog,r.manifest_checksum},&out)==ec::Error::identity_mismatch,"stale fresh source catalog on HOT accepted");same(out,before);
    catalog=f.catalog;catalog.tensors[0].planes[0].checksum^=1;need(c->acquire(p,{f.specs,catalog,r.manifest_checksum},&out)==ec::Error::identity_mismatch,"fresh source fingerprint changed on HOT");same(out,before);
    need(c->acquire(p,{f.specs,f.catalog,r.manifest_checksum^1},&out)==ec::Error::identity_mismatch,"changed publication fingerprint accepted");same(out,before);
    auto wrong_specs=f.specs;std::swap(wrong_specs[0],wrong_specs[1]);need(c->acquire(p,{wrong_specs,f.catalog,r.manifest_checksum},&out)==ec::Error::identity_mismatch,"fresh complete spec snapshot ordering changed silently");same(out,before);
    need(c->stats().provider_reads==stats.provider_reads&&c->stats().copy_bytes==stats.copy_bytes,"identity refusal reached provider/cache publish");
}
void ordinary_mutation(){
    ++bt::cases;
    for(int kind=0;kind<4;++kind){Fixture f;const auto path=f.temp.path/"ready";const auto r=f.pack(path);auto c=open(f,path,r);const auto p=pair(*c,0);ec::Batch out;need(c->acquire(p,fresh(f,r),&out)==ec::Error::ok,"mutation baseline HOT");const auto before=out;const auto stats=c->stats();
        const auto target=path/(kind==2?"manifest.json":"segment-00000000.bin");
        if(kind==0){auto b=read(target);b.back()^=std::byte{1};write(target,b);} // late record/padding corruption
        if(kind==1)fs::resize_file(target,fs::file_size(target)-1);
        if(kind==2){auto b=read(target);b[0]^=std::byte{1};write(target,b);}
        if(kind==3){fs::rename(target,path/"old-segment");const auto b=read(path/"old-segment");write(target,b);}
        need(c->acquire(p,fresh(f,r),&out)==ec::Error::identity_mismatch,"ordinary HOT file mutation/replacement accepted");same(out,before);bytes(f,out);need(c->stats().provider_reads==stats.provider_reads,"mutated HOT store reached provider");
    }
}
void bad_open_and_caps(){
    ++bt::cases;Fixture f;const auto path=f.temp.path/"ready";const auto r=f.pack(path);auto kept=open(f,path,r);auto* original=kept.get();
    for(int kind=0;kind<7;++kind){br::Limits l;if(kind==0)l.staging_bytes=32768;if(kind==1)l.max_payload_bytes=1;if(kind==2)l.max_snapshot_bytes=1;if(kind==3)l.max_metadata_bytes=1;if(kind==4)l.max_watch_files=12;if(kind==5)l.rank=4;if(kind==6)l.world=1;const auto e=br::RecordCache::open((f.temp.path/"absent").string(),fresh(f,r),l,&kept);need(e!=ec::Error::ok&&e!=ec::Error::io&&kept.get()==original,"explicit cap/rank did not refuse before missing-store IO or changed output");}
    auto catalog=f.catalog;catalog.source_id+="-stale";need(br::RecordCache::open(path.string(),{f.specs,catalog,r.manifest_checksum},br::Limits{},&kept)==ec::Error::identity_mismatch&&kept.get()==original,"stale open catalog published bridge");
    const auto target=path/"segment-00000011.bin";auto b=read(target);b.back()^=std::byte{1};write(target,b);need(br::RecordCache::open(path.string(),fresh(f,r),br::Limits{},&kept)==ec::Error::checksum&&kept.get()==original,"late corrupt native record published bridge");
}
void whole_union_refusal_and_cancel(){
    ++bt::cases;Fixture f;const auto path=f.temp.path/"ready";const auto r=f.pack(path);br::Limits l;l.cache.hot_bytes=0;l.cache.cold_bytes=57344;auto c=open(f,path,r,l);const auto p=pair(*c,0);ec::Batch out;need(c->acquire(std::span<const ec::RecordRef>(p.data(),1),fresh(f,r),&out)==ec::Error::ok,"small baseline under union budget");const auto before=out;const auto stats=c->stats();need(c->acquire(p,fresh(f,r),&out)==ec::Error::budget,"whole missing MTP union exceeded reserved cold budget");same(out,before);need(c->stats().provider_reads==stats.provider_reads,"full union budget refusal performed partial Reader IO");
    std::stop_source source;source.request_stop();need(c->acquire(std::span<const ec::RecordRef>(p.data()+1,1),fresh(f,r),&out,source.get_token())==ec::Error::cancelled,"cancelled acquire read native payload");same(out,before);need(c->stats().provider_reads==stats.provider_reads&&c->stats().staging_bytes==0,"cancel leaked Reader staging/read");
}
void concurrent_acquire(){
    ++bt::cases;Fixture f;const auto path=f.temp.path/"ready";const auto r=f.pack(path);auto c=open(f,path,r);const auto p=pair(*c,0);std::array<std::vector<std::byte>,2> expected{f.expected(spec(f,p[0])),f.expected(spec(f,p[1]))};std::array<ec::Batch,8> batches;std::latch go(1);std::atomic<bool> okay{true};std::vector<std::thread> workers;
    workers.reserve(batches.size());
    try{for(size_t thread=0;thread<batches.size();++thread)workers.emplace_back([&,thread]{go.wait();try{for(int i=0;i<30;++i){ec::Batch next;if(c->acquire(p,fresh(f,r),&next)!=ec::Error::ok||next.leases.size()!=2){okay=false;return;}for(size_t k=0;k<2;++k)if(!std::equal(expected[k].begin(),expected[k].end(),next.leases[k].bytes().begin(),next.leases[k].bytes().end()))okay=false;batches[thread]=std::move(next);}}catch(...){okay=false;}});}
    catch(...){go.count_down();for(auto& t:workers){t.join();}throw;}
    go.count_down();for(auto& t:workers){t.join();}need(okay,"concurrent exact native cache acquisition failed");const auto stats=c->stats();need(stats.provider_reads==2&&stats.cache.provider_reads==2&&stats.peak_staging_bytes==57344&&stats.staging_bytes==0,"concurrent overlap duplicated Reader payload or unbounded staging");for(size_t thread=1;thread<batches.size();++thread)for(size_t k=0;k<2;++k)need(batches[thread].leases[k].bytes().data()==batches[0].leases[k].bytes().data(),"concurrent overlapping immutable leases differ");
    const auto ledger=c->accounting();c.reset();need(ledger.usage().hot_bytes==90112,"concurrent leases not retained after bridge close");for(auto& b:batches){b.leases.clear();}need(ledger.usage().hot_bytes==0&&ledger.usage().cold_bytes==0,"concurrent last leases did not refund ledger");
}
void allocation_refusal(){
    ++bt::cases;Fixture f;const auto path=f.temp.path/"ready";const auto r=f.pack(path);auto kept=open(f,path,r);auto* const kept_pointer=kept.get();const auto p=pair(*kept,0);ec::Batch caller;need(kept->acquire(p,fresh(f,r),&caller)==ec::Error::ok,"OOM baseline live leases");const auto before=caller;
    const auto directory=path.string();
    for(int64_t n=0;n<8;++n){allocation_fault::remaining=n;const auto e=br::RecordCache::open(directory,fresh(f,r),br::Limits{},&kept);allocation_fault::remaining=-1;need(e==ec::Error::no_memory&&kept.get()==kept_pointer,"open allocation refusal changed published adapter");same(caller,before);}
    auto measured=open(f,path,r);ec::Batch measured_batch;allocation_fault::seen=0;allocation_fault::measure=true;const auto e=measured->acquire(p,fresh(f,r),&measured_batch);allocation_fault::measure=false;const size_t count=allocation_fault::seen;need(e==ec::Error::ok&&count>=5,"measure bounded miss allocation positions");measured_batch.leases.clear();measured.reset();
    const std::array<size_t,5> positions{0,count/4,count/2,count*3/4,count-1};
    for(const auto n:positions){auto c=open(f,path,r);allocation_fault::remaining=int64_t(n);const auto failed=c->acquire(p,fresh(f,r),&caller);allocation_fault::remaining=-1;need(failed==ec::Error::no_memory,"sampled allocation fault did not report no_memory");same(caller,before);need(c->stats().staging_bytes==0&&c->accounting().usage().cold_bytes==0,"allocation refusal leaked staged/cold payload reservation");}
    bytes(f,caller);
}

}
int main(){try{hot_and_ownership();identity_before_io();ordinary_mutation();bad_open_and_caps();whole_union_refusal_and_cancel();concurrent_acquire();allocation_refusal();std::cout<<"PASS "<<bt::cases<<" synthetic S1 Reader/cache bridge cases, "<<bt::checks.load()<<" checks, independent_native_bytes="<<bt::compared_bytes<<"; actual CPU Reader+Cache, explicit single-record staging/copy, no Engine/GPU/model claim\n";return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
