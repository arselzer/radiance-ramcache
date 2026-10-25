#include "record_cache_bridge.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
namespace glm53::record_bridge {
namespace {
struct Failure {cache::Error error;const char* text;};
void check(bool v,cache::Error e,const char* why){if(!v)throw Failure{e,why};}
uint64_t sum(uint64_t a,uint64_t b){check(b<=UINT64_MAX-a,cache::Error::too_large,"byte sum overflow");return a+b;}
uint64_t product(uint64_t a,uint64_t b){check(!a||b<=UINT64_MAX/a,cache::Error::too_large,"byte product overflow");return a*b;}
void explain(std::string* why,const char* message)noexcept{if(why)try{*why=message;}catch(...){}}
cache::Error map(store::Status s){switch(s){case store::Status::ok:return cache::Error::ok;case store::Status::budget:return cache::Error::budget;case store::Status::io:return cache::Error::io;case store::Status::checksum:return cache::Error::checksum;case store::Status::identity:return cache::Error::identity_mismatch;case store::Status::no_memory:return cache::Error::no_memory;default:return cache::Error::invalid;}}
struct Fd {
    int n=-1;Fd()=default;explicit Fd(int v):n(v){}~Fd(){if(n>=0)::close(n);}
    Fd(Fd&& v)noexcept:n(v.n){v.n=-1;}Fd& operator=(Fd&& v)noexcept{if(this!=&v){if(n>=0)::close(n);n=v.n;v.n=-1;}return *this;}Fd(const Fd&)=delete;
};
struct Stamp {
    dev_t dev;ino_t ino;off_t bytes;mode_t mode;timespec mt,ct;
    bool operator==(const Stamp& v)const{return dev==v.dev&&ino==v.ino&&bytes==v.bytes&&mode==v.mode&&mt.tv_sec==v.mt.tv_sec&&mt.tv_nsec==v.mt.tv_nsec&&ct.tv_sec==v.ct.tv_sec&&ct.tv_nsec==v.ct.tv_nsec;}
};
Stamp stamp(const struct stat& s){return {s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_mtim,s.st_ctim};}
Stamp fd_stamp(int fd,bool directory=false){struct stat s{};check(!::fstat(fd,&s)&&s.st_size>=0&&(directory?S_ISDIR(s.st_mode):S_ISREG(s.st_mode)),cache::Error::io,"fstat regular file/directory");return stamp(s);}
std::string hex(uint64_t n){char b[17];std::snprintf(b,sizeof b,"%016llx",static_cast<unsigned long long>(n));return b;}
const char* record_class(store::RecordClass c){if(c==store::RecordClass::main_mxfp4_e8m0)return "main_mxfp4_e8m0@synthetic_s1";if(c==store::RecordClass::mtp_fp8_f32)return "mtp_fp8_f32@synthetic_s1";throw Failure{cache::Error::identity_mismatch,"unknown native record class"};}
// Length-prefixed field encoding is canonical without copying/parsing the private manifest.
struct Snapshot {
    std::string bytes;size_t cap;const std::string* expected=nullptr;size_t offset=0;
    void append(std::string_view v){
        check(v.size()<=cap-offset,cache::Error::budget,"fresh metadata snapshot cap");
        if(expected){check(offset<=expected->size()&&v.size()<=expected->size()-offset&&std::string_view(*expected).substr(offset,v.size())==v,cache::Error::identity_mismatch,"fresh metadata snapshot mismatch");}
        else bytes.append(v);
        offset+=v.size();
    }
    void number(uint64_t v){std::array<char,8> b{};for(unsigned k=0;k<8;++k)b[k]=char(v>>(8*k));append({b.data(),b.size()});}
    void text(std::string_view s,size_t limit,bool empty=false){check((empty||!s.empty())&&s.size()<=limit&&s.find('\0')==std::string_view::npos,cache::Error::invalid,"bounded metadata string");number(s.size());append(s);}
    void fixed(const char* s,size_t capacity,bool empty=false){const char* end=static_cast<const char*>(std::memchr(s,0,capacity));check(end,cache::Error::invalid,"unterminated encoding string");text({s,size_t(end-s)},capacity,empty);}
};
std::string snapshot(const Fresh& f,const Limits& l,const std::string* expected=nullptr){
    check(f.specs.size()>0&&f.specs.size()<=l.store.max_records&&f.catalog.tensors.size()>0&&f.catalog.tensors.size()<=l.store.max_sources,cache::Error::budget,"fresh record/source counts");
    Snapshot s{{},l.max_snapshot_bytes,expected};s.text("synthetic-s1-cache-binding-v1",64);s.number(f.manifest_checksum);s.text(f.catalog.source_id,l.store.max_string_bytes);s.text(f.catalog.revision,l.store.max_string_bytes);s.number(f.catalog.tensors.size());
    for(const auto& t:f.catalog.tensors){s.text(t.logical,l.store.max_string_bytes);s.number(uint64_t(t.projection));s.number(uint64_t(t.shape[0]));s.number(uint64_t(t.shape[1]));const auto& e=t.encoding;check(e.n_planes==2&&e.reserved==0,cache::Error::invalid,"canonical encoding plane count/reserved");s.fixed(e.scheme,RAD_ENC_STR);s.fixed(e.transform,RAD_ENC_STR,true);s.number(uint64_t(e.n_planes));
        for(int k=0;k<e.n_planes;++k){const auto& p=e.plane[k];s.fixed(p.role,RAD_ENC_STR);s.number(p.dtype);s.number(p.kind);for(const auto x:p.block)s.number(uint64_t(x));for(const auto x:p.extent)s.number(uint64_t(x));}
        for(const auto& p:t.planes){s.text(p.path,4096);s.number(p.offset);s.number(p.bytes);s.number(p.checksum);}
    }
    s.number(f.specs.size());for(const auto& spec:f.specs){check(spec.data(),cache::Error::identity_mismatch,"empty native record spec");const auto key=spec.key();check(key.world==l.world&&key.rank<l.world&&key.layer>=0&&key.expert>=0,cache::Error::identity_mismatch,"native world/rank/layer/expert");check(spec.stride()>0&&spec.stride()<=l.staging_bytes&&spec.stride()<=l.store.max_read_bytes&&spec.stride()<=l.cache.max_record_bytes,cache::Error::budget,"record exceeds staging/read/cache cap");(void)record_class(key.record_class);const auto metadata=spec.metadata_json();s.text(metadata,65536);}
    if(expected)check(s.offset==expected->size(),cache::Error::identity_mismatch,"fresh metadata snapshot extent mismatch");
    return std::move(s.bytes);
}
uint64_t hash(std::string_view s){return store::checksum({reinterpret_cast<const std::byte*>(s.data()),s.size()});}
void limits(const Limits& l){
    check(l.world>=1&&l.world<=64&&l.rank<l.world&&l.staging_bytes&&l.staging_bytes<=SIZE_MAX&&l.max_payload_bytes&&l.max_payload_bytes<=(256u<<20),cache::Error::invalid,"world/staging/payload caps");
    check(l.max_snapshot_bytes>0&&l.max_snapshot_bytes<=(4u<<20)&&l.max_watch_files>0&&l.max_watch_files<=1025&&l.max_metadata_bytes>0&&l.max_metadata_bytes<=(64u<<20),cache::Error::invalid,"snapshot/watch/metadata caps");
    check(l.store.max_records>0&&l.store.max_records<=1024&&l.store.max_sources>0&&l.store.max_sources<=1024&&l.store.max_segments>0&&l.store.max_segments<l.max_watch_files&&l.store.max_manifest_nodes<=32768&&l.store.max_manifest_bytes<=(1u<<20),cache::Error::budget,"setup metadata counts");
    check(l.cache.max_records>0&&l.cache.max_records<=1024&&l.cache.max_catalog_records<=4096&&l.cache.max_batch>0&&l.cache.max_batch<=1024&&l.cache.max_identity_bytes>=128&&l.cache.max_identity_bytes<=512,cache::Error::budget,"cache metadata caps");
    check(sum(sum(l.cache.hot_bytes,l.cache.cold_bytes),l.staging_bytes)<=l.max_payload_bytes,cache::Error::budget,"HOT+COLD+Reader staging reservation");
}
uint64_t metadata_reservation(const Limits& l){
    auto n=sum(l.store.max_manifest_bytes,product(l.max_snapshot_bytes,2));
    n=sum(n,product(l.store.max_manifest_nodes,128));n=sum(n,product(l.store.max_records,1024));
    n=sum(n,product(l.max_watch_files,512));n=sum(n,product(l.cache.max_catalog_records,1024));
    check(n<=l.max_metadata_bytes,cache::Error::budget,"explicit bounded metadata reservation");return n;
}
struct Watch {std::string name;Fd file;Stamp initial;};
struct Entry {store::Key native;cache::RecordRef ref;};
struct Provider final:cache::Provider {
    std::unique_ptr<store::Reader> reader;std::vector<Entry> entries;std::mutex mutex;Stats* counters;uint64_t staging_cap;
    Provider(Stats& s,uint64_t cap):counters(&s),staging_cap(cap){}
    cache::ReadResult read(const cache::RecordRef& ref,std::span<std::byte> dst,std::stop_token stop)override{
        std::lock_guard lock(mutex);++counters->provider_reads;
        const auto found=std::find_if(entries.begin(),entries.end(),[&](const auto& e){return e.ref==ref;});
        if(found==entries.end()||dst.size()!=ref.length){++counters->provider_failures;return {cache::Error::identity_mismatch,0};}
        if(stop.stop_requested()){++counters->provider_failures;return {cache::Error::cancelled,0};}
        check(ref.length<=staging_cap&&counters->staging_bytes==0,cache::Error::budget,"serialized Reader staging bound");
        struct Reservation {Stats& stats;explicit Reservation(Stats& v,uint64_t n):stats(v){stats.staging_bytes=n;stats.peak_staging_bytes=std::max(stats.peak_staging_bytes,n);}~Reservation(){stats.staging_bytes=0;}} reserve(*counters,ref.length);
        std::vector<std::byte> ready;const auto result=map(reader->read(found->native,&ready));
        if(result!=cache::Error::ok){++counters->provider_failures;return {result,0};}
        counters->provider_success_bytes+=ready.size();
        if(ready.size()!=dst.size()||cache::fnv1a64(ready)!=ref.checksum){++counters->provider_failures;return {cache::Error::checksum,ready.size()};}
        if(stop.stop_requested()){++counters->provider_failures;return {cache::Error::cancelled,ready.size()};}
        std::memcpy(dst.data(),ready.data(),ready.size());counters->copy_bytes+=ready.size();return {cache::Error::ok,ready.size()};
    }
};
} // namespace
struct RecordCache::Impl {
    Limits limits;std::string path,snapshot;Fd directory;Stamp directory_stamp{};std::vector<Watch> watches;
    mutable std::mutex transaction;Stats counters;std::vector<cache::RecordRef> refs;
    Provider provider;std::unique_ptr<cache::Cache> payload_cache;
    explicit Impl(const Limits& l):limits(l),provider(counters,l.staging_bytes){}
    void verify_files(){
        struct stat named{};check(!::lstat(path.c_str(),&named)&&S_ISDIR(named.st_mode)&&stamp(named)==directory_stamp&&fd_stamp(directory.n,true)==directory_stamp,cache::Error::identity_mismatch,"store directory identity changed");
        for(const auto& w:watches){++counters.stat_checks;struct stat named_file{};check(fd_stamp(w.file.n)==w.initial&&!::fstatat(directory.n,w.name.c_str(),&named_file,AT_SYMLINK_NOFOLLOW)&&S_ISREG(named_file.st_mode)&&stamp(named_file)==w.initial,cache::Error::identity_mismatch,"manifest/segment ordinary mutation/replacement");}
    }
};
RecordCache::RecordCache(std::unique_ptr<Impl> p):impl_(std::move(p)){}RecordCache::~RecordCache()=default;
cache::Error RecordCache::open(const std::string& path,const Fresh& fresh,const Limits& l,std::unique_ptr<RecordCache>* out,std::string* why)noexcept{
    try{
        check(out&&path.size()>0&&path.size()<=4096&&path.find('\0')==std::string::npos,cache::Error::invalid,"bridge path/output");limits(l);const auto reservation=metadata_reservation(l);auto current=snapshot(fresh,l);
        const auto count=std::count_if(fresh.specs.begin(),fresh.specs.end(),[&](const auto& spec){return spec.key().rank==l.rank;});
        check(count>0&&size_t(count)<=l.cache.max_records&&size_t(count)<=l.cache.max_catalog_records,cache::Error::budget,"bound rank catalog count cap before I/O");
        auto p=std::make_unique<Impl>(l);p->path=path;p->snapshot=std::move(current);p->counters.snapshot_bytes=p->snapshot.size();p->counters.metadata_reservation=size_t(reservation);
        const auto opened=map(store::Reader::open(path,fresh.specs,fresh.catalog,fresh.manifest_checksum,l.store,&p->provider.reader,why));check(opened==cache::Error::ok,opened,"S1 Reader open refused");
        p->directory=Fd(::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));check(p->directory.n>=0,cache::Error::io,"bridge directory open");p->directory_stamp=fd_stamp(p->directory.n,true);
        const size_t segments=p->provider.reader->receipt().segments;check(segments+1<=l.max_watch_files,cache::Error::budget,"descriptor count cap");p->watches.reserve(segments+1);
        for(size_t i=0;i<=segments;++i){char name[48];if(i==0)std::snprintf(name,sizeof name,"manifest.json");else std::snprintf(name,sizeof name,"segment-%08zu.bin",i-1);Fd fd(::openat(p->directory.n,name,O_RDONLY|O_CLOEXEC|O_NOFOLLOW));check(fd.n>=0,cache::Error::io,"bridge watch open");const auto initial=fd_stamp(fd.n);p->watches.push_back({name,std::move(fd),initial});}
        p->counters.watched_files=p->watches.size();p->provider.entries.reserve(fresh.specs.size());p->refs.reserve(fresh.specs.size());p->verify_files();
        const auto& receipt=p->provider.reader->receipt();
        // S1 has no public record-checksum catalogue: derive it via its checked Reader,
        // one bounded private vector at a time. Do not parse/duplicate private manifests.
        for(const auto& spec:fresh.specs){
            p->counters.staging_bytes=spec.stride();p->counters.peak_staging_bytes=std::max(p->counters.peak_staging_bytes,spec.stride());
            std::vector<std::byte> bytes;++p->counters.setup_reads;const auto status=map(p->provider.reader->read(spec.key(),&bytes,why));check(status==cache::Error::ok,status,"startup native checksum pass refused");p->counters.setup_success_bytes+=bytes.size();check(bytes.size()==spec.stride(),cache::Error::io,"startup native record extent");
            const auto native=spec.key();if(native.rank==l.rank){
                cache::RecordRef ref;ref.key={"s1:"+hex(receipt.source_checksum)+":"+hex(receipt.manifest_checksum),"synthetic_not_checkpoint","s1meta:"+hex(hash(spec.metadata_json())),record_class(native.record_class),native.world,native.rank,native.layer,native.expert};
                // This ordinal is an opaque provider catalogue token, NOT a file offset.
                ref.offset=p->refs.size();ref.length=spec.stride();ref.checksum=cache::fnv1a64(bytes);p->provider.entries.push_back({native,ref});p->refs.push_back(std::move(ref));
            }
            p->counters.staging_bytes=0;
        }
        check(!p->refs.empty()&&p->refs.size()<=l.cache.max_records&&p->refs.size()<=l.cache.max_catalog_records,cache::Error::budget,"bound-rank catalogue cap");p->verify_files();p->payload_cache=std::make_unique<cache::Cache>(l.cache,p->provider);
        *out=std::unique_ptr<RecordCache>(new RecordCache(std::move(p)));return cache::Error::ok;
    }catch(const Failure& e){explain(why,e.text);return e.error;}catch(const std::bad_alloc&){explain(why,"bridge allocation failed");return cache::Error::no_memory;}catch(const std::exception& e){explain(why,e.what());return cache::Error::invalid;}catch(...){explain(why,"bridge unknown exception");return cache::Error::invalid;}
}
std::span<const cache::RecordRef> RecordCache::records()const noexcept{return impl_->refs;}
cache::Error RecordCache::acquire(std::span<const cache::RecordRef> refs,const Fresh& fresh,cache::Batch* out,std::stop_token stop)noexcept{
    try{
        std::lock_guard lock(impl_->transaction);auto& p=*impl_;++p.counters.acquire_calls;
        auto fail=[&](cache::Error e){++p.counters.refused_calls;return e;};
        try{
            if(!out)return fail(cache::Error::invalid);
            if(stop.stop_requested())return fail(cache::Error::cancelled);
            if(refs.empty()||refs.size()>p.limits.cache.max_batch)return fail(cache::Error::budget);
            // Compare streamed bounded fields against our owned snapshot; no second full
            // snapshot buffer is allocated on HOT/request paths.
            (void)snapshot(fresh,p.limits,&p.snapshot);
            for(const auto& ref:refs)if(std::find(p.refs.begin(),p.refs.end(),ref)==p.refs.end())return fail(cache::Error::identity_mismatch);
            p.verify_files();cache::Batch ready;const auto status=p.payload_cache->acquire(refs,&ready,stop);if(status!=cache::Error::ok)return fail(status);
            p.verify_files();if(stop.stop_requested())return fail(cache::Error::cancelled);
            out->leases.swap(ready.leases);return cache::Error::ok;
        }catch(const Failure& e){return fail(e.error);}catch(const std::bad_alloc&){return fail(cache::Error::no_memory);}catch(...){return fail(cache::Error::invalid);}
    }catch(...){return cache::Error::invalid;}
}
cache::Error RecordCache::trim()noexcept{try{std::lock_guard lock(impl_->transaction);impl_->verify_files();return impl_->payload_cache->trim();}catch(const Failure& e){return e.error;}catch(...){return cache::Error::invalid;}}
Stats RecordCache::stats()const noexcept{std::lock_guard lock(impl_->transaction);auto result=impl_->counters;result.cache=impl_->payload_cache->stats();return result;}
cache::Accounting RecordCache::accounting()const noexcept{return impl_->payload_cache->accounting();}
} // namespace glm53::record_bridge
