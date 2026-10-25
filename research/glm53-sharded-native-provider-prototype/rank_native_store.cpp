#include "rank_native_store.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
namespace glm53::sharded {
namespace detail {
struct Failure {Error error;const char* why;bool seal=false;};
[[noreturn]] void refuse(Error e,const char* why){throw Failure{e,why};}
void check(bool v,Error e,const char* why){if(!v)refuse(e,why);}
void verify(bool v,const char* why){if(!v)throw Failure{Error::identity,why,true};}
uint64_t add(uint64_t a,uint64_t b){check(b<=UINT64_MAX-a,Error::invalid,"addition overflow");return a+b;}
uint64_t mul(uint64_t a,uint64_t b){check(!a||b<=UINT64_MAX/a,Error::invalid,"multiplication overflow");return a*b;}
void inc(uint64_t& value,uint64_t n=1){value=n>UINT64_MAX-value?UINT64_MAX:value+n;}
Error convert(native::Status s){switch(s){case native::Status::ok:return Error::ok;case native::Status::budget:return Error::budget;case native::Status::io:return Error::io;case native::Status::checksum:return Error::checksum;case native::Status::identity:return Error::identity;case native::Status::no_memory:return Error::no_memory;case native::Status::unsupported:return Error::unsupported;default:return Error::invalid;}}
void explain(std::string* out,const char* why)noexcept{if(out)try{*out=why;}catch(...){}}
struct Token;
struct Process {
    ProcessLimits limits;mutable std::mutex mutex;ProcessUsage usage;
    struct Slot {std::weak_ptr<Token> owner;uint32_t world=0,rank=0;};
    std::vector<Slot> slots;
    explicit Process(ProcessLimits l):limits(l),slots(l.max_rank_owners){usage.metadata_reserved=256+l.max_rank_owners*128;usage.peak_metadata_reserved=usage.metadata_reserved;}
};
struct Token {
    std::shared_ptr<Process> process;uint32_t world=0,rank=0;size_t slot=0,fd_cap=0,fds=0;
    uint64_t metadata=0,cache=0;bool committed=false;
    ~Token(){if(!committed)return;std::lock_guard lock(process->mutex);process->usage.metadata_reserved-=metadata;
        process->usage.cache_reserved-=cache;--process->usage.rank_owners;}
    void reserve_metadata(uint64_t n){std::lock_guard lock(process->mutex);
        check(n<=process->limits.max_metadata_bytes-process->usage.metadata_reserved,Error::budget,"process metadata cap");
        metadata=add(metadata,n);process->usage.metadata_reserved+=n;
        process->usage.peak_metadata_reserved=std::max(process->usage.peak_metadata_reserved,process->usage.metadata_reserved);
    }
    void release_metadata(uint64_t n)noexcept{std::lock_guard lock(process->mutex);metadata-=n;process->usage.metadata_reserved-=n;}
    void reserve_fds(size_t n){std::lock_guard lock(process->mutex);
        check(n<=fd_cap-fds&&n<=process->limits.max_open_fds-process->usage.open_fd_credits,Error::budget,"rank/process FD cap");
        fds+=n;process->usage.open_fd_credits+=n;
        process->usage.peak_fd_credits=std::max(process->usage.peak_fd_credits,process->usage.open_fd_credits);
    }
    void release_fds(size_t n)noexcept{std::lock_guard lock(process->mutex);fds-=n;process->usage.open_fd_credits-=n;}
};
struct MetaReservation {std::shared_ptr<Token> token;uint64_t bytes=0;
    MetaReservation(std::shared_ptr<Token> t,uint64_t n):token(std::move(t)),bytes(n){token->reserve_metadata(bytes);}
    ~MetaReservation(){if(bytes)token->release_metadata(bytes);}
};
struct FdReservation {std::shared_ptr<Token> token;size_t n;
    FdReservation(std::shared_ptr<Token> t,size_t count):token(std::move(t)),n(count){token->reserve_fds(n);}
    ~FdReservation(){token->release_fds(n);}
};
struct Fd {
    int fd=-1;std::shared_ptr<Token> token;
    Fd()=default;~Fd(){reset();}
    Fd(Fd&& o)noexcept:fd(o.fd),token(std::move(o.token)){o.fd=-1;}
    Fd& operator=(Fd&& o)noexcept{if(this!=&o){reset();fd=o.fd;token=std::move(o.token);o.fd=-1;}return *this;}
    void reset()noexcept{if(fd>=0){::close(fd);fd=-1;token->release_fds(1);token.reset();}}
    static Fd open(std::shared_ptr<Token> token,int parent,const std::string& name,int flags){
        token->reserve_fds(1);const int n=::openat(parent,name.c_str(),flags|O_CLOEXEC|O_NOFOLLOW);
        if(n<0){const int failure=errno;token->release_fds(1);if(failure==ENOENT||failure==ELOOP||failure==ENOTDIR)throw Failure{Error::identity,"openat identity changed",true};refuse(Error::io,"openat refused");}
        Fd out;out.fd=n;out.token=std::move(token);return out;
    }
};
native::FileStamp stamp(const struct stat& s){return {uint64_t(s.st_dev),uint64_t(s.st_ino),uint64_t(s.st_size),int64_t(s.st_mtim.tv_sec),int64_t(s.st_mtim.tv_nsec),int64_t(s.st_ctim.tv_sec),int64_t(s.st_ctim.tv_nsec)};}
native::FileStamp fd_stamp(int fd,bool directory=false){struct stat s{};
    verify(!::fstat(fd,&s)&&s.st_size>=0&&(directory?S_ISDIR(s.st_mode):S_ISREG(s.st_mode)),"fstat type/extent");return stamp(s);
}
native::FileStamp at_stamp(int fd,const std::string& name,bool directory=false){struct stat s{};
    verify(!::fstatat(fd,name.c_str(),&s,AT_SYMLINK_NOFOLLOW)&&s.st_size>=0&&(directory?S_ISDIR(s.st_mode):S_ISREG(s.st_mode)),"named identity/type");return stamp(s);
}
bool leaf(const std::string& s){if(s.empty()||s.size()>64||s=="."||s=="..")return false;
    for(const unsigned char c:s)if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'))return false;
    return true;
}
struct GenRecord {native::VerifiedRecord native;uint32_t shard=0,file=0;int32_t unit=-1;};
struct Shard {std::string name;std::shared_ptr<const native::VerifiedCatalog> catalog;uint32_t first_file=0;};
struct Generation {
    std::shared_ptr<Token> token;const rad::Program* program=nullptr;
    uint32_t world=0,rank=0;uint64_t epoch=0;std::atomic<bool> sealed{false};
    std::string root_path;native::FileStamp root_stamp;
    std::vector<Shard> shards;std::vector<GenRecord> records;size_t file_count=0;
};
struct Ledger {
    std::shared_ptr<Token> token;Limits limits;mutable std::mutex mutex;Usage usage;
    Ledger(std::shared_ptr<Token> t,Limits l):token(std::move(t)),limits(l){}
    void reserve(uint64_t bytes,size_t count){std::lock_guard lock(mutex);
        check(bytes<=limits.cold_bytes-usage.cold_bytes&&count<=limits.max_live_records-usage.live_records,Error::budget,"whole missing union cold/record cap");
        usage.cold_bytes+=bytes;usage.live_records+=count;
        usage.peak_cold_bytes=std::max(usage.peak_cold_bytes,usage.cold_bytes);
    }
    void release(bool hot,uint64_t bytes,size_t count=1)noexcept{std::lock_guard lock(mutex);
        (hot?usage.hot_bytes:usage.cold_bytes)-=bytes;usage.live_records-=count;
    }
    bool promote(uint64_t n){std::lock_guard lock(mutex);if(n>limits.hot_bytes-usage.hot_bytes)return false;
        usage.hot_bytes+=n;usage.cold_bytes-=n;usage.peak_hot_bytes=std::max(usage.peak_hot_bytes,usage.hot_bytes);return true;
    }
    void reserve_leases(size_t n){std::lock_guard lock(mutex);
        check(n<=limits.max_live_leases-usage.live_leases,Error::budget,"live lease metadata cap");
        usage.live_leases+=n;usage.peak_live_leases=std::max(usage.peak_live_leases,usage.live_leases);
    }
    void release_leases(size_t n=1)noexcept{std::lock_guard lock(mutex);usage.live_leases-=n;}
    void reserve_slots(size_t n){std::lock_guard lock(mutex);check(n<=limits.max_live_leases-usage.lease_vector_slots,Error::budget,"retained lease-vector slot cap");usage.lease_vector_slots+=n;}
    void release_slots(size_t n)noexcept{std::lock_guard lock(mutex);usage.lease_vector_slots-=n;}
    Usage snapshot()const noexcept{std::lock_guard lock(mutex);auto out=usage;out.metadata_reserved=token->metadata;return out;}
};
struct BatchCredit {std::shared_ptr<Ledger> ledger;size_t slots=0;~BatchCredit(){if(slots)ledger->release_slots(slots);}};
struct Record {
    std::shared_ptr<const Generation> generation;std::shared_ptr<Ledger> ledger;
    uint32_t ordinal=0;std::vector<std::byte> bytes;bool hot=false,charged=false;
    Record(std::shared_ptr<const Generation> g,std::shared_ptr<Ledger> l,uint32_t n)
        :generation(std::move(g)),ledger(std::move(l)),ordinal(n),bytes(size_t(generation->records[n].native.length)){}
    ~Record(){const auto n=bytes.size();std::vector<std::byte>().swap(bytes);if(charged)ledger->release(hot,n);}
};
struct Reserve {std::shared_ptr<Ledger> ledger;uint64_t bytes=0;size_t count=0,leases=0,slots=0;
    ~Reserve(){if(count)ledger->release(false,bytes,count);if(leases)ledger->release_leases(leases);if(slots)ledger->release_slots(slots);}
};
struct Cell {std::weak_ptr<Record> weak;std::shared_ptr<Record> hot;int32_t previous=-1,next=-1;uint64_t seen=0;};
struct CachedFd {Fd file;uint32_t shard=0,segment=0;uint64_t touch=0;};
thread_local bool active=false;
struct Active {Active(){check(!active,Error::reentrant,"same-thread store reentry");active=true;}~Active(){active=false;}};
struct StoreImpl {
    Limits limits;std::shared_ptr<Generation> generation;std::shared_ptr<Ledger> ledger;Fd root;
    mutable std::mutex transaction;std::atomic<bool> closing{false};bool closed=false;
    Stats counters;uint64_t request_stamp=0,fd_stamp_counter=0;
    std::vector<Cell> cells;std::vector<CachedFd> fds;int32_t newest=-1,oldest=-1;
    std::vector<uint32_t> unique,selected_shards;std::vector<int32_t> file_heads,file_next;
    std::vector<uint64_t> shard_marks,file_marks;
    std::vector<std::shared_ptr<Record>> work;
    struct ClearWork {StoreImpl& s;~ClearWork(){s.work.clear();s.unique.clear();s.selected_shards.clear();}};
    StoreImpl(Limits l,std::shared_ptr<Generation> g):limits(l),generation(std::move(g)),ledger(std::make_shared<Ledger>(generation->token,l)){}
    void cancelled(std::stop_token stop){check(!stop.stop_requested()&&!closing.load(),Error::cancelled,"request cancelled/closing");}
    void root_audit(){inc(counters.stat_probes,2);
        verify(fd_stamp(root.fd,true)==generation->root_stamp&&at_stamp(AT_FDCWD,generation->root_path,true)==generation->root_stamp,"generation root changed");
    }
    void audit_shard(uint32_t s,bool all){const auto& shard=generation->shards[s];
        auto dir=Fd::open(generation->token,root.fd,shard.name,O_RDONLY|O_DIRECTORY);inc(counters.metadata_opens);
        inc(counters.stat_probes,3);
        verify(fd_stamp(dir.fd,true)==shard.catalog->directory_stamp()&&
              at_stamp(root.fd,shard.name,true)==shard.catalog->directory_stamp()&&
              at_stamp(dir.fd,"manifest.json")==shard.catalog->manifest_stamp(),"shard directory/manifest changed");
        auto audit_file=[&](uint32_t local){const auto& segment=shard.catalog->segments()[local];inc(counters.stat_probes);
            verify(at_stamp(dir.fd,segment.name)==segment.stamp,"selected segment changed");
        };
        if(all){for(uint32_t n=0;n<shard.catalog->segments().size();++n)audit_file(n);}
        else for(int32_t n=file_heads[s];n>=0;n=file_next[size_t(n)])audit_file(uint32_t(n)-shard.first_file);
    }
    void audit_selected(){root_audit();for(const auto s:selected_shards)audit_shard(s,false);}
    void select_files(){
        for(const auto n:unique){const auto& record=generation->records[n];
            if(shard_marks[record.shard]!=request_stamp){shard_marks[record.shard]=request_stamp;
                selected_shards.push_back(record.shard);file_heads[record.shard]=-1;}
            if(file_marks[record.file]!=request_stamp){file_marks[record.file]=request_stamp;
                file_next[record.file]=file_heads[record.shard];file_heads[record.shard]=int32_t(record.file);}
        }
    }
    int payload_fd(const GenRecord& record){const auto& shard=generation->shards[record.shard];const auto local=record.native.segment;
        check(fd_stamp_counter<UINT64_MAX,Error::exhausted,"FD LRU counter exhausted");++fd_stamp_counter;
        for(auto& entry:fds)if(entry.file.fd>=0&&entry.shard==record.shard&&entry.segment==local){
            inc(counters.stat_probes);verify(fd_stamp(entry.file.fd)==shard.catalog->segments()[local].stamp,"cached payload FD changed");
            entry.touch=fd_stamp_counter;return entry.file.fd;
        }
        auto target=std::find_if(fds.begin(),fds.end(),[](const auto& f){return f.file.fd<0;});
        if(target==fds.end()){target=std::min_element(fds.begin(),fds.end(),[](const auto& a,const auto& b){return a.touch<b.touch;});target->file.reset();inc(counters.fd_evictions);}
        auto directory=Fd::open(generation->token,root.fd,shard.name,O_RDONLY|O_DIRECTORY);inc(counters.metadata_opens);inc(counters.stat_probes,2);
        verify(fd_stamp(directory.fd,true)==shard.catalog->directory_stamp()&&at_stamp(root.fd,shard.name,true)==shard.catalog->directory_stamp(),"payload shard changed");
        const auto& segment=shard.catalog->segments()[local];auto file=Fd::open(generation->token,directory.fd,segment.name,O_RDONLY);
        inc(counters.payload_opens);inc(counters.stat_probes,2);
        verify(fd_stamp(file.fd)==segment.stamp&&at_stamp(directory.fd,segment.name)==segment.stamp,"payload identity mismatch");
        target->file=std::move(file);target->shard=record.shard;target->segment=local;target->touch=fd_stamp_counter;return target->file.fd;
    }
    void read_record(Record& out,std::stop_token stop){const auto& record=generation->records[out.ordinal];const int fd=payload_fd(record);
        const auto base=record.native.offset;check(base<=uint64_t(INT64_MAX)&&out.bytes.size()<=uint64_t(INT64_MAX)-base,Error::invalid,"pread range overflow");
        inc(counters.payload_reads);
        for(size_t done=0;done<out.bytes.size();){cancelled(stop);const auto chunk=std::min(limits.io_chunk_bytes,out.bytes.size()-done);
            const auto n=::pread(fd,out.bytes.data()+done,chunk,off_t(base+done));
            if(n<0&&errno==EINTR)continue;
            if(n==0)throw Failure{Error::io,"unexpected selected EOF",true};
            check(n>0,Error::io,"failed selected pread");done+=size_t(n);inc(counters.payload_read_bytes,uint64_t(n));
        }
        if(native::checksum(out.bytes)!=record.native.checksum)throw Failure{Error::checksum,"selected record checksum",true};
        uint64_t previous=0;for(size_t m=0;m<4;++m){const auto at=record.native.member_offsets[m],bytes=record.native.member_bytes[m];
            check(previous<=at&&at<=out.bytes.size()&&bytes<=out.bytes.size()-at,Error::identity,"bound member extent");
            check(std::all_of(out.bytes.begin()+size_t(previous),out.bytes.begin()+size_t(at),[](auto b){return b==std::byte{0};}),Error::checksum,"selected record padding");previous=at+bytes;
        }
        check(std::all_of(out.bytes.begin()+size_t(previous),out.bytes.end(),[](auto b){return b==std::byte{0};}),Error::checksum,"selected trailing padding");
    }
    void unlink(uint32_t n){auto& cell=cells[n];if(cell.previous>=0)cells[size_t(cell.previous)].next=cell.next;else newest=cell.next;
        if(cell.next>=0)cells[size_t(cell.next)].previous=cell.previous;else oldest=cell.previous;cell.previous=cell.next=-1;
    }
    void front(uint32_t n){auto& cell=cells[n];cell.previous=-1;cell.next=newest;if(newest>=0)cells[size_t(newest)].previous=int32_t(n);else oldest=int32_t(n);newest=int32_t(n);}
    bool evict_one(){for(int32_t n=oldest;n>=0;n=cells[size_t(n)].previous){auto& cell=cells[size_t(n)];if(cell.hot.use_count()==1){unlink(uint32_t(n));cell.hot.reset();inc(counters.evictions);return true;}}return false;}
    void admit(uint32_t n,const std::shared_ptr<Record>& record){auto& cell=cells[n];
        if(cell.hot){if(newest!=int32_t(n)){unlink(n);front(n);}return;}
        if(record->bytes.size()>limits.hot_bytes){inc(counters.admission_bypasses);return;}
        while(!ledger->promote(record->bytes.size()))if(!evict_one()){inc(counters.admission_bypasses);return;}
        record->hot=true;cell.hot=record;front(n);
    }
    Error acquire(std::span<const BoundRecord> request,Batch* out,std::stop_token stop){
        check(out&&request.size()<=limits.max_batch,Error::invalid,"batch/output cap");
        check(!closed&&!generation->sealed.load(),Error::closed,"closed/sealed generation");cancelled(stop);
        check(request_stamp<limits.max_transactions,Error::exhausted,"request stamp exhausted; reload unsupported");
        ++request_stamp;inc(counters.acquires);ClearWork clear{*this};
        for(const auto& handle:request){inc(counters.dedup_touches);
            check(handle.generation_.get()==generation.get()&&handle.ordinal_<cells.size(),Error::identity,"foreign/stale/rank handle");
            auto& cell=cells[handle.ordinal_];if(cell.seen==request_stamp){inc(counters.duplicates);continue;}
            cell.seen=request_stamp;unique.push_back(handle.ordinal_);
        }
        select_files();audit_selected();cancelled(stop);
        uint64_t missing_bytes=0;size_t misses=0;work.resize(unique.size());
        for(size_t i=0;i<unique.size();++i){const auto n=unique[i];auto& cell=cells[n];inc(counters.cell_lookups);
            work[i]=cell.weak.lock();if(work[i])inc(cell.hot?counters.hot_hits:counters.cold_hits);
            else{missing_bytes=add(missing_bytes,generation->records[n].native.length);++misses;}
        }
        Reserve reserved{ledger};ledger->reserve(missing_bytes,misses);reserved.bytes=missing_bytes;reserved.count=misses;
        ledger->reserve_leases(unique.size());reserved.leases=unique.size();
        ledger->reserve_slots(unique.size());reserved.slots=unique.size();
        Batch result;
        if(!unique.empty()){auto credit=std::make_shared<BatchCredit>();credit->ledger=ledger;credit->slots=unique.size();reserved.slots=0;result.storage_=std::move(credit);}
        result.leases.reserve(unique.size());
        for(size_t i=0;i<unique.size();++i)if(!work[i]){auto fresh=std::make_shared<Record>(generation,ledger,unique[i]);
            fresh->charged=true;reserved.bytes-=fresh->bytes.size();--reserved.count;work[i]=std::move(fresh);
            inc(counters.misses);read_record(*work[i],stop);
        }
        // Crucial transaction seam: ALL selected HOT/miss file audits complete again BEFORE
        // any new weak/HOT cell, output lease, or caller result is committed.
        cancelled(stop);audit_selected();cancelled(stop);
        for(size_t i=0;i<work.size();++i){result.leases.push_back(Lease(work[i]));--reserved.leases;}
        for(size_t i=0;i<unique.size();++i){cells[unique[i]].weak=work[i];admit(unique[i],work[i]);}
        *out=std::move(result);return Error::ok;
    }
};
} // namespace detail
using namespace detail;
const char* name(Error e)noexcept{switch(e){case Error::ok:return "ok";case Error::invalid:return "invalid";case Error::budget:return "budget";case Error::identity:return "identity";case Error::io:return "io";case Error::checksum:return "checksum";case Error::cancelled:return "cancelled";case Error::closed:return "closed";case Error::reentrant:return "reentrant";case Error::no_memory:return "no_memory";case Error::unsupported:return "unsupported";case Error::exhausted:return "exhausted";}return "unknown";}
namespace {std::mutex registry_mutex;std::weak_ptr<Process> registered_process;}
ProcessBudget::ProcessBudget(std::shared_ptr<Process> p):state_(std::move(p)){}
Error ProcessBudget::create(const ProcessLimits& l,std::shared_ptr<ProcessBudget>* out)noexcept{try{
    check(out&&!*out&&l.max_rank_owners&&l.max_rank_owners<=64&&l.max_open_fds>=4&&l.max_open_fds<=4096&&l.max_metadata_bytes>=256+l.max_rank_owners*128&&l.max_cache_reservations,Error::invalid,"process limits/output");
    std::lock_guard registry_lock(registry_mutex);
    check(registered_process.expired(),Error::budget,"one process-native budget already exists");
    auto state=std::make_shared<Process>(l);auto result=std::shared_ptr<ProcessBudget>(new ProcessBudget(state));
    registered_process=state;*out=std::move(result);return Error::ok;
}catch(const Failure& e){return e.error;}catch(const std::bad_alloc&){return Error::no_memory;}catch(...){return Error::invalid;}}
ProcessUsage ProcessBudget::usage()const noexcept{std::lock_guard lock(state_->mutex);return state_->usage;}
Lease::Lease(std::shared_ptr<Record> p):record_(std::move(p)){}
Lease::~Lease(){if(record_)record_->ledger->release_leases();}
Lease::Lease(Lease&& o)noexcept:record_(std::move(o.record_)){}
Lease& Lease::operator=(Lease&& o)noexcept{if(this!=&o){if(record_)record_->ledger->release_leases();record_=std::move(o.record_);}return *this;}
std::span<const std::byte> Lease::bytes()const noexcept{return record_?std::span<const std::byte>(record_->bytes):std::span<const std::byte>{};}
native::Key Lease::key()const noexcept{return record_?record_->generation->records[record_->ordinal].native.key:native::Key{};}
Batch::~Batch(){std::vector<Lease>().swap(leases);storage_.reset();}
Batch::Batch(Batch&& other)noexcept:leases(std::move(other.leases)),storage_(std::move(other.storage_)){}
Batch& Batch::operator=(Batch&& other)noexcept{if(this!=&other){std::vector<Lease>().swap(leases);storage_.reset();leases=std::move(other.leases);storage_=std::move(other.storage_);}return *this;}
Usage Accounting::usage()const noexcept{return ledger_?ledger_->snapshot():Usage{};}
RankNativeStore::RankNativeStore(std::unique_ptr<StoreImpl> p):impl_(std::move(p)){}
RankNativeStore::~RankNativeStore(){(void)close();}
Error RankNativeStore::open(std::shared_ptr<ProcessBudget> process,const rad::Program& program,const rad::Plan& plan,
                           const std::string& path,uint64_t epoch,std::span<const ShardInput> inputs,
                           const Limits& l,std::unique_ptr<RankNativeStore>* out,std::string* why)noexcept{
    try{
        Active active;
        check(process&&out&&!*out&&epoch&&epoch!=UINT64_MAX&&l.buffered_io&&
              l.integrity==IntegrityPolicy::ImmutableGenerationSelectedAudit,Error::unsupported,"profile/output/epoch");
        check(!path.empty()&&path[0]=='/'&&path.size()<=4096&&path.find('\0')==std::string::npos&&
              l.max_records&&l.max_records<=65536&&l.max_batch&&l.max_batch<=4096&&l.max_batch<=l.max_records&&
              l.max_shards&&l.max_shards<=256&&l.max_segments&&l.max_segments<=65536&&
              l.max_payload_fds&&l.max_payload_fds<=64&&l.max_rank_fd_credits>=l.max_payload_fds+2&&
              l.max_rank_fd_credits<=4096&&l.max_live_records&&l.max_live_records<=l.max_records&&
              l.max_live_leases&&l.max_live_leases<=65536&&l.max_transactions&&
              l.max_record_bytes&&l.max_record_bytes<=SIZE_MAX&&l.max_metadata_bytes&&
              l.max_metadata_bytes<=(512u<<20)&&l.hot_bytes<=UINT64_MAX-l.cold_bytes&&
              l.io_chunk_bytes>=4096&&l.io_chunk_bytes<=1u<<20&&
              !inputs.empty()&&inputs.size()<=l.max_shards,Error::invalid,"bounded rank limits");
        check(program.ctx.world_size>0&&program.ctx.world_size<=64&&program.ctx.rank>=0&&
              program.ctx.rank<program.ctx.world_size&&plan.world_size==program.ctx.world_size&&plan.rank==program.ctx.rank&&
              !plan.units.empty()&&plan.units.size()<=l.max_records+8192&&program.weights.size()<=4*l.max_records+8192,
              Error::identity,"actual Program/rank/Plan bounds");
        size_t count=0,segments_bound=0;uint64_t startup_meta=0;
        for(size_t i=0;i<inputs.size();++i){const auto& shard=inputs[i];
            check(leaf(shard.name)&&shard.sources&&!shard.specs.empty()&&shard.specs.size()<=l.max_records-count,
                  Error::invalid,"shard leaf/source/count");
            for(size_t j=0;j<i;++j)check(inputs[j].name!=shard.name,Error::identity,"duplicate shard");
            check(shard.limits.max_segments&&shard.limits.max_segments<=l.max_segments-segments_bound&&
                  shard.limits.max_segments+3<=l.max_rank_fd_credits&&
                  shard.limits.max_segments+3<=process->state_->limits.max_open_fds,
                  Error::budget,"startup Reader FD bound including root");
            check(shard.limits.max_record_bytes<=l.max_record_bytes&&shard.limits.max_records>=shard.specs.size(),
                  Error::budget,"shard record cap");
            count+=shard.specs.size();segments_bound+=shard.limits.max_segments;
            const auto parser=add(add(mul(shard.limits.max_manifest_bytes,4),mul(shard.limits.max_manifest_nodes,256)),
                add(add(mul(shard.limits.max_records,4096),mul(shard.limits.max_sources,2048)),shard.limits.io_chunk_bytes));
            startup_meta=std::max(startup_meta,parser);
        }
        const uint64_t persistent=add(4096,add(add(mul(count,1024),mul(segments_bound,1024)),
            add(add(mul(inputs.size(),2048),mul(l.max_batch,256)),add(mul(l.max_live_leases,256),mul(plan.units.size(),128)))));
        check(persistent<=l.max_metadata_bytes&&startup_meta<=l.max_metadata_bytes-persistent,Error::budget,"rank setup/catalog/lease metadata cap");
        auto token=std::make_shared<Token>();token->process=process->state_;token->world=uint32_t(program.ctx.world_size);
        token->rank=uint32_t(program.ctx.rank);token->fd_cap=l.max_rank_fd_credits;
        // Commit quota ownership atomically. An old lease/handle/accounting observer retains
        // this token, preventing a second owner for the same process/rank from hiding bytes.
        {
            std::lock_guard lock(token->process->mutex);size_t slot=token->process->slots.size();
            for(size_t i=0;i<token->process->slots.size();++i){const auto& s=token->process->slots[i];
                if(!s.owner.expired())check(s.world!=token->world||s.rank!=token->rank,Error::budget,"rank already owns budget token");
                else if(slot==token->process->slots.size())slot=i;
            }
            check(slot<token->process->slots.size()&&persistent<=token->process->limits.max_metadata_bytes-token->process->usage.metadata_reserved&&
                  l.hot_bytes+l.cold_bytes<=token->process->limits.max_cache_reservations-token->process->usage.cache_reserved,
                  Error::budget,"process owner/cache/metadata admission");
            token->slot=slot;token->metadata=persistent;token->cache=l.hot_bytes+l.cold_bytes;
            auto& usage=token->process->usage;usage.metadata_reserved+=persistent;usage.cache_reserved+=token->cache;++usage.rank_owners;token->committed=true;
            usage.peak_metadata_reserved=std::max(usage.peak_metadata_reserved,usage.metadata_reserved);
            token->process->slots[slot]={token,token->world,token->rank};
        }
        MetaReservation parser_reservation(token,startup_meta);
        auto generation=std::make_shared<Generation>();generation->token=token;generation->program=&program;
        generation->world=token->world;generation->rank=token->rank;generation->epoch=epoch;generation->root_path=path;
        auto p=std::make_unique<StoreImpl>(l,generation);p->root=Fd::open(token,AT_FDCWD,path,O_RDONLY|O_DIRECTORY);
        generation->root_stamp=fd_stamp(p->root.fd,true);
        check(at_stamp(AT_FDCWD,path,true)==generation->root_stamp,Error::identity,"root replaced during open");
        generation->shards.reserve(inputs.size());generation->records.reserve(count);
        std::vector<uint8_t> claimed(plan.units.size(),0);
        const auto& source_id=inputs.front().sources->source_id;const auto& revision=inputs.front().sources->revision;
        for(const auto& input:inputs){
            check(input.sources->source_id==source_id&&input.sources->revision==revision,Error::identity,"mixed root source identity/revision");
            FdReservation reader_fds(token,input.limits.max_segments+2);
            p->counters.startup_reserved_fd_peak=std::max(p->counters.startup_reserved_fd_peak,input.limits.max_segments+3);
            std::unique_ptr<native::Reader> reader;std::string error;
            const auto opened=native::Reader::open(path+"/"+input.name,input.specs,*input.sources,input.manifest_checksum,input.limits,&reader,&error);
            check(opened==native::Status::ok,convert(opened),"unchanged S1 startup validation refused");
            std::shared_ptr<const native::VerifiedCatalog> catalog;
            const auto exported=reader->export_catalog(&catalog,&error);
            check(exported==native::Status::ok,convert(exported),"verified catalog export refused");
            check(at_stamp(p->root.fd,input.name,true)==catalog->directory_stamp(),Error::identity,"shard replaced during setup");
            p->counters.startup_actual_fd_peak=std::max(p->counters.startup_actual_fd_peak,catalog->segments().size()+3);
            inc(p->counters.setup_verified_bytes,catalog->receipt().total_bytes);inc(p->counters.setup_manifest_bytes,catalog->manifest_stamp().bytes);
            const auto shard_index=uint32_t(generation->shards.size());const auto first_file=uint32_t(generation->file_count);
            check(catalog->segments().size()<=l.max_segments-generation->file_count,Error::budget,"normalized segment cap");
            generation->file_count+=catalog->segments().size();
            for(const auto& record:catalog->records()){
                uint64_t member_bytes=0;for(const auto bytes:record.member_bytes)member_bytes=add(member_bytes,bytes);
                check(member_bytes<=record.length,Error::identity,"normalized member accounting");
                inc(p->counters.setup_payload_io_bytes,add(record.length,record.length-member_bytes));
                check(record.key.world==generation->world&&record.key.rank==generation->rank&&record.length<=l.max_record_bytes,
                      Error::identity,"cross-rank native record");
                const auto unit=std::find_if(plan.units.begin(),plan.units.end(),[&](const auto& u){return u.layer==record.key.layer&&u.expert==record.key.expert;});
                check(unit!=plan.units.end(),Error::identity,"native record has no actual Program unit");
                const size_t index=size_t(unit-plan.units.begin());check(!claimed[index],Error::identity,"duplicate native unit across shards");
                native::RecordSpec fresh;const auto captured=native::capture_unit(program,plan,index,*input.sources,input.limits,&fresh,&error);
                check(captured==native::Status::ok,convert(captured),"fresh Program capture refused");
                const auto expected=std::find_if(input.specs.begin(),input.specs.end(),[&](const auto& spec){return spec.key()==record.key;});
                check(expected!=input.specs.end()&&fresh.metadata_json()==expected->metadata_json(),Error::identity,"actual Program/native layout mismatch");
                claimed[index]=1;generation->records.push_back({record,shard_index,first_file+record.segment,int32_t(index)});
            }
            generation->shards.push_back({input.name,std::move(catalog),first_file});
            reader.reset(); // ALL S1 descriptors close before their reserved credits return.
        }
        for(size_t i=0;i<plan.units.size();++i)if(plan.units[i].expert>=0)check(claimed[i],Error::identity,"incomplete actual expert unit coverage");
        check(generation->records.size()==count,Error::identity,"catalog count/coverage");
        p->cells.resize(count);p->fds.resize(l.max_payload_fds);p->unique.reserve(l.max_batch);p->work.reserve(l.max_batch);
        p->selected_shards.reserve(std::min(inputs.size(),l.max_batch));p->file_heads.resize(inputs.size(),-1);
        p->file_next.resize(generation->file_count,-1);p->file_marks.resize(generation->file_count,0);p->shard_marks.resize(inputs.size(),0);
        p->counters.catalog_records=count;p->counters.segments=generation->file_count;p->counters.shards=inputs.size();
        p->root_audit();for(uint32_t s=0;s<generation->shards.size();++s)p->audit_shard(s,true);
        auto result=std::unique_ptr<RankNativeStore>(new RankNativeStore(std::move(p)));*out=std::move(result);return Error::ok;
    }catch(const Failure& e){explain(why,e.why);return e.error;}catch(const std::bad_alloc&){explain(why,"allocation refused");return Error::no_memory;}catch(...){explain(why,"unexpected setup failure");return Error::invalid;}
}
Error RankNativeStore::bind(const native::Key& key,BoundRecord* out)const noexcept{try{
    Active active;auto& p=*impl_;std::lock_guard lock(p.transaction);
    check(out&&!p.closed&&!p.closing.load()&&!p.generation->sealed.load(),Error::closed,"bind output/closed generation");
    const auto found=std::find_if(p.generation->records.begin(),p.generation->records.end(),[&](const auto& r){return r.native.key==key;});
    check(found!=p.generation->records.end(),Error::identity,"unknown record identity");
    BoundRecord result;result.generation_=p.generation;result.ordinal_=uint32_t(found-p.generation->records.begin());*out=std::move(result);return Error::ok;
}catch(const Failure& e){return e.error;}catch(...){return Error::invalid;}}
Error RankNativeStore::acquire(std::span<const BoundRecord> request,Batch* out,std::stop_token stop)noexcept{
    if(active)return Error::reentrant;
    try{Active guard;auto& p=*impl_;std::lock_guard lock(p.transaction);
        try{return p.acquire(request,out,stop);}
        catch(const Failure& e){inc(p.counters.refused);if(e.seal)p.generation->sealed.store(true);return e.error;}
        catch(const std::bad_alloc&){inc(p.counters.refused);return Error::no_memory;}
        catch(...){inc(p.counters.refused);return Error::invalid;}
    }catch(...){return Error::invalid;}
}
Error RankNativeStore::audit_all()noexcept{
    if(active)return Error::reentrant;
    try{Active guard;auto& p=*impl_;std::lock_guard lock(p.transaction);
        try{check(!p.closed&&!p.closing.load()&&!p.generation->sealed.load(),Error::closed,"audit closed generation");
            inc(p.counters.global_audits);p.root_audit();
            for(uint32_t s=0;s<p.generation->shards.size();++s)p.audit_shard(s,true);
            return Error::ok;
        }catch(const Failure& e){if(e.seal)p.generation->sealed.store(true);return e.error;}
    }catch(...){return Error::invalid;}
}
Error RankNativeStore::trim()noexcept{try{Active active;auto& p=*impl_;std::lock_guard lock(p.transaction);
    check(!p.closed,Error::closed,"trim closed generation");while(p.evict_one()){}return Error::ok;
}catch(const Failure& e){return e.error;}catch(...){return Error::invalid;}}
Error RankNativeStore::close()noexcept{if(!impl_)return Error::ok;if(active)return Error::reentrant;
    auto& p=*impl_;p.closing.store(true);std::lock_guard lock(p.transaction);if(p.closed)return Error::ok;
    p.closed=true;p.generation->sealed.store(true);p.work.clear();
    for(auto& cell:p.cells){cell.hot.reset();}
    p.newest=p.oldest=-1;
    for(auto& fd:p.fds){fd.file.reset();}
    p.root.reset();return Error::ok;
}
bool RankNativeStore::is_closing()const noexcept{return impl_->closing.load();}
Stats RankNativeStore::stats()const noexcept{auto& p=*impl_;std::lock_guard lock(p.transaction);auto result=p.counters;
    for(const auto& fd:p.fds){result.cached_payload_fds+=fd.file.fd>=0;}
    result.usage=p.ledger->snapshot();return result;
}
Accounting RankNativeStore::accounting()const noexcept{return Accounting(impl_->ledger);}
} // namespace glm53::sharded
