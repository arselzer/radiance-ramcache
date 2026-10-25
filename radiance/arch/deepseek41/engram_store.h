/* Bounded original-IQ4_NL Engram row store. No mmap, full-table read or device work.
 * The owning installer must verify Asset::sha256 before opening; this component does
 * not scan a hundred-GiB payload to re-hash it. It pins the opened file descriptor's
 * identity/epoch and refuses subsequent visible size/mtime/ctime changes.
 *
 * Cache payload stays packed (18 bytes / 32 values). Move-only leases keep rows
 * immutable until the last reader releases them; decode writes a caller-owned span.
 * One fixed slab accounts for rows, slot/hash metadata, request bookkeeping and
 * in-flight coalesced-read scratch. A fixed control/allocator allowance is included
 * in the reported budget. Caller output/lease containers and an optional externally
 * owned I/O backend are outside that cache budget, and must be bounded by the caller.
 *
 * This is a row backend, not a registered serving graph or a measured I/O optimizer.
 */
#pragma once
#include "engram_ref.h"
#include <array>
#include <cassert>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <new>
#include <stop_token>
#include <sys/stat.h>
#include <unistd.h>

namespace rad::deepseek41::engram {

struct Iq4NlAsset {
    std::string path;
    std::string sha256; // installer-verified content pin; exactly 64 hexadecimal characters
    std::string format="gguf-iq4_nl-block-v1"; // names the byte layout, never a private GGML ID
    int layer_id=-1;
    int64_t file_bytes=0,data_offset=0,rows=0,width=0;
};
struct RowReadBackend {
    virtual ~RowReadBackend()=default;
    // Same result/errno contract as pread. Must not throw or retain caller buffers.
    // The optional backend's own resources need a separate owner/budget.
    virtual ssize_t read(int fd,void* destination,size_t bytes,off_t offset) noexcept=0;
};
struct Iq4NlCacheOptions {
    uint64_t budget_bytes=0;
    size_t max_batch_rows=4096,max_io_rows=64;
    size_t max_cached_rows=0; // zero chooses the largest capacity fitting the byte budget
    std::shared_ptr<RowReadBackend> backend; // null calls POSIX pread directly
};
struct Iq4NlCacheStats {
    uint64_t requests=0,hits=0,misses=0,duplicate_rows=0,waits=0;
    uint64_t loaded_rows=0,evictions=0,failures=0,pread_calls=0,bytes_read=0;
    uint64_t allocation_bytes=0,slab_bytes=0,fixed_bytes=0,payload_bytes=0,metadata_bytes=0;
    uint64_t io_scratch_bytes=0,in_flight_io_bytes=0,peak_in_flight_io_bytes=0;
    size_t capacity_rows=0,resident_rows=0,loading_rows=0,peak_loading_rows=0;
    size_t pinned_rows=0,active_references=0;
};
class Iq4NlStore;
class Iq4NlRowLease {
public:
    Iq4NlRowLease()=default;
    ~Iq4NlRowLease(){reset();}
    Iq4NlRowLease(const Iq4NlRowLease&)=delete;
    Iq4NlRowLease& operator=(const Iq4NlRowLease&)=delete;
    Iq4NlRowLease(Iq4NlRowLease&& other) noexcept {swap(other);}
    Iq4NlRowLease& operator=(Iq4NlRowLease&& other) noexcept {if(this!=&other){reset();swap(other);}return *this;}
    void reset() noexcept;
    explicit operator bool() const {return bool(owner_);}
    int64_t row() const {return owner_?row_:-1;}
    std::span<const uint8_t> packed() const;
    int decode_into(std::span<float> out,std::string* why=nullptr) const;
private:
    friend class Iq4NlStore;
    Iq4NlRowLease(std::shared_ptr<Iq4NlStore> owner,size_t slot,uint64_t generation,int64_t row)
        :owner_(std::move(owner)),slot_(slot),generation_(generation),row_(row){}
    void swap(Iq4NlRowLease& other) noexcept {
        owner_.swap(other.owner_);std::swap(slot_,other.slot_);std::swap(generation_,other.generation_);std::swap(row_,other.row_);
    }
    std::shared_ptr<Iq4NlStore> owner_;
    size_t slot_=0;uint64_t generation_=0;int64_t row_=-1;
};

class Iq4NlStore:public std::enable_shared_from_this<Iq4NlStore> {
    enum class State:uint8_t {empty,loading,ready};
    struct Slot {int64_t row=-1;uint64_t generation=0;size_t references=0;State state=State::empty;bool referenced=false;};
    struct Bucket {int64_t row=-1;size_t slot=0;};
    struct Request {int64_t row=0;size_t original=0,slot=0;bool reserved=false;};
    struct Layout {size_t slots=0,buckets=0,requests=0,payload=0,scratch=0,total=0,capacity=0,bucket_count=0;};
    struct SlabDelete {void operator()(std::byte* p)const noexcept {::operator delete[](p,std::align_val_t(alignof(std::max_align_t)));}};
    struct Epoch {
        dev_t device{};ino_t inode{};off_t size{};timespec modified{},changed{};
        static Epoch of(const struct stat& st){return {st.st_dev,st.st_ino,st.st_size,st.st_mtim,st.st_ctim};}
        bool same(const struct stat& st)const {
            return device==st.st_dev&&inode==st.st_ino&&size==st.st_size&&
                modified.tv_sec==st.st_mtim.tv_sec&&modified.tv_nsec==st.st_mtim.tv_nsec&&
                changed.tv_sec==st.st_ctim.tv_sec&&changed.tv_nsec==st.st_ctim.tv_nsec;
        }
    };
public:
    static int open(Iq4NlAsset asset,Iq4NlCacheOptions options,std::shared_ptr<Iq4NlStore>* out,std::string* why=nullptr) {
        if(!out)return refuse(why,RAD_E_INVAL,"missing Engram store destination");
        if(asset.format!="gguf-iq4_nl-block-v1")return refuse(why,RAD_E_UNSUPPORTED,"Engram store requires explicitly named original IQ4_NL layout");
        auto hex=[](unsigned char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F');};
        if(asset.path.empty()||asset.path.size()>4096||asset.sha256.size()!=64||
           !std::all_of(asset.sha256.begin(),asset.sha256.end(),hex)||asset.layer_id<0||
           asset.file_bytes<=0||asset.data_offset<0||asset.rows<=0||asset.width<=0||asset.width%32||
           options.max_batch_rows==0||options.max_batch_rows>(1u<<20)||options.max_io_rows==0||options.max_io_rows>4096||
           options.budget_bytes==0||options.budget_bytes>SIZE_MAX/2)
            return refuse(why,RAD_E_SHAPE,"invalid Engram asset identity, geometry or cache bounds");
        const uint64_t blocks=uint64_t(asset.width)/32;
        if(blocks>uint64_t(INT64_MAX)/18)return refuse(why,RAD_E_SHAPE,"Engram packed row overflows");
        const uint64_t row_bytes=blocks*18;
        if(uint64_t(asset.rows)>uint64_t(INT64_MAX)/row_bytes||
           uint64_t(asset.data_offset)>uint64_t(INT64_MAX)-uint64_t(asset.rows)*row_bytes||
           uint64_t(asset.data_offset)+uint64_t(asset.rows)*row_bytes>uint64_t(asset.file_bytes)||
           uint64_t(asset.file_bytes)>uint64_t(std::numeric_limits<off_t>::max())||
           uint64_t(asset.width)>SIZE_MAX/sizeof(float))
            return refuse(why,RAD_E_SHAPE,"Engram table extent overflows or leaves the pinned file size");
        const int fd=::open(asset.path.c_str(),O_RDONLY|O_CLOEXEC);
        if(fd<0)return refuse(why,RAD_E_IO,"cannot open Engram asset");
        struct FileGuard {int fd;~FileGuard(){if(fd>=0)::close(fd);}} file_guard{fd};
        struct stat st{};
        if(::fstat(fd,&st)<0||!S_ISREG(st.st_mode)||st.st_size!=asset.file_bytes)return refuse(why,RAD_E_FORMAT,"Engram asset is not a regular file of the pinned size");
        try {
            auto* raw=new Iq4NlStore(std::move(asset),std::move(options),fd,Epoch::of(st),size_t(row_bytes));
            file_guard.fd=-1;
            auto store=std::shared_ptr<Iq4NlStore>(raw);
            const int status=store->initialize(why);
            if(status!=RAD_OK)return status;
            *out=std::move(store);if(why)why->clear();
            return RAD_OK;
        } catch(const std::bad_alloc&) {return refuse(why,RAD_E_NOMEM,"cannot allocate bounded Engram cache");}
        catch(...) {return refuse(why,RAD_E_STATE,"cannot initialize Engram store controls");}
    }
    ~Iq4NlStore(){if(fd_>=0)::close(fd_);}
    Iq4NlStore(const Iq4NlStore&)=delete;Iq4NlStore& operator=(const Iq4NlStore&)=delete;
    const Iq4NlAsset& asset()const{return asset_;}
    size_t row_bytes()const{return row_bytes_;}
    Iq4NlCacheStats stats()const{std::lock_guard lock(mutex_);return stats_;}
    // Model shutdown only: individual inference cancellation uses a caller stop_token.
    // Stops new work and wakes waiters. Existing leases stay readable. POSIX pread
    // itself cannot be forcibly cancelled; its owner cleans reservations on return.
    void cancel(){std::lock_guard lock(mutex_);cancelled_=true;changed_.notify_all();}
    int lease(int64_t row,Iq4NlRowLease* out,std::string* why=nullptr,std::stop_token stop={}) {
        if(!out)return refuse(why,RAD_E_INVAL,"missing Engram row lease destination");
        if(row<0||row>=asset_.rows)return refuse(why,RAD_E_SHAPE,"Engram row is out of range");
        auto owner=shared_from_this();std::stop_callback wake(stop,[this]{std::lock_guard lock(mutex_);changed_.notify_all();});
        if(stop.stop_requested())return fail_count(RAD_E_CANCELLED,"Engram request was cancelled",why);
        const int epoch=check_epoch();
        if(epoch!=RAD_OK)return fail_count(epoch,"Engram file identity/epoch changed or cannot be read",why);
        std::unique_lock lock(mutex_);
        for(;;) {
            if(cancelled_||stop.stop_requested()){++stats_.failures;return refuse(why,RAD_E_CANCELLED,"Engram store or request is cancelled");}
            const size_t found=find(row);
            if(found==none)break;
            auto& slot=slots_[found];
            if(slot.state==State::loading){++stats_.waits;changed_.wait(lock);continue;}
            pin(found);++stats_.requests;++stats_.hits;slot.referenced=true;
            Iq4NlRowLease result(owner,found,slot.generation,row);lock.unlock();*out=std::move(result);if(why)why->clear();return RAD_OK;
        }
        lock.unlock();return fetch(std::span(&row,1),std::span(out,1),{},std::move(owner),why,stop);
    }
    // Caller owns exactly rows.size() lease objects. On failure they are unchanged.
    // The complete batch must fit concurrently; pinned-capacity pressure returns FULL.
    int lease_many(std::span<const int64_t> rows,std::span<Iq4NlRowLease> out,std::string* why=nullptr,std::stop_token stop={}) {
        if(out.size()!=rows.size())return refuse(why,RAD_E_SHAPE,"Engram lease batch shape mismatch");
        return fetch(rows,out,{},shared_from_this(),why,stop);
    }
    // Copies only requested decoded rows. Batches are bounded by max_batch_rows.
    // Failure may leave an output prefix written; callers must discard it on error.
    int copy_rows(std::span<const int64_t> rows,std::span<float> out,std::string* why=nullptr,std::stop_token stop={}) {
        if(rows.size()>SIZE_MAX/size_t(asset_.width)||out.size()!=rows.size()*size_t(asset_.width))
            return refuse(why,RAD_E_SHAPE,"Engram decoded row output shape mismatch");
        for(auto row:rows)if(row<0||row>=asset_.rows)return refuse(why,RAD_E_SHAPE,"Engram row is out of range");
        auto owner=shared_from_this();
        if(stop.stop_requested())return fail_count(RAD_E_CANCELLED,"Engram request was cancelled",why);
        const size_t band=std::min(options_.max_batch_rows,capacity_);
        for(size_t start=0;start<rows.size();) {
            const size_t fast_count=std::min({size_t(64),band,rows.size()-start});
            bool handled=false;
            int status=copy_hot(rows.subspan(start,fast_count),out.subspan(start*size_t(asset_.width),fast_count*size_t(asset_.width)),&handled,why,stop);
            if(status!=RAD_OK)return status;
            if(handled){start+=fast_count;continue;}
            const size_t count=std::min(band,rows.size()-start);
            status=fetch(rows.subspan(start,count),{},out.subspan(start*size_t(asset_.width),count*size_t(asset_.width)),owner,why,stop);
            if(status!=RAD_OK)return status;
            start+=count;
        }
        if(why)why->clear();
        return RAD_OK;
    }
    RowLookup row_lookup(std::stop_token stop={}) {
        auto owner=shared_from_this();
        return [owner,stop](int layer,std::span<const int64_t> rows,int64_t width,std::span<float> out,std::string* why) {
            if(layer!=owner->asset_.layer_id||width!=owner->asset_.width)
                return refuse(why,RAD_E_SHAPE,"Engram row source layer/width mismatch");
            return owner->copy_rows(rows,out,why,stop);
        };
    }
    static int decode_original(std::span<const uint8_t> packed,int64_t width,std::span<float> out,std::string* why=nullptr) {
        if(width<=0||width%32||uint64_t(width/32)>SIZE_MAX/18||out.size()!=uint64_t(width)||packed.size()!=uint64_t(width/32)*18)
            return refuse(why,RAD_E_SHAPE,"original IQ4_NL row shape mismatch");
        static constexpr int8_t levels[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        for(size_t block=0;block<size_t(width/32);++block) {
            const uint8_t* source=packed.data()+block*18;
            const float scale=half(uint16_t(source[0])|(uint16_t(source[1])<<8));
            for(size_t j=0;j<16;++j){out[block*32+j]=scale*float(levels[source[2+j]&15]);out[block*32+j+16]=scale*float(levels[source[2+j]>>4]);}
        }
        if(why)why->clear();
        return RAD_OK;
    }
private:
    friend class Iq4NlRowLease;
    static constexpr size_t none=SIZE_MAX;
    Iq4NlStore(Iq4NlAsset asset,Iq4NlCacheOptions options,int fd,Epoch epoch,size_t row_bytes)
        :asset_(std::move(asset)),options_(std::move(options)),fd_(fd),epoch_(epoch),row_bytes_(row_bytes){}
    static float half(uint16_t h) {
        uint32_t sign=uint32_t(h&0x8000)<<16,exponent=(h>>10)&31,mantissa=h&1023;
        if(exponent==0){if(!mantissa)return std::bit_cast<float>(sign);int e=-14;while(!(mantissa&1024)){mantissa<<=1;--e;}return std::bit_cast<float>(sign|(uint32_t(e+127)<<23)|((mantissa&1023)<<13));}
        if(exponent==31)return std::bit_cast<float>(sign|0x7f800000u|(mantissa<<13));
        return std::bit_cast<float>(sign|((exponent+112)<<23)|(mantissa<<13));
    }
    static bool add(size_t* cursor,size_t bytes,size_t alignment,size_t* offset) {
        if(*cursor>SIZE_MAX-(alignment-1))return false;
        const size_t begin=(*cursor+alignment-1)&~(alignment-1);
        if(bytes>SIZE_MAX-begin)return false;
        *offset=begin;*cursor=begin+bytes;return true;
    }
    bool layout(size_t capacity,Layout* out)const {
        if(!capacity||capacity>SIZE_MAX/2)return false;
        Layout x;x.capacity=capacity;x.bucket_count=2;
        while(x.bucket_count<capacity*2){if(x.bucket_count>SIZE_MAX/2)return false;x.bucket_count*=2;}
        auto multiply=[](size_t a,size_t b,size_t* value){if(a&&b>SIZE_MAX/a)return false;*value=a*b;return true;};
        size_t n=0,cursor=0;
        if(!multiply(capacity,sizeof(Slot),&n)||!add(&cursor,n,alignof(Slot),&x.slots)||
           !multiply(x.bucket_count,sizeof(Bucket),&n)||!add(&cursor,n,alignof(Bucket),&x.buckets)||
           !multiply(options_.max_batch_rows,sizeof(Request),&n)||!add(&cursor,n,alignof(Request),&x.requests)||
           !multiply(capacity,row_bytes_,&n)||!add(&cursor,n,16,&x.payload)||
           !multiply(options_.max_io_rows,row_bytes_,&n)||!add(&cursor,n,16,&x.scratch))return false;
        x.total=cursor;*out=x;return true;
    }
    int initialize(std::string* why) {
        // Conservative fixed allowance covers control block and allocation bookkeeping.
        const size_t fixed=sizeof(*this)+asset_.path.capacity()+asset_.sha256.capacity()+asset_.format.capacity()+3+256;
        if(options_.budget_bytes<=fixed)return refuse(why,RAD_E_NOMEM,"Engram budget cannot hold cache controls");
        const size_t available=size_t(options_.budget_bytes-fixed);
        size_t high=std::min(size_t(asset_.rows),available/std::max(size_t(1),row_bytes_));
        if(options_.max_cached_rows)high=std::min(high,options_.max_cached_rows);
        size_t low=0;
        while(low<high){const size_t middle=low+(high-low+1)/2;Layout x;if(layout(middle,&x)&&x.total<=available)low=middle;else high=middle-1;}
        Layout x;if(!low||!layout(low,&x))return refuse(why,RAD_E_NOMEM,"Engram budget cannot hold one row and its bounded workspace");
        slab_.reset(static_cast<std::byte*>(::operator new[](x.total,std::align_val_t(alignof(std::max_align_t)))));
        slots_=reinterpret_cast<Slot*>(slab_.get()+x.slots);buckets_=reinterpret_cast<Bucket*>(slab_.get()+x.buckets);
        requests_=reinterpret_cast<Request*>(slab_.get()+x.requests);payload_=reinterpret_cast<uint8_t*>(slab_.get()+x.payload);scratch_=reinterpret_cast<uint8_t*>(slab_.get()+x.scratch);
        for(size_t i=0;i<x.capacity;++i)std::construct_at(slots_+i);
        for(size_t i=0;i<x.bucket_count;++i)std::construct_at(buckets_+i);
        for(size_t i=0;i<options_.max_batch_rows;++i)std::construct_at(requests_+i);
        capacity_=x.capacity;bucket_count_=x.bucket_count;
        stats_.allocation_bytes=x.total+fixed;stats_.slab_bytes=x.total;stats_.fixed_bytes=fixed;
        stats_.payload_bytes=capacity_*row_bytes_;stats_.io_scratch_bytes=options_.max_io_rows*row_bytes_;
        stats_.metadata_bytes=x.total-stats_.payload_bytes-stats_.io_scratch_bytes;stats_.capacity_rows=capacity_;
        return RAD_OK;
    }
    int check_epoch()const {
        struct stat st{};if(::fstat(fd_,&st)<0)return RAD_E_IO;return epoch_.same(st)?RAD_OK:RAD_E_FORMAT;
    }
    int fail_count(int status,const char* text,std::string* why) {
        {std::lock_guard lock(mutex_);++stats_.failures;}return refuse(why,status,text);
    }
    // A bounded stack of row pins lets the actual RowLookup hot path bypass an
    // unrelated slow cold batch. No payload/workspace heap allocation is needed.
    int copy_hot(std::span<const int64_t> rows,std::span<float> out,bool* handled,std::string* why,std::stop_token stop) {
        *handled=false;assert(rows.size()<=64);
        if(stop.stop_requested())return fail_count(RAD_E_CANCELLED,"Engram request was cancelled",why);
        int status=check_epoch();
        if(status!=RAD_OK)return fail_count(status,"Engram file identity/epoch changed or cannot be read",why);
        std::array<size_t,64> pinned{};
        std::unique_lock lock(mutex_);
        if(cancelled_||stop.stop_requested()){++stats_.failures;return refuse(why,RAD_E_CANCELLED,"Engram store or request is cancelled");}
        for(size_t i=0;i<rows.size();++i){
            const size_t found=find(rows[i]);
            if(found==none||slots_[found].state!=State::ready){
                for(size_t j=0;j<i;++j)unpin(pinned[j]);
                changed_.notify_all();return RAD_OK;
            }
            pinned[i]=found;pin(found);slots_[found].referenced=true;
        }
        stats_.requests+=rows.size();
        for(size_t i=0;i<rows.size();++i){
            bool duplicate=false;for(size_t j=0;j<i;++j)duplicate=duplicate||rows[j]==rows[i];
            if(duplicate)++stats_.duplicate_rows;else ++stats_.hits;
        }
        lock.unlock();
        for(size_t i=0;i<rows.size();++i)
            decode_original(std::span(payload_+pinned[i]*row_bytes_,row_bytes_),asset_.width,
                            out.subspan(i*size_t(asset_.width),size_t(asset_.width)));
        status=check_epoch();lock.lock();
        for(size_t i=0;i<rows.size();++i)unpin(pinned[i]);
        if(cancelled_||stop.stop_requested())status=RAD_E_CANCELLED;
        if(status!=RAD_OK)++stats_.failures;
        changed_.notify_all();lock.unlock();*handled=true;
        if(status!=RAD_OK)return refuse(why,status,"Engram hot row copy was cancelled or source changed");
        return RAD_OK;
    }
    static uint64_t hash(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
    size_t find(int64_t row)const {
        const size_t start=size_t(hash(uint64_t(row)))&(bucket_count_-1);
        for(size_t i=0;i<bucket_count_;++i){const auto& b=buckets_[(start+i)&(bucket_count_-1)];if(b.row==-1)return none;if(b.row==row)return b.slot;}return none;
    }
    void insert(int64_t row,size_t slot) {
        const size_t start=size_t(hash(uint64_t(row)))&(bucket_count_-1);size_t tomb=none;
        for(size_t i=0;i<bucket_count_;++i){const size_t at=(start+i)&(bucket_count_-1);auto& b=buckets_[at];
            if(b.row==-2&&tomb==none)tomb=at;
            if(b.row==-1){const size_t use=tomb==none?at:tomb;if(tomb!=none)--tombstones_;buckets_[use]={row,slot};return;}}
        assert(tomb!=none);--tombstones_;buckets_[tomb]={row,slot};
    }
    void erase(int64_t row) {
        const size_t start=size_t(hash(uint64_t(row)))&(bucket_count_-1);
        for(size_t i=0;i<bucket_count_;++i){auto& b=buckets_[(start+i)&(bucket_count_-1)];if(b.row==row){b.row=-2;++tombstones_;return;}if(b.row==-1)break;}assert(false);
    }
    void rebuild(){for(size_t i=0;i<bucket_count_;++i)buckets_[i].row=-1;tombstones_=0;for(size_t i=0;i<capacity_;++i)if(slots_[i].state!=State::empty)insert(slots_[i].row,i);}
    void pin(size_t slot){auto& s=slots_[slot];if(s.references++==0)++stats_.pinned_rows;++stats_.active_references;}
    void unpin(size_t slot){auto& s=slots_[slot];assert(s.references);if(--s.references==0)--stats_.pinned_rows;--stats_.active_references;}
    size_t reserve(int64_t row) {
        if(tombstones_>bucket_count_/4)rebuild();
        for(size_t tries=0;tries<capacity_*2;++tries){const size_t at=clock_;clock_=(clock_+1)%capacity_;auto& s=slots_[at];
            if(s.state==State::loading||s.references)continue;
            if(s.state==State::ready&&s.referenced){s.referenced=false;continue;}
            if(s.state==State::ready){erase(s.row);--stats_.resident_rows;++stats_.evictions;}
            s.row=row;s.state=State::loading;s.referenced=true;++s.generation;pin(at);insert(row,at);
            ++stats_.loading_rows;stats_.peak_loading_rows=std::max(stats_.peak_loading_rows,stats_.loading_rows);return at;
        }return none;
    }
    void discard(size_t slot){auto& s=slots_[slot];assert(s.state==State::loading&&s.references==0);erase(s.row);s.state=State::empty;s.row=-1;--stats_.loading_rows;}
    void release(size_t slot,uint64_t generation,int64_t row)noexcept {
        std::lock_guard lock(mutex_);assert(slot<capacity_&&slots_[slot].generation==generation&&slots_[slot].row==row);unpin(slot);changed_.notify_all();
    }
    void unwind(size_t count) {
        for(size_t i=0;i<count;++i)unpin(requests_[i].slot);
        for(size_t i=0;i<count;++i)if(requests_[i].reserved)discard(requests_[i].slot);
    }
    struct WorkspaceGuard {
        Iq4NlStore* owner;
        ~WorkspaceGuard(){std::lock_guard lock(owner->mutex_);owner->workspace_busy_=false;owner->changed_.notify_all();}
    };
    int read_exact(uint64_t offset,size_t bytes,std::stop_token stop) {
        size_t done=0;
        while(done<bytes){
            {std::lock_guard lock(mutex_);if(cancelled_||stop.stop_requested())return RAD_E_CANCELLED;}
            const auto got=options_.backend?options_.backend->read(fd_,scratch_+done,bytes-done,off_t(offset+done)):
                                                     ::pread(fd_,scratch_+done,bytes-done,off_t(offset+done));
            const int error=got<0?errno:0;
            {std::lock_guard lock(mutex_);++stats_.pread_calls;if(got>0&&uint64_t(got)<=bytes-done)stats_.bytes_read+=uint64_t(got);}
            if(got<0&&error==EINTR)continue;
            if(got<0&&error==ECANCELED)return RAD_E_CANCELLED;
            if(got<=0||uint64_t(got)>bytes-done)return RAD_E_IO;
            done+=size_t(got);
        }return RAD_OK;
    }
    int fetch(std::span<const int64_t> rows,std::span<Iq4NlRowLease> leases,std::span<float> decoded,
                std::shared_ptr<Iq4NlStore> owner,std::string* why,std::stop_token stop={}) {
        if(rows.size()>options_.max_batch_rows)return refuse(why,RAD_E_FULL,"Engram batch exceeds bounded request workspace");
        for(auto row:rows)if(row<0||row>=asset_.rows)return refuse(why,RAD_E_SHAPE,"Engram row is out of range");
        if(rows.empty()){if(why)why->clear();return RAD_OK;}
        std::stop_callback wake(stop,[this]{std::lock_guard lock(mutex_);changed_.notify_all();});
        std::unique_lock lock(mutex_);
        if(workspace_busy_&&!cancelled_&&!stop.stop_requested())++stats_.waits;
        changed_.wait(lock,[&]{return !workspace_busy_||cancelled_||stop.stop_requested();});
        if(cancelled_||stop.stop_requested()){++stats_.failures;return refuse(why,RAD_E_CANCELLED,"Engram store or request is cancelled");}
        workspace_busy_=true;lock.unlock();WorkspaceGuard guard{this};
        int status=check_epoch();if(status!=RAD_OK)return fail_count(status,"Engram file identity/epoch changed or cannot be read",why);
        for(size_t i=0;i<rows.size();++i)requests_[i]={rows[i],i,0,false};
        std::sort(requests_,requests_+rows.size(),[](const auto& a,const auto& b){return a.row!=b.row?a.row<b.row:a.original<b.original;});
        lock.lock();stats_.requests+=rows.size();size_t prepared=0;
        if(cancelled_||stop.stop_requested())status=RAD_E_CANCELLED;
        for(;prepared<rows.size()&&status==RAD_OK;++prepared){auto& request=requests_[prepared];
            if(prepared&&request.row==requests_[prepared-1].row){request.slot=requests_[prepared-1].slot;pin(request.slot);++stats_.duplicate_rows;continue;}
            const size_t cached=find(request.row);
            if(cached!=none){assert(slots_[cached].state==State::ready);request.slot=cached;pin(cached);slots_[cached].referenced=true;++stats_.hits;}
            else {request.slot=reserve(request.row);if(request.slot==none){status=RAD_E_FULL;break;}request.reserved=true;++stats_.misses;}
        }
        if(status!=RAD_OK){unwind(prepared);++stats_.failures;changed_.notify_all();lock.unlock();return refuse(why,status,"Engram cache capacity is pinned, batch is too large, or work was cancelled");}
        lock.unlock();
        // Sort order makes duplicates and adjacent requested misses explicit. No unrequested
        // gap rows are read, so coalescing does not introduce speculative I/O amplification.
        for(size_t i=0;i<rows.size()&&status==RAD_OK;){
            if(!requests_[i].reserved){++i;continue;}
            const size_t begin=i;size_t count=1;int64_t last=requests_[i].row;++i;
            while(i<rows.size()){
                if(requests_[i].row==last){++i;continue;}
                if(count==options_.max_io_rows||!requests_[i].reserved||requests_[i].row!=last+1)break;
                last=requests_[i].row;++count;++i;
            }
            {std::lock_guard io_lock(mutex_);stats_.in_flight_io_bytes=count*row_bytes_;stats_.peak_in_flight_io_bytes=std::max(stats_.peak_in_flight_io_bytes,stats_.in_flight_io_bytes);}
            status=read_exact(uint64_t(asset_.data_offset)+uint64_t(requests_[begin].row)*row_bytes_,count*row_bytes_,stop);
            {std::lock_guard io_lock(mutex_);stats_.in_flight_io_bytes=0;}
            if(status!=RAD_OK)break;
            for(size_t j=begin;j<i;++j)if(requests_[j].reserved)
                std::memcpy(payload_+requests_[j].slot*row_bytes_,scratch_+size_t(requests_[j].row-requests_[begin].row)*row_bytes_,row_bytes_);
        }
        if(status==RAD_OK)status=check_epoch();
        lock.lock();if(cancelled_||stop.stop_requested())status=RAD_E_CANCELLED;
        if(status!=RAD_OK){unwind(rows.size());++stats_.failures;changed_.notify_all();lock.unlock();return refuse(why,status,"Engram read failed, was cancelled or source file changed");}
        for(size_t i=0;i<rows.size();++i)if(requests_[i].reserved){auto& s=slots_[requests_[i].slot];s.state=State::ready;--stats_.loading_rows;++stats_.resident_rows;++stats_.loaded_rows;}
        changed_.notify_all();lock.unlock();
        for(size_t i=0;i<rows.size();++i){const auto& request=requests_[i];const auto& slot=slots_[request.slot];
            if(!leases.empty())leases[request.original]=Iq4NlRowLease(owner,request.slot,slot.generation,request.row);
            else decode_original(std::span(payload_+request.slot*row_bytes_,row_bytes_),asset_.width,
                                 decoded.subspan(request.original*size_t(asset_.width),size_t(asset_.width)));
        }
        if(leases.empty()){
            lock.lock();for(size_t i=0;i<rows.size();++i)unpin(requests_[i].slot);
            if(cancelled_||stop.stop_requested()){status=RAD_E_CANCELLED;++stats_.failures;}
            changed_.notify_all();lock.unlock();
            if(status!=RAD_OK)return refuse(why,status,"Engram row copy was cancelled after loading");
        }
        if(why)why->clear();
        return RAD_OK;
    }
    Iq4NlAsset asset_;Iq4NlCacheOptions options_;int fd_=-1;Epoch epoch_;size_t row_bytes_=0;
    std::unique_ptr<std::byte[],SlabDelete> slab_;
    Slot* slots_=nullptr;Bucket* buckets_=nullptr;Request* requests_=nullptr;uint8_t* payload_=nullptr;uint8_t* scratch_=nullptr;
    size_t capacity_=0,bucket_count_=0,tombstones_=0,clock_=0;
    mutable std::mutex mutex_;std::condition_variable changed_;bool workspace_busy_=false,cancelled_=false;
    Iq4NlCacheStats stats_;
};
inline void Iq4NlRowLease::reset()noexcept {if(owner_){owner_->release(slot_,generation_,row_);owner_.reset();}}
inline std::span<const uint8_t> Iq4NlRowLease::packed()const {return owner_?std::span(owner_->payload_+slot_*owner_->row_bytes_,owner_->row_bytes_):std::span<const uint8_t>{};}
inline int Iq4NlRowLease::decode_into(std::span<float> out,std::string* why)const {
    if(!owner_)return refuse(why,RAD_E_STATE,"empty Engram row lease");
    return Iq4NlStore::decode_original(packed(),owner_->asset_.width,out,why);
}

// One owner/budget for every Engram layer. The same sidecar file may supply two
// different extents: layer and geometry remain separate cache namespaces. For the
// pinned model a 2-GiB total cap is divided across layers1 and14, not applied twice.
class Iq4NlLayerStores:public std::enable_shared_from_this<Iq4NlLayerStores> {
public:
    struct Budget {uint64_t limit=0,allocation=0,fixed=0;size_t stores=0;};
    static int open(std::string model_identity,std::vector<Iq4NlAsset> assets,uint64_t total_budget,
                     Iq4NlCacheOptions common,std::shared_ptr<Iq4NlLayerStores>* out,std::string* why=nullptr) {
        if(!out||model_identity.empty()||model_identity.size()>256||assets.empty()||assets.size()>64)
            return refuse(why,RAD_E_INVAL,"invalid Engram layer-store identity or layer list");
        for(size_t i=0;i<assets.size();++i)for(size_t j=0;j<i;++j)
            if(assets[i].layer_id==assets[j].layer_id)return refuse(why,RAD_E_DUPLICATE,"duplicate Engram layer store");
        try {
            auto group=std::shared_ptr<Iq4NlLayerStores>(new Iq4NlLayerStores);
            group->identity_=std::move(model_identity);group->stores_.reserve(assets.size());
            const uint64_t fixed=sizeof(*group)+group->identity_.capacity()+1+
                                 group->stores_.capacity()*sizeof(std::shared_ptr<Iq4NlStore>)+256;
            if(total_budget<=fixed)return refuse(why,RAD_E_NOMEM,"Engram total budget cannot hold layer controls");
            const uint64_t available=total_budget-fixed,each=available/assets.size(),extra=available%assets.size();
            group->budget_={total_budget,fixed,fixed,assets.size()};
            for(size_t i=0;i<assets.size();++i){
                auto options=common;options.budget_bytes=each+(i<extra);
                std::shared_ptr<Iq4NlStore> store;const int status=Iq4NlStore::open(std::move(assets[i]),std::move(options),&store,why);
                if(status!=RAD_OK)return status;
                group->budget_.allocation+=store->stats().allocation_bytes;group->stores_.push_back(std::move(store));
            }
            assert(group->budget_.allocation<=total_budget);
            *out=std::move(group);if(why)why->clear();
            return RAD_OK;
        } catch(const std::bad_alloc&){return refuse(why,RAD_E_NOMEM,"cannot allocate Engram layer controls");}
    }
    const std::string& identity()const{return identity_;}
    Budget budget()const{return budget_;}
    std::span<const std::shared_ptr<Iq4NlStore>> stores()const{return stores_;}
    void cancel(){for(const auto& store:stores_)store->cancel();}
    RowLookup row_lookup(std::stop_token stop={}){
        auto owner=shared_from_this();
        return [owner,stop](int layer,std::span<const int64_t> rows,int64_t width,std::span<float> out,std::string* why) {
            for(const auto& store:owner->stores_)if(store->asset().layer_id==layer){
                if(width!=store->asset().width)return refuse(why,RAD_E_SHAPE,"Engram layer row width mismatch");
                return store->copy_rows(rows,out,why,stop);
            }
            return refuse(why,RAD_E_NOTFOUND,"no Engram asset for the requested layer");
        };
    }
private:
    std::string identity_;std::vector<std::shared_ptr<Iq4NlStore>> stores_;Budget budget_;
};
} // namespace rad::deepseek41::engram
