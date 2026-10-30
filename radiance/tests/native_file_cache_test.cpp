/* Pure HOST controls for the real byte-owning NativeFileCache. No Mover, file,
 * model, source capability, driver, GPU stream or device work is constructed. */
#include "rad_test.h"
#include "place/native_file_cache.h"
#include "rad_types.h"

#include <array>
#include <atomic>
#include <climits>
#include <cstdint>
#include <thread>
#include <vector>

namespace {
using rad::NativeFileCache;
using Bytes = std::vector<uint8_t>;
constexpr int64_t setup_budget = 1 << 20;
constexpr uint8_t untouched = 0xd7;
constexpr size_t guard = 16;

Bytes payload(size_t identity, size_t n) {
    Bytes out(n);
    for (size_t i=0;i<n;++i) out[i]=uint8_t((identity*47+i*19+(i>>3))&255);
    return out;
}
void charge_bounds(const NativeFileCache::Stats& s) {
    CHECK(s.metadata_bytes>=0); CHECK(s.payload_charge>=0);
    CHECK(s.peak_charge>=s.metadata_bytes+s.payload_charge);
    CHECK(s.metadata_bytes+s.payload_charge<=s.budget_bytes);
    CHECK(s.peak_charge<=s.budget_bytes);
}
/* Choose external capacity boundaries using observed public charges, without
 * copying private Entry layout or the allocation-rounding implementation. */
struct Charge { int64_t metadata, payload; };
Charge observed_charge(size_t count, const Bytes& src) {
    NativeFileCache c;
    if(c.configure(setup_budget,count)!=RAD_OK || c.insert(0,src.data(),int64_t(src.size()))!=RAD_OK)
        throw std::runtime_error("cannot establish small cache test charge");
    const auto s=c.stats();
    if(s.entries!=1 || s.metadata_bytes<=0 || s.payload_charge<int64_t(src.size()))
        throw std::runtime_error("cache did not own its bounded setup payload");
    charge_bounds(s); return {s.metadata_bytes,s.payload_charge};
}
void expect_copy(NativeFileCache& c,size_t key,const Bytes& expected,bool wanted) {
    Bytes dst(expected.size()+2*guard,untouched);const auto before=dst;bool hit=false;
    CHECK_EQ(c.copy(key,int64_t(expected.size()),dst.data()+guard,int64_t(expected.size()),&hit),RAD_OK);
    CHECK_EQ(hit,wanted);
    if(wanted) {
        CHECK(std::equal(expected.begin(),expected.end(),dst.begin()+guard));
        CHECK(std::all_of(dst.begin(),dst.begin()+guard,[](auto x){return x==untouched;}));
        CHECK(std::all_of(dst.end()-guard,dst.end(),[](auto x){return x==untouched;}));
    } else CHECK(dst==before);
}
/* At most eight owned threads. A partial std::thread-construction failure opens
 * the start gate and joins every thread already made before the exception leaves. */
struct Threads {
    std::array<std::thread,8> values;
    size_t count=0;
    std::atomic<bool> go{false};
    ~Threads(){join();}
    void join(){
        go.store(true,std::memory_order_release);
        for(size_t i=0;i<count;++i)if(values[i].joinable())values[i].join();
    }
};
}

TEST(native_cache_disabled_and_configuration_refusals_clear_namespace) {
    NativeFileCache c;const auto a=payload(1,71);Bytes dst(a.size(),untouched);bool hit=true;
    CHECK_EQ(c.configure(0,0),RAD_OK);
    CHECK_EQ(c.insert(999,a.data(),int64_t(a.size())),RAD_OK);
    CHECK_EQ(c.copy(999,int64_t(a.size()),dst.data(),int64_t(dst.size()),&hit),RAD_OK);
    CHECK(!hit);CHECK(std::all_of(dst.begin(),dst.end(),[](auto x){return x==untouched;}));
    auto s=c.stats();CHECK_EQ(s.budget_bytes,0);CHECK_EQ(s.entries,size_t(0));CHECK_EQ(s.hits,uint64_t(0));CHECK_EQ(s.misses,uint64_t(0));
    REQUIRE_EQ(c.configure(setup_budget,4),RAD_OK);
    REQUIRE_EQ(c.insert(1,a.data(),int64_t(a.size())),RAD_OK);
    CHECK_EQ(c.configure(-1,4),RAD_E_INVAL);
    CHECK_EQ(c.stats().entries,size_t(0));CHECK_EQ(c.stats().budget_bytes,0);
    CHECK_EQ(c.configure(1024,0),RAD_E_INVAL);
    CHECK_EQ(c.configure(INT64_MAX,SIZE_MAX),RAD_E_INVAL);
    CHECK_EQ(c.configure(1,1),RAD_E_SCRATCH);
    CHECK_EQ(c.stats().entries,size_t(0));CHECK_EQ(c.stats().metadata_bytes,0);
}

TEST(native_cache_variable_size_lru_respects_hits_and_budget) {
    const auto a=payload(1,37),b=payload(2,139),cbytes=payload(3,521),d=payload(4,83);
    const auto qa=observed_charge(8,a),qb=observed_charge(8,b),qc=observed_charge(8,cbytes),qd=observed_charge(8,d);
    REQUIRE(qa.metadata==qb.metadata&&qb.metadata==qc.metadata&&qc.metadata==qd.metadata);
    REQUIRE(qd.payload>qa.payload&&qd.payload<=qb.payload); // makes the victim unambiguous
    NativeFileCache c;REQUIRE_EQ(c.configure(qa.metadata+qa.payload+qb.payload+qc.payload,8),RAD_OK);
    for(const auto& x:std::array<std::pair<size_t,const Bytes*>,3>{{{0,&a},{1,&b},{2,&cbytes}}})
        REQUIRE_EQ(c.insert(x.first,x.second->data(),int64_t(x.second->size())),RAD_OK);
    expect_copy(c,0,a,true); // order is A,C,B: B alone has room for D
    REQUIRE_EQ(c.insert(3,d.data(),int64_t(d.size())),RAD_OK);
    auto s=c.stats();CHECK_EQ(s.entries,size_t(3));CHECK_EQ(s.inserts,uint64_t(4));CHECK_EQ(s.evictions,uint64_t(1));charge_bounds(s);
    expect_copy(c,1,b,false);expect_copy(c,0,a,true);expect_copy(c,2,cbytes,true);expect_copy(c,3,d,true);
    s=c.stats();CHECK_EQ(s.hits,uint64_t(4));CHECK_EQ(s.misses,uint64_t(1));CHECK_EQ(s.hit_bytes,uint64_t(2*a.size()+cbytes.size()+d.size()));CHECK_EQ(s.miss_bytes,uint64_t(b.size()));charge_bounds(s);
}

TEST(native_cache_oversize_bypass_does_not_evict_resident_payload) {
    const auto small=payload(5,91),large=payload(6,4097);const auto q=observed_charge(3,small);
    NativeFileCache c;REQUIRE_EQ(c.configure(q.metadata+q.payload,3),RAD_OK);
    REQUIRE_EQ(c.insert(0,small.data(),int64_t(small.size())),RAD_OK);const auto before=c.stats();
    REQUIRE_EQ(c.insert(1,large.data(),int64_t(large.size())),RAD_OK);
    auto s=c.stats();CHECK_EQ(s.bypasses,uint64_t(1));CHECK_EQ(s.evictions,before.evictions);CHECK_EQ(s.entries,before.entries);CHECK_EQ(s.payload_charge,before.payload_charge);CHECK_EQ(s.inserts,before.inserts);
    expect_copy(c,0,small,true);expect_copy(c,1,large,false);charge_bounds(c.stats());
}

TEST(native_cache_miss_and_invalid_copy_never_partially_change_destination) {
    const auto a=payload(7,129);NativeFileCache c;REQUIRE_EQ(c.configure(setup_budget,2),RAD_OK);
    expect_copy(c,0,a,false);REQUIRE_EQ(c.insert(0,a.data(),int64_t(a.size())),RAD_OK);
    Bytes dst(a.size()+2*guard,untouched);const auto before=dst;bool hit=true;
    CHECK_EQ(c.copy(0,int64_t(a.size()),dst.data()+guard,int64_t(a.size()-1),&hit),RAD_E_INVAL);CHECK(dst==before);
    CHECK_EQ(c.copy(0,int64_t(a.size()-1),dst.data()+guard,int64_t(a.size()),&hit),RAD_E_STATE);CHECK(!hit);CHECK(dst==before);
    CHECK_EQ(c.copy(2,int64_t(a.size()),dst.data()+guard,int64_t(a.size()),&hit),RAD_E_INVAL);CHECK(dst==before);
    CHECK_EQ(c.copy(0,0,dst.data()+guard,int64_t(a.size()),&hit),RAD_E_INVAL);CHECK(dst==before);
    CHECK_EQ(c.copy(0,int64_t(a.size()),nullptr,int64_t(a.size()),&hit),RAD_E_INVAL);
    CHECK_EQ(c.copy(0,int64_t(a.size()),dst.data()+guard,int64_t(a.size()),nullptr),RAD_E_INVAL);CHECK(dst==before);
    CHECK_EQ(c.insert(2,a.data(),int64_t(a.size())),RAD_E_INVAL);
    CHECK_EQ(c.insert(0,nullptr,int64_t(a.size())),RAD_E_INVAL);
    CHECK_EQ(c.insert(0,a.data(),0),RAD_E_INVAL);
    CHECK_EQ(c.insert(0,a.data(),INT64_MAX),RAD_E_INVAL);
    expect_copy(c,0,a,true);CHECK_EQ(c.stats().misses,uint64_t(1));CHECK_EQ(c.stats().hits,uint64_t(1));charge_bounds(c.stats());
}

TEST(native_cache_duplicate_insert_is_idempotent_and_touches_lru) {
    const auto a=payload(8,81),b=payload(9,81),d=payload(10,81);const auto q=observed_charge(3,a);
    NativeFileCache c;REQUIRE_EQ(c.configure(q.metadata+2*q.payload,3),RAD_OK);
    REQUIRE_EQ(c.insert(0,a.data(),int64_t(a.size())),RAD_OK);REQUIRE_EQ(c.insert(1,b.data(),int64_t(b.size())),RAD_OK);
    const auto before=c.stats();REQUIRE_EQ(c.insert(0,a.data(),int64_t(a.size())),RAD_OK);
    auto s=c.stats();CHECK_EQ(s.inserts,before.inserts);CHECK_EQ(s.payload_charge,before.payload_charge);CHECK_EQ(s.entries,before.entries);
    REQUIRE_EQ(c.insert(2,d.data(),int64_t(d.size())),RAD_OK);
    expect_copy(c,1,b,false);expect_copy(c,0,a,true);expect_copy(c,2,d,true);CHECK_EQ(c.stats().evictions,uint64_t(1));charge_bounds(c.stats());
}

TEST(native_cache_reconfigure_is_a_fresh_program_namespace) {
    const auto first=payload(11,127),next=payload(12,127);NativeFileCache c;
    REQUIRE_EQ(c.configure(setup_budget,4),RAD_OK);REQUIRE_EQ(c.insert(1,first.data(),int64_t(first.size())),RAD_OK);expect_copy(c,1,first,true);
    REQUIRE_EQ(c.configure(setup_budget,4),RAD_OK);
    auto s=c.stats();CHECK_EQ(s.entries,size_t(0));CHECK_EQ(s.inserts,uint64_t(0));CHECK_EQ(s.hits,uint64_t(0));CHECK_EQ(s.misses,uint64_t(0));CHECK_EQ(s.payload_charge,0);CHECK_EQ(s.peak_charge,s.metadata_bytes);
    expect_copy(c,1,next,false);REQUIRE_EQ(c.insert(1,next.data(),int64_t(next.size())),RAD_OK);expect_copy(c,1,next,true);
    c.clear();s=c.stats();CHECK_EQ(s.budget_bytes,0);CHECK_EQ(s.metadata_bytes,0);CHECK_EQ(s.payload_charge,0);CHECK_EQ(s.entries,size_t(0));expect_copy(c,1,next,false);
}

TEST(native_cache_copy_is_owned_after_source_mutation_eviction_and_clear) {
    auto src=payload(13,257);const auto expected=src,other=payload(14,257);const auto q=observed_charge(2,src);
    NativeFileCache c;REQUIRE_EQ(c.configure(q.metadata+q.payload,2),RAD_OK);REQUIRE_EQ(c.insert(0,src.data(),int64_t(src.size())),RAD_OK);
    std::fill(src.begin(),src.end(),0); // insert owns bytes, not this caller's storage
    Bytes dst(expected.size()+2*guard,untouched);bool hit=false;
    REQUIRE_EQ(c.copy(0,int64_t(expected.size()),dst.data()+guard,int64_t(expected.size()),&hit),RAD_OK);REQUIRE(hit);
    REQUIRE_EQ(c.insert(1,other.data(),int64_t(other.size())),RAD_OK);CHECK_EQ(c.stats().evictions,uint64_t(1));
    expect_copy(c,0,expected,false);c.clear();
    CHECK(std::equal(expected.begin(),expected.end(),dst.begin()+guard));
    CHECK(std::all_of(dst.begin(),dst.begin()+guard,[](auto x){return x==untouched;}));CHECK(std::all_of(dst.end()-guard,dst.end(),[](auto x){return x==untouched;}));
}

TEST(native_cache_eight_threads_immutable_payload_pressure) {
    constexpr size_t keys=32,rounds=512,max_bytes=513;
    std::array<Bytes,keys> prepared;
    for(size_t k=0;k<keys;++k)prepared[k]=payload(100+k,65+(k*53)%449);
    const auto catalogue=std::move(prepared);const auto original=catalogue;
    const auto q=observed_charge(keys,catalogue[0]);NativeFileCache cache;
    REQUIRE_EQ(cache.configure(q.metadata+2048,keys),RAD_OK);
    std::atomic<uint64_t> errors{0},finished{0};Threads workers;
    for(size_t t=0;t<workers.values.size();++t) {
        workers.values[t]=std::thread([&,t]{
            while(!workers.go.load(std::memory_order_acquire))std::this_thread::yield();
            try {
                std::array<uint8_t,max_bytes+2*guard> dst;
                for(size_t j=0;j<rounds;++j) {
                    const size_t key=(j*7+t*11)%keys;const auto& expected=catalogue[key];
                    for(int attempt=0;attempt<2;++attempt) {
                        dst.fill(untouched);bool hit=false;
                        const int rc=cache.copy(key,int64_t(expected.size()),dst.data()+guard,int64_t(expected.size()),&hit);
                        bool good=rc==RAD_OK;
                        if(hit)good=good&&std::equal(expected.begin(),expected.end(),dst.begin()+guard);
                        else good=good&&std::all_of(dst.begin()+guard,dst.begin()+guard+expected.size(),[](auto x){return x==untouched;});
                        good=good&&std::all_of(dst.begin(),dst.begin()+guard,[](auto x){return x==untouched;})&&
                            std::all_of(dst.begin()+guard+expected.size(),dst.end(),[](auto x){return x==untouched;});
                        if(!good)errors.fetch_add(1,std::memory_order_relaxed);
                        if(!hit&&attempt==0&&cache.insert(key,expected.data(),int64_t(expected.size()))!=RAD_OK)errors.fetch_add(1,std::memory_order_relaxed);
                    }
                }
            } catch(...) {errors.fetch_add(1,std::memory_order_relaxed);}
            finished.fetch_add(1,std::memory_order_relaxed);
        });
        ++workers.count;
    }
    workers.join();CHECK_EQ(finished.load(),uint64_t(8));CHECK_EQ(errors.load(),uint64_t(0));CHECK(catalogue==original);
    const auto s=cache.stats();CHECK_EQ(s.hits+s.misses,uint64_t(8*rounds*2));CHECK(s.inserts>0);CHECK(s.evictions>0);CHECK(s.entries>0&&s.entries<keys);CHECK(s.hit_bytes+s.miss_bytes>=uint64_t(8*rounds*2*65));charge_bounds(s);
}
RAD_TEST_MAIN()
