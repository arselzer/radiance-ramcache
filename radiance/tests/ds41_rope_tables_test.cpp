#include "rad_test.h"
#include "../libds41/ds41_rope_tables.h"
#include <array>
#include <atomic>
#include <latch>
#include <bit>
#include <cfenv>
#include <thread>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace t=rad::ds41::transform;
namespace {
t::RopeSpec spec() {return {4,4096,40000.f,16.f,32.f,1.f};}
rad::deepseek41::Config pinned_config() {
    // Exact relevant metadata from official config.json at HF revision
    // 1bae0e1088993b9b82de08f59de5be5a34e42170; full-file SHA256
    // 8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879.
    // Only eight positions are built below; the model's 1M context is metadata.
    rad::deepseek41::Config c;c.n_layers=40;c.rope_head_dim=64;c.rope_type="yarn";
    c.rope_theta=10000;c.compress_rope_theta=160000;c.rope_factor=16;
    c.original_seq_len=65536;c.beta_fast=32;c.beta_slow=1;
    c.compress_ratios={0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    c.layers.resize(size_t(c.n_layers));
    for(size_t i=0;i<c.layers.size();++i)c.layers[i].compress_ratio=int(c.compress_ratios[i]);
    return c;
}
struct FenvRestore {
    fenv_t env{};
#if defined(__SSE__)
    unsigned csr=_mm_getcsr();
#endif
    FenvRestore() {if(std::fegetenv(&env))throw std::runtime_error("cannot save host floating-point environment");}
    ~FenvRestore() {
        std::fesetenv(&env);
#if defined(__SSE__)
        _mm_setcsr(csr);
#endif
    }
};
}

TEST(ds41_rope_tables_pinned_forty_layer_metadata_uses_exactly_two_tiny_tables) {
    const auto config=pinned_config();t::RopeTableStore store({4096,2});
    std::array<t::RopeTableStore::Lease,40> leases;
    for(int64_t layer=0;layer<config.n_layers;++layer) {
        t::RopeSpec selected;REQUIRE_EQ(t::select_spec(config,layer,&selected),RAD_OK);
        REQUIRE_EQ(store.get(selected,8,&leases[size_t(layer)]),RAD_OK);
        CHECK(leases[size_t(layer)].get()==leases[layer<2?0:2].get());
    }
    CHECK(leases[0].get()!=leases[2].get());
    CHECK_EQ(leases[0]->spec().theta,10000.f);CHECK_EQ(leases[0]->spec().original_context,0);
    CHECK_EQ(leases[2]->spec().theta,160000.f);CHECK_EQ(leases[2]->spec().original_context,65536);
    CHECK_EQ(store.stats().entries,2u);CHECK_EQ(store.stats().payload_bytes,4096u);
    for(size_t i:{size_t(0),size_t(2)}) {
        std::vector<float> original;REQUIRE_EQ(t::build_pairs(leases[i]->spec(),8,2048,&original),RAD_OK);
        CHECK(std::equal(original.begin(),original.end(),leases[i]->pairs().begin(),leases[i]->pairs().end()));
        CHECK_EQ(leases[i]->receipt().positions,8);CHECK_EQ(leases[i]->receipt().rotary,64);
        CHECK_EQ(leases[i]->receipt().bytes,2048);CHECK_EQ(leases[i]->key().algorithm_version,1u);
    }
    CHECK_EQ(store.clear_unleased(),0u);
    leases={};CHECK_EQ(store.clear_unleased(),2u);CHECK_EQ(store.stats().payload_bytes,0u);
}

TEST(ds41_rope_tables_key_preserves_every_spec_bit_positions_and_algorithm_version) {
    t::RopeTableStore store({4096,12});std::array<t::RopeSpec,7> variants;
    variants.fill(spec());variants[1].rotary=6;variants[2].original_context+=1;
    variants[3].theta=std::nextafter(variants[3].theta,INFINITY);
    variants[4].factor=std::nextafter(variants[4].factor,INFINITY);
    variants[5].beta_fast=std::nextafter(variants[5].beta_fast,INFINITY);
    variants[6].beta_slow=std::nextafter(variants[6].beta_slow,INFINITY);
    std::array<t::RopeTableStore::Lease,8> leases;
    for(size_t i=0;i<variants.size();++i) {
        REQUIRE_EQ(store.get(variants[i],8,&leases[i]),RAD_OK);
        for(size_t j=0;j<i;++j)CHECK(leases[i].get()!=leases[j].get());
        CHECK_EQ(leases[i]->key().theta_bits,std::bit_cast<uint32_t>(variants[i].theta));
        CHECK_EQ(leases[i]->key().factor_bits,std::bit_cast<uint32_t>(variants[i].factor));
        CHECK_EQ(leases[i]->key().beta_fast_bits,std::bit_cast<uint32_t>(variants[i].beta_fast));
        CHECK_EQ(leases[i]->key().beta_slow_bits,std::bit_cast<uint32_t>(variants[i].beta_slow));
    }
    REQUIRE_EQ(store.get(spec(),9,&leases.back()),RAD_OK);
    CHECK(leases.back().get()!=leases[0].get());CHECK_EQ(store.stats().entries,8u);
    auto replacement=leases[0];const auto before=store.stats();
    CHECK_EQ(store.get(spec(),8,&replacement,0),RAD_E_UNSUPPORTED);
    CHECK(replacement==leases[0]);CHECK_EQ(store.stats().entries,before.entries);
    CHECK_EQ(store.get(spec(),8,&replacement,UINT32_MAX),RAD_E_UNSUPPORTED);CHECK(replacement==leases[0]);
    REQUIRE_EQ(store.get(spec(),8,&replacement),RAD_OK);CHECK(replacement==leases[0]);
}

TEST(ds41_rope_tables_payload_and_entry_caps_do_not_evict_live_leases_or_mutate_replacement) {
    for(bool entry_cap:{false,true}) {
        t::RopeTableStore store({entry_cap?4096u:128u,entry_cap?1u:8u});
        t::RopeTableStore::Lease first,replacement;REQUIRE_EQ(store.get(spec(),8,&first),RAD_OK);
        replacement=first;auto other=spec();other.theta=50000;
        CHECK_EQ(store.get(other,8,&replacement),RAD_E_NOMEM);CHECK(replacement==first);
        CHECK_EQ(store.stats().entries,1u);CHECK_EQ(store.stats().payload_bytes,128u);
        CHECK_EQ(store.clear_unleased(),0u);first.reset();CHECK_EQ(store.clear_unleased(),0u);
        replacement.reset();CHECK_EQ(store.clear_unleased(),1u);
        CHECK_EQ(store.stats().entries,0u);CHECK_EQ(store.stats().payload_bytes,0u);
        REQUIRE_EQ(store.get(other,8,&replacement),RAD_OK);CHECK_EQ(replacement->spec().theta,50000);
    }
}

TEST(ds41_rope_tables_live_lease_survives_store_destruction_and_retains_charge_until_free) {
    t::RopeTableStore::Lease lease;std::shared_ptr<const t::RopeTableAccounting> ledger;
    std::vector<float> saved;
    {
        t::RopeTableStore store({128,1});ledger=store.accounting();
        REQUIRE_EQ(store.get(spec(),8,&lease),RAD_OK);saved.assign(lease->pairs().begin(),lease->pairs().end());
        CHECK_EQ(ledger->limits().payload_bytes,128u);CHECK_EQ(ledger->limits().max_entries,1u);
    }
    REQUIRE(lease);CHECK_EQ(ledger->stats().entries,1u);CHECK_EQ(ledger->stats().payload_bytes,128u);
    CHECK(std::equal(saved.begin(),saved.end(),lease->pairs().begin(),lease->pairs().end()));
    auto second=lease;lease.reset();CHECK_EQ(ledger->stats().payload_bytes,128u);
    second.reset();CHECK_EQ(ledger->stats().entries,0u);CHECK_EQ(ledger->stats().payload_bytes,0u);
    // A ledger observer retains no table and cannot keep its payload alive.
}

TEST(ds41_rope_tables_concurrent_same_key_has_one_payload_and_one_identity) {
    constexpr size_t n=12;t::RopeTableStore store({128,1});std::latch start{1};
    std::array<t::RopeTableStore::Lease,n> leases;std::array<int,n> codes{};
    {
        std::array<std::jthread,n> threads;
        try {
            for(size_t i=0;i<n;++i)threads[i]=std::jthread([&,i] {
                start.wait();codes[i]=store.get(spec(),8,&leases[i]);
            });
        } catch(...) {
            // Release already-created workers before their jthread destructors
            // join; a fixed-participant barrier would deadlock on thread failure.
            start.count_down();throw;
        }
        start.count_down();
    }
    REQUIRE(leases[0]);
    for(size_t i=0;i<n;++i) {CHECK_EQ(codes[i],RAD_OK);CHECK(leases[i]==leases[0]);}
    CHECK_EQ(store.stats().payload_bytes,128u);CHECK_EQ(store.stats().entries,1u);
    leases={};CHECK_EQ(store.clear_unleased(),1u);CHECK_EQ(store.stats().entries,0u);
}

TEST(ds41_rope_tables_invalid_shapes_overflow_and_caps_are_checked_without_output_replacement) {
    t::RopeTableStore store({128,1});t::RopeTableStore::Lease out;
    REQUIRE_EQ(store.get(spec(),8,&out),RAD_OK);const auto saved=out;
    for(int64_t positions:{int64_t(0),int64_t(-1),INT64_MAX,t::max_positions+1}) {
        CHECK_EQ(store.get(spec(),positions,&out),RAD_E_SHAPE);CHECK(out==saved);
    }
    for(int64_t rotary:{int64_t(0),int64_t(-2),int64_t(3),INT64_MAX,t::max_head_dim+2}) {
        auto bad=spec();bad.rotary=rotary;CHECK_EQ(store.get(bad,8,&out),RAD_E_SHAPE);CHECK(out==saved);
    }
    auto bad=spec();bad.original_context=-1;CHECK_EQ(store.get(bad,8,&out),RAD_E_SHAPE);CHECK(out==saved);
    bad=spec();bad.theta=NAN;CHECK_EQ(store.get(bad,8,&out),RAD_E_INVAL);CHECK(out==saved);
    bad=spec();bad.factor=INFINITY;CHECK_EQ(store.get(bad,8,&out),RAD_E_INVAL);CHECK(out==saved);
    CHECK_EQ(store.get(spec(),8,nullptr),RAD_E_INVAL);CHECK_EQ(store.stats().entries,1u);
    t::RopeTableStore no_bytes({0,UINT64_MAX}),no_entries({UINT64_MAX,0});
    CHECK_EQ(no_bytes.get(spec(),8,&out),RAD_E_NOMEM);CHECK(out==saved);
    CHECK_EQ(no_entries.get(spec(),8,&out),RAD_E_NOMEM);CHECK(out==saved);
    bad=spec();bad.rotary=t::max_head_dim;
    CHECK_EQ(no_bytes.get(bad,t::max_positions,&out),RAD_E_NOMEM);CHECK(out==saved);
    CHECK_EQ(no_bytes.stats().payload_bytes,0u);CHECK_EQ(no_entries.stats().entries,0u);
}

TEST(ds41_rope_tables_cold_arithmetic_refusal_refunds_reservation_and_keeps_output_atomic) {
    t::RopeTableStore store({256,2});t::RopeTableStore::Lease out;
    REQUIRE_EQ(store.get(spec(),8,&out),RAD_OK);const auto saved=out;
    auto bad=spec();bad.factor=std::numeric_limits<float>::denorm_min();
    CHECK_EQ(t::validate_spec(bad),RAD_OK);CHECK(store.get(bad,8,&out)<0);CHECK(out==saved);
    CHECK_EQ(store.stats().entries,1u);CHECK_EQ(store.stats().payload_bytes,128u);
    auto valid=spec();valid.factor=8;REQUIRE_EQ(store.get(valid,8,&out),RAD_OK);
    CHECK(out!=saved);CHECK_EQ(store.stats().entries,2u);CHECK_EQ(store.stats().payload_bytes,256u);
}

TEST(ds41_rope_tables_nondefault_rounding_and_flush_modes_refuse_hits_and_cold_construction) {
    FenvRestore restore;REQUIRE(t::rope_table_environment_supported());
    t::RopeTableStore store({256,2});t::RopeTableStore::Lease out;
    REQUIRE_EQ(store.get(spec(),8,&out),RAD_OK);const auto saved=out;auto cold=spec();cold.theta=50000;
    for(int mode:{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        REQUIRE_EQ(std::fesetround(mode),0);CHECK(!t::rope_table_environment_supported());
        CHECK_EQ(store.get(spec(),8,&out),RAD_E_UNSUPPORTED);CHECK(out==saved);
        CHECK_EQ(store.get(cold,8,&out),RAD_E_UNSUPPORTED);CHECK(out==saved);
    }
    REQUIRE_EQ(std::fesetround(FE_TONEAREST),0);
#if defined(__SSE__)
    const unsigned original=_mm_getcsr();
    for(unsigned flag:{1u<<15,1u<<6}) { // FTZ and DAZ individually.
        _mm_setcsr(original|flag);CHECK(!t::rope_table_environment_supported());
        CHECK_EQ(store.get(spec(),8,&out),RAD_E_UNSUPPORTED);CHECK(out==saved);
        CHECK_EQ(store.get(cold,8,&out),RAD_E_UNSUPPORTED);CHECK(out==saved);
        _mm_setcsr(original);
    }
#endif
    CHECK_EQ(store.stats().entries,1u);CHECK_EQ(store.stats().payload_bytes,128u);
    REQUIRE_EQ(store.get(cold,8,&out),RAD_OK);CHECK(out!=saved);
}
TEST(ds41_rope_tables_weak_promotion_racing_clear_retains_identity_and_exact_quota) {
    constexpr unsigned rounds=1024,stopped=UINT32_MAX;
    t::RopeTableStore store({128,1});std::vector<float> expected;
    REQUIRE_EQ(t::build_pairs(spec(),8,128,&expected),RAD_OK);
    struct Shared {
        std::atomic<unsigned> phase{0};
        std::weak_ptr<const t::RopeTable> weak;
        t::RopeTableStore::Lease promoted;
    } shared;
    auto wait=[&](unsigned target) {
        for(;;) {
            const auto value=shared.phase.load(std::memory_order_acquire);
            if(value==stopped)return false;
            if(value>=target)return true;
            shared.phase.wait(value,std::memory_order_acquire);
        }
    };
    // Only ONE worker is constructed. If construction throws there is no existing
    // participant waiting for it. Early test return/exception requests stop and
    // changes the wait value; notify without a value change would be insufficient.
    std::jthread promoter([&](std::stop_token token) {
        std::stop_callback wake_on_stop(token,[&] {
            shared.phase.store(stopped,std::memory_order_release);
            shared.phase.notify_all();
        });
        for(unsigned round=0;round<rounds;++round) {
            const unsigned base=round*4;
            if(!wait(base+1))return;
            auto held=shared.weak.lock();
            shared.promoted=held;
            unsigned value=base+1;
            if(!shared.phase.compare_exchange_strong(value,base+2,std::memory_order_release))return;
            shared.phase.notify_all();
            if(!wait(base+3))return;
            shared.promoted.reset();held.reset();
            value=base+3;
            if(!shared.phase.compare_exchange_strong(value,base+4,std::memory_order_release))return;
            shared.phase.notify_all();
        }
    });
    unsigned promoted=0,expired=0;
    for(unsigned round=0;round<rounds;++round) {
        const unsigned base=round*4;
        t::RopeTableStore::Lease initial;
        REQUIRE_EQ(store.get(spec(),8,&initial),RAD_OK);
        const auto* identity=initial.get();shared.weak=initial;initial.reset();
        // Endpoint1 deterministically expires the table before the worker locks.
        if(round==1)CHECK_EQ(store.clear_unleased(),1u);
        shared.phase.store(base+1,std::memory_order_release);shared.phase.notify_all();
        if(round==0) {
            // Endpoint0 deterministically holds a promoted lease before clear.
            REQUIRE(wait(base+2));CHECK_EQ(store.clear_unleased(),0u);
        } else if(round>1) {
            // The narrow use_count/reset window is stressed, not claimed to be
            // deterministically forced. Both endpoint orders are covered above.
            if(round&1u)std::this_thread::yield();
            static_cast<void>(store.clear_unleased());
        }
        REQUIRE(wait(base+2));
        auto held=shared.promoted;
        if(held) {++promoted;CHECK(held.get()==identity);}
        else ++expired;
        if(round==0)CHECK(bool(held));
        if(round==1)CHECK(!held);
        t::RopeTableStore::Lease recovered;
        // A live weak-promoted object already consumes the entire quota. Losing
        // its map identity would spuriously fail/rebuild here, which is the bug
        // this race test distinguishes from ordinary same-key construction.
        REQUIRE_EQ(store.get(spec(),8,&recovered),RAD_OK);
        if(held)CHECK(recovered==held);
        CHECK(std::equal(expected.begin(),expected.end(),recovered->pairs().begin(),recovered->pairs().end()));
        CHECK_EQ(store.stats().entries,1u);CHECK_EQ(store.stats().payload_bytes,128u);
        recovered.reset();held.reset();
        shared.phase.store(base+3,std::memory_order_release);shared.phase.notify_all();
        REQUIRE(wait(base+4));
        CHECK_EQ(store.clear_unleased(),1u);
        CHECK(shared.weak.expired());CHECK_EQ(store.stats().entries,0u);CHECK_EQ(store.stats().payload_bytes,0u);
    }
    CHECK(promoted>=1u);CHECK(expired>=1u);
}
RAD_TEST_MAIN()
