#pragma once
// Pure, internal quotes for the sealed synthetic TP1 startup profile. No allocation,
// backend call, model mutation or Engine activation. Not an allocator/RSS quota.
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
namespace rad {
class Program;
struct TinyStorageQuote {
    uint64_t device_bytes=0,pinned_bytes=0,mapped_bytes=0;
    uint64_t device_page_bytes=0,pinned_page_bytes=0,mapped_page_bytes=0;
    uint64_t device_requests=0,pinned_requests=0,mapped_requests=0;
    uint64_t heap_payload_bytes=0,embedded_object_bytes=0;
    // Explicit conservative allowance for container growth/control; not measured malloc cost.
    uint64_t heap_growth_allowance=0,control_allowance=0,event_count=0,stream_count=0;
    uint64_t requested_total=0,page_and_heap_total=0;
};
struct TinySamplerQuote {
    TinyStorageQuote storage;
    uint64_t sizeof_sampler=0,sizeof_sample_params=0,sizeof_couple_entry=0;
    uint64_t rows=0,history=0,candidates=0,mask_words=0,logprob_width=0;
    uint64_t pinned_request_bytes[11]{};
};
struct TinyCtxQuote {
    TinyStorageQuote storage;
    uint64_t sizeof_ctx=0,sizeof_tensor=0,sizeof_op_plan=0,sizeof_issue_plan=0;
    uint64_t sizeof_weight_table=0,sizeof_weight_member=0,sizeof_route_slot=0;
    uint64_t sizeof_expert_state=0,sizeof_expert_map=0,sizeof_expert_entry=0;
    uint64_t buffers=0,weights=0,ops=0,bands=0,tables=0,table_entries=0;
    uint64_t arena_device=0,arena_host=0,route_bank=0,route_device=0;
    uint64_t prepare_bank=0,prepare_maps=0,prepare_entries=0,prepare_seen=0;
    uint64_t first_use_heap_reservation=0;
};
struct TinyOwnerQuote {
    TinyStorageQuote storage;
    uint64_t sizeof_owner=0,sizeof_impl=0,sizeof_weight_slot=0;
    uint64_t sizeof_publication=0,sizeof_lease=0;
    uint64_t weights=0,slots=0,classes=0;
};
struct TinyStepQuote {
    TinyStorageQuote storage;
    uint64_t sizeof_step_slot=0,slots=0,input_eog=0,unique_eog=0;
};
static_assert(std::is_trivially_copyable_v<TinySamplerQuote>&&std::is_trivially_copyable_v<TinyCtxQuote>&&std::is_trivially_copyable_v<TinyOwnerQuote>&&std::is_trivially_copyable_v<TinyStepQuote>);
// This narrow profile uses the reviewed 4KiB backend page; unknown geometry refuses.
namespace tiny_quote {
inline constexpr uint64_t page=4096,control_cap=4u<<20,arena_cap=4u<<20;
inline constexpr uint64_t max_weights=512,max_buffers=256,max_ops=256,max_bands=512;
bool add(uint64_t& value,uint64_t n) noexcept;
bool mul(uint64_t a,uint64_t b,uint64_t* out) noexcept;
bool device(TinyStorageQuote&,uint64_t bytes,uint64_t requests=1) noexcept;
bool pinned(TinyStorageQuote&,uint64_t bytes,uint64_t requests=1) noexcept;
bool mapped(TinyStorageQuote&,uint64_t bytes,uint64_t requests=1) noexcept;
bool heap(TinyStorageQuote&,uint64_t count,uint64_t element) noexcept;
bool finish(TinyStorageQuote&) noexcept;
int program_profile(const Program&) noexcept;
}
// Actual StepSlot from sched/advance.h. Initial profile fixed to one rank/slot;
// eog input is Vocab::eog() before Engine's local sort/unique (bounded 0..319).
int tiny_step_state_quote(std::span<const int32_t> eog,TinyStepQuote* out) noexcept;
}
