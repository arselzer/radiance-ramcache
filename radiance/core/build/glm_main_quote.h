#pragma once
#include "tiny_component_quotes.h"
#include <initializer_list>
namespace rad {
// Same POD arithmetic, separate closed actual-profile cap. Never changes the
// synthetic cap and never claims allocator/container/driver RSS accounting.
inline bool glm_main_quote_finish(TinyStorageQuote& q)noexcept {
    uint64_t metadata=0;
    for(auto n:{q.heap_payload_bytes,q.embedded_object_bytes,q.heap_growth_allowance,q.control_allowance})
        if(!tiny_quote::add(metadata,n))return false;
    if(metadata>(128u<<20))return false;
    uint64_t requested=metadata,rounded=metadata;
    for(auto n:{q.device_bytes,q.pinned_bytes,q.mapped_bytes})if(!tiny_quote::add(requested,n))return false;
    for(auto n:{q.device_page_bytes,q.pinned_page_bytes,q.mapped_page_bytes})if(!tiny_quote::add(rounded,n))return false;
    q.requested_total=requested;q.page_and_heap_total=rounded;return true;
}
}
