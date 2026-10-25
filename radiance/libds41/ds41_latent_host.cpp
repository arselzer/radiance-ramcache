#include "ds41_latent_args.h"
namespace rad::ds41::latent {
int prepare(const RadArgs* a,bool store,Launch* out) {return validate(a,store,out);}
}
extern "C" int ds41_ref_latent_store(const RadArgs* a,RadStream) {
    using namespace rad::ds41::latent;
    Launch l;int rc=validate(a,true,&l);if(rc!=RAD_OK)return rc;
    const auto* t=a->t;const auto* pos=static_cast<const int64_t*>(t[1].data);
    const auto* slots=static_cast<const int64_t*>(t[2].data);
    // Full validation before writes, including duplicate/overlapping completed-entry slots.
    for(int64_t row=0;row<l.rows;++row) {
        int64_t first=0;rc=store_row(l,pos[row],slots[row],&first);if(rc!=RAD_OK)return rc;
        for(int64_t other=0;other<row;++other)if(slots[row]==slots[other])return RAD_E_INVAL;
    }
    for(int64_t row=0;row<l.rows;++row) {
        const int64_t first=slots[row]-l.layout.ratio+1;
        const auto* src=static_cast<const uint8_t*>(t[0].data)+row*l.pitch*l.layout.element_bytes;
        auto* dst=static_cast<uint8_t*>(t[3].data)+first*l.layout.slot_bytes;
        std::memcpy(dst,src,size_t(l.layout.entry_bytes));
        static_cast<int32_t*>(t[4].data)[row]=RAD_OK;
    }
    return RAD_OK;
}
extern "C" int ds41_ref_latent_gather(const RadArgs* a,RadStream) {
    using namespace rad::ds41::latent;
    Launch l;int rc=validate(a,false,&l);if(rc!=RAD_OK)return rc;
    const auto* t=a->t;const auto* slots=static_cast<const int64_t*>(t[1].data);
    for(int64_t row=0;row<l.rows;++row)if((rc=gather_row(l,slots[row]))!=RAD_OK)return rc;
    for(int64_t row=0;row<l.rows;++row) {
        const auto* src=static_cast<const uint8_t*>(t[0].data)+slots[row]*l.layout.slot_bytes;
        auto* dst=static_cast<uint8_t*>(t[2].data)+row*l.pitch*l.layout.element_bytes;
        std::memcpy(dst,src,size_t(l.layout.entry_bytes));
        static_cast<int32_t*>(t[3].data)[row]=RAD_OK;
    }
    return RAD_OK;
}
