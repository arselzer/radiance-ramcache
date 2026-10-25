/* Proposed DS41 state storage over the existing KV ABI. No architecture is registered here.
 * Opaque shared latent, not ordinary distinct K/V: each D-element compressed row occupies
 * `ratio` original-token slots in [block,1,block_size,2*head_dim]. Logical rows stay contiguous;
 * allocator, block tables, prefixes and window floors retain original-token coordinates. */
#pragma once
#include "rad_builder.h"
#include <cstdint>

#if defined(__HIPCC__)
#define DS41_STATE_HD __host__ __device__ inline
#else
#define DS41_STATE_HD inline
#endif

namespace rad::deepseek41::state {
struct Layout {
    int64_t dimension=0,ratio=0,block_size=0,element_bytes=0,head_dim=0;
    int64_t entry_bytes=0,slot_bytes=0,page_bytes=0;
};
struct Address {
    int64_t original_token=0,logical_block=0,slot=0,plane=0,column=0;
    int64_t page_element=0,page_byte=0;
};
struct Prefix {
    int64_t completed_entries=0,pending_tokens=0,pending_first_token=0;
};
struct Budget {
    int64_t completed_entries=0,pending_tokens=0,pages=0,payload_bytes=0,reserved_bytes=0;
};
DS41_STATE_HD bool product(int64_t a,int64_t b,int64_t* out) {
    if(!out||a<0||b<0||(b&&a>INT64_MAX/b))return false;
    *out=a*b;return true;
}
DS41_STATE_HD int make_layout(int64_t dimension,int64_t ratio,int64_t block_size,
                             int64_t element_bytes,Layout* out) {
    if(!out)return RAD_E_INVAL;
    if(dimension<=0||(ratio!=1&&ratio!=2)||block_size<=0||block_size%ratio||
       dimension%(2*ratio)||(element_bytes!=1&&element_bytes!=2&&element_bytes!=4&&element_bytes!=8))
        return RAD_E_SHAPE;
    Layout value;value.dimension=dimension;value.ratio=ratio;value.block_size=block_size;
    value.element_bytes=element_bytes;value.head_dim=dimension/(2*ratio);
    if(!product(dimension,element_bytes,&value.entry_bytes))return RAD_E_SHAPE;
    value.slot_bytes=value.entry_bytes/ratio;
    if(!product(block_size,value.slot_bytes,&value.page_bytes))return RAD_E_SHAPE;
    *out=value;return RAD_OK;
}
DS41_STATE_HD bool valid(const Layout& l) {
    Layout expected;
    return make_layout(l.dimension,l.ratio,l.block_size,l.element_bytes,&expected)==RAD_OK&&
           l.head_dim==expected.head_dim&&l.entry_bytes==expected.entry_bytes&&
           l.slot_bytes==expected.slot_bytes&&l.page_bytes==expected.page_bytes;
}
DS41_STATE_HD int map(const Layout& l,int64_t entry,int64_t element,Address* out) {
    if(!out)return RAD_E_INVAL;
    if(!valid(l)||entry<0||element<0||element>=l.dimension)return RAD_E_SHAPE;
    const int64_t per_token=l.dimension/l.ratio;
    const int64_t within_token=element%per_token,token_in_entry=element/per_token;
    int64_t first=0;
    if(!product(entry,l.ratio,&first)||first>INT64_MAX-token_in_entry)return RAD_E_SHAPE;
    Address a;a.original_token=first+token_in_entry;a.logical_block=a.original_token/l.block_size;
    a.slot=a.original_token%l.block_size;a.plane=within_token/l.head_dim;a.column=within_token%l.head_dim;
    // Actual ABI is K then V per original-token slot, never two page-sized planes.
    a.page_element=a.slot*(2*l.head_dim)+a.plane*l.head_dim+a.column;
    a.page_byte=a.page_element*l.element_bytes;*out=a;return RAD_OK;
}
DS41_STATE_HD int prefix(const Layout& l,int64_t committed_tokens,Prefix* out) {
    if(!out)return RAD_E_INVAL;
    if(!valid(l)||committed_tokens<0)return RAD_E_SHAPE;
    Prefix p;p.completed_entries=committed_tokens/l.ratio;p.pending_tokens=committed_tokens%l.ratio;
    p.pending_first_token=committed_tokens-p.pending_tokens;*out=p;return RAD_OK;
}
DS41_STATE_HD int map_completed(const Layout& l,int64_t committed_tokens,int64_t entry,
                               int64_t element,Address* out) {
    Prefix p;const int status=prefix(l,committed_tokens,&p);
    if(status!=RAD_OK)return status;
    if(entry<0||entry>=p.completed_entries)return RAD_E_STATE;
    return map(l,entry,element,out);
}
/* For ordinary FULL/fixed-WINDOW tables only. `first_block_pos` is the original-token position
 * table[0] covers; runtime reference/ring windows have a separate mapping and are not supported. */
DS41_STATE_HD int table_index(const Layout& l,int64_t entry,int64_t element,
                             int64_t first_block_pos,int64_t table_size,int64_t* out) {
    if(!out)return RAD_E_INVAL;
    Address a;const int status=map(l,entry,element,&a);if(status!=RAD_OK)return status;
    if(first_block_pos<0||first_block_pos%l.block_size||table_size<=0||a.original_token<first_block_pos)
        return RAD_E_SHAPE;
    const int64_t index=(a.original_token-first_block_pos)/l.block_size;
    if(index>=table_size)return RAD_E_SHAPE;
    *out=index;return RAD_OK;
}
/* A kernel must first look up its physical page in the sequence's block table. This helper
 * checks only that the resulting per-layer byte offset fits signed ABI addressing. */
DS41_STATE_HD int physical_byte(const Layout& l,int64_t physical_block,int64_t entry,
                               int64_t element,int64_t* out) {
    if(!out)return RAD_E_INVAL;
    Address a;const int status=map(l,entry,element,&a);if(status!=RAD_OK)return status;
    int64_t start=0;
    if(!product(physical_block,l.page_bytes,&start)||start>INT64_MAX-(a.page_byte+l.element_bytes))return RAD_E_SHAPE;
    *out=start+a.page_byte;return RAD_OK;
}
DS41_STATE_HD int budget(const Layout& l,int64_t original_tokens,int64_t bound_layers,Budget* out) {
    if(!out)return RAD_E_INVAL;
    Prefix p;const int status=prefix(l,original_tokens,&p);if(status!=RAD_OK)return status;
    if(bound_layers<=0)return RAD_E_SHAPE;
    Budget b;b.completed_entries=p.completed_entries;b.pending_tokens=p.pending_tokens;
    b.pages=original_tokens/l.block_size+(original_tokens%l.block_size!=0);
    int64_t payload=0,reserved=0;
    if(!product(b.completed_entries,l.entry_bytes,&payload)||!product(payload,bound_layers,&b.payload_bytes)||
       !product(b.pages,l.page_bytes,&reserved)||!product(reserved,bound_layers,&b.reserved_bytes))return RAD_E_SHAPE;
    *out=b;return RAD_OK;
}

/* Pure metadata constructors. The caller must bind real owners, declare a compatible consumer,
 * then check rad_kv_block_size()==layout.block_size. No group allocation or op is declared here.
 * Byte-addressable storage only; this does not implement DS41 FP4 payloads/scales. */
inline int paged_decl(const Layout& l,uint32_t dtype,int kind,int64_t window,RadKVGroupDecl* out) {
    if(!out)return RAD_E_INVAL;
    if(!valid(l)||(kind!=RAD_KV_FULL&&kind!=RAD_KV_WINDOW)||
       (kind==RAD_KV_FULL?window!=0:window<=0))return RAD_E_SHAPE;
    if(rad_dtype_bits(dtype)!=l.element_bytes*8)return RAD_E_DTYPE;
    RadKVGroupDecl d{};d.kind=kind;d.dtype=dtype;d.window=window;d.n_head_kv=1;d.head_dim=l.head_dim;
    *out=d;return RAD_OK;
}
inline int fixed_decl(uint32_t dtype,int64_t rows,int64_t cols,RadKVGroupDecl* out) {
    if(!out)return RAD_E_INVAL;
    const int64_t bytes=rad_dtype_bytes(dtype,1);int64_t count=0,total=0;
    if(bytes<=0||rad_dtype_bits(dtype)!=bytes*8)return RAD_E_DTYPE;
    if(rows<=0||cols<=0||!product(rows,cols,&count)||!product(count,bytes,&total))return RAD_E_SHAPE;
    RadKVGroupDecl d{};d.kind=RAD_KV_LINEAR;d.dtype=dtype;d.n_head_kv=1;d.state_dim[0]=rows;d.state_dim[1]=cols;
    *out=d;return RAD_OK;
}
inline int pending_c2_decl(int64_t dimension,uint32_t dtype,RadKVGroupDecl* out) {
    // One pending projected KV row plus one pending gate-score row. Phase is committed % 2.
    return fixed_decl(dtype,2,dimension,out);
}
inline int engram_history_decl(int64_t max_ngram_size,RadKVGroupDecl* out) {
    if(max_ngram_size<=1)return RAD_E_SHAPE;
    // Bind once to one real owner, not once per Engram consumer; DEAD=-1 must remain intact.
    return fixed_decl(RAD_I32,1,max_ngram_size-1,out);
}
} // namespace rad::deepseek41::state
#undef DS41_STATE_HD
