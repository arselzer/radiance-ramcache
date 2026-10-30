/* Exact supplemental ISTA Engram roles. Metadata/type/range policy only;
 * no generated controls, dequantization, global import mode or payload copies. */
#pragma once
#include "../text/gguf.h"
#include <climits>
#include <cstring>

namespace rad::ista_engram {
inline constexpr const char* policy_key = "deepseek41.engram_source";
inline constexpr const char* policy_value = "ista_iq4_nl_v1";
struct Role {
    const char* name;
    uint32_t type, rank;
    int64_t rows, cols, bytes;
};
inline constexpr Role roles[] = {
    {"engram_token_map.weight",26,1,129280,1,517120},
    {"engram_primes.weight",27,2,2,24,384},
    {"engram_offsets.weight",27,2,2,24,384},
    {"engram_multipliers.weight",27,2,2,4,64},
    {"blk.1.engram_embd.weight",20,2,384006168,256,55296888192LL},
    {"blk.14.engram_embd.weight",20,2,384016682,256,55298402208LL},
};
inline const Role* role(const char* name) {
    if (!name) return nullptr;
    for (const auto& r:roles) if (!std::strcmp(name,r.name)) return &r;
    return nullptr;
}
inline bool matches(const Role& r,const GgufTensor& t) {
    return t.name==r.name && t.ggml_type==r.type && t.n_dims==r.rank &&
           t.bytes==uint64_t(r.bytes) && t.ne[0]==(r.rank==1?r.rows:r.cols) &&
           (r.rank==1 || t.ne[1]==r.rows);
}
/* A derived dense slice must retain the original representation and lie wholly
 * inside its one owned mapped tensor. uintptr_t arithmetic avoids unrelated
 * C++ pointer subtraction; every sum is bounded before it is formed. */
inline bool view_valid(const GgufFile* g,const char* original,uint32_t type,
                       uint32_t dtype,const std::vector<int64_t>& shape,
                       const void* data,int64_t bytes) {
    if (!g || !g->is_open() || !original || !data || bytes<=0) return false;
    const auto* r=role(original); const auto* t=g->tensor(original);
    if (!r || !t || !matches(*r,*t) || type!=r->type || dtype!=gguf_to_rad_dtype(type)) return false;
    int64_t n=1;
    if (shape.empty() || shape.size()>4) return false;
    for (int64_t d:shape) { if (d<=0 || n>INT64_MAX/d) return false; n*=d; }
    const auto* dt=gguf_dtype(type);
    if (!dt || dt->blck<=0 || n%dt->blck || uint64_t(n/dt->blck)>uint64_t(INT64_MAX)/dt->block_bytes ||
        uint64_t(bytes)!=uint64_t(n/dt->blck)*dt->block_bytes) return false;
    const uintptr_t begin=reinterpret_cast<uintptr_t>(g->tensor_data(*t));
    const uintptr_t ptr=reinterpret_cast<uintptr_t>(data);
    return begin && t->bytes<=UINTPTR_MAX-begin && ptr>=begin && ptr<=begin+t->bytes &&
           uint64_t(bytes)<=begin+t->bytes-ptr;
}
} // namespace rad::ista_engram
