/* UNUSED, allocation-free Phase2 route metadata prerequisite. No Engine/P1/arch caller,
 * launch, allocation, environment flag or executor permission. Original piece-local sorts,
 * offsets and six-slot IDs are copied, never re-sorted into a span-wide GEMM.
 *
 * Actual native contract: MoeFP8::pass_head sorts rows*6 slots relative to the rebased FFN
 * suffix; IDs/weights use row_from, while sorted/offsets/counts start at zero. Only the local96
 * expert prefix is live. The remaining sorted tail is copied verbatim, not interpreted.
 * A caller must separately own/fence all GPU buffers and fund sizeof(Frame) per original piece. */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rad { namespace ds41_phase2 { namespace routes {
constexpr int kLocalExperts=96, kAllExperts=384, kTopK=6, kRanks=4, kLayers=40;
constexpr int kMaxPieceRows=4096, kMaxPieces=4, kMaxSpanRows=8192;
constexpr int kMaxSlots=kMaxPieceRows*kTopK;
enum class Phase:uint8_t { Unknown=0, Prefill=1, Decode=2, Draft=3 };
enum class Residence:uint8_t { Unknown=0, Resident=1, Nonresident=2 };
struct Binding {
    uint64_t sequence=0, span_epoch=0, frame_epoch=0, plan_cookie=0; // sequence0 valid; epochs/cookie0 unbound
    int32_t layer=-1, rank=-1, piece=-1;
    int64_t begin=0; // absolute position of the ORIGINAL piece, not its FFN suffix
    int32_t query_rows=0, row_from=0;
    Phase phase=Phase::Unknown;
};
struct View {
    const int32_t* ids=nullptr; std::size_t ids_size=0;       // [query_rows-row_from,6]
    const int32_t* sorted=nullptr; std::size_t sorted_size=0; // [rows*6], including dead tail
    const int32_t* counts=nullptr; std::size_t counts_size=0; // local96
    const int32_t* offsets=nullptr; std::size_t offsets_size=0; // local97 exclusive prefix
};
class Frame;
inline bool freeze(const Binding&,const View&,Frame*,const char**);
class Frame {
public:
    bool valid()const{return valid_;}
    const Binding& binding()const{return binding_;}
    int rows()const{return binding_.query_rows-binding_.row_from;}
    int slots()const{return rows()*kTopK;}
    int live_slots()const{return offsets_[kLocalExperts];}
    const int32_t* ids()const{return ids_.data();}
    const int32_t* sorted()const{return sorted_.data();}
    const int32_t* counts()const{return counts_.data();}
    const int32_t* offsets()const{return offsets_.data();}
    // No mutable view/pointer escapes: the snapshot cannot follow reused readback memory.
private:
    Binding binding_{};
    bool valid_=false;
    std::array<int32_t,kLocalExperts> counts_{};
    std::array<int32_t,kLocalExperts+1> offsets_{};
    std::array<int32_t,kMaxSlots> ids_{},sorted_{};
    friend bool freeze(const Binding&,const View&,Frame*,const char**);
};
inline bool refuse(const char* reason,const char**why){if(why)*why=reason;return false;}
inline bool binding_ok(const Binding& b) {
    return b.span_epoch&&b.frame_epoch&&b.plan_cookie&&b.phase==Phase::Prefill&&b.layer>=0&&b.layer<kLayers&&
        b.rank>=0&&b.rank<kRanks&&b.piece>=0&&b.piece<kMaxPieces&&b.begin>=0&&
        b.query_rows>0&&b.query_rows<=kMaxPieceRows&&b.row_from>=0&&b.row_from<=b.query_rows&&
        b.begin<=std::numeric_limits<int32_t>::max()-b.query_rows;
}
/* Validate before copying; refusal leaves the previous snapshot untouched. Zero FFN rows are
 * an explicit key-only piece with empty IDs/sorted and ALL counts/offsets zero. Positive rows may
 * select no local experts (all six choices foreign): no invented expert or compulsory wave. */
inline bool freeze(const Binding& b,const View& v,Frame*out,const char**why=nullptr) {
    if(!out||!binding_ok(b))return refuse("unbound/non-prefill/native piece scope",why);
    const int slots=(b.query_rows-b.row_from)*kTopK,base=b.rank*kLocalExperts;
    if(v.ids_size!=std::size_t(slots)||v.sorted_size!=std::size_t(slots)||
       v.counts_size!=kLocalExperts||v.offsets_size!=kLocalExperts+1||!v.counts||!v.offsets||
       (slots&&(!v.ids||!v.sorted)))return refuse("wrong original piece metadata extents",why);
    for(int t=0;t<slots;t+=kTopK)for(int j=0;j<kTopK;++j) {
        const int e=v.ids[t+j];
        if(e < -1 || e>=kAllExperts)return refuse("global selected expert out of native range",why);
        // -1 is the original dropped/unselected slot, not expert0; repeats are harmless.
        if(e>=0)for(int q=0;q<j;++q)if(v.ids[t+q]==e)return refuse("duplicate expert in one top-six tuple",why);
    }
    int live=0;
    if(v.offsets[0]!=0)return refuse("local offsets must begin at zero",why);
    for(int e=0;e<kLocalExperts;++e) {
        if(v.counts[e]<0||v.counts[e]>slots-live||v.offsets[e]!=live)
            return refuse("negative count or nonexclusive local offsets",why);
        live+=v.counts[e];
        if(v.offsets[e+1]!=live)return refuse("counts/offset prefix disagreement",why);
    }
    std::array<uint8_t,kMaxSlots> seen{};
    for(int e=0;e<kLocalExperts;++e)for(int j=v.offsets[e];j<v.offsets[e+1];++j) {
        const int s=v.sorted[j];
        if(s<0||s>=slots||seen[std::size_t(s)]||v.ids[s]!=base+e)
            return refuse("live sorted prefix is not a local expert/slot bijection",why);
        seen[std::size_t(s)]=1;
    }
    for(int s=0;s<slots;++s) {
        const bool local=v.ids[s]>=base&&v.ids[s]<base+kLocalExperts;
        if(local!=bool(seen[std::size_t(s)]))return refuse("local selected slot missing from original sort",why);
    }
    Frame got;got.binding_=b;
    for(int i=0;i<kLocalExperts;++i)got.counts_[i]=v.counts[i];
    for(int i=0;i<=kLocalExperts;++i)got.offsets_[i]=v.offsets[i];
    for(int i=0;i<slots;++i){got.ids_[i]=v.ids[i];got.sorted_[i]=v.sorted[i];}
    got.valid_=true;*out=got;if(why)*why=nullptr;return true;
}

struct Scope {
    uint64_t sequence=0,span_epoch=0,plan_cookie=0;
    int32_t layer=-1,rank=-1,pieces=0;
    int64_t begin=0,end=0;
};
struct UnitMap {
    uint64_t plan_cookie=0;
    int32_t layer=-1,rank=-1;
    std::array<int32_t,kLocalExperts> unit{}; // selected units: actual immutable Plan IDs, zero valid
    std::array<Residence,kLocalExperts> residence{}; // explicit effective classification at this layer
};
struct Range {int32_t lo=0,hi=0;};
struct Entry {
    int32_t local_expert=-1,global_expert=-1,unit=-1;
    Residence residence=Residence::Unknown;
    uint8_t pieces_mask=0;
    int64_t rows=0;
    std::array<Range,kMaxPieces> original_ranges{};
};
struct Union {
    Scope scope{};
    int32_t size=0;
    std::array<uint64_t,kMaxPieces> frame_epochs{}; // each ORIGINAL issued frame, never replaced by span_epoch
    std::array<Entry,kLocalExperts> entries{};
    int32_t expert_occurrences=0,duplicate_occurrences=0;
    int32_t nonresident_occurrences=0,unique_nonresident=0,duplicate_nonresident=0;
    int64_t local_rows=0;
};
/* A Plan unit appears ONCE in the union, and still has a separate ORIGINAL sorted range per
 * piece. Order is deterministic local expert order, not an instruction to change numerical
 * gather order or the P1 source arrival policy. Counts describe potential reuse, not DMA saved.
 * A common span_epoch groups pieces; their ORIGINAL frame epochs remain separate and ordered.
 * They may differ across ordinary steps, or share a frame in a future bounded span entry.
 * No source bytes, cache hit, stream/event lifetime or alias correctness is inferred here. */
inline bool make_union(const Scope&s,const Frame*frames,std::size_t count,const UnitMap&m,
                       Union*out,const char**why=nullptr) {
    if(!out||!frames||!s.span_epoch||!s.plan_cookie||s.pieces<1||s.pieces>kMaxPieces||
       count!=std::size_t(s.pieces)||s.layer<0||s.layer>=kLayers||s.rank<0||s.rank>=kRanks||
       s.begin<0||s.end<=s.begin||s.end-s.begin>kMaxSpanRows||s.end>std::numeric_limits<int32_t>::max()||
       m.plan_cookie!=s.plan_cookie||m.layer!=s.layer||m.rank!=s.rank)
        return refuse("unbound/mismatched native span or Plan",why);
    int64_t at=s.begin;uint64_t previous_frame=0;
    for(int p=0;p<s.pieces;++p) {
        const auto&f=frames[p];const auto&b=f.binding();
        if(!f.valid()||b.sequence!=s.sequence||b.span_epoch!=s.span_epoch||b.plan_cookie!=s.plan_cookie||
           b.layer!=s.layer||b.rank!=s.rank||b.piece!=p||b.begin!=at||b.frame_epoch<previous_frame)
            return refuse("stale/mixed/reordered/noncontiguous original piece",why);
        at+=b.query_rows;previous_frame=b.frame_epoch;
    }
    if(at!=s.end)return refuse("span end differs from original piece boundaries",why);
    Union got;got.scope=s;
    for(int p=0;p<s.pieces;++p)got.frame_epochs[p]=frames[p].binding().frame_epoch;
    for(int e=0;e<kLocalExperts;++e) {
        Entry u;u.local_expert=e;u.global_expert=s.rank*kLocalExperts+e;u.unit=m.unit[e];u.residence=m.residence[e];
        for(int p=0;p<s.pieces;++p) {
            const auto&f=frames[p];const int n=f.counts()[e];
            u.original_ranges[p]={f.offsets()[e],f.offsets()[e+1]};
            if(!n)continue;
            u.pieces_mask|=uint8_t(1u<<p);u.rows+=n;++got.expert_occurrences;
            if(u.residence==Residence::Nonresident)++got.nonresident_occurrences;
        }
        if(!u.pieces_mask)continue;
        if(u.unit<0||(u.residence!=Residence::Resident&&u.residence!=Residence::Nonresident))
            return refuse("selected expert lacks explicit Plan unit/residence",why);
        for(int j=0;j<got.size;++j)if(got.entries[j].unit==u.unit)
            return refuse("distinct selected experts alias one Plan unit",why);
        got.entries[got.size++]=u;got.local_rows+=u.rows;
        got.unique_nonresident+=u.residence==Residence::Nonresident;
    }
    got.duplicate_occurrences=got.expert_occurrences-got.size;
    got.duplicate_nonresident=got.nonresident_occurrences-got.unique_nonresident;
    *out=got;if(why)*why=nullptr;return true;
}
/* Explicit value-object metadata charge, including freeze's largest temporary snapshot and
 * validation bitmap, union temporary and a4KiB bookkeeping margin. This is NOT allocator
 * admission or a device/pinned-memory estimate.
 * Future readback/event/launch-frame storage is extra, as is the existing retention inventory. */
inline bool metadata_bytes(int pieces,int64_t*out) {
    if(!out||pieces<1||pieces>kMaxPieces)return false;
    *out=int64_t(pieces+1)*int64_t(sizeof(Frame))+2*sizeof(Union)+sizeof(UnitMap)+
        sizeof(Scope)+sizeof(Binding)+sizeof(View)+sizeof(Entry)+kMaxSlots+4096;
    return true;
}
}}} // namespace rad::ds41_phase2::routes
