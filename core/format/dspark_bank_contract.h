//Private pure contract only; not integrated into Engine/Mover and not an IO/crypto proof.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <sys/stat.h>
namespace rad::dspark_bank {
using BankId=uint32_t;
inline constexpr BankId canonical=0,draft=1;
struct Epoch {
    uint64_t device=0,inode=0,bytes=0,links=0;
    uint32_t mode=0;
    int64_t mtime_ns=0,ctime_ns=0;
    bool operator==(const Epoch&)const=default;
    bool valid()const{return S_ISREG(mode)&&inode!=0&&bytes>0&&bytes<=uint64_t(std::numeric_limits<int64_t>::max())&&links>0;}
};
//generation is private Program/cache lifetime identity; not a checkpoint content hash.
struct Snapshot {
    uint64_t generation=0;
    uint32_t count=0;
    std::array<Epoch,2> epoch{};
    bool valid()const{
        if(!generation||(count!=1&&count!=2)||!epoch[0].valid())return false;
        return count==1||(epoch[1].valid()&&
            (epoch[0].device!=epoch[1].device||epoch[0].inode!=epoch[1].inode));
    }
    bool current(const Snapshot&now)const{
        if(!valid()||!now.valid()||generation!=now.generation||count!=now.count)return false;
        for(uint32_t i=0;i<count;++i)if(epoch[i]!=now.epoch[i])return false;
        return true; //ALL active banks; never just the requested weight's bank.
    }
};
struct Owner {
    uint64_t generation=0;
    BankId bank=canonical;
    bool operator==(const Owner&)const=default;
};
struct Span {Owner owner{};uint64_t offset=0,bytes=0;};
inline bool end(const Span&s,uint64_t*out){
    if(!out||!s.bytes||s.offset>std::numeric_limits<uint64_t>::max()-s.bytes)return false;
    *out=s.offset+s.bytes;return true;
}
inline bool fits(const Snapshot&ns,const Span&s,uint64_t data_start){
    uint64_t hi=0;
    return ns.valid()&&s.owner.generation==ns.generation&&s.owner.bank<ns.count&&
        data_start<=ns.epoch[s.owner.bank].bytes&&s.offset>=data_start&&end(s,&hi)&&
        hi<=ns.epoch[s.owner.bank].bytes;
}
//Inputs already passed fits(); caller sorts bank/generation BEFORE numeric offset.
inline bool coalescible(const Span&a,const Span&b,uint64_t max_gap){
    uint64_t ae=0,be=0;
    if(a.owner!=b.owner||!a.owner.generation||a.owner.bank>draft||
       !end(a,&ae)||!end(b,&be)||b.offset<a.offset)return false;
    return b.offset<=ae||b.offset-ae<=max_gap;
}
struct Plane {Owner owner{};uint64_t entry_index=0,offset=0,bytes=0;};
//Ownership-only: EVERY plane must separately pass fits() against the owning bank.
inline bool one_entry(std::span<const Plane>p){
    if(p.empty()||!p[0].owner.generation||p[0].owner.bank>draft)return false;
    for(const auto&x:p)if(x.owner!=p[0].owner||x.entry_index!=p[0].entry_index)return false;
    return true;
}
struct CacheKey {
    Owner owner{};
    uint64_t weight_index=0;
    std::array<uint8_t,32> stored_layout_sha256{};
    bool operator==(const CacheKey&)const=default;
};
//An array implementation may index by weight_index only AFTER this immutable binding,
//whole-Snapshot validation and successful empty-lifetime reset are established.
inline bool cache_binding(const Snapshot&bound,const Snapshot&now,const CacheKey&key,const CacheKey&fixed){
    return key==fixed&&bound.current(now)&&key.owner.generation==bound.generation&&
        key.owner.bank<bound.count&&key.weight_index>0;
}
} //namespace rad::dspark_bank
