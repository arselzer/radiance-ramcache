#pragma once
// Checker-only COMPLETE allocation family. Requested-byte counters cover this
// rank thread's explicitly bracketed component call, not process RSS/runtime.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
namespace allocation_fault {
inline thread_local int64_t remaining=-1;
inline thread_local uint64_t epoch=0,calls=0,live=0,peak=0,total=0,max_request=0;
inline thread_local bool tracking=false;
struct Header {void* base;size_t bytes;uint64_t epoch,magic;};
constexpr uint64_t magic=0x434f4d5041435441ull;
inline void begin(){++epoch;calls=live=peak=total=max_request=0;tracking=true;}
inline void end(){tracking=false;remaining=-1;}
inline void* allocate(size_t bytes,size_t alignment=alignof(std::max_align_t)){
    if(remaining>=0&&remaining--==0){remaining=-1;throw std::bad_alloc();}
    alignment=std::max(alignment,alignof(Header));
    if(!alignment||(alignment&(alignment-1))||bytes>std::numeric_limits<size_t>::max()-sizeof(Header)-alignment)throw std::bad_alloc();
    void* base=std::malloc((bytes?bytes:1)+sizeof(Header)+alignment-1);if(!base)throw std::bad_alloc();
    const auto start=reinterpret_cast<uintptr_t>(base)+sizeof(Header);
    const auto address=(start+alignment-1)&~uintptr_t(alignment-1);
    auto* header=new(reinterpret_cast<void*>(address-sizeof(Header)))Header{base,bytes,tracking?epoch:0,magic};
    if(header->epoch){++calls;total+=bytes;live+=bytes;peak=std::max(peak,live);max_request=std::max<uint64_t>(max_request,bytes);}
    return reinterpret_cast<void*>(address);
}
inline void release(void* pointer)noexcept{
    if(!pointer)return;
    auto* header=reinterpret_cast<Header*>(reinterpret_cast<uintptr_t>(pointer)-sizeof(Header));
    if(header->magic!=magic)std::abort();
    if(header->epoch&&header->epoch==epoch){if(header->bytes>live)std::abort();live-=header->bytes;}
    void* base=header->base;header->magic=0;std::free(base);
}
}
void* operator new(size_t n){return allocation_fault::allocate(n);}
void* operator new[](size_t n){return allocation_fault::allocate(n);}
void* operator new(size_t n,const std::nothrow_t&)noexcept{try{return allocation_fault::allocate(n);}catch(...){return nullptr;}}
void* operator new[](size_t n,const std::nothrow_t&)noexcept{try{return allocation_fault::allocate(n);}catch(...){return nullptr;}}
void* operator new(size_t n,std::align_val_t a){return allocation_fault::allocate(n,size_t(a));}
void* operator new[](size_t n,std::align_val_t a){return allocation_fault::allocate(n,size_t(a));}
void* operator new(size_t n,std::align_val_t a,const std::nothrow_t&)noexcept{try{return allocation_fault::allocate(n,size_t(a));}catch(...){return nullptr;}}
void* operator new[](size_t n,std::align_val_t a,const std::nothrow_t&)noexcept{try{return allocation_fault::allocate(n,size_t(a));}catch(...){return nullptr;}}
void operator delete(void* p)noexcept{allocation_fault::release(p);}
void operator delete[](void* p)noexcept{allocation_fault::release(p);}
void operator delete(void* p,size_t)noexcept{allocation_fault::release(p);}
void operator delete[](void* p,size_t)noexcept{allocation_fault::release(p);}
void operator delete(void* p,const std::nothrow_t&)noexcept{allocation_fault::release(p);}
void operator delete[](void* p,const std::nothrow_t&)noexcept{allocation_fault::release(p);}
void operator delete(void* p,std::align_val_t)noexcept{allocation_fault::release(p);}
void operator delete[](void* p,std::align_val_t)noexcept{allocation_fault::release(p);}
void operator delete(void* p,size_t,std::align_val_t)noexcept{allocation_fault::release(p);}
void operator delete[](void* p,size_t,std::align_val_t)noexcept{allocation_fault::release(p);}
void operator delete(void* p,std::align_val_t,const std::nothrow_t&)noexcept{allocation_fault::release(p);}
void operator delete[](void* p,std::align_val_t,const std::nothrow_t&)noexcept{allocation_fault::release(p);}
