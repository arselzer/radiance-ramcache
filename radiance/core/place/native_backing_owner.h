#pragma once
#include "native_inventory.h"
namespace rad {
struct NativeBackingStats {
    uint64_t prepares=0,source_batches=0,source_bytes=0,copy_bytes=0,published_units=0,retirements=0;
    size_t held_leases=0,peak_held_leases=0;
};
struct TinyOwnerQuote;
class NativeBackingOwner final:public ExpertPrepareOwner {
public:
    static int tiny_startup_quote(const Program&,const Plan&,TinyOwnerQuote*) noexcept;
    static int open(Program&,const Plan&,Mover&,std::shared_ptr<NativeRecordSource>,
                    std::unique_ptr<NativeBackingOwner>* out);
    ~NativeBackingOwner();
    int check_program(const Program&)const override;
    bool accepts_native_inventory(const NativeInventory*)const override;
    int bind_native_context(const NativeInventory*,const Ctx*,RadStream)override;
    int prepare(const ExpertPrepareView&,bool*)override;
    int end_pass(RadStream,int)override;
    int quiesce();
    // Required after Ctx::shutdown and successful quiesce. Native pointers are withdrawn,
    // never replaced with an unallocated original base. Pools/Mover still own the backing.
    int close_after_context_shutdown();
    NativeBackingStats stats()const noexcept;
private:
    struct Impl;
    explicit NativeBackingOwner(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> p_;
};
} // namespace rad
