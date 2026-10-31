/* Real-card byte oracle for the new bank backup mechanism. No model/kernel
 * arithmetic is changed. Native Program/Plan/alias scope is covered by the host
 * fixture and actual model startup, separately from this raw-copy proof.
 * Run ONLY inside the two-lock stop/restore window, before model lending. */
#include "place/draft_bank_backup.h"
#include "place/draft_bank_complement.h"
#include "rad_device.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace rad;
namespace d = rad::draft_bank_backup_detail;
namespace {
void need(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
void checked(int rc, const std::string& why) { need(rc >= 0, why); }
int number(const char* s) {
    char* end = nullptr;
    errno = 0;
    const long n = std::strtol(s, &end, 10);
    need(!errno && end != s && !*end && n > 0 && n <= 1048576, "invalid positive integer");
    return int(n);
}
struct Cohort {
    DraftBankBackupRanks ranks{};
    std::array<d::ImageRank, kDraftRanks> images{};
    std::array<void*, kDraftRanks> memory{};
    void* seed = nullptr;
    int hot_n = 0;
    int64_t raw_bytes = 0;
    ~Cohort() {
        for (int r = 0; r < kDraftRanks; ++r) {
            if (ranks[r].device < 0) continue;
            if (rad_dev_set(ranks[r].device) < 0) continue;
            if (ranks[r].stream) (void)rad_stream_sync(ranks[r].stream);
            if (memory[r]) rad_dev_free(memory[r], RAD_MEM_DEVICE);
            if (ranks[r].stream) rad_stream_destroy(ranks[r].stream);
        }
        if (seed && rad_dev_set(0) >= 0) rad_dev_free(seed, RAD_MEM_HOST_PINNED);
    }
    void init(int n) {
        need(rad_dev_count() == kDraftRanks, "oracle requires exactly four visible real cards");
        hot_n = n;
        raw_bytes = int64_t(kDraftStages) * n * kDraftPairBytes;
        for (int r = 0; r < kDraftRanks; ++r) {
            RadDeviceProps props{};
            checked(rad_dev_props(r, &props), "device properties failed");
            need(!props.is_host_backend && !std::strcmp(props.arch, "gfx1201"), "requires four gfx1201 cards; host fallback is not evidence");
            ranks[r].rank = r; ranks[r].device = r;
            checked(rad_dev_set(r), "select card failed");
            checked(rad_stream_create(&ranks[r].stream, 0), "copy stream creation failed");
            memory[r] = rad_dev_alloc(raw_bytes + 3 * 512, RAD_MEM_DEVICE);
            need(memory[r], "card allocation failed");
            auto& image = images[r];
            image.scratch_bytes = kDraftPairBytes;
            for (int i = 0; i < 3 * n; ++i)
                image.spans[image.nspans++] = {static_cast<char*>(memory[r]) + int64_t(i) * kDraftPairBytes, kDraftPairBytes, true};
            for (int i = 0; i < 3; ++i)
                image.spans[image.nspans++] = {static_cast<char*>(memory[r]) + raw_bytes + i * 512, 512, false};
        }
    }
    void seed_bytes(int rank, int span, int64_t bytes) {
        auto* p = static_cast<unsigned char*>(seed);
        for (int64_t i = 0; i < bytes; ++i)
            p[i] = static_cast<unsigned char>((uint64_t(i) * 29 + rank * 31 + span * 7) ^ (uint64_t(i) >> 8));
        p[bytes - 1] = 0; // short copies of the zero tail must still be detected
    }
    void fill_original(DraftBankBackupDeviceIO& io) {
        for (int r = 0; r < kDraftRanks; ++r)
            for (int i = 0; i < images[r].nspans; ++i) {
                const auto& s = images[r].spans[i];
                seed_bytes(r, i, s.bytes);
                checked(io.copy_sync(r, s.base, seed, s.bytes), "initial full-span copy failed");
            }
    }
    void overwrite(int rank, void* where, int64_t bytes) {
        checked(rad_dev_set(rank), "overwrite select failed");
        checked(rad_memset_async(where, 0xa5, bytes, ranks[rank].stream), "target overwrite failed");
        checked(rad_stream_sync(ranks[rank].stream), "target overwrite drain failed");
        checked(rad_dev_set(0), "overwrite rank0 restore failed");
    }
    void overwrite_all() { for (int r = 0; r < kDraftRanks; ++r) overwrite(r, memory[r], raw_bytes); }
};
struct FaultIO final : DraftBankBackupIO {
    DraftBankBackupDeviceIO& real;
    const void* short_destination = nullptr;
    bool armed = false, fired = false;
    explicit FaultIO(DraftBankBackupDeviceIO& value) : real(value) {}
    int64_t allocation_charge(int64_t bytes) const noexcept override { return real.allocation_charge(bytes); }
    bool room(int64_t n, int64_t still, int64_t margin, std::string* why) override { return real.room(n, still, margin, why); }
    void* alloc_pinned(int64_t bytes) override { return real.alloc_pinned(bytes); }
    void free_pinned(void* p) noexcept override { real.free_pinned(p); }
    int copy_sync(int rank, void* dst, const void* src, int64_t bytes) override {
        if (armed && rank == 3 && dst == short_destination && bytes == kDraftPairBytes) {
            armed = false; fired = true;
            // Actual short H2D copy, reported successful, to prove readback catches it.
            return real.copy_sync(rank, dst, src, bytes - 1);
        }
        return real.copy_sync(rank, dst, src, bytes);
    }
};
void timing_proof(const d::Image& image, int n, const DraftBankTimingCounter& original_fnv) {
    const auto& s=image.stats();const auto& t=s.timings;
    need(t.enabled&&!t.overflow&&image.ready(),"timing branch is not a complete valid image");
    const uint64_t raw=uint64_t(4)*uint64_t(3*n)*uint64_t(kDraftPairBytes), masks=4*3*512;
    const uint64_t spans=uint64_t(4)*uint64_t(3*n+3), native_spans=uint64_t(4)*uint64_t(3*n);
    auto c=[&](DraftBankTimingPart p)->const DraftBankTimingCounter&{return t.part[size_t(p)];};
    need(c(DraftBankTimingPart::Allocation).attempts==4&&c(DraftBankTimingPart::Allocation).ok==4&&
         c(DraftBankTimingPart::Allocation).ok_bytes==uint64_t(s.pinned_requested_bytes),"timed allocation accounting differs");
    for(auto p:{DraftBankTimingPart::Zero,DraftBankTimingPart::CaptureD2H,DraftBankTimingPart::CaptureFNV,
                DraftBankTimingPart::CaptureComplement,DraftBankTimingPart::CaptureVerifyD2H,DraftBankTimingPart::CaptureMemcmp}) {
        const auto& q=c(p);need(q.attempts==spans&&q.ok==spans&&q.requested_bytes==raw+masks&&q.ok_bytes==raw+masks,
                              "timed capture count/bytes differ from full native image");
    }
    for(auto p:{DraftBankTimingPart::RestoreH2D,DraftBankTimingPart::RestoreComplement,DraftBankTimingPart::RestoreD2H,DraftBankTimingPart::RestoreMemcmp}) {
        const auto& q=c(p);need(q.attempts==3*native_spans&&q.ok==3*native_spans&&q.requested_bytes==3*raw&&q.ok_bytes==3*raw,
                              "timed restore count/bytes differ from all three full restores");
    }
    const auto& hash=c(DraftBankTimingPart::CaptureFNV);
    need(hash.ns==original_fnv.ns&&hash.attempts==original_fnv.attempts&&hash.ok==original_fnv.ok&&
         hash.requested_bytes==original_fnv.requested_bytes&&hash.ok_bytes==original_fnv.ok_bytes,
         "restore repeated the original capture hash");
    need(c(DraftBankTimingPart::CaptureVerifyD2H).ok_bytes+c(DraftBankTimingPart::RestoreD2H).ok_bytes+
         c(DraftBankTimingPart::InvariantD2H).ok_bytes==uint64_t(s.verify_copy_bytes)&&
         c(DraftBankTimingPart::RestoreH2D).ok_bytes==uint64_t(s.restore_copy_bytes),"timed full-copy totals disagree");
}
struct CopyCall { int rank=0, direction=0; int64_t offset=0, bytes=0; };
struct CopyOrder {
    // One capture (two copies/span) plus one restore (two copies/raw span
    // and two mask observations). N32 is the largest supported native bank.
    std::array<CopyCall, 4*kDraftRanks*d::kSpans> calls{};
    size_t size=0;
};
struct OrderedIO final : DraftBankBackupIO {
    FaultIO& io; const Cohort& cards; CopyOrder& order;
    bool compare=false, tracking=true; size_t cursor=0;
    OrderedIO(FaultIO& f,const Cohort& c,CopyOrder& o,bool old_new):io(f),cards(c),order(o),compare(old_new){}
    int64_t allocation_charge(int64_t n)const noexcept override{return io.allocation_charge(n);}
    bool room(int64_t n,int64_t still,int64_t margin,std::string* why)override{return io.room(n,still,margin,why);}
    void* alloc_pinned(int64_t n)override{return io.alloc_pinned(n);}
    void free_pinned(void* p)noexcept override{io.free_pinned(p);}
    int copy_sync(int rank,void* dst,const void* src,int64_t bytes)override{
        if(tracking){
            need(rank>=0&&rank<kDraftRanks,"invalid copy-order rank");
            const uintptr_t base=uintptr_t(cards.memory[rank]);
            const uint64_t extent=uint64_t(cards.raw_bytes+3*512);
            const bool write=uintptr_t(dst)>=base&&uint64_t(uintptr_t(dst)-base)<extent;
            const uintptr_t ptr=write?uintptr_t(dst):uintptr_t(src);
            need(ptr>=base&&uint64_t(ptr-base)<extent,"copy-order endpoint is not a native card span");
            CopyCall current{rank,int(write),int64_t(ptr-base),bytes};
            if(compare){
                need(cursor<order.size,"new complement submitted extra copies");
                const auto& old=order.calls[cursor];
                need(old.rank==current.rank&&old.direction==current.direction&&old.offset==current.offset&&old.bytes==current.bytes,
                     "new complement changed rank/span/direction/byte copy order");
            }else{
                need(cursor<order.calls.size(),"bounded native copy-order ledger exhausted");
                order.calls[cursor]=current;order.size=cursor+1;
            }
            ++cursor;
        }
        return io.copy_sync(rank,dst,src,bytes);
    }
};
void complement_proof(Cohort& cards,DraftBankBackupDeviceIO& real,DraftBankBackupBudget budget){
    CopyOrder order;
    // All fixed oracle-only trace/call locals are charged alongside the original
    // initializer. No second snapshot or second pair relay is retained.
    constexpr int64_t workspace=sizeof(CopyOrder)+sizeof(OrderedIO)+sizeof(FaultIO)+
        2*sizeof(d::Image)+sizeof(DraftBankBackupBudget)+256;
    need(budget.controller_metadata_bytes>=0&&budget.controller_metadata_bytes<=INT64_MAX-workspace,
         "complement oracle metadata charge overflow");
    budget.controller_metadata_bytes+=workspace;
    std::string why;uint64_t hash=0;
    int64_t capture_bytes=0,restore_bytes=0,verify_bytes=0;
    cards.fill_original(real);
    {
        FaultIO fault(real);OrderedIO io(fault,cards,order,false);d::Image old;
        budget.complement_words=false;
        checked(old.capture(cards.images,budget,sizeof(old),&io,&why),why);
        old.mark_borrowed();cards.overwrite_all();checked(old.restore_verify_all(&why),why);
        need(old.ready()&&!old.failed(),"old complement full-byte baseline is incomplete");
        hash=old.stats().image_hash;capture_bytes=old.stats().capture_copy_bytes;
        restore_bytes=old.stats().restore_copy_bytes;verify_bytes=old.stats().verify_copy_bytes;
    }
    {
        FaultIO fault(real);OrderedIO io(fault,cards,order,true);d::Image words;
        budget.complement_words=true;
        checked(words.capture(cards.images,budget,sizeof(words),&io,&why),why);
        words.mark_borrowed();cards.overwrite_all();checked(words.restore_verify_all(&why),why);
        need(words.ready()&&!words.failed()&&words.stats().complement_words&&words.stats().image_hash==hash,
             "word complement changed the immutable full-bank hash or readiness");
        need(io.cursor==order.size&&words.stats().capture_copy_bytes==capture_bytes&&
             words.stats().restore_copy_bytes==restore_bytes&&words.stats().verify_copy_bytes==verify_bytes,
             "old/new native copy coverage differs");
        // Real successful-but-short final-byte D2H on the last card. The source
        // device bytes are correct, so the complement itself must make the
        // missing readback byte disagree (including the seeded zero tail).
        io.tracking=false;words.mark_borrowed();cards.overwrite_all();
        fault.short_destination=words.scratch(3);fault.armed=true;
        need(words.restore_verify_all(&why)<0&&fault.fired&&words.failed()&&!words.ready()&&
             why.find("strict byte")!=std::string::npos,"word complement did not catch a short real readback");
        need(words.restore_verify_all(&why)<0&&!words.ready(),"failed word complement became query-ready");
    }
    std::printf("PASS: DSBANK_COMPLEMENT_CARD ranks=4 hot_n=%d word_bytes=8 old_new_hash=1 full_memcmp=1 copy_order=exact copy_bytes=exact short_readback_fatal=1\n",cards.hot_n);
}
void run(int n, int backup_mib, bool timings, bool complement_words) {
    Cohort cards;
    cards.init(n);
    DraftBankBackupDeviceIO real(cards.ranks);
    std::string why;
    constexpr int64_t headroom = int64_t{2} << 30;
    const int64_t seed_charge = real.allocation_charge(kDraftPairBytes);
    need(seed_charge > 0 && real.room(seed_charge, 0, headroom, &why), why);
    cards.seed = real.alloc_pinned(kDraftPairBytes);
    need(cards.seed, "bounded initial-pattern scratch allocation failed");
    cards.fill_original(real);
    DraftBankBackupBudget budget;
    budget.backup_mib = backup_mib;
    budget.timings = timings;
    budget.complement_words = complement_words;
    budget.system_headroom_bytes = headroom;
    // Explicitly charge oracle-only initializer/metadata alongside image storage.
    budget.controller_metadata_bytes = seed_charge + sizeof(cards) + sizeof(real) + sizeof(FaultIO);
    {
        FaultIO io(real);
        d::Image image;
        checked(image.capture(cards.images, budget, sizeof(image), &io, &why), why);
        need(image.ready() && !image.failed(), "capture did not publish a complete ready image");
        std::printf("bank-card captured ranks=4 hot_n=%d raw_bytes=%lld charged=%lld hash=%016llx\n", n,
                    (long long)image.stats().raw_bytes, (long long)image.stats().total_charge_bytes,
                    (unsigned long long)image.stats().image_hash);
        const auto capture_fnv=image.stats().timings.part[size_t(DraftBankTimingPart::CaptureFNV)];
        for (int pass = 0; pass < 2; ++pass) {
            image.mark_borrowed();
            need(!image.ready(), "borrowed image remained ready");
            cards.overwrite_all();
            checked(image.restore_verify_all(&why), why);
            need(image.ready() && !image.failed(), "repeated restore was not fully verified");
        }
        // A later-rank target-fill failure must allow a complete, verified rollback.
        image.mark_borrowed();
        cards.overwrite(0, cards.memory[0], kDraftPairBytes);
        cards.overwrite(3, cards.memory[3], kDraftPairBytes / 2 + 31);
        checked(image.restore_verify_all(&why), why);
        need(image.ready() && image.stats().completed_restores == 3, "partial-fill rollback incomplete");
        if(timings)timing_proof(image,n,capture_fnv);
        // A successful-but-short later-rank restore must poison the whole cohort.
        image.mark_borrowed();
        cards.overwrite_all();
        io.short_destination = cards.memory[3]; io.armed = true;
        need(image.restore_verify_all(&why) < 0 && io.fired && image.failed() && !image.ready(), "short restore escaped strict readback");
        need(image.restore_verify_all(&why) < 0 && !image.ready(), "poisoned cohort became query-ready");
    }
    if(complement_words)complement_proof(cards,real,budget);
    // The correction planes are checked, never silently repaired. Keep their
    // original stable pointers and test a real final-byte corruption separately.
    {
        std::array<d::ImageRank, kDraftRanks> masks{};
        for (int r = 0; r < kDraftRanks; ++r) {
            masks[r].scratch_bytes = 512;
            for (int i = 0; i < 3; ++i)
                masks[r].spans[masks[r].nspans++] = cards.images[r].spans[3 * n + i];
        }
        DraftBankBackupBudget small;
        small.backup_mib = 32; small.system_headroom_bytes = headroom;
        small.timings = timings;
        small.complement_words = complement_words;
        d::Image image;
        checked(image.capture(masks, small, sizeof(image), &real, &why), why);
        void* changed = static_cast<char*>(masks[2].spans[1].base) + 511;
        cards.overwrite(2, changed, 1);
        need(image.restore_verify_all(&why) < 0 && image.failed() && !image.ready(), "changed correction plane was repaired or admitted");
        checked(real.copy_sync(2, cards.seed, changed, 1), "mask fault readback failed");
        need(*static_cast<unsigned char*>(cards.seed) == 0xa5, "failed mask invariant was silently repaired");
    }
    std::printf("PASS: DSBANK_BACKUP_CARD ranks=4 hot_n=%d restores=3 partial_fill_rollback=1 short_restore_fatal=1 masks=1\n", n);
    if(timings)std::printf("PASS: DSBANK_TIMING_CARD ranks=4 hot_n=%d restores=3 copy_counts=exact copy_bytes=exact repeated_fnv=0 strict_readback=1\n",n);
}
} // namespace
int main(int argc, char** argv) {
    try {
        bool inside = false, timings = false, complement_words = false;
        int n = 0, cap = 0;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--inside-guarded-window") && !inside) inside = true;
            else if (!std::strcmp(argv[i], "--timings") && !timings) timings = true;
            else if (!std::strcmp(argv[i], "--complement-words") && !complement_words) complement_words = true;
            else if (!std::strcmp(argv[i], "--hot-n") && !n && i + 1 < argc) n = number(argv[++i]);
            else if (!std::strcmp(argv[i], "--backup-mib") && !cap && i + 1 < argc) cap = number(argv[++i]);
            else throw std::runtime_error("unknown/duplicate/incomplete argument");
        }
        need(inside && n >= 1 && n <= 32 && cap > 0, "requires --inside-guarded-window --hot-n 1..32 --backup-mib N before any device access");
        run(n, cap, timings, complement_words);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL DSBANK_BACKUP_CARD: %s\n", e.what());
        return 1;
    }
}
