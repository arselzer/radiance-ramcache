/* Real-card byte oracle for the new bank backup mechanism. No model/kernel
 * arithmetic is changed. Native Program/Plan/alias scope is covered by the host
 * fixture and actual model startup, separately from this raw-copy proof.
 * Run ONLY inside the two-lock stop/restore window, before model lending. */
#include "place/draft_bank_backup.h"
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
void run(int n, int backup_mib) {
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
        // A successful-but-short later-rank restore must poison the whole cohort.
        image.mark_borrowed();
        cards.overwrite_all();
        io.short_destination = cards.memory[3]; io.armed = true;
        need(image.restore_verify_all(&why) < 0 && io.fired && image.failed() && !image.ready(), "short restore escaped strict readback");
        need(image.restore_verify_all(&why) < 0 && !image.ready(), "poisoned cohort became query-ready");
    }
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
        d::Image image;
        checked(image.capture(masks, small, sizeof(image), &real, &why), why);
        void* changed = static_cast<char*>(masks[2].spans[1].base) + 511;
        cards.overwrite(2, changed, 1);
        need(image.restore_verify_all(&why) < 0 && image.failed() && !image.ready(), "changed correction plane was repaired or admitted");
        checked(real.copy_sync(2, cards.seed, changed, 1), "mask fault readback failed");
        need(*static_cast<unsigned char*>(cards.seed) == 0xa5, "failed mask invariant was silently repaired");
    }
    std::printf("PASS: DSBANK_BACKUP_CARD ranks=4 hot_n=%d restores=3 partial_fill_rollback=1 short_restore_fatal=1 masks=1\n", n);
}
} // namespace
int main(int argc, char** argv) {
    try {
        bool inside = false;
        int n = 0, cap = 0;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--inside-guarded-window") && !inside) inside = true;
            else if (!std::strcmp(argv[i], "--hot-n") && !n && i + 1 < argc) n = number(argv[++i]);
            else if (!std::strcmp(argv[i], "--backup-mib") && !cap && i + 1 < argc) cap = number(argv[++i]);
            else throw std::runtime_error("unknown/duplicate/incomplete argument");
        }
        need(inside && n >= 1 && n <= 32 && cap > 0, "requires --inside-guarded-window --hot-n 1..32 --backup-mib N before any device access");
        run(n, cap);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL DSBANK_BACKUP_CARD: %s\n", e.what());
        return 1;
    }
}
