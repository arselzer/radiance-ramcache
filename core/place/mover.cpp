/* mover.cpp -- the mover's mechanics; mover.h has the design.
 *
 * Everything here runs on the dispatch thread. Transfers run asynchronously on the mover's
 * stream, gated behind an event recorded after the layer's launches, but every table edit is
 * applied by this thread at a layer boundary (see Mover::Move).
 */
#include "mover.h"
#include "format/dspark_rad_banks.h"
#include "../device/device.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <queue>
#include <thread>
#include <unordered_map>

#include <sys/mman.h>

namespace rad {

/* A buffer for O_DIRECT reads of a unit's planes (Mover::read_unit_planes): 2 MiB-aligned, a whole
 * number of 2 MiB, and on transparent huge pages where the kernel grants them. A buffer of
 * scattered 4 KiB pages limits a drive request to its 256 segments (1 MiB); on huge pages one
 * request spans the drive's 2 MiB max_sectors (measured: an 18.8 MB miss read 3.03 -> 2.83 ms,
 * 6.2 -> 6.7 GB/s). *cap gets the size; freed with std::free. */
uint8_t* alloc_plane_buffer(int64_t bytes, int64_t* cap) {
    /* RADIANCE_EXPERIMENTAL_DS41_PLANE_HUGEPAGES=0: 4 KiB pages, as before (an A/B switch). */
    static const bool huge = [] {
        const char* s = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PLANE_HUGEPAGES");
        return !s || std::strcmp(s, "0") != 0;
    }();
    const int64_t a = huge ? (int64_t)2 << 20 : DirectFile::alignment();
    const int64_t n = (bytes + a - 1) / a * a;
    void* p = std::aligned_alloc((size_t)a, (size_t)n);
    if (!p) { *cap = 0; return nullptr; }
    if (huge) (void)::madvise(p, (size_t)n, MADV_HUGEPAGE);
    *cap = n;
    return static_cast<uint8_t*>(p);
}

/* ------------------------------------------------------------------ the file tier */
int FileTier::open(const std::string& path) {
    /* No RAD_DIO_ALLOW_BUFFERED: a silent buffered fallback would make the tier a page-cache pump
     * that is several times slower, evicts the container's own mapped pages and does not show in
     * a profile. A tier pointed at tmpfs gets an error instead. */
    const int rc = f_.open(path.c_str(), RAD_DIO_READ);
    if (rc < 0)
        RAD_ERR("file tier: cannot open %s with O_DIRECT. The tier needs a filesystem that "
                "supports it -- tmpfs and some overlay mounts do not.", path.c_str());
    return rc;
}

int FileTier::read(uint64_t offset, int64_t bytes, void* dst, int64_t dst_cap) {
    if (!f_.is_open()) return RAD_E_STATE;
    if (!dst || bytes <= 0) return RAD_E_INVAL;
    /* The extent rounded up to the alignment: the unit stride already covers it, and reading
     * one extra block is cheaper than a branch on the step path. The rounding may pass end of
     * file when the extent is the container's last; only `bytes` must arrive. */
    const int64_t want = direct_align(bytes);
    if (want > dst_cap) {
        RAD_ERR("file tier: a %s read needs a %s aligned buffer, the staging slot is %s",
                humanb(bytes).c_str(), humanb(want).c_str(), humanb(dst_cap).c_str());
        return RAD_E_SCRATCH;
    }
    RAD_TRY(f_.read_covering(dst, (int64_t)offset, want, bytes));
    return RAD_OK;
}

int FileTier::remember_epoch() {
    epoch_ready_ = false;
    if (!f_.is_open() || ::fstat(f_.fd(), &epoch_) || !S_ISREG(epoch_.st_mode))
        return RAD_E_IO;
    epoch_ready_ = true;
    return RAD_OK;
}

bool FileTier::same_epoch() const {
    struct stat current{};
    return epoch_ready_ && f_.is_open() && !::fstat(f_.fd(), &current) &&
        current.st_dev == epoch_.st_dev && current.st_ino == epoch_.st_ino &&
        current.st_size == epoch_.st_size && current.st_mtim.tv_sec == epoch_.st_mtim.tv_sec &&
        current.st_mtim.tv_nsec == epoch_.st_mtim.tv_nsec &&
        current.st_ctim.tv_sec == epoch_.st_ctim.tv_sec &&
        current.st_ctim.tv_nsec == epoch_.st_ctim.tv_nsec;
}

/* ------------------------------------------------------------------ residency */
void Residency::init(const Program& prog) {
    slot_.assign(prog.weights.size(), WeightSlot{});
    noted_.assign(prog.weights.size(), 0);
    changed_.clear();
    ++epoch_;
}

/* The generation bump is the whole staleness protocol, with no lock: at this point both copies of
 * the bytes are valid (the mover wrote a shadow slot and has not freed the old one), so a reader
 * on either side of the write names correct bytes. The quarantine keeps that true. Why both
 * writers bump: mover.h. */
void Residency::begin_transfer(rad_weight w, void* ptr, RadEvent ready) {
    if ((size_t)w >= slot_.size()) return;
    slot_[w].ptr = ptr;
    slot_[w].ready = ready;
    ++slot_[w].generation;
    ++epoch_;
    note(w);
}

/* ------------------------------------------------------------------ the mover */
Mover::~Mover() {
    /* No CPU reader may retain RelayIn/window/cache references through teardown. */
    relay_workers_.reset();
    //No error path may destroy a mapped owner while an uncertain queued copy can read it.
    if (mapped_file_drain() != RAD_OK) {
        RAD_ERR("mapped file cache: transfer completion unknown at teardown; retaining process owners");
        std::_Exit(88);
    }
    /* Drain the stream first: its queued gate waits, event records and copies refer to the
     * events and memory freed below. */
    if (mover_) {
        (void)flush_moves();
        rad_stream_sync(mover_);
        rad_stream_destroy(mover_);
    }
    for (Move& m : moves_) if (m.ev) rad_event_destroy(m.ev);
    for (RadEvent e : lead_) if (e) rad_event_destroy(e);
    for (RadEvent e : win_ev_) if (e) rad_event_destroy(e);
    if (gate_) rad_event_destroy(gate_);
    /* Only what this object allocated; a Pool reservation belongs to the Pool. */
    if (owns_memory_) {
        for (ClassPools& c : pools_) {
            if (c.slab_base) rad_dev_free(c.slab_base, RAD_MEM_DEVICE);
            if (c.pool_base) rad_dev_free(c.pool_base, RAD_MEM_HOST_PINNED);
        }
        if (static_vram_) rad_dev_free(static_vram_, RAD_MEM_DEVICE);
        if (stage_) rad_dev_free(stage_, RAD_MEM_HOST_PINNED);
        if (file_vram_) rad_dev_free(file_vram_, RAD_MEM_DEVICE);
        if (windows_) rad_dev_free(windows_, RAD_MEM_HOST_PINNED);
    }
    /* The pageable region is always ours: Pools has no pageable pool, since pageable memory is
     * not a device-visible budget and nothing else contends for it. */
    if (pageable_) rad_dev_free(pageable_, RAD_MEM_HOST);
}

/* The one place the mover asks for memory. Device and pinned memory come out of the resolved
 * budgets (spec §6); the direct allocation is for the placement test, which runs without a Pools
 * (init() warns). */
void* Mover::reserve(const char* what, int64_t bytes, int kind) {
    if (bytes <= 0) return nullptr;
    if (budget_) {
        Pool& p = kind == RAD_MEM_DEVICE ? budget_->weights() : budget_->host_pinned();
        return p.reserve(what, bytes, RAD_ALIGN_UNIT);
    }
    owns_memory_ = true;
    return rad_dev_alloc(bytes, kind);
}

bool Mover::source_file_epoch() const {
    if(!cfg_.source_banks)return file_.same_epoch();
    return cfg_.source_banks->same_epoch()&&(!file_.ok()||file_.same_epoch())&&
        (!draft_file_.ok()||draft_file_.same_epoch());
}
int Mover::read_source_file(uint32_t bank,uint64_t offset,int64_t bytes,void*dst,int64_t cap) {
    if(bank>1||(bank&&!cfg_.source_banks))return RAD_E_STATE;
    if(cfg_.source_banks&&!source_file_epoch())return RAD_E_STATE;
    const int st=(bank?draft_file_:file_).read(offset,bytes,dst,cap);if(st<0)return st;
    return cfg_.source_banks&&!source_file_epoch()?RAD_E_STATE:RAD_OK;
}
int Mover::init(const Program& prog, const Plan& plan, const MoverConfig& cfg, Pools* budget) {
    if (cfg.native_file_cache_bytes < 0) return RAD_E_INVAL;
    if(cfg.source_banks){
        if(!cfg.source_banks->dual()||!cfg.source_banks->same_epoch())return RAD_E_STATE;
        for(size_t i=1;i<prog.weights.size();++i){const auto&w=prog.weights[i];
            if(w.decl.optional&&w.source_entry_index<0)continue;
            DsparkRadBanks::Entry owner;RAD_TRY(cfg.source_banks->bound(w,&owner));}
    }else for(const auto&w:prog.weights)if(w.file_id||w.source_bank_generation)return RAD_E_STATE;
    const bool want_native_file_workers = native_read_workers_opt_in(
        std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_FILE_WORKERS"));
    const char* mapped_env = std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SOURCE");
    const bool want_mapped = mapped_env && std::strcmp(mapped_env,"1") == 0;
    const char* fill_env=std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_DIRECT_FILL");
    const bool want_direct_fill=fill_env&&std::strcmp(fill_env,"1")==0;
    const char* slabs_env=std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_MAPPED_SLABS");
    if(want_direct_fill&&(!want_mapped||cfg.native_file_cache_bytes<=0||!slabs_env||std::strcmp(slabs_env,"1")||
        !std::any_of(plan.units.begin(),plan.units.end(),[](const MoveUnit& mu){return mu.tier==Tier::SSD;}))){
        RAD_ERR("mapped direct fill requires mapped COPY-source cache and retained slabs");return RAD_E_UNSUPPORTED;
    }
    /* Retire all callbacks before cache/RelayIn/window/namespace lifetimes change. */
    relay_workers_.reset();
    RAD_TRY(mapped_file_drain());
    RAD_TRY(mapped_file_cache_.clear());
    mapped_file_cache_on_ = false;
    mapped_direct_fill_on_ = want_direct_fill;
    native_file_cache_on_ = false;
    RAD_TRY(native_file_cache_.clear());
    /* A refused cache clear must retain the previous namespace and worker policy. */
    native_file_workers_on_ = want_native_file_workers;
    prog_ = &prog;
    plan_ = &plan;
    cfg_ = cfg;
    draft_file_.close(); //old copies/cache users are proven retired before owner replacement
    budget_ = budget;
    stats_ = MoverStats{};
    if (!budget_)
        RAD_WARN("mover: no Pools given; allocating the slab and host pool directly (test only)");

    unit_.assign(plan.units.size(), UnitState{});
    copying_.assign(plan.units.size(), 0);
    pool_layer_.assign(plan.classes.size(), {});
    donors_.assign(plan.classes.size(), DonorRank{});
    n_layers_ = 0;
    for (const MoveUnit& mu : plan.units) n_layers_ = std::max(n_layers_, mu.layer + 1);
    pools_.clear();
    pools_.resize(plan.classes.size());

    /* ---- how many slots each class needs, counted before anything is allocated ------------- */
    std::vector<int64_t> n_slab(plan.classes.size(), 0), n_pool(plan.classes.size(), 0);
    int64_t static_need = 0, pageable_need = 0;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        UnitState& us = unit_[u];
        us.tier = mu.tier;
        us.site = mu.site;
        us.klass = mu.slab_class;
        us.bytes = mu.bytes;
        if (mu.slab_class < 0) continue;
        if (mu.tier == Tier::VRAM && mu.slab_slot >= 0) {
            n_slab[(size_t)mu.slab_class] = std::max(n_slab[(size_t)mu.slab_class],
                                                     (int64_t)mu.slab_slot + 1);
        } else if (mu.tier == Tier::VRAM) {
            us.static_off = align_up(static_need, RAD_ALIGN_UNIT);
            static_need = us.static_off + mu.bytes;
        } else if (mu.tier == Tier::HostPinned) {
            us.pool_slot = n_pool[(size_t)mu.slab_class]++;
            pool_layer_[(size_t)mu.slab_class].push_back(mu.layer);
        } else if (mu.tier == Tier::HostPageable) {
            us.static_off = align_up(pageable_need, RAD_ALIGN_UNIT);
            pageable_need = us.static_off + mu.bytes;
        }
    }

    /* ---- allocate ------------------------------------------------------------------------- */
    int64_t total_shadow = 0, total_turn = 0, total_slab = 0, total_pool = 0;
    for (size_t c = 0; c < plan.classes.size(); ++c) {
        const SlabClass& sc = plan.classes[c];
        ClassPools& cp = pools_[c];
        cp.stride = sc.stride;

        /* Shadow slots follow the resident ones, so resident indices do not depend on the engine;
         * a static plan has none. Use the plan's count, which the planner budgeted. */
        const int64_t shadow = plan.dynamic ? sc.n_shadow : 0;
        /* Shadow and staging slots share one free list: both are somewhere to copy into that
         * nothing reads. */
        const int64_t slab_slots = n_slab[c] + shadow + sc.n_stage;
        /* Flex slots are added later by lend_from(), once the lender's memory is carved. */
        cp.n_fixed = slab_slots;
        cp.n_work = shadow + sc.n_stage;
        cp.n_turn = std::min<int64_t>(std::max(cfg.turnover_slots, 0),
                                      shadow * RAD_PLACE_TURNOVER_SLOTS / RAD_PLACE_SHADOW_SLOTS);
        total_turn += cp.n_turn;
        cp.slab.init(slab_slots, sc.stride);
        total_slab += slab_slots;
        total_pool += n_pool[c];
        if (slab_slots > 0) {
            /* Tail padding: a kernel assembling a packed code window from a wide read may read a
             * few bytes past the last slot; without padding that runs off the allocation and
             * faults the card. */
            cp.slab_base = reserve(sc.name.c_str(), slab_slots * sc.stride + RAD_ALIGN_UNIT,
                                   RAD_MEM_DEVICE);
            if (!cp.slab_base) {
                RAD_ERR("mover: no VRAM for the '%s' slab: %lld slots of %s",
                        sc.name.c_str(), (long long)slab_slots, humanb(sc.stride).c_str());
                return RAD_E_NOMEM;
            }
        }
        for (int64_t s = n_slab[c]; s < slab_slots; ++s) cp.slab.add_free(s);
        total_shadow += shadow + sc.n_stage;

        cp.pool.init(n_pool[c], sc.stride);
        if (n_pool[c] > 0) {
            cp.pool_base = reserve(sc.name.c_str(), n_pool[c] * sc.stride, RAD_MEM_HOST_PINNED);
            if (!cp.pool_base) {
                RAD_ERR("mover: no pinned host memory for the '%s' pool: %lld slots of %s",
                        sc.name.c_str(), (long long)n_pool[c], humanb(sc.stride).c_str());
                return RAD_E_NOMEM;
            }
            /* The pool's base as the card addresses it (equal to the host address on a unified
             * backend). Without one there is no zero-copy tier. */
            cp.pool_dev = rad_dev_device_ptr(cp.pool_base);
            /* Warned, not refused: a pool nothing zero-copies out of is not an error. The host
             * pointer is valid on the device only where the two coincide. A deployment's pool is
             * RAD_MEM_HOST_MAPPED (core/mem/pools.cpp), so this fires only on the test path. */
            if (!cp.pool_dev) {
                RAD_WARN("mover: the '%s' host pool has no device address; a kernel reading it "
                         "zero-copy would fault", sc.name.c_str());
                cp.pool_dev = cp.pool_base;
            }
        }
        /* The pool free list starts empty: every pooled unit occupies its slot. The engine primes
         * off the shadow slots and then alternates, a promotion freeing the pool slot a demotion
         * needs.
         *
         * Nothing is memset; the load phase writes every weight. rad_memset_async on
         * RAD_MEM_HOST fails on a real card but succeeds on the host backend, so use std::memset
         * if a host buffer ever needs clearing. */
    }

    if (static_need > 0) {
        static_vram_ = reserve("static weights", static_need, RAD_MEM_DEVICE);
        if (!static_vram_) {
            /* Log the budget arithmetic too: the slab, with its shadow and staging slots, was
             * reserved first, so this failure means the budget covered the resident set but not
             * the static weights beside it. */
            if (budget_)
                RAD_ERR("mover: the weight budget is %s and the slab already took %s of it, "
                        "leaving %s",
                        humanb(budget_->weights().bytes()).c_str(),
                        humanb(budget_->weights().used()).c_str(),
                        humanb(budget_->weights().avail()).c_str());
            RAD_ERR("mover: no VRAM for the %s of statically placed weights",
                    humanb(static_need).c_str());
            return RAD_E_NOMEM;
        }
        static_bytes_ = static_need;
    }
    if (pageable_need > 0) {
        pageable_ = rad_dev_alloc(pageable_need, RAD_MEM_HOST);
        if (!pageable_) {
            RAD_ERR("mover: no pageable host memory for %s of weights",
                    humanb(pageable_need).c_str());
            return RAD_E_NOMEM;
        }
        pageable_bytes_ = pageable_need;
    }

    /* ---- the file tier -------------------------------------------------------------------- */
    bool want_file = false;
    for (const MoveUnit& mu : plan.units) if (mu.tier == Tier::SSD) want_file = true;
    if (want_file) {
        if (cfg_.container_path.empty()) {
            RAD_ERR("mover: the plan puts weights on the file tier but no container path was "
                    "given");
            return RAD_E_INVAL;
        }
        RAD_TRY(file_.open(cfg_.container_path));
        if(cfg_.source_banks){
            RAD_TRY(cfg_.source_banks->direct_owner_matches(0,file_.borrowed_fd_for_owner_identity()));
            bool want_draft=false;
            for(const auto&mu:plan.units)if(mu.tier==Tier::SSD)for(const auto h:mu.weights)
                if(prog.weights[h].file_id==1)want_draft=true;
            if(want_draft){RAD_TRY(draft_file_.open(cfg_.source_banks->file(1)->path()));
                RAD_TRY(cfg_.source_banks->direct_owner_matches(1,draft_file_.borrowed_fd_for_owner_identity()));
                RAD_TRY(draft_file_.remember_epoch());}
        }
        if(cfg_.native_file_cache_bytes>0||cfg_.source_banks)RAD_TRY(file_.remember_epoch());
        if (cfg_.native_file_cache_bytes > 0) {
            if (want_mapped) {
                RAD_TRY(mapped_file_cache_.configure(size_t(cfg_.native_file_cache_bytes),
                                                     prog.weights.size(),sizeof(mapped_windows_) +
                                                         sizeof(mapped_copy_hooks_) + 128,cfg_.native_cache_allocation_device)); //fixed flags/stats/alignment reserve
                mapped_file_cache_on_ = true;
                if(mapped_direct_fill_on_)RAD_INFO("native mapped direct fill: whole-weight unpublished reservation, parallel CPU fill, atomic first lease; unchanged H2D event retirement");
                if (const char* pg = std::getenv("RADIANCE_EXPERIMENTAL_NATIVE_CACHE_PREGROW");
                    pg && !std::strcmp(pg, "1")) {
                    const auto t0 = std::chrono::steady_clock::now();
                    size_t slabs = 0;
                    RAD_TRY(mapped_file_cache_.pregrow(&slabs));
                    RAD_INFO("native mapped cache: %zu slabs pinned at startup in %.2f s, so no "
                             "request pays for growing them", slabs,
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                RAD_INFO("native mapped COPY-source cache: %s policy cap/rank; conservative no-SVM "
                         "retained backing + metadata (not physical RSS); legacy cache disabled",
                         humanb(cfg_.native_file_cache_bytes).c_str());
            } else {
                RAD_TRY(native_file_cache_.configure(cfg_.native_file_cache_bytes, prog.weights.size()));
                native_file_cache_on_ = true;
                RAD_INFO("native file RAM cache: %s for this rank, including %s of entry metadata",
                         humanb(cfg_.native_file_cache_bytes).c_str(),
                         humanb(native_file_cache_.stats().metadata_bytes).c_str());
            }
        }
        /* A weight the file tier relays on each read needs the relayout (set_file_relayout). */
        int64_t relaid = 0;
        for (const MoveUnit& mu : plan.units) {
            if (mu.tier != Tier::SSD) continue;
            for (rad_weight w : mu.weights) {
                const WeightInfo& wi = prog.weights[w];
                if (wi.widen) {
                    RAD_ERR("mover: '%s' is on the file tier and widened at load, which a file "
                            "read does not do", wi.name.c_str());
                    return RAD_E_STATE;
                }
                if (!wi.identity) ++relaid;
            }
        }
        if (relaid && !relayout_) {
            RAD_ERR("mover: the plan puts %lld weight(s) whose stored form is a relayout on the "
                    "file tier, and the mover was given no relayout to make it on a read",
                    (long long)relaid);
            return RAD_E_STATE;
        }
        if (relaid) {
            const unsigned hw = std::thread::hardware_concurrency();
            relay_threads_ = (int)std::max(1u, std::min(8u, hw / 2));
            relay_in_.clear();
            relay_in_.resize((size_t)relay_threads_);
        }
        /* Staging slots are sized by the widest unit actually on the file tier, not the widest
         * class. The widest class is usually the embedding table (gigabytes, and never on the
         * file tier), and sizing on it would reserve pinned memory for a weight that never
         * arrives, possibly failing the file tier for lack of memory. */
        int64_t widest = 0;
        for (const MoveUnit& mu : plan.units) {
            if (mu.tier != Tier::SSD || mu.slab_class < 0) continue;
            widest = std::max(widest, plan.classes[(size_t)mu.slab_class].stride);
        }
        stage_stride_ = FileTier::direct_align(widest);
        if (cfg_.stage_slots > 0 && stage_stride_ > 0) {
            stage_ = reserve("file-tier staging", (int64_t)cfg_.stage_slots * stage_stride_, RAD_MEM_HOST_PINNED);
            if (!stage_) {
                RAD_ERR("mover: no pinned memory for the file tier's %d staging slots of %s",
                        cfg_.stage_slots, humanb(stage_stride_).c_str());
                return RAD_E_NOMEM;
            }
        }
    }

    /* ---- the file stager's buffers ---------------------------------------------------------
     *
     * Exactly what the plan charged: two VRAM buffers from the weight budget and the read windows
     * from the host pool's file-staging reserve. The plan sizes them only when a routed unit is
     * on the file tier. */
    if (plan.file_stage_vram > 0) {
        if (!file_.ok()) {
            RAD_ERR("mover: the plan reads routed experts from the container and the file tier "
                    "did not open");
            return RAD_E_STATE;
        }
        file_vram_ = (char*)reserve("file-tier layer buffers", plan.file_stage_vram, RAD_MEM_DEVICE);
        if (!file_vram_) {
            RAD_ERR("mover: no VRAM for the two %s buffers a routed layer is read into from the "
                    "container", humanb(plan.file_layer_max).c_str());
            return RAD_E_NOMEM;
        }
        /* Two halves of what was charged, each at least the largest layer and aligned, so the
         * tail pad the plan added stays past the second buffer's end. */
        file_buf_bytes_ = (plan.file_stage_vram - RAD_ALIGN_UNIT) / 2 / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
        if (file_buf_bytes_ < plan.file_layer_max) {
            RAD_ERR("mover: the plan charged %s for two file-tier layer buffers of %s",
                    humanb(plan.file_stage_vram).c_str(), humanb(plan.file_layer_max).c_str());
            return RAD_E_STATE;
        }
        if (plan.file_windows < (int64_t)RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES) {
            RAD_ERR("mover: the plan reads routed experts from the container but reserved no read "
                    "windows for them");
            return RAD_E_STATE;
        }
        windows_ = (char*)reserve("file-tier read windows",
                                  (int64_t)RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES,
                                  RAD_MEM_HOST_PINNED);
        if (!windows_) {
            RAD_ERR("mover: no pinned memory for the file tier's %d read windows of %s",
                    RAD_PLACE_FILE_WINDOWS, humanb(RAD_PLACE_FILE_WINDOW_BYTES).c_str());
            return RAD_E_NOMEM;
        }
        n_windows_ = RAD_PLACE_FILE_WINDOWS;
    }

    /* ---- streams, events, move records ------------------------------------------------------ */
    RAD_TRY(rad_stream_create(&mover_, /*high_priority=*/0));
    /* Every event here is local: they order the mover's queue against this card's compute queue
     * and tell the host when a slot is free, and the host never reads the ordered bytes through
     * them. A copy out to host memory carries system scope on its own packets. */
    RAD_TRY(rad_event_create_local(&gate_));
    /* A window's event is waited on by the HOST, before the window is read into again. */
    win_ev_.assign((size_t)n_windows_, nullptr);
    win_rec_.assign((size_t)n_windows_, 0);
    for (RadEvent& e : win_ev_) RAD_TRY(rad_event_create_as(&e, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));

    const int lead = std::max(1, cfg_.max_lead_dispatches);
    lead_.assign((size_t)lead + 1, nullptr);
    for (RadEvent& e : lead_) RAD_TRY(rad_event_create_as(&e, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));

    /* Two records per free slot (shadow or turnover): a swap is two transfers, and with one
     * record per slot the demotion could never be in flight beside its promotion, so the resident
     * set would only grow. The floor lets a one-shadow plan still hold a swap. */
    const int64_t n_records = std::max<int64_t>(2 * (total_shadow + total_turn), 4);
    moves_.assign((size_t)n_records, Move{});
    for (Move& m : moves_) RAD_TRY(rad_event_create_local(&m.ev));

    /* ---- publish the initial addresses ------------------------------------------------------ */
    res_.init(prog);
    for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) {
        unit_[(size_t)u].slab_slot = plan.units[(size_t)u].slab_slot;
        republish(u);
    }

    RAD_INFO("mover: %lld slab slots (%lld shadow, %lld more kept free out of a loan), %lld pinned "
             "pool slots, %s static vram%s",
             (long long)total_slab, (long long)total_shadow, (long long)total_turn,
             (long long)total_pool,
             humanb(static_bytes_).c_str(), file_.ok() ? ", file tier open" : "");
    /* The pageable region is charged to no budget, so this is the only place it is reported. */
    if (pageable_bytes_ > 0)
        RAD_INFO("mover: %s of pageable host memory for the units no tier could hold",
                 humanb(pageable_bytes_).c_str());
    return RAD_OK;
}

int Mover::bind_compute_stream(RadStream s) {
    if (!s) return RAD_E_INVAL;
    compute_ = s;
    return RAD_OK;
}

void* Mover::unit_base(int32_t unit) const {
    if (unit < 0 || (size_t)unit >= unit_.size()) return nullptr;
    const UnitState& us = unit_[(size_t)unit];
    if (us.klass < 0) return nullptr;
    const ClassPools& cp = pools_[(size_t)us.klass];
    if (us.slab_slot >= 0 && (cp.is_flex(us.slab_slot) || cp.slab_base))
        return cp.slot_ptr(us.slab_slot);
    if (us.tier == Tier::VRAM && us.static_off >= 0 && static_vram_)
        return (char*)static_vram_ + us.static_off;
    if (us.tier == Tier::HostPinned && us.pool_slot >= 0 && cp.pool_dev)
        return (char*)cp.pool_dev + us.pool_slot * cp.stride;
    if (us.tier == Tier::HostPageable && us.static_off >= 0 && pageable_)
        return (char*)pageable_ + us.static_off;
    /* An unstaged file-tier unit has no address; a kernel cannot read from disk, so a caller
     * that gets null must stage the unit or refuse. */
    return nullptr;
}

void Mover::republish(int32_t unit) {
    const MoveUnit& mu = plan_->units[(size_t)unit];
    char* base = (char*)unit_base(unit);
    for (size_t i = 0; i < mu.weights.size(); ++i)
        res_.publish(mu.weights[i], base ? base + mu.offset[i] : nullptr);
}

/* Whether a copy into this unit's slot is still in flight. A slot whose staging copy has not
 * landed cannot be released: the quarantine barrier is on the compute stream and says nothing
 * about an H2D copy on the mover's. Only Copying counts; a record past it has its bytes in place,
 * and counting it would hold the slot until the record recycles. */
bool Mover::unit_in_flight(int32_t unit) const {
    return unit >= 0 && (size_t)unit < copying_.size() && copying_[(size_t)unit] > 0;
}

bool Mover::stageable(int32_t unit) const {
    const UnitState& us = unit_[(size_t)unit];
    return us.tier == Tier::HostPinned && us.pool_slot >= 0 && us.slab_slot < 0 && us.klass >= 0 &&
           !unit_in_flight(unit);
}

int Mover::free_move_record() const {
    const size_t n = moves_.size();
    for (size_t k = 0; k < n; ++k) {
        const size_t i = (rec_cursor_ + k) % n;
        if (moves_[i].state == Move::State::Free) { rec_cursor_ = (i + 1) % n; return (int)i; }
    }
    return -1;
}

/* The ordering event, recorded on the compute stream once per dispatch and only when a move goes
 * out. Everything the mover issues this dispatch waits on it, and it is downstream of every
 * launch so far; re-recording it per move would restate the same ordering. */
int Mover::record_gate() {
    if (!compute_) {
        RAD_ERR("mover: no compute stream bound; a transfer cannot be ordered against the "
                "kernels that read the slot it targets");
        return RAD_E_STATE;
    }
    if (gate_dispatch_ == dispatch_) return RAD_OK;
    RAD_TRY(rad_event_record(gate_, compute_));
    gate_dispatch_ = dispatch_;
    return RAD_OK;
}

int Mover::begin_dispatch(int dispatch, HeatEngine* heat) {
    dispatch_ = dispatch;
    ++stats_.dispatches;
    RAD_TRY(retire(heat));

    /* The lead gate: a host wait bounding how far ahead of the cards this thread queues work,
     * so quarantine events on the compute stream signal soon enough to recycle slots
     * (MoverConfig::max_lead_dispatches). It never waits on a transfer. */
    if (compute_ && cfg_.max_lead_dispatches > 0) {
        RadEvent old = lead_[(size_t)lead_next_];
        if (stats_.dispatches > (uint64_t)lead_.size()) RAD_TRY(rad_event_sync(old));
        RAD_TRY(rad_event_record(old, compute_));
        lead_next_ = (lead_next_ + 1) % (int)lead_.size();
    }
    return RAD_OK;
}

/* ---- the exact prefetch: layer offload's whole policy -----------------------------------------
 *
 * Declare recorded every op with its weight operands in order, so each staged unit's deadline is
 * known. Transfers start as soon as a slot is free and stop at the first unit that cannot get
 * one rather than skipping ahead: the schedule is in deadline order, and serving a later unit
 * first would turn the prefetch into a cache.
 */
int Mover::prefetch_to_op(int32_t op_index) {
    if (!plan_) return RAD_E_STATE;

    /* A backwards op index means the next step started; the schedule is per step, so the cursor
     * rewinds. */
    if (op_index < last_op_) prefetch_cursor_ = 0;
    last_op_ = op_index;

    /* Release first, so slots this op finished with are free for the units it needs next. A
     * staged unit's slot goes back once its last_use_op has passed; the declared order rules out
     * a read outside [first_use_op, last_use_op]. */
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        const UnitState& us = unit_[(size_t)u];
        if (us.site != Site::DeviceStaged || us.slab_slot < 0) continue;
        if (unit_in_flight(u)) continue;   /* its copy has not landed; the slot is not ours yet */
        const MoveUnit& mu = plan_->units[(size_t)u];
        if (mu.last_use_op >= 0 && mu.last_use_op < op_index) release_slot(u);
    }

    while (prefetch_cursor_ < (int32_t)plan_->prefetch.size()) {
        const Plan::Prefetch& pf = plan_->prefetch[(size_t)prefetch_cursor_];
        UnitState& us = unit_[(size_t)pf.unit];
        if (us.slab_slot >= 0) { ++prefetch_cursor_; continue; }   /* already in a slot */
        if (us.klass < 0) { ++prefetch_cursor_; continue; }

        const int64_t slot = pools_[(size_t)us.klass].slab.take();
        if (slot < 0) { ++stats_.starved_slab; break; }
        const int rc = start_stage(pf.unit, slot);
        if (rc < 0) {
            pools_[(size_t)us.klass].slab.add_free(slot);
            return rc;
        }
        ++prefetch_cursor_;
    }
    return RAD_OK;
}

int Mover::start_stage(int32_t unit, int64_t slab_slot) {
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return RAD_E_FULL; }
    RAD_TRY(record_gate());

    UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    char* dst = cp.slot_ptr(slab_slot);

    const void* src = nullptr;
    if (us.tier == Tier::HostPinned && us.pool_slot >= 0)
        src = (char*)cp.pool_base + us.pool_slot * cp.stride;
    else if (us.tier == Tier::HostPageable && us.static_off >= 0)
        src = (char*)pageable_ + us.static_off;
    else if (us.tier == Tier::SSD) {
        /* A blocking host read inside the dispatch, counted in ssd_reads. */
        if (!file_.ok() || !stage_) return RAD_E_STATE;
        /* The ring has wrapped: the copies out of its slots may still be queued on the mover's
         * stream, and the pread below would overwrite a source they have not read yet. */
        if (stage_next_ == 0 && stats_.ssd_reads > 0) RAD_TRY(rad_stream_sync(mover_));
        char* stage = (char*)stage_ + (int64_t)stage_next_ * stage_stride_;
        stage_next_ = (stage_next_ + 1) % std::max(1, cfg_.stage_slots);
        /* Weight by weight, each to its own offset in the unit. A container entry holds a
         * weight's planes at 256-byte alignment while a unit lays weights out at their declared
         * alignment, so one read of the unit's span would misplace every weight after the first. */
        const MoveUnit& mu = plan_->units[(size_t)unit];
        for (size_t k = 0; k < mu.weights.size(); ++k) {
            const WeightInfo& w = prog_->weights[mu.weights[k]];
            if (!w.identity || ((native_file_cache_on_ || (mapped_file_cache_on_ && mapped_file_eligible(w))) && w.stored_bytes <= RAD_PLACE_FILE_WINDOW_BYTES)) {
                if (relay_in_.empty()) relay_in_.resize(1);
                bool cached = false;
                RAD_TRY(read_native_file(w, stage + mu.offset[k], stage_stride_ - mu.offset[k],
                                        &relay_in_[0], &cached));
                if (!w.identity && !cached) {
                    ++stats_.file_relaid;
                    stats_.file_relaid_bytes += (uint64_t)w.stored_bytes;
                }
                continue;
            }
            RAD_TRY(read_source_file(w.file_id,w.file_offset, w.stored_bytes, stage + mu.offset[k],
                               stage_stride_ - mu.offset[k]));
        }
        ++stats_.ssd_reads;
        stats_.ssd_bytes += (uint64_t)us.bytes;
        src = stage;
    }
    if (!src) return RAD_E_STATE;

    RAD_TRY(rad_event_wait(mover_, gate_));
    RAD_TRY(rad_memcpy_async(dst, src, us.bytes, mover_));
    RAD_TRY(rad_event_record(moves_[(size_t)rec].ev, mover_));

    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Copying;
    m.kind = Move::Kind::Stage;
    m.unit = unit;
    ++copying_[(size_t)unit];
    m.slab_slot = slab_slot;
    m.pool_slot = -1;
    stats_.h2d_bytes += (uint64_t)us.bytes;
    stats_.prefetch_bytes += (uint64_t)us.bytes;
    ++stats_.prefetched;

    /* Publish the destination now, with the event, not at retire. Staging is the one move whose
     * destination the run phase reads: work already issued uses the old address (the pinned
     * pool), and work issued from here on names the slab slot and waits on `ev` on the compute
     * stream. That wait is the only guard against a race; begin_transfer bumps the generation so
     * the run phase performs it. */
    us.slab_slot = slab_slot;
    char* base = (char*)unit_base(unit);
    const MoveUnit& mu = plan_->units[(size_t)unit];
    for (size_t i = 0; i < mu.weights.size(); ++i)
        res_.begin_transfer(mu.weights[i], base + mu.offset[i], m.ev);
    return RAD_OK;
}

/* ---- staging within a pass (see mover.h) ---------------------------------------------------- */
int Mover::stage_units(const std::vector<int32_t>& units, char* dst, int64_t cap, RadEvent after,
                       RadEvent done, std::vector<int32_t>* staged) {
    const StageSpan one{dst, cap};
    return stage_units(units, &one, 1, after, done, staged);
}

int Mover::stage_units(const std::vector<int32_t>& units, const StageSpan* spans, int n_spans,
                       RadEvent after, RadEvent done, std::vector<int32_t>* staged) {
    if (!plan_ || !spans || n_spans < 1 || n_spans > 2 || !staged || !done) return RAD_E_INVAL;
    for (int k = 0; k < n_spans; ++k)
        if (!spans[k].dst) return RAD_E_INVAL;
    staged->clear();
    stage_picks_.clear();
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size()) return RAD_E_INVAL;
        if (!stageable(u)) continue;
        const UnitState& us = unit_[(size_t)u];
        const ClassPools& cp = pools_[(size_t)us.klass];
        const char* src = (const char*)cp.pool_base + us.pool_slot * cp.stride;
        stage_picks_.push_back(StagePick{src, u, us.bytes, align_up(us.bytes, RAD_ALIGN_UNIT),
                                         nullptr});
    }
    if (stage_picks_.empty()) return RAD_OK;
    /* Spans are filled in pool order, so units on neighbouring pool slots land side by side and
     * go as one copy. A layer's pooled experts were assigned slots together and mostly keep them,
     * so its ~90 units take far fewer copies, each a launch or DMA packet whatever its size. A
     * class's stride is its units' aligned size, so a run's source and destination advance alike.
     * A unit that does not fit the current span starts the next; one that fits no span is read
     * across the link in place. */
    std::sort(stage_picks_.begin(), stage_picks_.end(),
              [](const StagePick& a, const StagePick& b) { return a.src < b.src; });
    int k = 0;
    int64_t off = 0;
    size_t taken = 0;
    for (StagePick& p : stage_picks_) {
        if (off + p.need > spans[k].cap && k + 1 < n_spans && p.need <= spans[k + 1].cap) {
            ++k;
            off = 0;
        }
        if (off + p.need > spans[k].cap) { ++stats_.stage_overflow; continue; }
        p.dst = spans[k].dst + off;
        off += p.need;
        ++taken;
    }
    if (taken == 0) return RAD_OK;
    if (after) RAD_TRY(rad_event_wait(mover_, after));
    stage_segs_.clear();
    for (const StagePick& p : stage_picks_) {
        if (!p.dst) continue;
        RadCopySeg* run = stage_segs_.empty() ? nullptr : &stage_segs_.back();
        if (run && p.src - (const char*)run->src == p.dst - (char*)run->dst)
            run->bytes = (int64_t)(p.dst - (char*)run->dst) + p.bytes;
        else
            stage_segs_.push_back(RadCopySeg{p.src, p.dst, p.bytes});
        staged->push_back(p.u);
        stats_.h2d_bytes += (uint64_t)p.bytes;
        stats_.staged_bytes += (uint64_t)p.bytes;
    }
    RAD_TRY(rad_memcpy_list_async(stage_segs_.data(), (int)stage_segs_.size(), RAD_COPY_YIELD,
                                  mover_));
    RAD_TRY(rad_event_record(done, mover_));
    for (const StagePick& p : stage_picks_) {
        if (!p.dst) continue;
        const MoveUnit& mu = plan_->units[(size_t)p.u];
        for (size_t i = 0; i < mu.weights.size(); ++i) res_.publish(mu.weights[i], p.dst + mu.offset[i]);
    }
    ++stats_.stage_layers;
    stats_.staged_units += (uint64_t)staged->size();
    stats_.stage_copies += (uint64_t)stage_segs_.size();
    return RAD_OK;
}

void Mover::unstage_units(const std::vector<int32_t>& staged) {
    for (int32_t u : staged) republish(u);
}

void Mover::alias_file_units(const std::vector<int32_t>& units, char* base,
                             std::vector<int32_t>* staged) {
    if (!plan_ || !base || !staged) return;
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size()) continue;
        const UnitState& us = unit_[(size_t)u];
        if (us.tier != Tier::SSD || us.slab_slot >= 0 || us.klass < 0) continue;
        const MoveUnit& mu = plan_->units[(size_t)u];
        for (rad_weight w : mu.weights) res_.publish(w, base);
        staged->push_back(u);
        ++stats_.file_aliased;
    }
}

bool Mover::borrow_host_unit(int32_t u, HostUnit* out) {
    if (!out || out->n != 0 || !plan_ || !prog_ || !mapped_file_cache_on_ || u < 0 ||
        (size_t)u >= unit_.size())
        return false;
    const UnitState& us = unit_[(size_t)u];
    if (us.tier != Tier::SSD || us.slab_slot >= 0 || us.klass < 0) return false;
    const MoveUnit& mu = plan_->units[(size_t)u];
    if (mu.weights.empty() || mu.weights.size() > 2 || !source_file_epoch()) return false;
    for (rad_weight h : mu.weights) {
        const WeightInfo& w = prog_->weights[h];
        NativeMappedFileCache::Lease& l = out->lease[out->n];
        if (w.stored_bytes <= 0 || !mapped_file_eligible(w) ||
            mapped_file_cache_.borrow((size_t)(&w - prog_->weights.data()), (size_t)w.stored_bytes,
                                      &l) != NativeMappedFileCache::Status::Hit ||
            !l.active() || !l.host() || l.bytes() != (size_t)w.stored_bytes) {
            (void)l.release();
            release_host_unit(out);
            return false;
        }
        out->host[out->n] = l.host();
        out->bytes[out->n] = w.stored_bytes;
        ++out->n;
    }
    out->unit = u;
    return true;
}

bool Mover::read_unit_planes(HostPlanes* const* units, int n, int rows) {
    if (!units || n <= 0 || !plan_ || !prog_ || !relayout_ || !file_.ok() || !mapped_file_cache_on_)
        return false;
    /* As read_native_file: the container must still be the one the Program was loaded from,
     * before the reads and after them. */
    if (!source_file_epoch()) return false;
    /* A container extent [off, off+len) into dst (cap bytes there), or with m the mirror's bytes
     * [from, from+len) of *m, which are the container's extent if the mirror fails. */
    struct Read { uint32_t file; int64_t off, len; uint8_t* dst; int64_t cap;
                  const PlaneMirror::Ref* m; uint64_t from; };
    std::vector<Read> reads;
    /* Passes of min_rows or more take the mirror's share, smaller ones (decode) its decode share. */
    PlaneMirror* const pm = PlaneMirror::get();
    const int64_t tail = !pm ? 0 : rows >= pm->min_rows() ? mirror_tail_ : mirror_tail_decode_;
    PlaneMirror* const mirror = tail > 0 && pm->on() ? pm : nullptr;
    const int64_t A = DirectFile::alignment();
    int weights = 0;
    for (int i = 0; i < n; ++i) {
        HostPlanes* p = units[i];
        const int32_t u = p ? p->unit : -1;
        if (u < 0 || (size_t)u >= unit_.size()) return false;
        const UnitState& us = unit_[(size_t)u];
        if (us.tier != Tier::SSD || us.slab_slot >= 0 || us.klass < 0) return false;
        const MoveUnit& mu = plan_->units[(size_t)u];
        if (mu.weights.size() != 2) return false;
        p->n = 0;
        for (int k = 0; k < 2; ++k) {
            const WeightInfo& w = prog_->weights[mu.weights[(size_t)k]];
            RadTensor pt[RAD_ENC_MAX_PLANES];
            weight_planes(w, pt);
            if (!mapped_file_eligible(w) || w.n_sel != 2 || pt[0].dtype != RAD_FP4E2M1 ||
                pt[1].dtype != RAD_E8M0 || pt[0].shape[0] != pt[1].shape[0] ||
                pt[0].shape[1] != pt[1].shape[1] * 32)
                return false;
            int64_t lo = INT64_MAX, hi = 0;
            for (int j = 0; j < 2; ++j) {
                const int64_t bytes = pt[j].shape[0] * rad_dtype_bytes(pt[j].dtype, pt[j].shape[1]);
                lo = std::min(lo, (int64_t)w.file_plane_offset[j]);
                hi = std::max(hi, (int64_t)w.file_plane_offset[j] + bytes);
            }
            const int64_t a0 = lo / A * A, need = align_up(hi - a0, A);
            if (p->cap[k] < need) {
                std::free(p->buf[k]);
                p->buf[k] = alloc_plane_buffer(need, &p->cap[k]);
                if (!p->buf[k]) return false;
            }
            p->w[k] = mu.weights[(size_t)k];
            p->rows[k] = pt[0].shape[0];
            p->cols[k] = pt[0].shape[1];
            p->codes[k] = p->buf[k] + ((int64_t)w.file_plane_offset[0] - a0);
            p->scales[k] = p->buf[k] + ((int64_t)w.file_plane_offset[1] - a0);
            p->a0[k] = a0;
            p->len[k] = hi - a0;
            p->n = k + 1;
            ++weights;
            /* The gate_up codes' tail from the mirror (mirror_init checked the layout: codes, then
             * scales, both 4 KiB-aligned): the container reads the head and the scales. */
            const PlaneMirror::Ref* m = mirror && (size_t)u < mirror_ref_.size() &&
                mirror_ref_[(size_t)u].shard >= 0 && mirror_k_[(size_t)u] == k ? &mirror_ref_[(size_t)u] : nullptr;
            if (m) {
                const int64_t codes = (int64_t)w.file_plane_offset[0] - a0;
                const int64_t end = codes + (int64_t)(2 * m->bytes), cut = end - tail;
                reads.insert(reads.begin(), Read{ w.file_id, a0 + cut, tail, p->buf[k] + cut,
                                                  tail, m, m->bytes - (uint64_t)tail });
                reads.push_back(Read{ w.file_id, a0, cut, p->buf[k], cut, nullptr, 0 });
                reads.push_back(Read{ w.file_id, a0 + end, hi - a0 - end, p->buf[k] + end,
                                      p->cap[k] - end, nullptr, 0 });
                continue;
            }
            reads.push_back(Read{ w.file_id, a0, hi - a0, p->buf[k], p->cap[k], nullptr, 0 });
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> rc(reads.size(), RAD_OK);
    /* Readers each take the next read (the mirror's first: they take the longest): up to 8 kept
     * workers (plane_workers_, this thread one of them), or with ..._READ_POOL=0 up to 16 threads
     * started for the call; a reader that cannot be started leaves its share to the others. */
    static const bool pooled = [] {
        const char* s = std::getenv("RADIANCE_EXPERIMENTAL_DS41_READ_POOL");
        return !s || std::strcmp(s, "0") != 0;
    }();
    std::atomic<size_t> next{ 0 };
    auto reader = [&](int) {
        for (size_t j; (j = next.fetch_add(1)) < reads.size();) {
            const Read& r = reads[j];
            rc[j] = r.m ? mirror->read(*r.m, r.from, (uint64_t)r.len, r.dst)
                        : read_source_file(r.file, (uint64_t)r.off, r.len, r.dst, r.cap);
        }
    };
    const NativeReadWorkers::Result pr = !pooled || reads.size() < 2
        ? NativeReadWorkers::Result::Invalid
        : plane_workers_.run(8, (int)std::min<size_t>(reads.size(), 8), reader);
    if (pr == NativeReadWorkers::Result::CallbackFailed) return false;   /* reads left undone */
    if (pr == NativeReadWorkers::Result::Invalid) {
        std::vector<std::thread> th;
        try {
            for (size_t t = 1; t < std::min<size_t>(reads.size(), 16); ++t) th.emplace_back(reader, (int)t);
        } catch (...) {
        }
        reader(0);
        for (std::thread& t : th) t.join();
    }
    /* A mirror read that failed: the mirror is off for good, and the container has the bytes. */
    for (size_t j = 0; j < reads.size(); ++j)
        if (reads[j].m && rc[j] < 0) {
            mirror->turn_off(rc[j]);
            const Read& r = reads[j];
            rc[j] = read_source_file(r.file, (uint64_t)r.off, r.len, r.dst, r.cap);
        }
    host_plane_read_ns_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
    host_plane_reads_.fetch_add((uint64_t)weights, std::memory_order_relaxed);
    for (int r : rc) if (r < 0) return false;
    return source_file_epoch();
}

int64_t Mover::mapped_cache_pregrow() {
    if (!mapped_file_cache_on_) return 0;
    size_t slabs = 0;
    if (mapped_file_cache_.pregrow(&slabs) < 0) return 0;
    const auto st = mapped_file_cache_.stats();
    if (!st.slab_policy) return 0;
    return (int64_t)std::min(st.slabs.classes[0].capacity_slots, st.slabs.classes[1].capacity_slots);
}

int Mover::mirror_init() {
    PlaneMirror* m = PlaneMirror::get();
    mirror_tail_ = mirror_tail_decode_ = 0;
    if (!m || !plan_ || !prog_ || !mapped_file_cache_on_ || !file_.ok()) return RAD_OK;
    mirror_ref_.assign(unit_.size(), PlaneMirror::Ref{});
    mirror_k_.assign(unit_.size(), -1);
    const int64_t A = DirectFile::alignment();
    int64_t up = 0;
    std::vector<int32_t> found;
    for (size_t u = 0; u < unit_.size(); ++u) {
        if (!host_planes_eligible((int32_t)u)) continue;
        const MoveUnit& mu = plan_->units[u];
        for (int k = 0; k < 2; ++k) {
            const WeightInfo& w = prog_->weights[mu.weights[(size_t)k]];
            int layer = -1, expert = -1;
            char tail[16] = {};
            if (std::sscanf(w.name.c_str(), "blk.%d.ffn_gate_up_exps.%d.%15s", &layer, &expert, tail) != 3 ||
                std::strcmp(tail, "weight") != 0)
                continue;
            RadTensor pt[RAD_ENC_MAX_PLANES];
            weight_planes(w, pt);
            const int64_t codes = pt[0].shape[0] * rad_dtype_bytes(pt[0].dtype, pt[0].shape[1]);
            PlaneMirror::Ref r;
            /* Codes then scales, each 4 KiB-aligned, w3 the codes' second half. */
            if (!m->up_codes(layer, expert, &r) || (int64_t)r.bytes * 2 != codes ||
                w.file_plane_offset[0] % A || (int64_t)w.file_plane_offset[1] != (int64_t)w.file_plane_offset[0] + codes ||
                codes % A)
                continue;
            mirror_ref_[u] = r;
            mirror_k_[u] = (int8_t)k;
            up = (int64_t)r.bytes;
            found.push_back((int32_t)u);
        }
    }
    if (found.empty()) {
        RAD_WARN("plane mirror: none of rank %d's file-tier experts is in the mirror", plan_->rank);
        return RAD_OK;
    }
    const int64_t unit = plan_->units[(size_t)found[0]].bytes;
    auto tail_of = [&](double share) { return std::min<int64_t>((int64_t)(share * (double)unit) / A * A, up / A * A); };
    const int64_t prefill_tail = tail_of(m->share()), decode_tail = tail_of(m->decode_share());
    /* The longer tail is checked; the shorter is its suffix. */
    const int64_t tail = std::max(prefill_tail, decode_tail);
    if (tail <= 0) return RAD_OK;
    /* Eight units spread over the rank: the mirror's tail against the container's, byte for byte. */
    std::vector<uint8_t*> bufs(2, nullptr);
    for (auto& b : bufs) b = (uint8_t*)std::aligned_alloc((size_t)A, (size_t)tail);
    bool same = bufs[0] && bufs[1];
    for (int i = 0; same && i < 8; ++i) {
        const int32_t u = found[(size_t)i * found.size() / 8];
        const WeightInfo& w = prog_->weights[plan_->units[(size_t)u].weights[(size_t)mirror_k_[(size_t)u]]];
        const PlaneMirror::Ref& r = mirror_ref_[(size_t)u];
        const int64_t end = (int64_t)w.file_plane_offset[0] + 2 * (int64_t)r.bytes;
        same = read_source_file(w.file_id, (uint64_t)(end - tail), tail, bufs[0], tail) == RAD_OK &&
               m->read(r, r.bytes - (uint64_t)tail, (uint64_t)tail, bufs[1]) == RAD_OK &&
               std::memcmp(bufs[0], bufs[1], (size_t)tail) == 0;
    }
    for (auto& b : bufs) std::free(b);
    if (!same) {
        RAD_WARN("plane mirror: rank %d's sampled experts differ from the container (or could not be "
                 "read); reading the container only", plan_->rank);
        mirror_ref_.clear();
        mirror_k_.clear();
        return RAD_OK;
    }
    mirror_tail_ = prefill_tail;
    mirror_tail_decode_ = decode_tail;
    RAD_INFO("plane mirror: rank %d reads %s of each missed expert's gate_up from the mirror in passes "
             "of %d+ tokens and %s in smaller ones (%zu experts mapped, 8 sampled and identical)",
             plan_->rank, humanb(prefill_tail).c_str(), m->min_rows(), humanb(decode_tail).c_str(),
             found.size());
    return RAD_OK;
}

int Mover::insert_unit_planes(const HostPlanes& p, std::vector<uint8_t>* scratch) {
    if (!scratch || !relayout_ || !mapped_file_cache_on_) return RAD_E_STATE;
    for (int k = 0; k < p.n; ++k) {
        /* The check-insert-check bracket of read_native_file: never admit another epoch's bytes. */
        if (!source_file_epoch()) return RAD_E_STATE;
        const WeightInfo& w = prog_->weights[p.w[k]];
        RadTensor pt[RAD_ENC_MAX_PLANES];
        weight_planes(w, pt);
        pt[0].data = const_cast<uint8_t*>(p.codes[k]);
        pt[1].data = const_cast<uint8_t*>(p.scales[k]);
        if ((int64_t)scratch->size() < w.stored_bytes) scratch->resize((size_t)w.stored_bytes);
        const int st = relayout_(w, pt, scratch->data(), w.stored_bytes);
        if (st < 0) return st;
        using S = NativeMappedFileCache::Status;
        const auto rc = mapped_file_cache_.insert((size_t)(&w - prog_->weights.data()), scratch->data(),
                                                  (size_t)w.stored_bytes);
        if (rc != S::Ok && rc != S::Existing && rc != S::Bypass && rc != S::Allocation) return RAD_E_STATE;
        if (!source_file_epoch()) return RAD_E_STATE;
        host_plane_inserts_.fetch_add(1, std::memory_order_relaxed);
    }
    return RAD_OK;
}

bool Mover::read_weight_file(const WeightInfo& w, void* dst) {
    if (!dst || !file_.ok() || !w.identity || w.stored_bytes <= 0 ||
        (w.decl.shard != RAD_SHARD_NONE && plan_ && plan_->world_size > 1))
        return false;
    const int64_t A = DirectFile::alignment();
    const int64_t off = (int64_t)w.file_offset, a0 = off / A * A;
    const int64_t len = off + w.stored_bytes - a0, cap = align_up(len, A);
    uint8_t* buf = (uint8_t*)std::aligned_alloc((size_t)A, (size_t)cap);
    if (!buf) return false;
    const int rc = read_source_file(w.file_id, (uint64_t)a0, len, buf, cap);
    if (rc >= 0) std::memcpy(dst, buf + (off - a0), (size_t)w.stored_bytes);
    std::free(buf);
    return rc >= 0;
}

bool Mover::unit_cached(int32_t u) {
    if (!plan_ || !prog_ || !mapped_file_cache_on_ || u < 0 || (size_t)u >= unit_.size()) return false;
    for (rad_weight h : plan_->units[(size_t)u].weights)
        if (!mapped_file_cache_.present((size_t)h)) return false;
    return true;
}

bool Mover::host_planes_eligible(int32_t u) const {
    if (!plan_ || !prog_ || !mapped_file_cache_on_ || u < 0 || (size_t)u >= unit_.size()) return false;
    const UnitState& us = unit_[(size_t)u];
    if (us.tier != Tier::SSD || us.slab_slot >= 0 || us.klass < 0) return false;
    const MoveUnit& mu = plan_->units[(size_t)u];
    if (mu.weights.size() != 2) return false;
    for (rad_weight h : mu.weights) {
        const WeightInfo& w = prog_->weights[h];
        if (!mapped_file_eligible(w) || w.n_sel != 2) return false;
    }
    return true;
}

void Mover::release_host_unit(HostUnit* hu) {
    if (!hu) return;
    for (int i = 0; i < hu->n; ++i) {
        (void)hu->lease[i].release();
        hu->host[i] = nullptr;
        hu->bytes[i] = 0;
    }
    hu->n = 0;
    hu->unit = -1;
}

/* The same choice of units as stage_units, without a cap. */
int64_t Mover::stage_need(const std::vector<int32_t>& units, int64_t* largest) const {
    int64_t total = 0, most = 0;
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size() || !stageable(u)) continue;
        const int64_t need = align_up(unit_[(size_t)u].bytes, RAD_ALIGN_UNIT);
        total += need;
        most = std::max(most, need);
    }
    if (largest) *largest = most;
    return total;
}

/* ---- a routed layer read from the container (see mover.h) ---------------------------------- */
namespace {

/* One weight's bytes: where they are in the container and where they go. */
struct FileExt {
    uint32_t bank = 0;
    int64_t foff = 0, bytes = 0;
    char*   dst = nullptr;
};

/* The largest gap a run reads through instead of starting another pread. A routed layer's
 * weights sit together in the container, each on RAD_ALIGN_UNIT, so a gap between two file-tier
 * weights is either alignment padding (cheaper to read than a second pread) or units on another
 * tier (megabytes read for nothing). 256 KiB is above every padding and below every expert this
 * engine serves. */
constexpr int64_t kReadThrough = (int64_t)256 << 10;

}  /* namespace */

/* One weight relaid on the read: its selected planes' span, aligned out to the O_DIRECT block,
 * read into this thread's host input in one pread (the aligned path, which touches no shared
 * bounce, so threads read at once), then the relayout writes the stored form to `dst`. Each plane
 * is whole (weight_file_relaid), so it is one range of the container. */
int Mover::read_relaid(const WeightInfo& w, void* dst, int64_t dst_cap, RelayIn* in) {
    if (!relayout_ || !in || w.n_sel <= 0) return RAD_E_STATE;
    if (w.stored_bytes > dst_cap) {
        RAD_ERR("file tier: '%s' relays to %s and the pinned space for it is %s", w.name.c_str(),
                humanb(w.stored_bytes).c_str(), humanb(dst_cap).c_str());
        return RAD_E_SCRATCH;
    }
    RadTensor pt[RAD_ENC_MAX_PLANES];
    weight_planes(w, pt);
    int64_t lo = INT64_MAX, hi = 0;
    for (int k = 0; k < w.n_sel; ++k) {
        const int64_t len = pt[k].shape[0] * rad_dtype_bytes(pt[k].dtype, pt[k].shape[1]);
        lo = std::min(lo, (int64_t)w.file_plane_offset[k]);
        hi = std::max(hi, (int64_t)w.file_plane_offset[k] + len);
    }
    const int64_t A = DirectFile::alignment();
    const int64_t a0 = lo / A * A, n = align_up(hi - a0, A);
    if (in->cap < n) {
        std::free(in->p);
        in->p = (uint8_t*)std::aligned_alloc((size_t)A, (size_t)n);
        in->cap = in->p ? n : 0;
        if (!in->p) return RAD_E_NOMEM;
    }
    const auto t0 = std::chrono::steady_clock::now();
    RAD_TRY(read_source_file(w.file_id,(uint64_t)a0, hi - a0, in->p, in->cap));   /* reads n, needs hi - a0 */
    const auto t1 = std::chrono::steady_clock::now();
    for (int k = 0; k < w.n_sel; ++k) pt[k].data = in->p + ((int64_t)w.file_plane_offset[k] - a0);
    const int st = relayout_(w, pt, dst, w.stored_bytes);
    relaid_read_ns_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                              std::memory_order_relaxed);
    relaid_pack_ns_.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - t1).count(), std::memory_order_relaxed);
    relaid_reads_.fetch_add(1, std::memory_order_relaxed);
    if (st < 0)
        RAD_ERR("file tier: relaying '%s' (%s) on the read failed: %s", w.name.c_str(),
                w.layout_tag.c_str(), rad_strerror(st));
    return st;
}

bool Mover::mapped_file_eligible(const WeightInfo& w) {
    //Only byte transport changes. The Program/file epoch/key still select exact original bytes.
    //Keep this first experiment inside the native DS4.1 trunk expert representation.
    return w.decl.access == RAD_ACCESS_CONDITIONAL && w.decl.shard == RAD_SHARD_NONE &&
        w.decl.group.layer >= 0 && w.decl.group.layer < 40 &&
        w.decl.group.expert >= 0 && w.decl.group.expert < 96 &&
        !w.widen && !w.narrow && w.layout_tag == "ggml_mxfp4" &&
        (w.stored_bytes == int64_t(NativeMappedFileCache::GU) ||
         w.stored_bytes == int64_t(NativeMappedFileCache::DN));
}

int Mover::file_copy_event_record(RadEvent event) {
    if (mapped_file_cache_on_ && mapped_copy_hooks_.record)
        return mapped_copy_hooks_.record(mapped_copy_hooks_.context,event,mover_);
    return rad_event_record(event,mover_);
}
int Mover::file_copy_event_sync(RadEvent event) {
    if (mapped_file_cache_on_ && mapped_copy_hooks_.sync)
        return mapped_copy_hooks_.sync(mapped_copy_hooks_.context,event);
    return rad_event_sync(event);
}
int Mover::file_copy_stream_sync() {
    if (mapped_file_cache_on_ && mapped_copy_hooks_.drain)
        return mapped_copy_hooks_.drain(mapped_copy_hooks_.context,mover_);
    return mover_ ? rad_stream_sync(mover_) : RAD_E_STATE;
}
int Mover::mapped_copy_to_vram(void* dst,const void* src,int64_t bytes) {
    if (mapped_file_cache_on_ && mapped_copy_hooks_.copy)
        return mapped_copy_hooks_.copy(mapped_copy_hooks_.context,dst,src,bytes,mover_);
    return rad_memcpy_async(dst,src,bytes,mover_);
}

int Mover::retire_file_window(int window) {
    if (window < 0 || window >= n_windows_) return RAD_E_INVAL;
    auto& batch = mapped_windows_[(size_t)window];
    if (win_rec_[(size_t)window]) {
        //This exact event generation follows every copy consuming these leases.
        RAD_TRY(file_copy_event_sync(win_ev_[(size_t)window]));
    } else if (batch.copy_may_be_queued) {
        //An enqueue or event-record failure may have left a prefix of copies queued.
        RAD_TRY(file_copy_stream_sync());
    }
    //No copy was issued, or its event/drain is proven complete. A failure above retains all.
    for (auto& lease : batch.leases) {
        const auto rc = lease.release();
        if (rc != NativeMappedFileCache::Status::Ok && rc != NativeMappedFileCache::Status::Empty)
            return RAD_E_STATE;
    }
    batch.copy_may_be_queued = false;
    win_rec_[(size_t)window] = 0;
    return RAD_OK;
}

int Mover::retire_file_windows() {
    if (!mapped_file_cache_on_) return RAD_OK;
    for (int w = 0; w < n_windows_; ++w) {
        const MappedWindow& batch = mapped_windows_[(size_t)w];
        bool held = win_rec_[(size_t)w] || batch.copy_may_be_queued;
        for (const auto& l : batch.leases) held = held || l.active();
        if (held) RAD_TRY(retire_file_window(w));
    }
    return RAD_OK;
}

int Mover::mapped_file_drain() {
    bool pending = false;
    for (const auto& batch : mapped_windows_) pending = pending || batch.copy_may_be_queued;
    //On init/error/destruction, a single successful stream drain closes all generations.
    if (pending) RAD_TRY(file_copy_stream_sync());
    for (auto& batch : mapped_windows_) {
        for (auto& lease : batch.leases) {
            const auto rc = lease.release();
            if (rc != NativeMappedFileCache::Status::Ok && rc != NativeMappedFileCache::Status::Empty)
                return RAD_E_STATE;
        }
        batch.copy_may_be_queued = false;
    }
    if (pending) std::fill(win_rec_.begin(),win_rec_.end(),0);
    return RAD_OK;
}

/* The cache stores the final native bytes, not their container encoding. Its lock covers only
 * CPU copies; GPU copies still use the existing pinned windows and their existing drain events.
 * The one-Mover Program and held file epoch define the namespace, so a hit requires no payload
 * hash or relayout. Direct-layout weights use each worker's aligned input, avoiding a shared
 * O_DIRECT bounce buffer when several workers fill differently aligned window offsets. */
bool Mover::prefetch_trace_slot(const M1PrefetchTrace& t) {
    if(!m1_trace_tag_valid(t))return false;
    const uint64_t n=prefetch_trace_lines_.fetch_add(1,std::memory_order_relaxed);
    if(n==kM1PrefetchTraceMaxLines)
        RAD_INFO("DS41M1TRACE v=1 kind=truncated rank=%lld limit=%llu; incomplete trace, no recall estimate",
            (long long)t.rank,(unsigned long long)kM1PrefetchTraceMaxLines);
    return n<kM1PrefetchTraceMaxLines;
}

void Mover::prefetch_trace_route(const M1PrefetchTrace& t,const int32_t* counts,size_t SSD_units) {
    if(!m1_trace_tag_valid(t) || !counts || SSD_units>6)return;
    char ids[128]{};size_t at=0;int sum=0;
    for(int e=0;e<96;++e){
        if(counts[e]<0 || counts[e]>1)return;
        sum+=counts[e];if(sum>6)return;
        if(counts[e]){
            const int n=std::snprintf(ids+at,sizeof(ids)-at,"%s%d",at?",":"",e);
            if(n<0 || size_t(n)>=sizeof(ids)-at)return;
            at+=size_t(n);
        }
    }
    if(prefetch_trace_slot(t))
        RAD_INFO("DS41M1TRACE v=1 kind=route rank=%lld gen=%llu producer=%llu layer=%lld ready_ns=%llu base=%lld local=[%s] ssd_units=%zu",
            (long long)t.rank,(unsigned long long)t.generation,(unsigned long long)t.producer,(long long)t.layer,
            (unsigned long long)t.ready_ns,(long long)t.expert_base,ids,SSD_units);
}

void Mover::prefetch_trace_stage_done(const M1PrefetchTrace& t,int status) {
    if(prefetch_trace_slot(t))
        RAD_INFO("DS41M1TRACE v=1 kind=stage_return rank=%lld gen=%llu producer=%llu layer=%lld cpu_return_ns=%llu rc=%d; not GPU completion",
            (long long)t.rank,(unsigned long long)t.generation,(unsigned long long)t.producer,
            (long long)t.layer,(unsigned long long)m1_trace_now_ns(),status);
}

int Mover::read_native_file(const WeightInfo& w,void* dst,int64_t dst_cap,RelayIn* in,
                            bool* cache_hit,NativeFileCacheReadPolicy policy,
                            NativeMappedFileCache::Lease* mapped_lease,M1PrefetchTrace trace) {
    if(!m1_trace_weight(trace,w.decl.group.layer,w.decl.group.expert,w.decl.group.slot,w.stored_bytes) ||
        w.layout_tag!="ggml_mxfp4" || policy!=NativeFileCacheReadPolicy::Normal)
        return read_native_file_impl(w,dst,dst_cap,in,cache_hit,policy,mapped_lease);
    const uint64_t began=m1_trace_now_ns();
    const bool eligible=native_file_cache_on_ || (mapped_file_cache_on_ && mapped_file_eligible(w));
    const int rc=read_native_file_impl(w,dst,dst_cap,in,cache_hit,policy,mapped_lease);
    const uint64_t ended=m1_trace_now_ns();
    // Eligibility is not an entered-probe claim on error. Known0+eligible means
    // the existing cache API was consulted; hit is its actual returned outcome.
    if(prefetch_trace_slot(trace))
        RAD_INFO("DS41M1TRACE v=1 kind=read rank=%lld gen=%llu producer=%llu layer=%lld expert=%lld slot=%lld key=%zu bank=%u bank_generation=%llu entry=%lld bytes=%lld begin_ns=%llu end_ns=%llu rc=%d cache_eligible=%d hit=%d identity=%d",
            (long long)trace.rank,(unsigned long long)trace.generation,(unsigned long long)trace.producer,
            (long long)trace.layer,(long long)w.decl.group.expert,(long long)w.decl.group.slot,
            size_t(&w-prog_->weights.data()),unsigned(w.file_id),(unsigned long long)w.source_bank_generation,
            (long long)w.source_entry_index,(long long)w.stored_bytes,(unsigned long long)began,
            (unsigned long long)ended,rc,int(eligible),rc==RAD_OK && cache_hit?int(*cache_hit):-1,int(w.identity));
    return rc;
}

int Mover::read_native_file_impl(const WeightInfo& w, void* dst, int64_t dst_cap, RelayIn* in,
                            bool* cache_hit, NativeFileCacheReadPolicy policy,
                            NativeMappedFileCache::Lease* mapped_lease) {
    if (!in || !dst || !cache_hit || w.stored_bytes <= 0 || dst_cap < w.stored_bytes ||
        !native_file_cache_policy_valid(policy))
        return RAD_E_INVAL;
    *cache_hit = false;
    const size_t key = (size_t)(&w - prog_->weights.data());
    const bool mapped_scope = mapped_file_cache_on_ && mapped_file_eligible(w);
    if (mapped_scope && !source_file_epoch()) return RAD_E_STATE;
    //Slab ReadOnly borrows preserve decode LRU order. Per-entry ReadOnly still bypasses.
    const bool mapped = mapped_scope && (policy == NativeFileCacheReadPolicy::Normal ||
                                        mapped_file_cache_.supports_readonly_borrow());
    if (mapped) {
        if (!source_file_epoch()) return RAD_E_STATE;
        NativeMappedFileCache::Lease local;
        auto* lease = mapped_lease ? mapped_lease : &local;
        const auto rc = policy == NativeFileCacheReadPolicy::ReadOnly ?
            mapped_file_cache_.borrow_readonly(key,size_t(w.stored_bytes),lease) :
            mapped_file_cache_.borrow(key,size_t(w.stored_bytes),lease);
        if (rc == NativeMappedFileCache::Status::Hit) {
            *cache_hit = true;
            //Demand promotion still uses its pinned stage; routed copies keep this lease.
            if (!mapped_lease) std::memcpy(dst,lease->host(),size_t(w.stored_bytes));
            return source_file_epoch() ? RAD_OK : RAD_E_STATE;
        }
        if (rc != NativeMappedFileCache::Status::Miss) return RAD_E_STATE;
    }
    bool skip_mapped_insert=false;
    if(mapped&&mapped_direct_fill_on_&&mapped_lease&&policy==NativeFileCacheReadPolicy::Normal){
        NativeMappedFileCache::Write write;
        const auto reserved=mapped_file_cache_.reserve_fill(key,size_t(w.stored_bytes),&write);
        using S=NativeMappedFileCache::Status;
        if(reserved==S::Ok){
            // The charged unpublished slot is owned exclusively by this retired-before-return reader.
            // No owner/state/writer cache lock spans disk input or whole-weight native relayout.
            int rc=RAD_OK;
            if(!w.identity)rc=read_relaid(w,write.host(),int64_t(write.bytes()),in);
            else{
                const int64_t A=DirectFile::alignment();
                if(w.file_offset>uint64_t(INT64_MAX))return RAD_E_INVAL;
                const int64_t offset=int64_t(w.file_offset),lo=offset/A*A,delta=offset-lo;
                if(w.stored_bytes>INT64_MAX-delta-A)return RAD_E_INVAL;
                const int64_t need=delta+w.stored_bytes,allocation=align_up(need,A);
                if(in->cap<allocation){std::free(in->p);in->p=(uint8_t*)std::aligned_alloc(size_t(A),size_t(allocation));in->cap=in->p?allocation:0;}
                if(!in->p)return RAD_E_NOMEM;
                rc=read_source_file(w.file_id,uint64_t(lo),need,in->p,in->cap);
                if(rc>=0)std::memcpy(write.host(),in->p+delta,size_t(w.stored_bytes));
            }
            if(rc<0)return rc; // Write destructor cancels; a partial payload is never published
            if(!source_file_epoch())return RAD_E_STATE;
            const auto published=write.publish(mapped_lease); // atomic publication + consumer pin
            if(published!=S::Ok)return RAD_E_STATE;
            return source_file_epoch()?RAD_OK:RAD_E_STATE;
        }
        if(reserved==S::Existing){
            const auto hit=mapped_file_cache_.borrow(key,size_t(w.stored_bytes),mapped_lease);
            if(hit==S::Hit){*cache_hit=true;return source_file_epoch()?RAD_OK:RAD_E_STATE;}
            if(hit!=S::Miss)return RAD_E_STATE;
        }else if(reserved!=S::Bypass&&reserved!=S::Allocation)return RAD_E_STATE;
        // Gross cap/class/held-tail/allocation pressure keeps the original window path, without
        // repeating a failed reservation or copying that fallback into the cache a second time.
        skip_mapped_insert=true;
    }
    if (native_file_cache_on_) {
        if (!source_file_epoch()) return RAD_E_STATE;
        RAD_TRY(native_file_cache_.copy(key, w.stored_bytes, dst, dst_cap, cache_hit, policy));
        if (*cache_hit) return source_file_epoch() ? RAD_OK : RAD_E_STATE;
    }
    if (!w.identity) {
        RAD_TRY(read_relaid(w, dst, dst_cap, in));
    } else {
        const int64_t A = DirectFile::alignment();
        if (w.file_offset > (uint64_t)INT64_MAX) return RAD_E_INVAL;
        const int64_t offset = (int64_t)w.file_offset, lo = offset / A * A, delta = offset - lo;
        if (w.stored_bytes > INT64_MAX - delta - A) return RAD_E_INVAL;
        const int64_t need = delta + w.stored_bytes, allocation = align_up(need, A);
        if (in->cap < allocation) {
            std::free(in->p);
            in->p = (uint8_t*)std::aligned_alloc((size_t)A, (size_t)allocation);
            in->cap = in->p ? allocation : 0;
            if (!in->p) return RAD_E_NOMEM;
        }
        RAD_TRY(read_source_file(w.file_id,(uint64_t)lo, need, in->p, in->cap));
        std::memcpy(dst, in->p + delta, (size_t)w.stored_bytes);
    }
    if (mapped_scope && !source_file_epoch()) return RAD_E_STATE;
    if (mapped && !skip_mapped_insert && policy == NativeFileCacheReadPolicy::Normal) {
        if (!source_file_epoch()) return RAD_E_STATE;
        const auto rc = mapped_file_cache_.insert(key,dst,size_t(w.stored_bytes));
        //Admission pressure and a failed new allocation preserve the original window fallback.
        using S = NativeMappedFileCache::Status;
        if (rc != S::Ok && rc != S::Existing && rc != S::Bypass && rc != S::Allocation)
            return RAD_E_STATE;
        if (!source_file_epoch()) return RAD_E_STATE;
    }
    if (native_file_cache_on_) {
        if (!source_file_epoch()) return RAD_E_STATE;
        if (policy == NativeFileCacheReadPolicy::Normal)
            RAD_TRY(native_file_cache_.insert(key, dst, w.stored_bytes));
        if (!source_file_epoch()) return RAD_E_STATE;
    }
    return RAD_OK;
}

/* The file stager's relaid weights (stage_file_units): packed into the read windows, each
 * window's weights read and relaid on up to relay_threads_ threads at once, then copied to the
 * layer buffer on the mover's stream like the direct runs, the window's event recorded behind
 * them. A window comes round again only after its copies drained, as for the direct runs. */
int Mover::stage_relaid(const std::vector<Relaid>& rl, NativeFileCacheReadPolicy policy) {
    const int64_t W = RAD_PLACE_FILE_WINDOW_BYTES;
    if (!windows_ || n_windows_ <= 0) return RAD_E_STATE;
    if ((int)relay_in_.size() < relay_threads_) relay_in_.resize((size_t)relay_threads_);
    const auto t_all = std::chrono::steady_clock::now();
    struct Job { size_t r; int64_t wo; int mapped_slot; };
    std::vector<Job> jobs;
    size_t i = 0;
    while (i < rl.size()) {
        const int w = win_next_;
        win_next_ = (win_next_ + 1) % n_windows_;
        char* win = windows_ + (int64_t)w * W;
        if (win_rec_[(size_t)w] || mapped_file_cache_on_) {
            const auto t0 = std::chrono::steady_clock::now();
            RAD_TRY(retire_file_window(w));
            stats_.file_wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - t0).count();
            win_rec_[(size_t)w] = 0;
        }
        jobs.clear();
        int64_t used = 0;
        size_t mapped_slots = 0;
        while (i < rl.size()) {
            const WeightInfo& wi = *rl[i].w;
            const int64_t a = std::max<int64_t>(wi.align > 0 ? wi.align : RAD_ALIGN_UNIT, 256);
            const int64_t wo = align_up(used, a);
            if (wo + wi.stored_bytes > W) break;
            const bool eligible = mapped_file_cache_on_ && mapped_file_eligible(wi) &&
                (policy == NativeFileCacheReadPolicy::Normal || mapped_file_cache_.supports_readonly_borrow());
            if (eligible && mapped_slots >= kMappedPerWindow) return RAD_E_STATE;
            jobs.push_back(Job{ i, wo, eligible ? int(mapped_slots++) : -1 });
            used = wo + wi.stored_bytes;
            ++i;
        }
        if (jobs.empty()) {
            RAD_ERR("file tier: '%s' relays to %s, more than a %s read window", rl[i].w->name.c_str(),
                    humanb(rl[i].w->stored_bytes).c_str(), humanb(W).c_str());
            return RAD_E_SCRATCH;
        }
        std::vector<int> rc(jobs.size(), RAD_OK);
        std::vector<uint8_t> cache_hits(jobs.size(), 0);
        std::atomic<size_t> next{ 0 };
        auto work = [&, policy](int t) {
            for (size_t j; (j = next.fetch_add(1)) < jobs.size();) {
                const Job& jb = jobs[j];
                bool cached = false;
                rc[j] = read_native_file(*rl[jb.r].w, win + jb.wo, W - jb.wo,
                                         &relay_in_[(size_t)t], &cached, policy,
                                         jb.mapped_slot < 0 ? nullptr :
                                             &mapped_windows_[(size_t)w].leases[(size_t)jb.mapped_slot],rl[jb.r].trace);
                cache_hits[j] = cached;
            }
        };
        const int nt = (int)std::min<size_t>((size_t)relay_threads_, jobs.size());
        if (native_file_workers_on_) {
            const auto result = relay_workers_.run(relay_threads_, nt, work);
            if (result == NativeReadWorkers::Result::Invalid ||
                result == NativeReadWorkers::Result::CallbackFailed)
                return RAD_E_STATE;  /* run() has already retired every CPU reader */
        } else {
            std::vector<std::thread> th;
            for (int t = 1; t < nt; ++t) th.emplace_back(work, t);
            work(0);
            for (std::thread& x : th) x.join();
        }
        for (size_t j = 0; j < jobs.size(); ++j) RAD_TRY(rc[j]);
        if (policy == NativeFileCacheReadPolicy::ReadOnly) {
            for (size_t j = 0; j < jobs.size(); ++j) {
                const uint64_t bytes = uint64_t(rl[jobs[j].r].w->stored_bytes);
                if (cache_hits[j]) { ++stats_.file_prefill_ro_hits; stats_.file_prefill_ro_hit_bytes += bytes; }
                else { ++stats_.file_prefill_ro_misses; stats_.file_prefill_ro_miss_bytes += bytes; }
            }
        }
        if (mapped_file_cache_on_) {
            //All CPU readers/packers retired. Publish CPU writes before DEVICE reads.
            rad_dev_host_wrote();
            //Also retain ordinary fallback windows on partial enqueue/record failure.
            mapped_windows_[(size_t)w].copy_may_be_queued = true;
        }
        for (size_t j = 0; j < jobs.size(); ++j) {
            const Job& jb = jobs[j];
            const WeightInfo& wi = *rl[jb.r].w;
            const auto* lease = jb.mapped_slot < 0 ? nullptr :
                &mapped_windows_[(size_t)w].leases[(size_t)jb.mapped_slot];
            if (lease && lease->active()) {
                //Ownership was already transferred into this persistent window before enqueue.
                RAD_TRY(mapped_copy_to_vram(rl[jb.r].dst,lease->device(),wi.stored_bytes));
                ++stats_.mapped_source_copies;
                stats_.mapped_source_bytes += uint64_t(wi.stored_bytes);
            } else {
                RAD_TRY(mapped_copy_to_vram(rl[jb.r].dst, win + jb.wo, wi.stored_bytes));
            }
            if (!wi.identity && !cache_hits[j]) {
                ++stats_.file_relaid;
                stats_.file_relaid_bytes += (uint64_t)wi.stored_bytes;
            }
        }
        RAD_TRY(file_copy_event_record(win_ev_[(size_t)w]));
        win_rec_[(size_t)w] = 1;
    }
    stats_.file_relayout_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - t_all).count();
    return RAD_OK;
}

int Mover::stage_file_units(const std::vector<int32_t>& units, char* dst, int64_t cap,
                            RadEvent after, RadEvent done, std::vector<int32_t>* staged,
                            NativeFileCacheReadPolicy policy, M1PrefetchTrace trace) {
    if (!plan_ || !dst || !staged || !done || !native_file_cache_policy_valid(policy)) return RAD_E_INVAL;
    const auto cache_policy = (native_file_cache_on_ || mapped_file_cache_on_) ? policy : NativeFileCacheReadPolicy::Normal;
    staged->clear();
    if (!file_.ok() || !windows_ || n_windows_ <= 0) return RAD_E_STATE;
    if(cfg_.source_banks&&!source_file_epoch())return RAD_E_STATE;

    /* The layout in `dst`, unit by unit in the order given, and every weight's extent; a weight
     * the read relays goes through stage_relaid instead of the windows' direct copies. */
    std::vector<FileExt> ext;
    std::vector<Relaid>  relaid;
    std::vector<int64_t> at;
    int64_t off = 0;
    for (int32_t u : units) {
        if (u < 0 || (size_t)u >= unit_.size()) return RAD_E_INVAL;
        const UnitState& us = unit_[(size_t)u];
        if (us.tier != Tier::SSD || us.slab_slot >= 0 || us.klass < 0) continue;
        const int64_t need = align_up(us.bytes, RAD_ALIGN_UNIT);
        if (off + need > cap) {
            RAD_ERR("file stager: a routed layer's file-tier units need more than the %s buffer "
                    "the plan sized for the largest one", humanb(cap).c_str());
            return RAD_E_SCRATCH;
        }
        const MoveUnit& mu = plan_->units[(size_t)u];
        for (size_t k = 0; k < mu.weights.size(); ++k) {
            const WeightInfo& w = prog_->weights[mu.weights[k]];
            if (w.stored_bytes <= 0) continue;          /* nothing to read, and nothing to place */
            if (!w.identity || ((native_file_cache_on_ || (mapped_file_cache_on_ && mapped_file_eligible(w))) && w.stored_bytes <= RAD_PLACE_FILE_WINDOW_BYTES)) {
                relaid.push_back(Relaid{ &w, dst + off + mu.offset[k], trace });
                continue;
            }
            ext.push_back(FileExt{ w.file_id,(int64_t)w.file_offset, w.stored_bytes, dst + off + mu.offset[k] });
        }
        staged->push_back(u);
        at.push_back(off);
        off += need;
    }
    if (staged->empty()) return RAD_OK;
    /* Container order, so a window is one long run of the drive rather than a seek per weight. */
    std::stable_sort(ext.begin(), ext.end(),
                     [](const FileExt& a, const FileExt& b) { return a.bank!=b.bank?a.bank<b.bank:a.foff<b.foff; });

    if (after) RAD_TRY(rad_event_wait(mover_, after));

    const int64_t A = DirectFile::alignment();
    const int64_t W = RAD_PLACE_FILE_WINDOW_BYTES;
    struct Seg { uint32_t bank=0; int64_t fs = 0, fe = 0, ws = 0; };     /* file [fs, fe) read to window offset ws */
    struct Cp  { int64_t wo = 0, bytes = 0; char* dst = nullptr; };
    std::vector<Seg> segs;
    std::vector<Cp>  cps;
    size_t  i = 0;
    int64_t piece = 0;                                  /* bytes of ext[i] already placed */
    while (i < ext.size()) {
        const int w = win_next_;
        win_next_ = (win_next_ + 1) % n_windows_;
        char* win = windows_ + (int64_t)w * W;
        if (win_rec_[(size_t)w] || mapped_file_cache_on_) {
            /* The window's copies from the round before must have left it. */
            const auto t0 = std::chrono::steady_clock::now();
            RAD_TRY(retire_file_window(w));
            stats_.file_wait_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - t0).count();
            win_rec_[(size_t)w] = 0;
        }

        /* Pack the window: extend the current run while the next weight starts within the
         * read-through gap of its end, start another run otherwise, and split a weight the window
         * cannot finish across this window and the next. */
        segs.clear();
        cps.clear();
        while (i < ext.size()) {
            const FileExt& e = ext[i];
            const int64_t fo = e.foff + piece, left = e.bytes - piece;
            int64_t wo;
            if (!segs.empty() && e.bank==segs.back().bank && fo >= segs.back().fs && fo <= segs.back().fe + kReadThrough) {
                wo = segs.back().ws + (fo - segs.back().fs);
            } else {
                const int64_t ws = segs.empty()
                    ? 0 : segs.back().ws + align_up(segs.back().fe - segs.back().fs, A);
                const int64_t fs = fo / A * A;
                wo = ws + (fo - fs);
                if (wo >= W) break;
                segs.push_back(Seg{ e.bank,fs,fs,ws });
            }
            const int64_t take = std::min(left, W - wo);
            if (take <= 0) break;
            Seg& s = segs.back();
            s.fe = std::max(s.fe, fo + take);
            cps.push_back(Cp{ wo, take, e.dst + piece });
            piece += take;
            if (piece == e.bytes) { ++i; piece = 0; }
            if (take < left) break;                     /* the window is full */
        }
        if (cps.empty()) return RAD_E_STATE;            /* a window that takes nothing cannot end */

        const auto t0 = std::chrono::steady_clock::now();
        for (const Seg& s : segs) {
            RAD_TRY(read_source_file(s.bank,(uint64_t)s.fs, s.fe - s.fs, win + s.ws, W - s.ws));
            ++stats_.file_preads;
        }
        stats_.file_read_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - t0).count();
        if (mapped_file_cache_on_) mapped_windows_[(size_t)w].copy_may_be_queued = true;
        for (const Cp& c : cps) RAD_TRY(mapped_copy_to_vram(c.dst, win + c.wo, c.bytes));
        RAD_TRY(file_copy_event_record(win_ev_[(size_t)w]));
        win_rec_[(size_t)w] = 1;
    }
    if (!relaid.empty()) RAD_TRY(stage_relaid(relaid,cache_policy));
    RAD_TRY(rad_event_record(done, mover_));

    for (size_t k = 0; k < staged->size(); ++k) {
        const int32_t u = (*staged)[k];
        const MoveUnit& mu = plan_->units[(size_t)u];
        for (size_t j = 0; j < mu.weights.size(); ++j)
            res_.publish(mu.weights[j], dst + at[k] + mu.offset[j]);
        stats_.file_bytes += (uint64_t)unit_[(size_t)u].bytes;
    }
    ++stats_.file_layers;
    stats_.file_units += (uint64_t)staged->size();
    stats_.h2d_bytes += (uint64_t)off;
    return RAD_OK;
}

void Mover::release_slot(int32_t unit) {
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return; }
    UnitState& us = unit_[(size_t)unit];
    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Published;    /* nothing to copy; it is already "landed" */
    m.kind = Move::Kind::Release;
    m.unit = unit;
    m.slab_slot = us.slab_slot;
    m.pool_slot = -1;
    us.slab_slot = -1;
    republish(unit);                     /* the pointer stops naming the slot before the barrier */
}

int Mover::issue(const Swap& s, HeatEngine* heat) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;

    auto one = [&](int32_t unit, bool promote) -> bool {
        if (unit < 0) return false;
        UnitState& us = unit_[(size_t)unit];
        if (us.klass < 0) return false;
        ClassPools& cp = pools_[(size_t)us.klass];

        /* A unit that kept its host copy is dropped, not demoted: its pool slot still holds its
         * bytes, so there is nothing to copy and no second pool slot to take. */
        if (!promote && us.pool_slot >= 0 && us.slab_slot >= 0) return drop(unit, heat) >= 0;

        if (promote) {
            const int64_t slot = cp.slab.take();
            if (slot < 0) { ++stats_.starved_slab; return false; }
            if (promote_to(unit, slot) < 0) { cp.slab.add_free(slot); return false; }
            return true;
        }

        /* Every refusal comes before pool_slot_for, which may take a donor's host copy
         * (donor_slot) that a failed demotion could not give back. */
        if (us.slab_slot < 0) return false;
        const int rec = free_move_record();
        if (rec < 0) { ++stats_.no_record; return false; }
        if (record_gate() < 0) return false;
        const int64_t g = pool_slot_for(unit, heat);
        if (g < 0) { ++stats_.starved_pool; return false; }
        /* D2H out of the slab; the slab layout is the pool layout, so a straight copy. It is not
         * free even though zero-copy reads use the other direction: both directions share one
         * budget and a demotion costs about what a promotion does, the exchange rate heat.h's
         * min_gain is tuned against. */
        if (rad_event_wait(mover_, gate_) < 0) { cp.pool.add_free(g); return false; }
        queue_move_copy((char*)cp.pool_base + g * cp.stride, cp.slot_ptr(us.slab_slot), us.bytes,
                        rec);
        Move& m = moves_[(size_t)rec];
        m.kind = Move::Kind::Demote;
        m.slab_slot = us.slab_slot;
        m.pool_slot = g;
        stats_.d2h_bytes += (uint64_t)us.bytes;
        m.state = Move::State::Copying;
        m.unit = unit;
        ++copying_[(size_t)unit];
        return true;
    };

    const bool up = one(s.promote, true);
    const bool down = one(s.demote, false);
    /* A half-issued swap is legal: the halves supply each other, so the missing one goes out on
     * a later dispatch. A unit no move will publish must not stay busy, or the heat engine would
     * never consider it again, with no counter showing why. */
    if (heat) {
        if (!up) heat->set_busy(s.promote, false);
        if (!down) heat->set_busy(s.demote, false);
    }
    return (up || down) ? RAD_OK : RAD_E_FULL;
}

/* ------------------------------------------------------------------ the loan */
static const char* lender_name(int lender) {
    return lender == Mover::kLendKV ? "KV cache" : "activation arena";
}

static char* align_down_ptr(char* p) {
    const uintptr_t v = (uintptr_t)p;
    return (char*)(v - v % (uintptr_t)RAD_ALIGN_UNIT);
}

int Mover::lend_from(int lender, const std::vector<LoanStrip>& strips) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;      /* a static plan never moves a unit */
    if (lender < 0 || lender >= kLenders) return RAD_E_INVAL;
    for (const ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent)
            if (l.lender == lender) {
                RAD_ERR("mover: lend_from called twice for one lender; its slots are carved once");
                return RAD_E_STATE;
            }

    /* Slots go to classes with promotable units in the pinned pool, in proportion to their
     * bytes, and never more slots than such units, since a slot no unit can fill is wasted. */
    const size_t nc = pools_.size();
    std::vector<int64_t> want(nc, 0), got(nc, 0);
    for (size_t u = 0; u < unit_.size(); ++u) {
        const UnitState& us = unit_[u];
        if (us.klass < 0 || us.tier != Tier::HostPinned || us.pool_slot < 0) continue;
        if (!plan_->units[u].movable) continue;
        ++want[(size_t)us.klass];
    }
    std::vector<double> weight(nc, 0.0);
    for (size_t c = 0; c < nc; ++c)
        weight[c] = pools_[c].stride > 0 ? (double)want[c] * (double)pools_[c].stride : 0.0;

    /* Every stretch, top down. Classes take turns in proportion to their weight, so each stretch
     * holds each class's share regardless of the order the stretches arrive in. */
    struct Cut { int64_t need; char* p; };
    std::vector<std::vector<Cut>> cuts(nc);
    for (const LoanStrip& st : strips) {
        if (!st.top || st.bytes <= (int64_t)RAD_ALIGN_UNIT || st.unit_bytes <= 0 ||
            st.unit_tokens <= 0)
            continue;
        char* const bottom = st.top - st.bytes;
        /* The top alignment unit stays empty. A kernel assembling a packed code window reads a
         * few bytes past its slot; past the stretch's top is memory that is not the slab's (the
         * next layer's blocks, or at the end of the carve possibly unmapped memory). */
        char* cur = align_down_ptr(st.top - RAD_ALIGN_UNIT);
        for (;;) {
            int best = -1;
            double best_k = 0.0;
            for (size_t c = 0; c < nc; ++c) {
                if (weight[c] <= 0.0 || got[c] >= want[c]) continue;
                if (cur - bottom < pools_[c].stride) continue;
                if (align_down_ptr(cur - pools_[c].stride) < bottom) continue;
                const double k = (double)(got[c] + 1) * (double)pools_[c].stride / weight[c];
                if (best < 0 || k < best_k) { best = (int)c; best_k = k; }
            }
            if (best < 0) break;
            char* p = align_down_ptr(cur - pools_[(size_t)best].stride);
            /* The loan that covers this slot reaches its lowest byte. */
            const int64_t depth = st.top - p;
            const int64_t need = (depth + st.unit_bytes - 1) / st.unit_bytes * st.unit_tokens;
            cuts[(size_t)best].push_back(Cut{ need, p });
            ++got[(size_t)best];
            cur = p;
        }
    }

    /* Smallest loan first, so any loan covers a prefix of the slots and the deepest slot, which
     * the owner wants back first, is given back first. */
    int64_t n_new = 0, bytes_new = 0;
    for (size_t c = 0; c < nc; ++c) {
        if (cuts[c].empty()) continue;
        ClassPools& cp = pools_[c];
        std::stable_sort(cuts[c].begin(), cuts[c].end(),
                         [](const Cut& x, const Cut& y) { return x.need < y.need; });
        ClassPools::Lent l;
        l.lender = lender;
        l.seg = cp.slab.add_segment((int64_t)cuts[c].size());
        l.first = cp.slab.seg_lo(l.seg);
        /* Segments are appended in the same order as their pointers, so slot_ptr's
         * `slot - n_fixed` indexes flex_ptr directly. */
        if (l.first != cp.n_fixed + (int64_t)cp.flex_ptr.size()) {
            RAD_ERR("mover: a lent segment starts at slot %lld with %zu lent slots before it",
                    (long long)l.first, cp.flex_ptr.size());
            return RAD_E_STATE;
        }
        for (const Cut& k : cuts[c]) { cp.flex_ptr.push_back(k.p); l.need.push_back(k.need); }
        cp.lent.push_back(std::move(l));
        n_new += (int64_t)cuts[c].size();
        bytes_new += (int64_t)cuts[c].size() * cp.stride;
    }
    if (n_new <= 0) return RAD_OK;

    /* A record for every lendable slot. A fill promotes into every free slot at once and a recall
     * drops every unit past the loan at once, each holding a record until its barrier clears;
     * with only the shadow count, records rather than the link would limit how fast a loan is
     * used. */
    const size_t had = moves_.size();
    moves_.resize(had + (size_t)n_new);
    for (size_t i = had; i < moves_.size(); ++i) RAD_TRY(rad_event_create_local(&moves_[i].ev));

    RAD_INFO("mover: %lld slots can be lent out of the %s (%s), filled as it lends them",
             (long long)n_new, lender_name(lender),
             humanb(bytes_new).c_str());
    count_flex();
    return RAD_OK;
}

int Mover::set_loan(int lender, int64_t amount) {
    if (lender < 0 || lender >= kLenders) return RAD_E_INVAL;
    loan_[lender] = amount > 0 ? amount : 0;
    for (ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent) {
            if (l.lender != lender) continue;
            const int64_t n = std::upper_bound(l.need.begin(), l.need.end(), loan_[lender]) -
                              l.need.begin();
            if (cp.slab.target(l.seg) != l.first + n) {
                if (l.first + n < cp.slab.target(l.seg)) cp.strand_check = true;
                cp.slab.set_target(l.seg, l.first + n);
            }
        }
    count_flex();
    return RAD_OK;
}

int64_t Mover::loan_held(int lender) const {
    int64_t t = 0;
    for (const ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent) {
            if (l.lender != lender) continue;
            const int64_t top = cp.slab.top_held(l.seg);
            if (top >= l.first) t = std::max(t, l.need[(size_t)(top - l.first)]);
        }
    return t;
}

int64_t Mover::flex_capacity() const {
    int64_t n = 0;
    for (const ClassPools& cp : pools_) n += (int64_t)cp.flex_ptr.size() * cp.stride;
    return n;
}

int64_t Mover::loan_ceiling(int lender) const {
    int64_t t = 0;
    for (const ClassPools& cp : pools_)
        for (const ClassPools::Lent& l : cp.lent)
            if (l.lender == lender && !l.need.empty()) t = std::max(t, l.need.back());
    return t;
}

void Mover::count_flex() {
    int64_t used = 0, offered = 0;
    for (const ClassPools& cp : pools_) {
        if (cp.flex_ptr.empty()) continue;
        used += cp.slab.held_from(cp.n_fixed) * cp.stride;
        for (const ClassPools::Lent& l : cp.lent)
            offered += (cp.slab.target(l.seg) - l.first) * cp.stride;
    }
    flex_used_.store(used, std::memory_order_relaxed);
    flex_offered_.store(offered, std::memory_order_relaxed);
}

int Mover::drop(int32_t unit, HeatEngine* heat) {
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return RAD_E_FULL; }
    UnitState& us = unit_[(size_t)unit];
    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Published;       /* nothing to copy: the host copy never went away */
    m.kind = Move::Kind::Drop;
    m.unit = unit;
    m.slab_slot = us.slab_slot;
    m.pool_slot = -1;
    us.slab_slot = -1;
    us.tier = Tier::HostPinned;
    us.site = Site::DeviceZeroCopy;
    republish(unit);                        /* the pointer stops naming the slot before the barrier */
    ++stats_.demotions;
    ++stats_.dropped;
    if (heat) {
        heat->set_tier(unit, Tier::HostPinned);
        heat->set_busy(unit, true);
    }
    return RAD_OK;
}

/* The pool is sized to its units, so it has a free slot only while a promotion passes through
 * quarantine. Without donors a cold budgeted unit stays and the engine's swap drops a warmer lent
 * unit instead; replaying mixed routing at TP1, the hit rate is 72.7% that way, 76.7% with up to
 * n_turn donors per class, and 78.5% for one resident set with no regions.
 *
 * A donor loses its cheap recall: a recall or shrink must move it into a free budgeted slot or
 * back across the link (reclaim, balance). There is always room once quarantine has emptied,
 * since each donation left a budgeted slot free. Donors are capped at the turnover slots and come
 * only from the KV cache's loan, so the arena's per-prompt recall stays an address change.
 *
 * The hottest unit (of the victim's layer, when asked) donates, since a donor later demoted needs
 * a pool slot of its own. Its pool slot has no reader: the unit is resident and the copy into the
 * slot waits on the gate. */
int64_t Mover::donor_slot(size_t klass, int32_t layer, HeatEngine* heat) {
    ClassPools& cp = pools_[klass];
    if (cp.n_donated >= cp.n_turn) return -1;
    /* Not while the turnover slots are held (around a prompt): a donation releases a starved
     * swap, two copies on the stream the next prompt stages on. A decode run reaches the donor
     * count within a few steps anyway. */
    if (hold_turnover_) return -1;
    DonorRank& d = donors_[klass];
    if (d.at != dispatch_) {
        d.at = dispatch_;
        d.by_layer.assign((size_t)std::max(n_layers_, 1), {});
        for (const ClassPools::Lent& l : cp.lent) {
            if (l.lender != kLendKV) continue;
            const int64_t lo = cp.slab.seg_lo(l.seg), hi = cp.slab.target(l.seg);
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)klass || us.slab_slot < lo || us.slab_slot >= hi) continue;
                const int32_t ul = plan_->units[(size_t)u].layer;
                if (ul < 0 || ul >= n_layers_) continue;
                d.by_layer[(size_t)ul].push_back({ heat ? heat->heat(u) : 0.0f, u });
            }
        }
        for (auto& v : d.by_layer) std::sort(v.begin(), v.end());
    }
    for (;;) {
        /* The hottest candidate of the layer asked for, or of every layer. */
        std::vector<std::pair<float, int32_t>>* pick = nullptr;
        if (layer >= 0) {
            if (layer < (int32_t)d.by_layer.size() && !d.by_layer[(size_t)layer].empty())
                pick = &d.by_layer[(size_t)layer];
        } else {
            for (auto& v : d.by_layer)
                if (!v.empty() && (!pick || v.back() > pick->back())) pick = &v;
        }
        if (!pick) return -1;
        const int32_t u = pick->back().second;
        pick->pop_back();
        UnitState& us = unit_[(size_t)u];
        if (us.klass != (int32_t)klass || us.pool_slot < 0 || us.slab_slot < 0) continue;
        if (us.tier != Tier::VRAM || us.site != Site::Device || !cp.slab.covered(us.slab_slot))
            continue;
        if ((heat && heat->busy(u)) || unit_in_flight(u)) continue;
        const int64_t g = us.pool_slot;
        us.pool_slot = -1;
        ++cp.n_donated;
        ++stats_.donated;
        return g;
    }
}

/* The prefill stager copies runs of neighbouring pool slots, and the pool starts in unit order,
 * so a demotion into another layer's slot costs an extra copy in both layers on every later
 * prompt. Prefer a slot the unit's own layer started in, then a same-layer donor, then any free
 * slot, then any donor. At TP1 (26K-token prompts, 256 decoded tokens between) this takes a
 * layer's staging from 162 copies to 112. */
int64_t Mover::pool_slot_for(int32_t unit, HeatEngine* heat) {
    const UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    const int32_t layer = plan_->units[(size_t)unit].layer;
    const std::vector<int32_t>& home = pool_layer_[(size_t)us.klass];
    int64_t g = -1;
    if (layer >= 0) {
        g = cp.pool.take_where([&](int64_t p) { return (size_t)p < home.size() && home[(size_t)p] == layer; });
        if (g < 0) g = donor_slot((size_t)us.klass, layer, heat);
    }
    if (g < 0) g = cp.pool.take();
    if (g < 0) g = donor_slot((size_t)us.klass, -1, heat);
    return g;
}

bool Mover::migrate(int32_t unit, HeatEngine* heat) {
    UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    const int64_t to = cp.slab.take();
    if (to < 0) return false;
    if (to >= cp.n_fixed) {
        cp.slab.add_free(to);
        return false;
    }
    const int rec = free_move_record();
    if (rec < 0 || record_gate() < 0) {
        if (rec < 0) ++stats_.no_record;
        cp.slab.add_free(to);
        return false;
    }
    if (rad_event_wait(mover_, gate_) < 0) {
        cp.slab.add_free(to);
        return false;
    }
    queue_move_copy(cp.slot_ptr(to), cp.slot_ptr(us.slab_slot), us.bytes, rec);
    Move& m = moves_[(size_t)rec];
    m.state = Move::State::Copying;
    m.kind = Move::Kind::Migrate;
    m.unit = unit;
    m.slab_slot = us.slab_slot;
    m.pool_slot = -1;
    m.to_slot = to;
    ++copying_[(size_t)unit];
    if (heat) heat->set_busy(unit, true);
    return true;
}

int Mover::promote_to(int32_t unit, int64_t slot) {
    UnitState& us = unit_[(size_t)unit];
    ClassPools& cp = pools_[(size_t)us.klass];
    const int rec = free_move_record();
    if (rec < 0) { ++stats_.no_record; return RAD_E_FULL; }
    RAD_TRY(record_gate());
    if (us.pool_slot < 0) return RAD_E_STATE;
    const void* src = (const char*)cp.pool_base + us.pool_slot * cp.stride;
    Move& m = moves_[(size_t)rec];
    RAD_TRY(rad_event_wait(mover_, gate_));
    queue_move_copy(cp.slot_ptr(slot), src, us.bytes, rec);
    m.kind = Move::Kind::Promote;
    m.slab_slot = slot;
    /* Into a lent slot the host copy stays: the loan can be recalled at any time, and a unit
     * that kept its pool slot goes back by a change of address instead of needing a pool slot and
     * a link copy just as the cache wants its memory. The quarantine then has no pool slot to
     * free. */
    m.pool_slot = cp.is_flex(slot) ? -1 : us.pool_slot;
    m.state = Move::State::Copying;
    m.unit = unit;
    ++copying_[(size_t)unit];
    stats_.h2d_bytes += (uint64_t)us.bytes;
    return RAD_OK;
}

int Mover::fill(size_t klass, int64_t room, HeatEngine* heat) {
    if (room <= 0) return RAD_OK;
    struct Cand { float h; int32_t layer; int32_t unit; };
    std::vector<Cand> cand;
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        const UnitState& us = unit_[(size_t)u];
        if (us.klass != (int32_t)klass || us.tier != Tier::HostPinned || us.pool_slot < 0) continue;
        if (!plan_->units[(size_t)u].movable) continue;
        if (heat && heat->busy(u)) continue;
        cand.push_back(Cand{ heat ? heat->heat(u) : 0.0f, plan_->units[(size_t)u].layer, u });
    }
    std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.h > b.h; });

    /* Hottest first; on an exact tie, the layer with the fewest resident units. Every promotion
     * here is issued before any retires, so the heat engine's resident count does not move during
     * the batch and `added` tracks it instead; otherwise a fill of a cold slab (all heats zero)
     * would put the whole batch into whichever layer was thinnest at the start. */
    std::unordered_map<int32_t, int32_t> added;
    int64_t issued = 0;
    for (size_t i = 0; i < cand.size() && issued < room;) {
        size_t j = i;
        while (j < cand.size() && cand[j].h == cand[i].h) ++j;
        std::map<int32_t, std::vector<int32_t>> by_layer;            /* one heat, per layer */
        for (size_t k = j; k-- > i;) by_layer[cand[k].layer].push_back(cand[k].unit);
        using Key = std::pair<int32_t, int32_t>;                      /* (residents, layer) */
        std::priority_queue<Key, std::vector<Key>, std::greater<Key>> q;
        for (auto& kv : by_layer) {
            const int32_t rc = heat ? heat->layer_residents(kv.second.back()) : 0;
            q.push({ rc + added[kv.first], kv.first });
        }
        while (!q.empty() && issued < room) {
            const Key k = q.top();
            q.pop();
            std::vector<int32_t>& us = by_layer[k.second];
            const int32_t u = us.back();
            us.pop_back();
            /* A lent slot, never a working one. Shadow and staging slots are what let a swap
             * happen, and a unit filled into one sits in the budgeted range where no loan shrink
             * returns it, so the reserve would be lost. */
            const int64_t slot = pools_[klass].slab.take_at_least(pools_[klass].n_fixed);
            if (slot < 0) return RAD_OK;
            if (heat) heat->set_busy(u, true);
            if (promote_to(u, slot) < 0) {                   /* no record left: next time */
                pools_[klass].slab.add_free(slot);
                if (heat) heat->set_busy(u, false);
                return RAD_OK;
            }
            ++issued;
            ++added[k.second];
            if (!us.empty()) q.push({ k.first + 1, k.second });
        }
        i = j;
    }
    return RAD_OK;
}

int Mover::balance(HeatEngine* heat, bool turnover) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;

    for (size_t c = 0; c < pools_.size(); ++c) {
        ClassPools& cp = pools_[c];
        if (cp.flex_ptr.empty()) continue;
        bool drained = false;

        /* ---- Empty what the loan no longer covers, all of it, first: drops cost no link
         * traffic and the lender is waiting. A busy unit is still on its way in and goes on a
         * later call. The sweep walks every unit, so it runs only after a shrink
         * (ClassPools::strand_check). */
        if (cp.strand_check) {
            bool left = false;
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)c || us.slab_slot < 0 || cp.slab.covered(us.slab_slot)) continue;
                /* Anything else past the loan (an unretired promotion, a staged unit) becomes
                 * droppable later, so the sweep must come back for it. */
                if (us.tier != Tier::VRAM || us.site != Site::Device || (heat && heat->busy(u))) {
                    left = true;
                    continue;
                }
                /* A donor has no host copy to drop to: it moves into a free budgeted slot, or
                 * else back across the link, and stays past the loan until the copy lands. */
                if (us.pool_slot < 0) {
                    if (migrate(u, heat)) {
                        drained = true;
                    } else {
                        if (heat) heat->set_busy(u, true);
                        Swap s;
                        s.demote = u;
                        if (issue(s, heat) >= 0) drained = true;
                    }
                    left = true;
                    continue;
                }
                if (drop(u, heat) < 0) { left = true; break; }
                drained = true;
            }
            cp.strand_check = left;
        }

        /* ---- The working reserve. Shadow and staging slots must stay free whatever the loan,
         * or the mover cannot turn over; after a shrink the coldest lent units are dropped to
         * restore them. A slot already on its way back (`pending`) counts as free, or the drain
         * would repeat and empty the slab.
         *
         * The turnover slots are rebuilt the same way. They are the top of the lent range, which
         * a recall takes first, and the loan regrows only a prefill chunk at a time, so the fill
         * alone would leave them gone. A swap whose demotion found no pool slot is completed the
         * same way. */
        int64_t pending = 0;
        for (const Move& m : moves_) {
            if (m.state == Move::State::Free || m.unit < 0 || m.slab_slot < 0) continue;
            if (unit_[(size_t)m.unit].klass != (int32_t)c || !cp.slab.covered(m.slab_slot)) continue;
            if (m.frees_slab()) ++pending;
        }
        const int64_t turn = turnover ? cp.n_turn : 0;
        int64_t short_by = cp.n_work + turn - (cp.slab.n_free() + pending);
        if (short_by > 0) {
            drop_cand_.clear();
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)c || us.slab_slot < 0 || !cp.is_flex(us.slab_slot)) continue;
                if (us.tier != Tier::VRAM || us.site != Site::Device || us.pool_slot < 0) continue;
                if (heat && heat->busy(u)) continue;
                drop_cand_.push_back({ heat ? heat->heat(u) : 0.0f, u });
            }
            /* The coldest first, and on a tie the lower unit, so the choice is the same every run. */
            auto colder = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                return a.first != b.first ? a.first < b.first : a.second < b.second;
            };
            const size_t k = std::min(drop_cand_.size(), (size_t)short_by);
            std::partial_sort(drop_cand_.begin(), drop_cand_.begin() + (ptrdiff_t)k, drop_cand_.end(),
                              colder);
            for (size_t i = 0; i < k; ++i) {
                if (drop(drop_cand_[i].second, heat) < 0) break;
                drained = true;
                --short_by;
            }
        }
        /* With no lent unit left to drop, demote the coldest budgeted unit (a real copy, one per
         * call), for the working reserve only: link traffic to keep turnover slots would cost what
         * they save. A shrinking loan can take the free slots a swap left in the lent range, so
         * the reserve must be rebuildable from the budgeted slots. */
        if (short_by > turn) {
            int32_t victim = -1;
            float coldest = 0.0f;
            for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
                const UnitState& us = unit_[(size_t)u];
                if (us.klass != (int32_t)c || us.slab_slot < 0 || cp.is_flex(us.slab_slot)) continue;
                if (us.tier != Tier::VRAM || us.site != Site::Device || us.pool_slot >= 0) continue;
                if (!plan_->units[(size_t)u].movable) continue;
                if (heat && heat->busy(u)) continue;
                const float h = heat ? heat->heat(u) : 0.0f;
                if (victim < 0 || h < coldest) { victim = u; coldest = h; }
            }
            if (victim >= 0) {
                if (heat) heat->set_busy(victim, true);
                Swap s;
                s.demote = victim;
                if (issue(s, heat) >= 0) drained = true;
            }
        }
        /* A drain and a fill in the same call fight: the fill would take the slots the drain is
         * trying to hand back. */
        if (drained) continue;

        /* ---- Then fill every free slot past the working reserve and the turnover slots. */
        RAD_TRY(fill(c, cp.slab.n_free() - cp.n_work - turn, heat));
    }
    count_flex();
    return flush_moves();
}

int Mover::idle_tick(HeatEngine* heat) {
    if (!plan_ || !plan_->dynamic) return RAD_OK;
    RAD_TRY(retire(heat));
    return balance(heat, !hold_turnover_);
}

int Mover::reclaim(int lender, int64_t amount, HeatEngine* heat) {
    RAD_TRY(set_loan(lender, amount));
    if (loan_held(lender) <= loan_[lender]) return RAD_OK;
    if (!compute_ || !mover_) {
        RAD_ERR("mover: a loan cannot be recalled with no compute stream bound; nothing orders "
                "the kernels that may still read the slots");
        return RAD_E_STATE;
    }
    /* No copy may still be writing a slot: drain the mover's stream, then retire to publish
     * promotions that landed in slots about to be taken back. */
    RAD_TRY(flush_moves());
    RAD_TRY(rad_stream_sync(mover_));
    RAD_TRY(retire(heat));
    /* Drain quarantine too: the slots owed to the donors below come back through it (see
     * donor_slot). Nothing goes on the compute stream from here to the end of the recall. */
    RAD_TRY(rad_stream_sync(compute_));
    RAD_TRY(retire(heat));

    /* Every unit past the loan goes back to its host copy. A donor has none, so it moves into a
     * free budgeted slot or else across the link into a free pool slot, published once the
     * copies land. A donor with nowhere to go stays and the recall finishes short by its slot
     * (a later recall or balance() moves it); aborting instead would leak the slots already
     * taken off their units. */
    std::vector<std::pair<size_t, int64_t>> freed;
    std::vector<int32_t> moved;
    int stuck = 0;
    for (int32_t u = 0; u < (int32_t)unit_.size(); ++u) {
        UnitState& us = unit_[(size_t)u];
        if (us.klass < 0 || us.slab_slot < 0) continue;
        ClassPools& cp = pools_[(size_t)us.klass];
        if (!cp.is_flex(us.slab_slot) || cp.slab.covered(us.slab_slot)) continue;
        if (us.pool_slot < 0) {
            const int64_t from = us.slab_slot;
            int64_t to = cp.slab.take();
            if (to >= cp.n_fixed) {
                cp.slab.add_free(to);
                to = -1;
            }
            const int64_t g = to < 0 ? cp.pool.take() : -1;
            if (to >= 0) {
                RAD_TRY(rad_memcpy_async(cp.slot_ptr(to), cp.slot_ptr(from), us.bytes, mover_));
                us.slab_slot = to;
            } else if (g >= 0) {
                RAD_TRY(rad_memcpy_async((char*)cp.pool_base + g * cp.stride, cp.slot_ptr(from),
                                         us.bytes, mover_));
                us.pool_slot = g;
                us.slab_slot = -1;
                us.tier = Tier::HostPinned;
                us.site = Site::DeviceZeroCopy;
                stats_.d2h_bytes += (uint64_t)us.bytes;
                ++stats_.demotions;
                if (heat) {
                    heat->set_tier(u, Tier::HostPinned);
                    heat->set_busy(u, false);
                }
            } else {
                ++stuck;
                continue;
            }
            --cp.n_donated;
            ++stats_.relocated;
            freed.push_back({ (size_t)us.klass, from });
            moved.push_back(u);
            continue;
        }
        freed.push_back({ (size_t)us.klass, us.slab_slot });
        us.slab_slot = -1;
        us.tier = Tier::HostPinned;
        us.site = Site::DeviceZeroCopy;
        republish(u);
        ++stats_.demotions;
        ++stats_.dropped;
        if (heat) {
            heat->set_tier(u, Tier::HostPinned);
            heat->set_busy(u, false);
        }
    }

    if (!moved.empty()) {
        RAD_TRY(rad_stream_sync(mover_));
        for (int32_t u : moved) republish(u);
    }

    /* No kernel still names a freed slot: the compute stream is the slab's only reader (the
     * quarantine relies on the same fact), it drained above with nothing issued since, and work
     * issued from here reads the host copies. */
    for (const auto& f : freed) pools_[f.first].slab.add_free(f.second);

    /* Every record waiting on one of those slots is done: its barrier is behind the drain. A
     * promotion's record has nothing to free (its unit kept the pool slot); a drop's frees the
     * slot it left. */
    for (Move& m : moves_) {
        if (m.state == Move::State::Free || m.unit < 0 || m.slab_slot < 0) continue;
        const int32_t klass = unit_[(size_t)m.unit].klass;
        if (klass < 0) continue;
        ClassPools& cp = pools_[(size_t)klass];
        if (!cp.is_flex(m.slab_slot) || cp.slab.covered(m.slab_slot)) continue;
        if (m.frees_slab()) cp.slab.add_free(m.slab_slot);
        if (heat && m.kind != Move::Kind::Stage && m.kind != Move::Kind::Release)
            heat->set_busy(m.unit, false);
        if (m.state == Move::State::Copying) --copying_[(size_t)m.unit];
        m.clear();
    }
    count_flex();
    if (stuck > 0) {
        stats_.stranded += (uint64_t)stuck;
        RAD_ERR("mover: %d unit(s) that gave their host copy away found no free slot to move to; "
                "the %s loan stops at %lld instead of %lld until they move", stuck,
                lender_name(lender), (long long)loan_held(lender),
                (long long)loan_[lender]);
        return RAD_E_FULL;
    }
    if (loan_held(lender) > loan_[lender]) {
        RAD_ERR("mover: recalled the %s loan to %lld and a slot at %lld is still held",
                lender_name(lender), (long long)loan_[lender],
                (long long)loan_held(lender));
        return RAD_E_STATE;
    }
    return RAD_OK;
}

void Mover::queue_move_copy(void* dst, const void* src, int64_t bytes, int rec) {
    move_segs_.push_back(RadCopySeg{ src, dst, bytes });
    move_recs_.push_back((int32_t)rec);
}

/* Swap copies go to the DMA engine as one list. Copy kernels would compete with the step's
 * kernels: at 64 swaps a step, Qwen3.8-Flash-Next on one card spends about 10 ms of a 36 ms step
 * in them. Without a DMA queue of this process's own, rad_memcpy_list_async falls back to copy
 * kernels. The list yields to the link's collectives, like the prefill stager's. */
int Mover::flush_moves() {
    if (move_segs_.empty()) return RAD_OK;
    int rc = rad_memcpy_list_async(move_segs_.data(), (int)move_segs_.size(), RAD_COPY_YIELD,
                                   mover_);
    for (int32_t rec : move_recs_) {
        const int er = rad_event_record(moves_[(size_t)rec].ev, mover_);
        if (rc >= 0) rc = er;
    }
    move_segs_.clear();
    move_recs_.clear();
    return rc;
}

int Mover::retire(HeatEngine* heat) {
    RAD_TRY(flush_moves());
    for (Move& m : moves_) {
        if (m.state != Move::State::Copying) continue;
        const int q = rad_event_query(m.ev);
        if (q < 0) return RAD_E_DEVICE;
        if (q == 0) continue;

        --copying_[(size_t)m.unit];
        UnitState& us = unit_[(size_t)m.unit];
        switch (m.kind) {
            case Move::Kind::Promote:
                us.slab_slot = m.slab_slot;
                /* The source pool slot is freed after quarantine, unless the destination was
                 * lent, in which case the unit keeps it (see promote_to). */
                if (m.pool_slot >= 0) us.pool_slot = -1;
                us.tier = Tier::VRAM;
                us.site = Site::Device;
                ++stats_.promotions;
                if (heat) heat->set_tier(m.unit, Tier::VRAM);
                break;
            case Move::Kind::Demote:
                /* Out of a lent slot, the unit was a donor (donor_slot): any other unit there is
                 * dropped, not demoted. */
                if (pools_[(size_t)us.klass].is_flex(m.slab_slot)) --pools_[(size_t)us.klass].n_donated;
                us.pool_slot = m.pool_slot;
                us.slab_slot = -1;
                us.tier = Tier::HostPinned;
                /* The unit joins the zero-copy tier: the GEMM streams it from the pinned pool as
                 * it computes. */
                us.site = Site::DeviceZeroCopy;
                ++stats_.demotions;
                if (heat) heat->set_tier(m.unit, Tier::HostPinned);
                break;
            case Move::Kind::Stage:
                /* The destination was published at issue with the event. The copy has landed, so
                 * clear the event and later reads of the staged unit cost no runtime call. No
                 * generation bump: the address did not change, and a bump would make the run
                 * phase wait again on an event that has already signalled. */
                for (rad_weight w : plan_->units[(size_t)m.unit].weights) res_.settle(w);
                /* Straight to Free, skipping quarantine: a staging copy returns no slot (the
                 * later Release does, and is quarantined). Quarantining it would only hold the
                 * record for two more dispatches and stall the prefetch on a real card. */
                m.clear();
                continue;
            case Move::Kind::Migrate:
                us.slab_slot = m.to_slot;
                --pools_[(size_t)us.klass].n_donated;
                ++stats_.relocated;
                break;
            case Move::Kind::Release:
            case Move::Kind::Drop:
                break;
        }
        /* Safe without synchronisation: both copies of the bytes are valid now (see
         * Residency::publish). */
        republish(m.unit);
        m.state = Move::State::Published;
    }

    /* Quarantine, as a separate pass. Kernels issued in earlier layers may still name the freed
     * slot. Re-recording the move's event on the compute stream is the barrier: when it signals,
     * every launch made before the publish has retired and the slot is safe to write. It must be
     * downstream of the publish in stream order, so it is not folded into the loop above. */
    for (Move& m : moves_) {
        if (m.state != Move::State::Published) continue;
        if (!compute_) {
            RAD_ERR("mover: no compute stream to quarantine the slot just freed by unit %d",
                    (int)m.unit);
            return RAD_E_STATE;
        }
        RAD_TRY(rad_event_record(m.ev, compute_));
        m.state = Move::State::Quarantine;
        ++stats_.quarantined;
    }

    for (Move& m : moves_) {
        if (m.state != Move::State::Quarantine) continue;
        const int q = rad_event_query(m.ev);
        if (q < 0) return RAD_E_DEVICE;
        if (q == 0) continue;

        /* A Stage record never gets here: it is freed as soon as it lands. */
        ClassPools& cp = pools_[(size_t)unit_[(size_t)m.unit].klass];
        if (m.kind == Move::Kind::Promote) cp.pool.add_free(m.pool_slot);
        else if (m.frees_slab())           cp.slab.add_free(m.slab_slot);
        if (heat && m.kind != Move::Kind::Release) heat->set_busy(m.unit, false);
        m.clear();
    }
    count_flex();
    return RAD_OK;
}

std::string Mover::report() const {
    const double gib = 1.0 / 1073741824.0;
    std::string s = fmt("mover: %llu dispatches, %llu promotions, %llu demotions, %llu staged "
                        "(%.2f GiB h2d, %.2f GiB d2h)\n",
                        (unsigned long long)stats_.dispatches,
                        (unsigned long long)stats_.promotions,
                        (unsigned long long)stats_.demotions,
                        (unsigned long long)stats_.prefetched,
                        (double)stats_.h2d_bytes * gib, (double)stats_.d2h_bytes * gib);
    /* Supply counters: they separate a mover at its budget from one that never got to spend it. */
    s += fmt("  %llu found no free move record, %llu starved of a slab slot, %llu of a pool "
             "slot; %llu slots quarantined\n",
             (unsigned long long)stats_.no_record, (unsigned long long)stats_.starved_slab,
             (unsigned long long)stats_.starved_pool, (unsigned long long)stats_.quarantined);
    if (stats_.donated)
        s += fmt("  %llu host copies given by lent units to budgeted victims, %llu such units moved "
                 "by a recall, %llu left past one with no free slot\n",
                 (unsigned long long)stats_.donated, (unsigned long long)stats_.relocated,
                 (unsigned long long)stats_.stranded);
    if (stats_.ssd_reads)
        s += fmt("  file tier: %llu demand reads (%.2f GiB), each a blocking pread in the middle "
                 "of a dispatch. Raise --host-pool-mib.\n",
                 (unsigned long long)stats_.ssd_reads, (double)stats_.ssd_bytes * gib);
    /* The file stager's cost to the issuing thread: read is time in the drive, wait is time on a
     * window whose copies had not drained when it came round again. */
    if (stats_.file_layers) {
        const double rs = (double)stats_.file_read_ns * 1e-9;
        if (native_file_cache_on_ || mapped_file_cache_on_)
            s += fmt("  file stager: %llu routed layers, %llu units, %.2f GiB native bytes staged "
                     "from RAM or disk; %.2f s waiting on windows\n",
                     (unsigned long long)stats_.file_layers, (unsigned long long)stats_.file_units,
                     (double)stats_.file_bytes * gib, (double)stats_.file_wait_ns * 1e-9);
        else s += fmt("  file stager: %llu routed layers read, %llu units, %.2f GiB in %llu preads; "
                 "%.2f s reading (%.2f GB/s), %.2f s waiting on windows\n",
                 (unsigned long long)stats_.file_layers, (unsigned long long)stats_.file_units,
                 (double)stats_.file_bytes * gib, (unsigned long long)stats_.file_preads, rs,
                 rs > 0 ? (double)stats_.file_bytes / rs * 1e-9 : 0.0,
                 (double)stats_.file_wait_ns * 1e-9);
    }
    if (stats_.file_relaid)
        s += fmt("  file tier relayout: %llu weights relaid on the read (%.2f GiB), %.2f s of the "
                 "issuing thread in the file stager's read+relayout\n",
                 (unsigned long long)stats_.file_relaid, (double)stats_.file_relaid_bytes * gib,
                 (double)stats_.file_relayout_ns * 1e-9);
    if (host_plane_reads_.load())
        s += fmt("  host experts' plane reads: %llu weights read for the CPU in %.2f s of issuing-thread "
                 "time (%.2f ms a batch weight), %llu relaid into the RAM cache beside it\n",
                 (unsigned long long)host_plane_reads_.load(), (double)host_plane_read_ns_.load() * 1e-9,
                 (double)host_plane_read_ns_.load() * 1e-6 / (double)host_plane_reads_.load(),
                 (unsigned long long)host_plane_inserts_.load());
    if (mirror_tail_ > 0 && PlaneMirror::get()) {
        const PlaneMirror::Stats ms = PlaneMirror::get()->stats();
        if (ms.reads)
            s += fmt("  plane mirror (all ranks): %llu reads, %.2f GiB, %.2f ms a read\n",
                     (unsigned long long)ms.reads, (double)ms.bytes * gib, (double)ms.ns * 1e-6 / (double)ms.reads);
    }
    if (relaid_reads_.load())
        s += fmt("  file tier relayout split: %llu reads; %.2f s in preads and %.2f s relaying, summed "
                 "over the relay threads (%.2f ms and %.2f ms a weight)\n",
                 (unsigned long long)relaid_reads_.load(), (double)relaid_read_ns_.load() * 1e-9,
                 (double)relaid_pack_ns_.load() * 1e-9,
                 (double)relaid_read_ns_.load() * 1e-6 / (double)relaid_reads_.load(),
                 (double)relaid_pack_ns_.load() * 1e-6 / (double)relaid_reads_.load());
    if (native_file_cache_on_) {
        const NativeFileCache::Stats c = native_file_cache_.stats();
        s += fmt("  native file RAM cache: %llu hits (%.2f GiB), %llu misses (%.2f GiB), "
                 "%llu evictions, %llu bypasses; %s charged of %s\n",
                 (unsigned long long)c.hits, (double)c.hit_bytes * gib,
                 (unsigned long long)c.misses, (double)c.miss_bytes * gib,
                 (unsigned long long)c.evictions, (unsigned long long)c.bypasses,
                 humanb(c.metadata_bytes + c.payload_charge).c_str(), humanb(c.budget_bytes).c_str());
    }
    if (stats_.file_prefill_ro_hits || stats_.file_prefill_ro_misses) {
        s += fmt("  native prefill read-only: %llu cached weights (%.2f GiB), "
                 "%llu read/relayout weights (%.2f GiB) not admitted; completed CPU windows, no LRU touch\n",
                 (unsigned long long)stats_.file_prefill_ro_hits, (double)stats_.file_prefill_ro_hit_bytes * gib,
                 (unsigned long long)stats_.file_prefill_ro_misses, (double)stats_.file_prefill_ro_miss_bytes * gib);
    }
    if (mapped_file_cache_on_) {
        const auto c = mapped_file_cache_.stats();
        s += c.slab_policy ?
            "  mapped cache ReadOnly policy: retained-slab hit lease without LRU touch; miss read/pack windows without admission\n" :
            "  mapped cache ReadOnly policy: bypass to original read/pack windows; no lease/LRU touch/admission\n";
        s += fmt(c.slab_policy ?
                 "  mapped retained-slab COPY-source cache: %llu hits, %llu misses, %llu evictions, %llu bypasses; "
                 "%s owned logical payload (resident/idle), %s conservative mapped backing, %s metadata, %s policy cap; "
                 "%llu copies/%.2f GiB without CPU bounce; %llu mapped allocations/%llu public free calls\n" :
                 "  mapped COPY-source cache: %llu hits, %llu misses, %llu evictions, %llu bypasses; "
                 "%s owned logical payload (resident/idle), %s conservative mapped backing, %s metadata, %s policy cap; "
                 "%llu copies/%.2f GiB without CPU bounce; %llu mapped allocations/%llu public free calls\n",
                 (unsigned long long)c.cache.hits,(unsigned long long)c.cache.misses,
                 (unsigned long long)c.cache.evictions,(unsigned long long)c.cache.bypasses,
                 humanb(c.logical_payload).c_str(),humanb(c.mapped_policy_charge).c_str(),
                 humanb(c.external_metadata+c.cache.metadata).c_str(),
                 humanb(c.external_metadata+c.cache.budget).c_str(),
                 (unsigned long long)stats_.mapped_source_copies,double(stats_.mapped_source_bytes)*gib,
                 (unsigned long long)c.allocations,(unsigned long long)c.public_free_invocations);
        if(c.slab_policy){
            const auto& a=c.slabs;
            if(mapped_direct_fill_on_)s+=fmt("  mapped direct fill: %llu reservations, %llu committed (%s), %llu cancelled (%s), %zu active; %llu same-key busy bypasses; no fill-duration cache locks; full-weight only\n",
                (unsigned long long)a.fill_reservations,(unsigned long long)a.fill_commits,humanb(a.fill_bytes).c_str(),
                (unsigned long long)a.fill_aborts,humanb(a.fill_abort_bytes).c_str(),a.fills_active,(unsigned long long)a.fill_busy_bypasses);
            s += fmt("  mapped retained-slab ledger: %zu owners/%zu terminal free-invoked; %s retained backing, %s terminal charge, %s growth reserved; "
                     "%llu growth attempts/%llu failures; %llu device-bind failures; same-class LRU, no per-eviction public free; policy charge is not RSS/unpin proof\n",
                     a.retained_slab_owners,a.terminal_free_invoked_owners,humanb(a.retained_backing_charge).c_str(),
                     humanb(a.terminal_backing_charge).c_str(),humanb(a.cache.reserved_charge).c_str(),
                     (unsigned long long)a.growth_attempts,(unsigned long long)a.growth_failures,(unsigned long long)c.api.device_bind_failures);
            for(int k=0;k<2;++k){const auto& cl=a.classes[k];
                s += fmt("  mapped retained-slab class %s: %zu capacity/%zu resident/%zu idle/%zu busy/%zu leased slots; "
                         "%s logical capacity/%s resident/%s idle/%s leased bytes; %llu class bypasses/%llu leased-tail bypasses\n",
                         k==0?"GU":"DN",cl.capacity_slots,cl.resident_slots,cl.idle_slots,cl.busy_slots,cl.leased_slots,
                         humanb(cl.capacity_logical_bytes).c_str(),humanb(cl.resident_logical_bytes).c_str(),
                         humanb(cl.idle_logical_bytes).c_str(),humanb(cl.leased_logical_bytes).c_str(),
                         (unsigned long long)cl.bypasses,(unsigned long long)cl.leased_tail_bypasses);
            }
        }
        s += fmt("  mapped cache API timing (aggregate before final clear): %llu alloc attempts, "
                 "%llu null returns, %llu alias rejections; alloc %llu ns total/%llu ns max; "
                 "%llu free invocations, %llu ns total/%llu ns max; monotonic HOST elapsed, "
                 "not per-request/pure pin time/confirmed unpin\n",
                 (unsigned long long)c.api.allocate_attempts,(unsigned long long)c.api.allocate_null_returns,
                 (unsigned long long)c.api.alias_rejections,(unsigned long long)c.api.allocate_ns,
                 (unsigned long long)c.api.allocate_max_ns,(unsigned long long)c.api.free_invocations,
                 (unsigned long long)c.api.free_ns,(unsigned long long)c.api.free_max_ns);
    }
    if (native_file_workers_on_) {
        const auto readers = relay_workers_.stats();
        s += fmt("  native read workers: %llu pooled windows, %llu serial windows, "
                 "%llu startup-fallback windows, %llu threads started\n",
                 (unsigned long long)readers.pooled_dispatches,
                 (unsigned long long)readers.serial_dispatches,
                 (unsigned long long)readers.fallback_dispatches,
                 (unsigned long long)readers.thread_starts);
    }
    return s;
}

}  /* namespace rad */
