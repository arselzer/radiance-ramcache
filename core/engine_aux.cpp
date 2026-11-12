/* engine_aux.cpp -- the engine's part of the aux expert worker (place/aux_expert_worker.h): discovery, the
 * start (unit set, lanes, load, tables, warm-up, binding) and turning it off. Off (the default) none of it
 * runs past the switch's parse: no pin set, no allocation, no sysfs read, no log line. */
#include "engine_priv.h"
#include "rad_internal.h"
#include "device/device.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>

namespace rad {

void Engine::aux_off(const std::string& why, bool warn) {
    aux_ctl_stop();   /* the control thread first: it touches the lanes and the arena */
    for (auto& r : ranks_)
        if (r) r->fstager.aux_unbind();
    /* The process line, as at exit: whether every lane drained, or which are left to the process exit with the
     * arena (a load or warm-up whose copies did not complete leaves them queued). */
    if (aux_) {
        const std::string line = aux_->finish(0, 0, 0);
        if (aux_->drained()) RAD_INFO("%s", line.c_str());
        else RAD_WARN("%s", line.c_str());
    }
    if (aux_ctl_) {   /* the state file's last word: the worker is off for the process's life */
        const bool held = aux_ && !aux_->drained() && aux_->has_arena();
        aux_ctl_->write_final(held ? fmt("dead held %lld", (long long)aux_cfg_.cap_mib) : std::string("dead freed"),
                              "the worker is off: " + why);
        aux_ctl_.reset();
    }
    aux_.reset();
    if (warn) RAD_WARN("%s%s", ds41aux::kLogOff, why.c_str());
    else RAD_INFO("%s%s", ds41aux::kLogOff, why.c_str());
}

int Engine::aux_discover(const std::vector<int>& devs) {
    std::string why;
    const int p = ds41aux::parse([](const char* k) { return (const char*)std::getenv(k); }, &aux_cfg_, &why);
    if (p == 0) return RAD_OK;   /* off: nothing else is read */
    if (p < 0) {
        RAD_ERR("%s", why.c_str());
        return RAD_E_INVAL;
    }
    /* What the worker serves must be the configuration its name says, or the arm runs without it. */
    auto env = [](const char* k) {
        const char* v = std::getenv(k);
        return std::string(v ? v : "");
    };
    const std::string hx = env("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS");
    std::string bad;
    if (hx.empty() || hx == "0")
        bad = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS is unset or 0: there is no CPU path to serve";
    else if (env("RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY") != "all")
        bad = "RADIANCE_EXPERIMENTAL_DS41_HOST_EXPERTS_POLICY is not all (greedy sends units to the copy link, verify "
              "is the card-against-CPU mode)";
    else if (env("RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS").empty())
        bad = "RADIANCE_EXPERIMENTAL_DS41_FILE_BUFFER_UNITS is unset: without the cap not every file-tier unit is the "
              "CPU's";
    else if (env("RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE") != "1")
        bad = "RADIANCE_EXPERIMENTAL_DS41_PACKED_CACHE is not 1: the worker loads the packed cache's slots";
    else if (env("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED") != "1")
        bad = "RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED is not 1: the worker runs gemv_vp, the decode band's row under it";
    else if (env("RADIANCE_EXPERIMENTAL_SCHED_TBO") == "1")
        bad = "RADIANCE_EXPERIMENTAL_SCHED_TBO=1 slices the routed layers";
    else if (!env("RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE").empty())
        bad = "RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE is set: its choice record has no code for the worker's units";
    if (!bad.empty()) {
        RAD_ERR("%s=%lld refused: %s", ds41aux::kEnv, (long long)aux_cfg_.cap_mib, bad.c_str());
        return RAD_E_INVAL;
    }
    /* The pin set, as soon as the switch is on: whatever the decision, pinned memory maps only to the ranks'
     * cards from here on, so a card discovery leaves out maps nothing, as with four visible cards. */
    const std::vector<int> ranks(devs.begin(), devs.begin() + std::min<size_t>(devs.size(), (size_t)cfg_.tp));
    const int pr = dev_pin_devices(ranks);
    if (pr == RAD_E_UNSUPPORTED) {
        aux_off("this device backend maps pinned memory to every card", false);
        return RAD_OK;
    }
    if (pr < 0) {
        aux_off(fmt("the pin set was refused: %s", rad_dev_last_error()), true);
        return RAD_OK;
    }
    const std::vector<ds41aux::Card> cards = aux::inventory();
    const ds41aux::Decision d = ds41aux::decide(cards, ranks, aux_cfg_, aux::selector_bdf(aux_cfg_));
    if (!d.on) {
        aux_off(d.why, d.warn);
        return RAD_OK;
    }
    const ds41aux::Card* c = nullptr;
    for (const ds41aux::Card& x : cards)
        if (x.ord == d.ord) c = &x;
    auto w = std::make_unique<aux::Worker>(aux_cfg_, *c, ranks);
    if (aux_cfg_.controller()) {
        /* Card sharing or manual control (work/aux-dyn/SPEC.md A.5): the controller decides before any reserve
         * (the mode, the holders, the probe), writes `starting` and watches from then on, so a job started through
         * aux_yield.sh during the load finds the card given back. A busy card starts released. */
        aux_ctl_ = std::make_unique<aux::Control>(aux_cfg_, *c, (int)ranks.size());
        if (aux_ctl_->init_share(aux::Roots{}, (long)::getpid(), &why) < 0) {
            RAD_ERR("%s", why.c_str());
            aux_ctl_.reset();
            return RAD_E_INVAL;
        }
        std::string busy = d.released ? d.why : std::string();
        bool held = !d.released && aux_ctl_->may_reserve(&busy);
        if (held) {
            aux_ctl_->watch(w.get(), /*reserving=*/true);
            const int rc = w->reserve(devs[0], &why);
            if (rc >= 0) aux_ledger_ = w->arena_bytes();   /* in the device ledger even if given back at once */
            if (!aux_ctl_->reserved(rc >= 0)) {
                if (rc >= 0) {   /* the watch asked for the card back during the reserve */
                    w->release_arena(devs[0]);
                    busy = "released during the reserve";
                } else {
                    busy = why;
                }
                held = false;
            }
        }
        if (!held) aux_ctl_->watch(w.get(), /*reserving=*/false);
        aux_released_why_ = held ? std::string() : busy;
        if (held)
            RAD_INFO("%s%d (%s, %s, %s): %s free of %s; reserved %lld MiB (headroom %lld MiB, %s free after); %s; "
                     "pinned host memory maps to the %zu rank cards only", ds41aux::kLogDevice, c->ord, c->bdf.c_str(),
                     c->name.c_str(), c->arch.c_str(), humanb(c->vram_free).c_str(), humanb(c->vram_bytes).c_str(),
                     (long long)aux_cfg_.cap_mib, (long long)aux_cfg_.headroom_mib, humanb(w->free_after()).c_str(),
                     d.topology.c_str(), ranks.size());
        else
            RAD_INFO("%s%d (%s, %s, %s): %s free of %s; released at start (%s): nothing reserved; %s; pinned host "
                     "memory maps to the %zu rank cards only", ds41aux::kLogDevice, c->ord, c->bdf.c_str(),
                     c->name.c_str(), c->arch.c_str(), humanb(c->vram_free).c_str(), humanb(c->vram_bytes).c_str(),
                     busy.c_str(), d.topology.c_str(), ranks.size());
        aux_ = std::move(w);
        return RAD_OK;
    }
    if (w->reserve(devs[0], &why) < 0) {
        aux_off(why, true);
        return RAD_OK;
    }
    aux_ledger_ = w->arena_bytes();
    RAD_INFO("%s%d (%s, %s, %s): %s free of %s; reserved %lld MiB (headroom %lld MiB, %s free after); %s; pinned host "
             "memory maps to the %zu rank cards only", ds41aux::kLogDevice, c->ord, c->bdf.c_str(), c->name.c_str(),
             c->arch.c_str(), humanb(c->vram_free).c_str(), humanb(c->vram_bytes).c_str(), (long long)aux_cfg_.cap_mib,
             (long long)aux_cfg_.headroom_mib, humanb(w->free_after()).c_str(), d.topology.c_str(), ranks.size());
    aux_ = std::move(w);
    return RAD_OK;
}

int Engine::aux_start() {
    if (!aux_) return RAD_OK;
    const auto t0 = std::chrono::steady_clock::now();
    /* The late configuration checks: refusals, so an arm never runs without what its name says. */
    if (!cfg_.deterministic || (!ranks_.empty() && ranks_[0]->heat.enabled())) {
        RAD_ERR("%s=%lld refused: a dynamic plan (the heat engine moves units); the worker's set is static and needs "
                "--deterministic", ds41aux::kEnv, (long long)aux_cfg_.cap_mib);
        return RAD_E_INVAL;
    }
    for (auto& r : ranks_)
        if (!r->fstager.host_experts()) {
            RAD_ERR("%s=%lld refused: rank %d's file stager has no host experts (the DSpark head tail owns its routed "
                    "layers?)", ds41aux::kEnv, (long long)aux_cfg_.cap_mib, r->index);
            return RAD_E_INVAL;
        }
    std::string why;
    for (auto& r : ranks_)
        if (r->fstager.aux_resolve(&why, (int)aux_cfg_.rows) < 0) {
            aux_off(fmt("rank %d: %s", r->index, why.c_str()), true);
            return RAD_OK;
        }
    /* The plugins' device code must run on the extra card as on rank 0's (the warm-up job below runs it). */
    {
        std::string cw;
        const size_t mark = dev_code_mark();
        if (mark > 0 && dev_code_runs(0, mark, ranks_[0]->device, nullptr) == 1 &&
            dev_code_runs(0, mark, aux_->device(), &cw) == 0) {
            aux_off(fmt("the kernel libraries' device code does not run on device %d: %s", aux_->device(), cw.c_str()),
                    true);
            return RAD_OK;
        }
    }
    /* The unit set: every rank's candidates in the planner's order, cut at what the arena holds. */
    std::vector<ds41aux::Cand> all;
    std::vector<std::vector<ds41aux::Cand>> per((size_t)ranks_.size());
    double share_all = 0;
    for (auto& r : ranks_) {
        if (r->fstager.aux_candidates(&per[(size_t)r->index]) < 0) {
            aux_off(fmt("rank %d's candidates could not be listed", r->index), true);
            return RAD_OK;
        }
        for (const ds41aux::Cand& c : per[(size_t)r->index]) {
            all.push_back(c);
            share_all += c.share;
        }
    }
    if (all.empty()) {
        aux_off("no rank has a file-tier unit the worker can hold", true);
        return RAD_OK;
    }
    const int64_t gu_slot = ranks_[0]->mover.packed_slot_bytes(0), dn_slot = ranks_[0]->mover.packed_slot_bytes(1);
    if (gu_slot <= 0 || dn_slot <= 0) {
        aux_off("the RAM cache is not packed", true);
        return RAD_OK;
    }
    const ds41aux::Layout L = ds41aux::layout(gu_slot, dn_slot);
    const int R = (int)ranks_.size();
    const int layers = (int)ranks_[0]->fstager.aux_routed_layers();
    const int64_t N = ds41aux::capacity(aux_cfg_.cap_mib, R, layers, L.stride, (int64_t)all.size(), (int)aux_cfg_.rows);
    if (N <= 0) {
        aux_off(fmt("%lld MiB hold no unit beside the tables and lanes (%s)", (long long)aux_cfg_.cap_mib,
                    humanb(ds41aux::fixed_bytes(R, layers, (int)aux_cfg_.rows)).c_str()), true);
        return RAD_OK;
    }
    /* The online set keeps `spare` slots free for its copies (work/aux-dyn/SPEC.md §3.4): the set holds N - spare
     * units; without it spare is 0 and the quotas are the merged ones. */
    const int64_t spare = aux_cfg_.online ? aux_cfg_.spare : 0;
    if (spare > 0 && spare >= N) {
        RAD_ERR("%s=%lld refused: the arena holds %lld unit slots, and %s=%lld keeps that many or more free",
                ds41aux::kEnv, (long long)aux_cfg_.cap_mib, (long long)N, ds41aux::kOnlineSpareEnv, (long long)spare);
        return RAD_E_INVAL;
    }
    const std::vector<int64_t> q = ds41aux::quotas(all, N - spare, R);
    /* Card sharing: a released start loads nothing (the arena was never reserved, or the controller's watch gave
     * it back before the load); its lanes are made, and the first acquire carves the arena they work in. */
    std::string rel_why = aux_released_why_;
    /* claim() transfers the startup arena under start_mu_: the watch may free a reserved arena until then.
     * Inspecting has_arena() first would race that release. Once claimed, the watch only aborts the load. */
    bool released = aux_ctl_ ? !aux_ctl_->claim(&rel_why) : !aux_->has_arena();
    if (released) {
        aux_->plan(layers, N, L);
        if (aux_->make_lanes(ranks_[0]->device, &why) < 0) {
            aux_off(why, true);
            return RAD_OK;
        }
    } else if (aux_->build(layers, N, L, ranks_[0]->device, &why) < 0) {
        aux_off(why, true);
        return RAD_OK;
    }
    /* Each rank's list in the global order, loaded on its own thread through its own lane, up to its quota:
     * a unit that does not load is replaced by the rank's next candidate, so the set does not depend on the
     * loaders' timing. */
    std::vector<std::vector<int32_t>> lists((size_t)R);
    for (auto& v : per) std::sort(v.begin(), v.end(), ds41aux::before);
    for (int r = 0; r < R; ++r)
        for (const ds41aux::Cand& c : per[(size_t)r]) lists[(size_t)r].push_back(c.unit);
    std::vector<FileStager::AuxLoadStats> st((size_t)R);
    std::vector<std::string> err((size_t)R);
    std::vector<int> rc((size_t)R, RAD_OK);
    const std::atomic<bool>* abort = aux_ctl_ ? aux_ctl_->load_abort() : nullptr;
    if (!released) {
        {
            std::vector<std::thread> th;
            int64_t first = 0;
            for (int r = 0; r < R; ++r) {
                th.emplace_back([&, r, first] {
                    rc[(size_t)r] = ranks_[(size_t)r]->fstager.aux_load(aux_.get(), r, lists[(size_t)r], first,
                                                                        q[(size_t)r], &st[(size_t)r], &err[(size_t)r],
                                                                        abort);
                });
                first += q[(size_t)r];
            }
            for (auto& t : th) t.join();
        }
        for (int r = 0; r < R; ++r)
            if (rc[(size_t)r] < 0) {
                aux_off(fmt("the load failed: %s", err[(size_t)r].c_str()), true);
                return RAD_OK;
            }
        if (abort && abort->load(std::memory_order_acquire)) {
            /* The controller's watch asked for the card back during the load: every loader synced its copies, so
             * the lanes drain at once; the arena goes and the worker starts released. */
            for (int r = 0; r < R; ++r)
                if (aux::Lane* l = aux_->lane(r); l && !l->drain(2'000'000'000LL)) {
                    aux_off(fmt("released during the load, but rank %d's lane did not drain", r), true);
                    return RAD_OK;
                }
            aux_->release_arena(ranks_[0]->device);
            for (int r = 0; r < R; ++r) {
                aux::RankView& v = aux_->view(r);
                std::fill(v.resident.begin(), v.resident.end(), (uint8_t)0);
                std::fill(v.slot.begin(), v.slot.end(), -1);
                std::fill(v.gu_tab_entries.begin(), v.gu_tab_entries.end(), nullptr);
                std::fill(v.dn_tab_entries.begin(), v.dn_tab_entries.end(), nullptr);
                v.units = 0;
                v.share = 0;
            }
            released = true;
            rel_why = "released during the load";
        }
    }
    if (!released) {
        if (aux_->units() <= 0) {
            aux_off("no unit loaded", true);
            return RAD_OK;
        }
        if (aux_->load_tables(&why) < 0) {
            aux_off(why, true);
            return RAD_OK;
        }
        /* A warm-up job a lane, against the CPU pool's rows: libr4d's code object loads on the extra card here,
         * not in a rank thread's first launch, and its VRAM is in the headroom check below. */
        for (int r = 0; r < R; ++r)
            if (ranks_[(size_t)r]->fstager.aux_warm(aux_.get(), r, &why) < 0) {
                aux_off(why, true);
                return RAD_OK;
            }
        if (aux_->check_headroom(&why) < 0) {
            aux_off(why, true);
            return RAD_OK;
        }
    }
    /* The control thread (work/aux-dyn/SPEC.md §3), when a switch asks for it: the gate (card sharing) and the
     * online set's candidates go to each stager with its binding. */
    std::vector<std::vector<uint8_t>> cand_bits((size_t)R);
    if (aux_cfg_.control_thread() && !aux_ctl_) aux_ctl_ = std::make_unique<aux::Control>(aux_cfg_, aux_->card(), R);
    for (int r = 0; r < R; ++r) {
        FileStager& fs = ranks_[(size_t)r]->fstager;
        fs.aux_bind(aux_->lane(r), &aux_->view(r), aux_cfg_);
        if (aux_ctl_) {
            if (aux_cfg_.online) {
                cand_bits[(size_t)r].assign(fs.aux_layer_index().size(), 0);
                for (const ds41aux::Cand& c : per[(size_t)r])
                    if (c.unit >= 0 && (size_t)c.unit < cand_bits[(size_t)r].size()) cand_bits[(size_t)r][(size_t)c.unit] = 1;
            }
            fs.aux_set_dynamic(aux_ctl_->gate(), r, aux_cfg_.online ? &cand_bits[(size_t)r] : nullptr);
        }
    }
    if (aux_ctl_) {
        std::vector<aux::RankUnits> units((size_t)R);
        for (int r = 0; r < R; ++r) {
            const std::vector<int32_t> li = ranks_[(size_t)r]->fstager.aux_layer_index();
            aux::RankUnits& u = units[(size_t)r];
            u.cands = per[(size_t)r];
            u.quota = q[(size_t)r];
            u.n_plan_units = (int32_t)li.size();
            for (const ds41aux::Cand& c : u.cands) u.li.push_back(c.unit >= 0 && (size_t)c.unit < li.size() ? li[(size_t)c.unit] : -1);
        }
        aux::Hooks h;
        h.source = [this, L](int r, int32_t u, aux::Hooks::Source* s) {
            auto* hu = new Mover::HostUnit();
            if (!ranks_[(size_t)r]->fstager.aux_source(u, L, hu, &s->gu, &s->gu_n, &s->dn, &s->dn_n)) {
                delete hu;
                return false;
            }
            s->lease = hu;
            return true;
        };
        h.release = [this](int r, aux::Hooks::Source* s) {
            if (auto* hu = (Mover::HostUnit*)s->lease) {
                ranks_[(size_t)r]->mover.release_host_unit(hu);
                delete hu;
            }
            s->lease = nullptr;
        };
        h.warm = [this](int r, int32_t u, int32_t li, const void* g, const void* d, std::string* w) {
            return ranks_[(size_t)r]->fstager.aux_warm_unit(aux_->lane(r), u, li, g, d, w);
        };
        h.hit_tokens = [this](int r) { return ranks_[(size_t)r]->fstager.aux_hit_tokens(); };
        h.swap_hits = [this](int r, std::vector<uint32_t>* e) { return ranks_[(size_t)r]->fstager.aux_swap_hits(e); };
        h.guard_tripped = [this](int r) { return ranks_[(size_t)r]->fstager.aux_guard_tripped(); };
        const int dev = aux_->device();
        h.kfd_available = [dev] { return dev_card_available(dev); };
        if (aux_ctl_->attach(aux_.get(), std::move(h), std::move(units), layers, N, L, released, rel_why, &why) < 0) {
            aux_off("the control thread: " + why, true);
            return RAD_OK;
        }
    }
    (void)rad_dev_set(ranks_[0]->device);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    /* RADIANCE_EXPERIMENTAL_DS41_AUX_ROWS above 16 (ds41_aux.h, "taller passes"): said on the startup line. */
    const std::string tall = aux_cfg_.rows > ds41aux::kMaxRows
        ? fmt("; %s=%lld: passes of 17-%lld rows by %s, the deadline D * ceil(rows / 16), lanes of %s each",
              ds41aux::kRowsEnv, (long long)aux_cfg_.rows, (long long)aux_cfg_.rows, ds41p1::kTwinRow,
              humanb(ds41aux::ws((int)aux_cfg_.rows).bytes).c_str())
        : std::string();
    int64_t cached = 0, packed = 0, skipped = 0;
    double share = 0;
    std::string per_rank;
    for (int r = 0; r < R; ++r) {
        cached += st[(size_t)r].cached;
        packed += st[(size_t)r].packed;
        skipped += st[(size_t)r].skipped;
        share += st[(size_t)r].share;
        per_rank += fmt("%srank %d %lld", r ? ", " : "", r, (long long)st[(size_t)r].loaded);
    }
    if (released)
        RAD_INFO("%s%s; nothing loaded on device %d (%lld slots, %zu candidate units, profile share %.4f); "
                 "passes of up to %d rows, policy %s, deadline %lld us, backoff %lld, headroom %lld MiB, cap %lld MiB, "
                 "forced lateness %lld (0 off)%s",
                 ds41aux::kLogReleased, rel_why.c_str(), aux_->device(), (long long)N, all.size(), share_all,
                 (int)aux_cfg_.rows, aux_cfg_.verify ? "verify" : "serve", (long long)aux_cfg_.deadline_us,
                 (long long)aux_cfg_.backoff, (long long)aux_cfg_.headroom_mib, (long long)aux_cfg_.cap_mib,
                 (long long)aux_cfg_.force_late, tall.c_str());
    else
    RAD_INFO("%s%lld units (%s) resident on device %d: %s; profile share %.4f of the ranks' %.4f over %zu candidate "
             "units; loaded in %.1f s (%lld from the packed RAM cache, %lld packed from the container, %lld skipped), "
             "read back identical; warm-up rows equal the CPU pool's; passes of up to %d rows, policy %s, deadline "
             "%lld us, backoff %lld, headroom %lld MiB, cap %lld MiB, forced lateness %lld (0 off)%s",
             ds41aux::kLogOn, (long long)aux_->units(), humanb(aux_->units() * L.stride).c_str(), aux_->device(),
             per_rank.c_str(), share, share_all, all.size(), secs, (long long)cached, (long long)packed,
             (long long)skipped, (int)aux_cfg_.rows, aux_cfg_.verify ? "verify" : "serve",
             (long long)aux_cfg_.deadline_us, (long long)aux_cfg_.backoff, (long long)aux_cfg_.headroom_mib,
             (long long)aux_cfg_.cap_mib, (long long)aux_cfg_.force_late, tall.c_str());
    if (aux_ctl_) {
        for (const std::string& l : aux_ctl_->startup_lines()) RAD_INFO("%s", l.c_str());
        aux_ctl_->start();
    }
    return RAD_OK;
}

}  // namespace rad
