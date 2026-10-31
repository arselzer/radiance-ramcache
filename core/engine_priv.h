/* engine_priv.h -- shared by the Engine's translation units only.
 *
 * `Engine` is one class split across several files by phase, which share little: bring-up runs
 * once and touches every subsystem, the step loop runs constantly and touches a few, the drafters
 * are extra passes, and the probe is debug-only. The public surface in engine.h is unchanged.
 *
 * `Engine::Rank` is defined here, not in engine.h, because it names every subsystem's concrete
 * type; engine.h is included by main.cpp and the tools, which then need only a forward
 * declaration.
 */
#pragma once
#include "engine.h"

#include "mem/pools.h"
#include "mem/kv.h"
#include "mem/prefix.h"
#include "mem/kvtier.h"
#include "runtime/tierexec.h"
#include "build/rad_build.h"
#include "lora/declare.h"
#include "place/heat.h"
#include "place/planner.h"
#include "place/mover.h"
#include "place/draft_bank_backup.h"
#include "place/stager.h"
#include "place/xstats.h"
#include "runtime/ctx.h"
#include "sample/sampler.h"
#include "sched/advance.h"

namespace rad {

/* One bounded transaction, owned solely by the scheduler thread. No worker may publish a
 * loan. Backup accounts this object's full size, including all target/slot metadata. */
struct Engine::DraftBankLoan {
    static constexpr int kSlots = kDraftStages * kDraftRankExperts;
    explicit DraftBankLoan(draft_bank::Config c): controller(c) {}
    draft_bank::Controller controller;
    DraftBankBackup backup;
    std::array<std::array<int32_t, kSlots>, kDraftRanks> target{};
    std::array<int, kDraftRanks> count{};
    std::array<uint64_t,kDraftRanks> selected_at_lend{};
    // Sequential source extent reader; never overlaps relay/readback or another rank's fill.
    Mover::HostPlanes source;
    // Two independent 2MiB-rounded, 2MiB-aligned source buffers plus conservative
    // alignment allocator overhead. Native plane sizes/4KiB extent rounding fit 18MiB.
    static constexpr int64_t kSourcePayload = 18ll << 20;
    static constexpr int64_t kSourceBound = kSourcePayload + (4ll << 20);
    bool published = false, resume_pending = false;
    uint64_t loan_epoch = 0, resume_epoch = 0, query_seq = 0;
    int64_t query_step = -1;
    uint64_t lends = 0, restores = 0, rollbacks = 0, fill_bytes = 0;
    uint64_t source_drive_bytes = 0, source_ram_bytes = 0, relay_bytes = 0;
    uint64_t startup_ns = 0, drain_ns = 0, fill_ns = 0, restore_ns = 0;
};

/* ================================================================== per-rank state */
/* One tensor-parallel rank's state; see Engine::ranks_. */
struct Engine::Rank {
    int       index = 0;
    int       device = 0;
    Program   program;          /* declare runs once per rank: dimensions are already divided */
    /* Runtime LoRA: this rank's sites (core/lora/lora.h), the loader's map from an adapter's
     * modules to this rank's slot bytes. Empty with LoRA off. */
    std::vector<lora::Site> lora_sites;
    Plan      plan;
    Pools     pools;
    Mover     mover;
    /* The policy half of tiered placement, bound to the plan at bring-up and off unless the plan
     * is dynamic. `heat_seen[l]` is the last step whose layer-l histogram was credited: a
     * histogram stays in its bank until the next report overwrites it, and must not be counted
     * twice. */
    HeatEngine       heat;
    std::vector<int> heat_seen;
    /* One sweep's histograms, found before any is credited; kept so a sweep allocates nothing. */
    struct HeatRow { int layer; int bank; const int32_t* h; };
    std::vector<HeatRow> heat_rows;
    /* Credited histograms, and layer sweeps that found nothing new. A histogram is skipped
     * either because its copy has not landed (the host ran ahead and the sample was lost) or
     * because its bank holds an already-credited step (the layer did not route). Counting these
     * apart keeps a placement engine starved of evidence from looking like idle layers. */
    uint64_t heat_credited = 0, heat_idle = 0;
    /* Trunk steps in a row whose pass the prefill stager did not arm (Engine::tiering_dispatch). */
    int unstaged_steps = 0;
    KVManager kv;
    PrefixCache prefix;
    /* Idle session tiers (spec 7.3). Rank 0 owns the policy and session index, since a
     * conversation is not sharded; the transfer is per rank, since each holds its own shard of
     * every KV block. Both are inert unless --prefix-cache-mib sizes the host tier. */
    IdleTiers tiers;
    TierExec  tierx;
    /* The activation arena's smaller levels, smallest step first (Engine::probe_arena_levels). */
    std::vector<ArenaLevel> arena_levels;
    /* Per Ctx level (0 is the full plan): the bytes at the top of this rank's arena that level
     * leaves to the expert slab. */
    std::vector<int64_t>    arena_lent;
    /* Where the prefill stager's ring is at the span's shared level, from the arena's base; -1
     * without that level. */
    int64_t                 arena_low_off = -1;
    /* Prefill stager (core/place/stager.h): copies the next routed layer's pooled experts into the
     * top of this rank's arena while a full-plan step runs the layer before. */
    PrefillStager stager;
    /* File stager (core/place/stager.h): reads a routed layer's file-tier experts from the
     * container into one of the mover's two file buffers before the layer's first expert op, on
     * every pass. With both stagers in use, the chain hands each the passes it armed for. */
    FileStager  fstager;
    StagerChain stagers;
    Ctx       ctx;
    Sampler   sampler;          /* one per rank: each samples its own vocab shard (spec §9) */
    /* Scored rows' logits (Request::token_scores), this rank's vocabulary shard, copied into pinned
     * memory ahead of the sampler. Grown to the most rows a step has scored. */
    float*    score_host = nullptr;
    int64_t   score_rows_cap = 0;
    /* RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH (engine_hash.cpp): this rank's lines for the step and
     * each prompt's digest, and a pinned buffer one KV layer's pool is copied into. */
    std::vector<std::string> hash_lines;
    std::vector<uint64_t>    hash_digest;
    uint8_t*  hash_host = nullptr;
    int64_t   hash_cap = 0;
    rad_buf   tok_buf = 0;      /* where the chain writes the sampled ids */
    /* The card's step state (sched/advance.h): one StepSlot per scheduler slot, and the sorted
     * end-of-generation ids acceptance stops at. Every rank keeps its own and computes the same
     * values from the same sampled ids. */
    StepSlot* step_slots = nullptr;
    int32_t*  eog = nullptr;
    int32_t   n_eog = 0;
};

/* RADIANCE_EXPERIMENTAL_DS41_PREFILL_HASH (engine_hash.cpp): the prompts whose prefill the step
 * ends, and for each what its KV groups hold for it on rank 0's manager (every rank's tables are
 * rank 0's block ids), staged with the ranks parked. */
struct Engine::PrefillHashJob {
    struct Group {
        int32_t              g = 0;           /* handle */
        int                  kind = 0;        /* RAD_KV_* */
        int64_t              block = 0;       /* block_size */
        int64_t              window = 0;      /* a WINDOW group's positions */
        int64_t              first = 0;       /* the position the table starts at */
        int32_t              state = -1;      /* a stateful group's slot */
        std::vector<int32_t> table;
    };
    struct Seq {
        uint64_t           seq = 0;
        int64_t            n = 0;             /* positions the prompt wrote */
        int32_t            row = -1;          /* its logits row */
        int64_t            cut = 0;           /* StepEntry::consumer_end */
        std::vector<Group> groups;
    };
    int              step = -1;
    std::vector<Seq> seqs;
};

/* The tiers' work in flight (Engine::tier_tick): one copy batch per rank's transfer stream at a
 * time. Each rank's result is kept until all ranks have one, so a landed rank is not polled again
 * and a failure is not lost. */
struct TierPump {
    std::vector<TierJob> copy;
    std::vector<int>     copy_st;    /* per rank: 0 in flight, 1 landed, negative failed */
    int64_t              max_copy = 0;
    int                  slow_budget = 0;   /* capacity drops one plan may make */
};

/* The buffer probe (engine_probe.cpp): the buffers RADIANCE_DUMP_BUF names, at -vv only. The step
 * loop dumps after a trunk step, and each drafter after its own passes, since a drafter's buffers
 * still hold the previous step's values when the step loop dumps. */
void dump_buffers(Ctx& ctx, int step, const char* phase = "");

}  /* namespace rad */
