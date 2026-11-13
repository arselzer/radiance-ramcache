/* p1_hybrid.h -- hybrid card-path decode (RADIANCE_EXPERIMENTAL_DS41_P1_HYBRID=<pct>, ds41_p1.h kHybridEnv):
 * in a decode pass armed for the card path (..._P1_DECODE_ROWS), each routed layer gives some of its
 * RAM-cache units to the CPU pool, which computes them while the card streams the others through its ring.
 *
 * ---- why it is exact ----------------------------------------------------------------------------------
 *
 * A routed layer's down projection writes one row a sorted slot (expert e's slots are [off[e], off[e+1]),
 * the scatter's order); the gather reads those rows by slot and weighs them in its own fixed order. Which
 * side wrote a slot's row never reaches the gather: it reads bytes. The card path's rows are the P1 twin's,
 * held byte for byte to the decode band's at every M up to 4096 (device check `p1`); the CPU pool's are
 * ds41hx's, bit-exact with the decode band (host_experts_test; the CPU path of every pass below the
 * threshold). So every slot's bytes are the same whichever side computes its expert, and the gather's
 * sum is the base's. In a hybrid layer:
 *   - a CPU unit is published to the plugin's ops with kMarkSkip, as every non-resident unit of an armed
 *     layer is, and is in no wave, so no card kernel writes its slots (the twin's skip: no rows, no bytes;
 *     its zero-fill is [off[96], T), the other ranks' slots, never a local expert's);
 *   - its rows are the CPU path's own: hx_jobs_into takes expert e's live slots of [off[e], off[e+1]) from
 *     the sorted order fetched with the counts, refuses a layer whose count disagrees with the scatter's,
 *     and hx_capture refuses a gate_up that reads other rows than the ones the pool was given;
 *   - hx_finish copies them over the down projection's rows at the gather's operands, on the compute
 *     stream behind the layer's waves (after_op runs them right behind the down projection, before any
 *     later op), exactly as the CPU path copies its rows before the gather today.
 * Residency: a CPU unit is borrowed (Mover::borrow_host_unit) with the same LRU touch the ring's add_ram
 * gives it, in the same order (the layer's selected units in order), and a borrow that fails sends the
 * unit to the ring as before; so which units the RAM cache holds evolves as without the switch, except
 * that background inserts that wait for an idle pool (HostExpertPool::wait_idle) may land later.
 *
 * ---- the split ----------------------------------------------------------------------------------------
 *
 * choose(): of a layer's off-card selected units (its ring units without the switch), only RAM-cache hits
 * may go to the CPU (a drive miss stays on the card: its planes are read and packed for the ring). They
 * are taken fewest rows first (the CPU's cost grows with rows, the card's link time does not), while
 *     (pool backlog + CPU cost of the taken) / cpu rate  <=  share * (bytes left to the card) / link rate,
 * with the pool's own cost model (HostExpertPool::cost_rows; backlog over every rank, reservations
 * included, so ranks deciding at once see each other). The first unit that does not fit stops the scan:
 * later ones have as many rows or more.
 *
 * Share: the switch's percent, adapted per rank by Share::layer() from each hybrid layer's outcome at the
 * gather: the CPU done before the issuing thread got there (early: grow by 5% up to the switch's value),
 * the thread waited but the card was still on the layer's waves (hold), or the card had finished them and
 * waited for the CPU's rows (late: shrink by 20%, down to 1%). ..._P1_HYBRID_ADAPT=0 holds it fixed.
 */
#pragma once
#include "host_expert_pool.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace rad {
namespace p1hyb {

/* One off-card selected unit of the layer: its rows this pass, its bytes over the link, and whether the
 * RAM cache holds it (only then may the CPU take it). */
struct Cand {
    int     rows = 0;
    int64_t bytes = 0;
    bool    cached = false;
};

struct Split {
    int     n_cpu = 0;
    int64_t cpu_cost = 0;     /* the pool's cost of the taken units (HostExpertPool::cost_rows) */
    int64_t card_bytes = 0;   /* bytes left to the card's ring */
    double  cpu_ns = 0;       /* predicted: (backlog + cpu_cost) / cpu rate */
    double  card_ns = 0;      /* predicted: card_bytes / link rate */
};

/* to_cpu[i] (n entries) set for the units the CPU takes. cpu_bpns / link_bpns: bytes a ns
 * (HostExpertPool::cpu_bytes_per_ns, link_bytes_per_ns); share > 0. `order` is scratch. */
inline Split choose(const Cand* c, int n, int64_t backlog, double cpu_bpns, double link_bpns, double share,
                    uint8_t* to_cpu, std::vector<int>* order) {
    Split s;
    for (int i = 0; i < n; ++i) {
        to_cpu[i] = 0;
        s.card_bytes += c[i].bytes;
    }
    s.card_ns = link_bpns > 0 ? (double)s.card_bytes / link_bpns : 0;
    s.cpu_ns = cpu_bpns > 0 ? (double)std::max<int64_t>(backlog, 0) / cpu_bpns : 0;
    if (n <= 0 || !(share > 0) || !(cpu_bpns > 0) || !(link_bpns > 0)) return s;
    order->clear();
    for (int i = 0; i < n; ++i)
        if (c[i].cached && c[i].rows > 0 && c[i].bytes > 0) order->push_back(i);
    std::stable_sort(order->begin(), order->end(), [&](int a, int b) { return c[a].rows < c[b].rows; });
    const double base = (double)std::max<int64_t>(backlog, 0);
    for (int i : *order) {
        const int64_t cost = s.cpu_cost + HostExpertPool::cost_rows(c[i].rows);
        const int64_t left = s.card_bytes - c[i].bytes;
        const double cpu = (base + (double)cost) / cpu_bpns, card = (double)left / link_bpns;
        if (cpu > share * card) break;
        to_cpu[i] = 1;
        ++s.n_cpu;
        s.cpu_cost = cost;
        s.card_bytes = left;
        s.cpu_ns = cpu;
        s.card_ns = card;
    }
    return s;
}

/* The share a rank runs with (see the header). */
class Share {
public:
    static constexpr double kMin = 0.01, kGrow = 1.05, kShrink = 0.8;
    void init(int pct, bool adapt) {
        max_ = pct / 100.0;
        cur_ = max_;
        min_seen_ = max_;
        adapt_ = adapt;
    }
    double share() const { return cur_; }
    double min_seen() const { return min_seen_; }
    /* A hybrid layer's outcome at the gather: `early` the CPU was done before the issuing thread came,
     * `late` the card had run the layer's waves and waited for the CPU. */
    void layer(bool early, bool late) {
        if (!adapt_) return;
        if (late) cur_ = std::max(kMin, cur_ * kShrink);
        else if (early) cur_ = std::min(max_, cur_ * kGrow);
        min_seen_ = std::min(min_seen_, cur_);
    }

private:
    double max_ = 0, cur_ = 0, min_seen_ = 0;
    bool   adapt_ = true;
};

}  /* namespace p1hyb */
}  /* namespace rad */
