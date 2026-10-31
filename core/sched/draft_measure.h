/* Fixed-depth early-readback MEASUREMENT, no probability policy. Fixed storage and integer ns.
 * B_t is the host return from step_end_ before issue_ahead; B_t..B_(t+1) is tagged by the
 * earlier frame's seq/epoch/k and covers that verify plus the following query/observer/wait.
 */
#pragma once
#include <cstdint>
namespace rad {
/* The native DSpark chain's full5 or fixed-from1 shorter query, never a cap-sized query. */
inline bool draft_measure_profile(int64_t short_from, int64_t full_rows, int64_t depth,
                                  int64_t short_rows) noexcept {
    return full_rows == 5 && depth == 5 &&
           (short_from == 0 || (short_from == 1 && short_rows >= 2 && short_rows <= 4));
}
struct DraftMeasureFrame {
    uint64_t seq = 0, epoch = 0, ready_ns = 0, wait_ns = 0, output = 0;
    int step = -1, card = -1, k = -1, query_rows = 0;
    int32_t ctx = -1, anchor = -1, ids[5] = {};
    bool pure_decode = false;
};
struct DraftMeasureStats {
    uint64_t n[3] = {}, ns[3] = {}, wait_ns[3] = {}, accepted[3] = {}, emitted[3] = {};
    uint64_t queries = 0, query_rows = 0, waits = 0, wait_all_ns = 0;
    uint64_t transition = 0, no_query = 0, end_cancel = 0, unpublished = 0;
};
struct DraftMeasure {
    DraftMeasureFrame pending{}, current{}, closing{};
    DraftMeasureStats stats{};
    uint64_t closing_ns = 0;
    bool close = false;
    void no_query() {
        if (pending.epoch) ++stats.no_query;
        pending = {}; current = {}; closing = {}; close = false;
    }
    void ready(const DraftMeasureFrame& frame, bool previous_matches) {
        current = frame;
        ++stats.queries; stats.query_rows += frame.query_rows;
        ++stats.waits; stats.wait_all_ns += frame.wait_ns;
        close = false;
        if (pending.epoch) {
            if (!previous_matches || !pending.pure_decode || !frame.pure_decode ||
                pending.seq != frame.seq || pending.card != frame.card || pending.k != frame.k ||
                frame.epoch <= pending.epoch || pending.query_rows != frame.query_rows ||
                frame.ready_ns <= pending.ready_ns) ++stats.transition;
            else { closing = pending; closing_ns = frame.ready_ns - pending.ready_ns; close = true; }
        }
        pending = {};
    }
    /* Called only after the SAME step committed and its new prefix was published. Terminal,
     * cancelled, capped/mismatched or otherwise unpublished cycles are excluded, never C0. */
    bool committed(bool live, bool published, uint64_t output, int accepted) {
        bool counted = false;
        if (close) {
            if (!live) ++stats.end_cancel;
            else if (!published) ++stats.unpublished;
            else if (closing.k < 0 || closing.k > 2 || accepted < 0 || accepted > closing.k ||
                     output < closing.output || output - closing.output != (uint64_t)(1 + accepted))
                ++stats.transition;
            else {
                const int k = closing.k;
                ++stats.n[k]; stats.ns[k] += closing_ns; stats.wait_ns[k] += current.wait_ns;
                stats.accepted[k] += (uint64_t)accepted; stats.emitted[k] += output - closing.output;
                counted = true;
            }
        }
        if (live && published) { current.output = output; pending = current; }
        current = {}; closing = {}; close = false;
        return counted;
    }
};
}
