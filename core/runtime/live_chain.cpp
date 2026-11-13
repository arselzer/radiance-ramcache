/* live_chain.cpp -- the live issue path's overlap bookkeeping, trace and summary (live_chain.h).
 * The per-launch glue (operand ranges, the window around the launch) is Ctx::live_before /
 * live_after in issue.cpp. */
#include "live_chain.h"
#include "../rad_core.h"
#include "../rad_internal.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace rad {

/* ================================================================== the hazard */

static bool meets(const LiveRange* a, int na, const LiveRange* b, int nb) {
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j)
            if (a[i].lo < b[j].hi && b[j].lo < a[i].hi) return true;
    return false;
}

int live_dep(const LiveOp& b, const LiveOp& a) {
    if (a.opaque || b.opaque) return kLiveUnknown;
    int d = 0;
    if (meets(b.rd, b.n_rd, a.wr, a.n_wr)) d |= kLiveRaw;
    if (meets(b.wr, b.n_wr, a.rd, a.n_rd)) d |= kLiveWar;
    if (meets(b.wr, b.n_wr, a.wr, a.n_wr)) d |= kLiveWaw;
    return d;
}

const char* live_why_name(int why) {
    switch (why) {
    case kLiveFree:    return "free";
    case kLiveForeign: return "foreign";
    case kLiveOpaque:  return "opaque";
    case kLiveHost:    return "host";
    case kLiveAfter:   return "after";
    case kLiveCap:     return "cap";
    case kLiveClash:   return "clash";
    default:           return "?";
    }
}

/* ================================================================== the queues */

int LiveQueues::claim() {
    const int s = head_;
    head_ = (head_ + 1) % kRing;
    return s;
}

const LiveQueues::Q* LiveQueues::find(uint32_t id) const {
    for (const Q& q : qs_)
        if (q.id == id) return &q;
    return nullptr;
}

LiveQueues::Q& LiveQueues::get(uint32_t id) {
    for (Q& q : qs_)
        if (q.id == id) return q;
    qs_.emplace_back();
    qs_.back().id = id;
    qs_.back().infl.reserve(kTapeOverlapRun + 1);
    return qs_.back();
}

LiveWhy LiveQueues::check(int s, const Pos& at, uint64_t* clash, int64_t* foreign) const {
    *clash = 0;
    *foreign = -1;
    const LiveOp& op = ring_[(size_t)s];
    const Q* q = find(at.queue);
    if (!q || !q->known) return kLiveForeign;
    *foreign = (int64_t)(at.wr - q->end);
    /* Both say the same when only this thread writes the queue; either alone catches a packet the
     * issue path did not write. */
    if (at.wr != q->end || at.last_tag != q->last_tag) return kLiveForeign;
    if (op.opaque) return kLiveOpaque;
    if (op.host) return kLiveHost;
    if (q->blind) return kLiveAfter;
    if (q->run >= kTapeOverlapRun) return kLiveCap;
    for (const Inflight& f : q->infl) {
        const LiveOp& o = ring_[(size_t)f.slot];
        /* Its slot was reused: the launch is no longer known, so it could touch anything. */
        if (o.tag != f.tag) return kLiveAfter;
        /* Disjoint operands, but the same op: the row's instance state is shared (LiveOp::op). */
        if (live_dep(op, o) || (op.op != 0 && op.op == o.op)) {
            *clash = o.tag;
            return kLiveClash;
        }
    }
    return kLiveFree;
}

void LiveQueues::commit(int s, const Pos& before, const Win& w, const Pos& after) {
    if (w.packets <= 0) return;   /* nothing on this queue: its in-flight set is unchanged */
    const LiveOp& op = ring_[(size_t)s];
    Q& q = get(before.queue);
    /* Whether the in-flight set the launch was checked against is the one it joined. */
    const bool was_known = q.known && before.wr == q.end && before.last_tag == q.last_tag;
    if (after.queue != before.queue || after.last_tag != op.tag) {
        /* Something followed the launch's packets on the queue: start over at the next one. */
        q.known = false;
        q.blind = true;
        q.run = 0;
        q.infl.clear();
        return;
    }
    q.known = true;
    q.end = after.wr;
    q.last_tag = op.tag;
    if (w.flags & LIVE_BARRIER) {
        /* A barrier packet inside the launch: not a kernel the ranges describe. */
        q.blind = true;
        q.run = 0;
        q.infl.clear();
        return;
    }
    if ((w.flags & LIVE_FIRST_UNORDERED) && w.packets == 1) {
        q.infl.push_back(Inflight{ s, op.tag });
        ++q.run;
        /* Never the case with the permits this path gives (the check needs a known queue), but a
         * packet the backend left unordered behind an unknown set leaves the set unknown. */
        if (!was_known) q.blind = true;
        return;
    }
    /* Its last packet had the barrier bit: everything before it on the queue has finished. */
    q.infl.clear();
    q.infl.push_back(Inflight{ s, op.tag });
    q.run = 0;
    q.blind = op.opaque || op.host;
}

int LiveQueues::in_flight(uint32_t queue) const {
    const Q* q = find(queue);
    return q && q->known ? (int)q->infl.size() : 0;
}

/* ================================================================== the run phase's side */

LiveChain::~LiveChain() {
    if (ops_out_) std::fclose(ops_out_);
}

void LiveChain::init(int rank, const Program* prog) {
    rank_ = rank;
    prog_ = prog;
    mode_ = every_ = pf_every_ = 0;
    pick_ = LiveTracePick();
    if (const char* m = std::getenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP"); m && *m)
        mode_ = std::clamp(std::atoi(m), 0, 2);
    if (const char* t = std::getenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE"); t && *t)
        every_ = std::max(0, std::atoi(t));
    if (const char* t = std::getenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_PREFILL"); t && *t)
        pf_every_ = std::max(0, std::atoi(t));
    pick_ = LiveTracePick(every_, pf_every_);
    if (!on()) return;
    if (!live_supported() || rad_dev_chain_enabled()) {
        if (rank_ == 0)
            RAD_WARN("live chain: %s; RADIANCE_EXPERIMENTAL_DS41_CHAIN_OVERLAP / _TRACE / "
                     "_TRACE_PREFILL are off",
                     !live_supported() ? "the device backend has no live issue path"
                                       : "launches chain (rad_dev_chain_enabled), and a chained "
                                         "kernel's wait counts chained blocks only");
        mode_ = every_ = pf_every_ = 0;
        pick_ = LiveTracePick();
        return;
    }
    const char* o = std::getenv("RADIANCE_EXPERIMENTAL_DS41_CHAIN_TRACE_OUT");
    prefix_ = o && *o ? std::string(o) : fmt("/tmp/radiance-chain-%d", (int)getpid());
    /* Unique per Ctx in the process, never 0 (the backend's "untracked"): two contexts sharing a
     * stream (CtxDesc::stream) must not mistake each other's packets for their own. */
    static std::atomic<uint64_t> contexts{0};
    tag_base_ = (contexts.fetch_add(1, std::memory_order_relaxed) + 1) << 40;
    tag_n_ = 0;
    q_.reset();
    if (pick_.on() && rank_ == 0) rows_.resize(2048);
    if (rank_ == 0) {
        const char* what = mode_ == 1   ? "independent dispatches overlap in decode passes"
                           : mode_ == 2 ? "independent dispatches overlap in every pass issued live"
                                        : "overlap off, the check only counts";
        std::string trace =
            every_ > 0 ? fmt("; one trunk decode pass in %d on rank 0 traced", every_) : std::string();
        if (pf_every_ > 0)
            trace += fmt("; one armed card-path prefill pass in %d on rank 0 traced", pf_every_);
        if (pick_.on())
            trace += fmt(" into %s.ops.csv and %s.stamps.csv", prefix_.c_str(), prefix_.c_str());
        RAD_INFO("live chain: %s%s", what, trace.c_str());
    }
}

/* ================================================================== which passes are traced */

const char* live_trace_kind_name(int kind) {
    return kind == kLiveTraceDecode ? "decode" : kind == kLiveTracePrefill ? "prefill" : "none";
}

int LiveTracePick::candidate(const RadBatch* b, int rank, bool card_pass) {
    if (!b || rank != 0 || b->draft_pass != 0) return kLiveTraceNone;
    if (b->phase == RAD_PHASE_DECODE) return kLiveTraceDecode;
    return card_pass ? kLiveTracePrefill : kLiveTraceNone;
}

int LiveTracePick::next(const RadBatch* b, int rank, bool card_pass) {
    const int c = candidate(b, rank, card_pass);
    if (c == kLiveTraceDecode && every_ > 0 && ++decode_seen_ % (uint64_t)every_ == 0) return c;
    if (c == kLiveTracePrefill && pf_every_ > 0 && ++pf_seen_ % (uint64_t)pf_every_ == 0) return c;
    return kLiveTraceNone;
}

void LiveChain::pass_begin(const RadBatch* b, bool card_pass) {
    /* Chaining is decided before any Ctx exists (engine bring-up); checked again per pass anyway,
     * since a chained kernel's own missing bit would read here as one this path cleared. */
    overlap_pass_ = traced_ = false;
    in_pass_ = on() && !rad_dev_chain_enabled();
    if (!in_pass_) return;
    const bool decode = b && b->phase == RAD_PHASE_DECODE;
    ++st_.passes;
    if (decode) ++st_.decode_passes;
    overlap_pass_ = mode_ == 2 || (mode_ == 1 && decode);
    if (overlap_pass_) ++st_.overlap_passes;
    traced_ = false;
    pass_kind_ = pick_.next(b, rank_, card_pass);
    if (pass_kind_ != kLiveTraceNone) {
        traced_ = true;
        pass_rows_ = b->n_tok;
        ++trace_id_;
        ++st_.traced;
        if (pass_kind_ == kLiveTracePrefill) ++st_.traced_prefill;
        n_rows_ = 0;
        pass_tag0_ = tag_base_ | (tag_n_ + 1);
        live_stamp(true, trace_id_, (prefix_ + ".stamps.csv").c_str());
    }
}

void LiveChain::pass_end() {
    if (!in_pass_) return;
    in_pass_ = false;
    if (traced_) {
        live_stamp(false, trace_id_, nullptr);
        write_trace();
        traced_ = false;
    }
}

LiveTraceRow& LiveChain::trace_row() {
    if (n_rows_ == rows_.size()) rows_.resize(rows_.size() * 2 + 64);
    LiveTraceRow& r = rows_[n_rows_];
    r = LiveTraceRow{};
    r.seq = (uint32_t)n_rows_++;
    return r;
}

/* "blk.<n>." in a weight's name, or -1. */
static int layer_of(const std::string& name) {
    const size_t p = name.find("blk.");
    if (p == std::string::npos) return -1;
    return std::atoi(name.c_str() + p + 4);
}

static void csv_field(std::string* out, const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) {
        *out += s;
        return;
    }
    *out += '"';
    for (char c : s) {
        if (c == '"') *out += '"';
        *out += c;
    }
    *out += '"';
}

/* One traced pass, written after it was issued, so the formatting costs the trace nothing in the
 * timed launches. */
void LiveChain::write_trace() {
    if (ops_failed_) return;
    if (!ops_out_) {
        const std::string path = prefix_ + ".ops.csv";
        ops_out_ = std::fopen(path.c_str(), "a");
        if (!ops_out_) {
            ops_failed_ = true;
            RAD_WARN("live chain: %s cannot be opened (%s); the trace is dropped", path.c_str(),
                     std::strerror(errno));
            return;
        }
        std::fseek(ops_out_, 0, SEEK_END);
        if (std::ftell(ops_out_) == 0)
            std::fputs("pass,seq,layer,op,label,kernel,lane,queue,idx,packets,unordered,would,why,"
                       "clash,foreign,deps,reads,writes,kind,rows\n", ops_out_);
    }
    const Program& P = *prog_;
    std::string line;
    /* A launch's layer is its first weight's; one with no weight takes the last one's on its lane,
     * so the next layer's weights read ahead on lane 1 (DS41's pf_attn_tail) do not pull this
     * layer's later launches into the next one. */
    int lane_layer[2] = { -1, -1 };
    /* How far back a dependence is looked for: a layer is ~40 launches, two lanes' worth. */
    constexpr size_t kBack = 96;
    for (size_t j = 0; j < n_rows_; ++j) {
        const LiveTraceRow& r = rows_[j];
        int& layer = lane_layer[r.lane ? 1 : 0];
        std::string label;
        if (r.info && !r.info->weights.empty() && (size_t)r.info->weights[0] < P.weights.size()) {
            label = P.weights[(size_t)r.info->weights[0]].name;
            const int l = layer_of(label);
            if (l >= 0) layer = l;
        }
        std::string reads, writes;
        auto add = [](std::string* to, const std::string& n) {
            if (!to->empty()) *to += '|';
            *to += n;
        };
        for (int i = 0; i < r.n_opd; ++i) {
            const LiveOpd& o = r.opd[i];
            std::string n;
            switch (o.kind) {
            case RAD_OPK_BUF:
                n = o.handle < P.buffers.size() ? P.buffers[o.handle].name : fmt("buf%u", o.handle);
                break;
            case RAD_OPK_KV:
                n = fmt("kv:%s@%lld", o.handle < P.kv_groups.size() ? P.kv_groups[o.handle].name.c_str() : "?",
                        (long long)o.offset);
                break;
            case RAD_OPK_RAW:    n = "raw"; break;
            case RAD_OPK_WTAB:   n = "wtab"; break;
            default:             continue;   /* weights name the op (label); absent operands */
            }
            if (o.role == RAD_OPD_IN || o.role == RAD_OPD_INOUT || o.role == RAD_OPD_WTAB ||
                o.role == RAD_OPD_WEIGHT)
                add(&reads, n);
            if (o.role == RAD_OPD_OUT || o.role == RAD_OPD_INOUT) add(&writes, n);
        }
        if (r.scratch) {
            add(&reads, "scratch");
            add(&writes, "scratch");
        }
        if (label.empty()) label = writes.empty() ? "-" : writes.substr(0, writes.find('|'));
        std::string deps;
        if (r.ranges.opaque) {
            deps = "opaque";
        } else {
            for (size_t i = j > kBack ? j - kBack : 0; i < j; ++i) {
                const int d = live_dep(r.ranges, rows_[i].ranges);
                if (!d) continue;
                if (!deps.empty()) deps += ' ';
                deps += fmt("%u%s%s%s%s", rows_[i].seq, d & kLiveRaw ? "r" : "", d & kLiveWar ? "a" : "",
                            d & kLiveWaw ? "w" : "", d & kLiveUnknown ? "?" : "");
            }
        }
        line.clear();
        line += fmt("%" PRIu64 ",%u,%d,", trace_id_, r.seq, layer);
        csv_field(&line, r.info ? r.info->op : std::string("?"));
        line += ',';
        csv_field(&line, label);
        line += ',';
        csv_field(&line, r.row && r.row->info && r.row->info->name ? r.row->info->name : "?");
        line += fmt(",%u,%u,%" PRIu64 ",%d,%d,%d,%s,%lld,%lld,", (unsigned)r.lane, r.queue, r.first,
                    r.packets, (int)r.unordered, (int)r.would, live_why_name(r.why),
                    (long long)r.clash, (long long)r.foreign);
        csv_field(&line, deps);
        line += ',';
        csv_field(&line, reads);
        line += ',';
        csv_field(&line, writes);
        line += fmt(",%s,%lld\n", live_trace_kind_name(pass_kind_), (long long)pass_rows_);
        std::fputs(line.c_str(), ops_out_);
    }
    std::fflush(ops_out_);
}

std::string LiveChain::report() const {
    if (!on()) return std::string();
    const LiveStats& s = st_;
    auto pct = [](uint64_t a, uint64_t b) { return b ? 100.0 * (double)a / (double)b : 0.0; };
    std::string r = fmt(
        "live chain (%s): %llu passes issued live (%llu decode); %llu device launches checked, "
        "%llu with nothing in flight to wait for (%.1f%%), %llu first dispatches went unordered "
        "(%llu permits the backend kept ordered); the others kept the bit: %llu behind packets the "
        "issue path did not write, %llu behind an op they depend on, %llu opaque, %llu "
        "host-visible, %llu behind an opaque or unknown op, %llu at the run cap; %llu launches of "
        "several packets, %llu of none on their queue",
        mode_ == 1 ? "overlap in decode passes" : mode_ == 2 ? "overlap in every live pass" : "dump only",
        (unsigned long long)s.passes, (unsigned long long)s.decode_passes,
        (unsigned long long)s.launches, (unsigned long long)s.free, pct(s.free, s.launches),
        (unsigned long long)s.unordered, (unsigned long long)s.refused,
        (unsigned long long)s.why[kLiveForeign], (unsigned long long)s.why[kLiveClash],
        (unsigned long long)s.why[kLiveOpaque], (unsigned long long)s.why[kLiveHost],
        (unsigned long long)s.why[kLiveAfter], (unsigned long long)s.why[kLiveCap],
        (unsigned long long)s.multi, (unsigned long long)s.silent);
    if (s.decode_passes)
        r += fmt("; a decode pass: %.1f launches, %.1f free, %.1f unordered",
                 (double)s.decode_launches / (double)s.decode_passes,
                 (double)s.decode_free / (double)s.decode_passes,
                 (double)s.decode_unordered / (double)s.decode_passes);
    if (s.traced)
        r += fmt("; %llu passes traced (%llu prefill) into %s.ops.csv", (unsigned long long)s.traced,
                 (unsigned long long)s.traced_prefill, prefix_.c_str());
    return r;
}

}  /* namespace rad */
