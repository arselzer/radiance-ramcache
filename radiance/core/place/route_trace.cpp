/* route_trace.cpp -- see route_trace.h. */
#include "route_trace.h"

#include "../rad_internal.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>

namespace rad {

namespace rtrace {
float sqrt_softplus(float x) { return std::sqrt(x > 20.0f ? x : std::log1p(std::exp(x))); }

void encode_pass(std::vector<uint8_t>& b, const PassRec& p) {
    Rec r(b, kPass, 0);
    r.put(p.step);
    r.put(p.pass);
    r.put(p.phase);
    r.put(p.draft_pass);
    r.put(p.n_tok);
    r.put(p.n_seq);
    r.put(p.n_spec);
    r.put(p.n_tok_decode);
    r.put(p.max_ctx_len);
    r.put(uint32_t{ 0 });
    r.put(p.t_ns);
    r.finish();
}

void encode_layer(std::vector<uint8_t>& b, const LayerRec& L) {
    Rec r(b, kLayer, 0);
    r.put(L.pass);
    r.put(L.step);
    r.put(L.layer);
    r.put(L.rows);
    r.put((uint16_t)L.n_picks);
    r.put(L.flags);
    r.put(uint8_t{ 0 });
    r.put(L.us);
    r.raw(L.ram, sizeof L.ram);
    for (int i = 0; i < L.n_picks; ++i) {
        r.put(L.picks[i].expert);
        r.put(L.picks[i].rows);
        r.put(L.picks[i].status);
    }
    if (L.flags & kRoute)
        for (int row = 0; row < L.rows; ++row) {
            for (int k = 0; k < kTopK; ++k) r.put((uint16_t)L.ids[row * kTopK + k]);
            r.raw(L.weights + row * kTopK, kTopK * sizeof(float));
            r.raw(L.keys + row * kTopK, kTopK * sizeof(float));
        }
    if (L.flags & kLogits) r.raw(L.logits, (size_t)L.rows * kExperts * sizeof(float));
    r.finish();
}

void encode_bias(std::vector<uint8_t>& b, uint16_t layer, const float* bias) {
    Rec r(b, kBias, 0);
    r.put(layer);
    r.put((uint16_t)kExperts);
    r.raw(bias, kExperts * sizeof(float));
    r.finish();
}

void encode_tiers(std::vector<uint8_t>& b, const std::array<uint8_t, kLocal>* tier) {
    Rec r(b, kTiers, 0);
    r.put((uint16_t)kLayers);
    r.put((uint16_t)kLocal);
    for (int l = 0; l < kLayers; ++l) {
        r.put((uint16_t)l);
        r.put(uint16_t{ 0 });
        r.raw(tier[l].data(), kLocal);
    }
    r.finish();
}
}  /* namespace rtrace */

namespace {
uint64_t now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  /* namespace */

RouteTrace* RouteTrace::get() {
    /* Never destroyed: a rank thread's late record must not meet a freed object. close() runs at
     * the engine's shutdown and, for a process that leaves without one, at exit. */
    static RouteTrace* const t = []() -> RouteTrace* {
        const char* p = std::getenv("RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE");
        if (!p || !*p) return nullptr;
        RouteTrace* r = new RouteTrace(p);
        std::atexit([] { if (RouteTrace* x = RouteTrace::get()) x->close(); });
        RAD_INFO("route trace: every EP-routed layer's picks, weights and residency to "
                 "%s.rank<r>.bin, the step loop's requests to %s.steps.bin "
                 "(RADIANCE_EXPERIMENTAL_DS41_ROUTE_TRACE)", p, p);
        return r;
    }();
    return t;
}

RouteTrace::RouteTrace(std::string prefix) : prefix_(std::move(prefix)), t0_ns_(now_ns()) {
    writer_ = std::thread(&RouteTrace::loop, this);
}

RouteTrace::~RouteTrace() { close(); }

/* Under st.mu. The file, and its header as the buffer's first bytes, so the writer thread is the
 * only one that writes the file. */
bool RouteTrace::open_locked(int s, Stream& st) {
    if (st.tried) return st.fp != nullptr;
    st.tried = true;
    const std::string path = prefix_ + (s == kSteps ? std::string(".steps.bin")
                                                    : ".rank" + std::to_string(s) + ".bin");
    st.fp = std::fopen(path.c_str(), "wb");
    if (!st.fp) {
        RAD_WARN("route trace: cannot open %s; its records are dropped", path.c_str());
        open_failed_.store(true, std::memory_order_relaxed);
        return false;
    }
    uint8_t h[rtrace::kHeaderBytes] = {};
    const uint32_t f[8] = { rtrace::kVersion, (uint32_t)s, (uint32_t)rtrace::kWorld,
                            (uint32_t)rtrace::kExperts, (uint32_t)rtrace::kTopK,
                            (uint32_t)rtrace::kLocal, (uint32_t)rtrace::kLayers,
                            rtrace::kHeaderBytes };
    std::memcpy(h, rtrace::kMagic, 8);
    std::memcpy(h + 8, f, sizeof f);
    std::memcpy(h + 8 + sizeof f, &t0_ns_, 8);
    st.cur.insert(st.cur.end(), h, h + sizeof h);
    return true;
}

/* The issuing threads' only cost: a copy into the stream's buffer under a lock no other thread
 * takes except at close (each stream has one producer), and at every kHandoff bytes a queue push. */
void RouteTrace::put(int s, const void* rec, size_t n) {
    if (s < 0 || s >= kStreams || !rec || !n) return;
    Stream& st = st_[(size_t)s];
    std::vector<uint8_t> full;
    {
        std::lock_guard<std::mutex> lk(st.mu);
        if (closed_.load(std::memory_order_acquire) || !open_locked(s, st)) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (st.cur.capacity() < kHandoff) {
            /* A buffer the writer has written, else a new one. */
            {
                std::lock_guard<std::mutex> q(qmu_);
                if (!free_.empty()) {
                    std::vector<uint8_t> b = std::move(free_.back());
                    free_.pop_back();
                    b.insert(b.end(), st.cur.begin(), st.cur.end());
                    st.cur.swap(b);
                }
            }
            if (st.cur.capacity() < kHandoff) st.cur.reserve(kHandoff + (64 << 10));
        }
        const uint8_t* p = (const uint8_t*)rec;
        st.cur.insert(st.cur.end(), p, p + n);
        ++st.records;
        st.bytes += n;
        if (st.cur.size() >= kHandoff) full.swap(st.cur);
    }
    if (!full.empty()) handoff(s, full);
}

/* A full buffer to the writer; past the backlog cap it is dropped (counted). */
void RouteTrace::handoff(int s, std::vector<uint8_t>& b) {
    {
        std::lock_guard<std::mutex> lk(qmu_);
        if (backlog_ + b.size() > kBacklogCap) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        backlog_ += b.size();
        max_backlog_ = std::max<uint64_t>(max_backlog_, backlog_);
        q_.push_back(Chunk{ s, std::move(b) });
    }
    qcv_.notify_one();
}

void RouteTrace::loop() {
    /* Below the rank, reader and pool threads: a file write may wait, they may not. */
    (void)setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10);
    std::unique_lock<std::mutex> lk(qmu_);
    for (;;) {
        qcv_.wait(lk, [&] { return stop_ || !q_.empty(); });
        if (q_.empty()) break;   /* stopping, and everything written */
        Chunk c = std::move(q_.front());
        q_.pop_front();
        lk.unlock();
        /* The stream's file was opened (under its lock) before its first buffer was queued. */
        FILE* fp = st_[(size_t)c.s].fp;
        const uint64_t t0 = now_ns();
        const bool ok = fp && std::fwrite(c.b.data(), 1, c.b.size(), fp) == c.b.size();
        const uint64_t dt = now_ns() - t0;
        lk.lock();
        backlog_ -= c.b.size();
        ++writes_;
        write_ns_ += dt;
        if (!ok) dropped_.fetch_add(1, std::memory_order_relaxed);
        c.b.clear();
        if (free_.size() < 8) free_.push_back(std::move(c.b));
    }
}

void RouteTrace::step(int32_t step, int phase, int draft_pass, int64_t n_tok, bool ahead,
                      const RtStepEntry* e, int n) {
    n = std::clamp(n, 0, 65535);
    std::vector<uint8_t> b;
    b.reserve(32 + 20 * (size_t)n);
    rtrace::Rec r(b, rtrace::kStep, 0);
    r.put(step);
    r.put((int16_t)phase);
    r.put((int16_t)draft_pass);
    r.put((int32_t)n_tok);
    r.put((uint16_t)n);
    r.put((uint8_t)(ahead ? 1 : 0));
    r.put(uint8_t{ 0 });
    r.put(now_ns());
    for (int i = 0; i < n; ++i) {
        r.put(e[i].seq);
        r.put(e[i].ctx_len);
        r.put(e[i].n_tokens);
        r.put(e[i].flags);
        r.put(e[i].n_spec);
        r.put(uint16_t{ 0 });
    }
    r.finish();
    put(kSteps, b.data(), b.size());
}

void RouteTrace::end(int32_t step, int32_t ahead_step, uint64_t seq, uint64_t n_out) {
    std::vector<uint8_t> b;
    rtrace::Rec r(b, rtrace::kEnd, 0);
    r.put(step);
    r.put(ahead_step);
    r.put(seq);
    r.put(n_out);
    r.finish();
    put(kSteps, b.data(), b.size());
}

void RouteTrace::close() {
    if (closed_.exchange(true, std::memory_order_acq_rel)) return;
    /* A put that read closed_ false finishes under its stream's lock before this takes it; every
     * later one is dropped. */
    for (int s = 0; s < kStreams; ++s) {
        std::vector<uint8_t> rest;
        {
            std::lock_guard<std::mutex> lk(st_[(size_t)s].mu);
            rest.swap(st_[(size_t)s].cur);
        }
        if (rest.empty()) continue;
        std::lock_guard<std::mutex> lk(qmu_);
        backlog_ += rest.size();
        max_backlog_ = std::max<uint64_t>(max_backlog_, backlog_);
        q_.push_back(Chunk{ s, std::move(rest) });
    }
    {
        std::lock_guard<std::mutex> lk(qmu_);
        stop_ = true;
    }
    qcv_.notify_all();
    if (writer_.joinable()) writer_.join();
    for (Stream& st : st_) {
        std::lock_guard<std::mutex> lk(st.mu);
        if (st.fp && std::fclose(st.fp) != 0) dropped_.fetch_add(1, std::memory_order_relaxed);
        st.fp = nullptr;
    }
}

RouteTrace::Stats RouteTrace::stats() const {
    Stats s;
    {
        std::lock_guard<std::mutex> lk(qmu_);
        s.writes = writes_;
        s.write_ns = write_ns_;
        s.max_backlog = max_backlog_;
    }
    s.dropped = dropped_.load(std::memory_order_relaxed);
    s.open_failed = open_failed_.load(std::memory_order_relaxed);
    for (int i = 0; i < kStreams; ++i) {
        const Stream& st = st_[(size_t)i];
        std::lock_guard<std::mutex> lk(st.mu);
        s.records[i] = st.records;
        s.bytes[i] = st.bytes;
    }
    return s;
}

std::string RouteTrace::report() const {
    const Stats s = stats();
    uint64_t bytes = 0;
    for (uint64_t b : s.bytes) bytes += b;
    return fmt("route trace: %s.rank<r>.bin and .steps.bin: records rank0..3 %llu/%llu/%llu/%llu, "
               "steps %llu; %.1f MB in %llu writes (%.1f ms writing, at most %.1f MB waiting); %llu "
               "records or buffers dropped%s",
               prefix_.c_str(), (unsigned long long)s.records[0], (unsigned long long)s.records[1],
               (unsigned long long)s.records[2], (unsigned long long)s.records[3],
               (unsigned long long)s.records[kSteps], (double)bytes / 1e6,
               (unsigned long long)s.writes, (double)s.write_ns / 1e6,
               (double)s.max_backlog / 1e6, (unsigned long long)s.dropped,
               s.open_failed ? "; a file could not be opened" : "");
}

}  /* namespace rad */
