/* plane_mirror.h -- a second copy of DeepSeek-V4.1's routed expert planes on another drive, read
 * beside the container so a missed expert arrives sooner.
 *
 * The checkpoint's safetensors hold the container's expert planes byte for byte: a gate_up
 * weight's codes are w1's then w3's, its scales s1's then s3's, a down weight's are w2's. When the
 * checkpoint sits on a different drive from the container, the CPU expert path's plane reads
 * (Mover::read_unit_planes) take the tail of each gate_up codes plane -- the end of w3 -- from the
 * mirror while the container's drive reads the rest, both at once. With the container on an NVMe
 * drive (~6.5 GB/s) and the checkpoint on a SATA stripe (~1.45 GB/s) an 18% share adds ~22% drive
 * bandwidth: a prefill chunk's many misses, bound by bandwidth, arrive sooner (TTFT -10%), while a
 * decode miss, bound by its slowest read's latency, does not (decode -4%), so by default only
 * passes of 2+ tokens use it.
 *
 * Reads only. The mirror is trusted no further than checked: its shards' size, modification and
 * change times are taken at open and compared around every read, every expert must appear once,
 * and Mover samples units of every rank against the container before it uses the mirror at all.
 * A read that fails turns the mirror off for the process; the reader takes those bytes from the
 * container instead.
 *
 * Off unless asked for: --expert-plane-mirror <directory of model-*.safetensors>,
 * --expert-plane-mirror-share <fraction of a unit's bytes> (0.18; at most the whole of w3, 0.31)
 * for passes of --expert-plane-mirror-min-rows <tokens> (2) or more, and
 * --expert-plane-mirror-decode-share (0, off) for smaller passes, reach it through configure()
 * (Config::expert_plane_mirror*). A test hook stays in the environment:
 * RADIANCE_EXPERIMENTAL_DS41_PLANE_MIRROR_FAULT=<n> fails the n-th read, to exercise the fallback
 * to the container. */
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

class PlaneMirror {
    friend struct PlaneMirrorTest;   /* tests/plane_mirror_test.cpp builds instances of its own */
public:
    /* The command line's choice (Config::expert_plane_mirror*). Set once at startup, before the
     * first get(); an empty dir, the default, means no mirror. */
    struct Options {
        std::string dir;
        double      share = 0.18;         /* passes of min_rows+ tokens (prefill chunks) */
        int         min_rows = 2;
        double      decode_share = 0.0;   /* smaller passes (decode); 0 reads the container only */
    };
    static void configure(const Options& o);
    /* The process's mirror, or null when no directory was configured or it cannot be read (warned
     * once). */
    static PlaneMirror* get();

    struct Ref { int shard = -1; uint64_t off = 0, bytes = 0; };   /* a tensor's data, absolute */
    /* The up projection's codes (w3) of global expert `expert` in layer `layer`. */
    bool up_codes(int layer, int expert, Ref* out) const;
    /* The `bytes` bytes of r from `from` on into dst (any alignment), O_DIRECT through this
     * thread's bounce buffer. RAD_OK, or a negative RAD_E_*; a shard whose size or mtime changed
     * since open fails. */
    int read(const Ref& r, uint64_t from, uint64_t bytes, uint8_t* dst);
    double share() const { return share_; }
    double decode_share() const { return decode_share_; }
    /* Passes of fewer tokens take decode_share(), by default none: a decode miss waits on its
     * slowest read, and the SATA stripe's reads, alone or queued behind other ranks', come back
     * later than the NVMe drive's larger share (measured: decode -4% at an 18% share, +~3% at 10%;
     * prefill TTFT -7 to -10% at 18%). */
    int min_rows() const { return min_rows_; }
    /* False once a read failed: the readers then take every byte from the container. */
    bool on() const { return on_.load(std::memory_order_acquire); }
    void turn_off(int rc);

    struct Stats { uint64_t reads = 0, bytes = 0, ns = 0; };
    Stats stats() const;

private:
    PlaneMirror() = default;
    ~PlaneMirror();
    bool open(const std::string& dir);
    bool unchanged(int shard) const;

    struct Shard { std::string path; int fd = -1; int64_t size = 0, mtime_ns = 0, ctime_ns = 0; };
    std::vector<Shard> shards_;
    std::unordered_map<std::string, Ref> index_;     /* "L.E" -> w3 codes */
    double share_ = 0.18;
    int    min_rows_ = 2;
    double decode_share_ = 0.0;
    std::atomic<bool> on_{ true };
    std::atomic<int64_t> fault_{ -1 };   /* ..._FAULT=N: the N-th read fails (tests the fallback) */
    mutable std::mutex stats_mu_;
    Stats stats_;
};

}  /* namespace rad */
