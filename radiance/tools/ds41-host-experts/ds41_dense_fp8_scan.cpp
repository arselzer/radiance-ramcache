/* ds41_dense_fp8_scan.cpp -- the CPU-only exactness scan of ds41's FP8-sourced dense weights for
 * gemm_nt_bf16f8 (RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8).
 *
 * The container holds these linears as bf16, cast from the release's E4M3 codes and E8M0 32x32
 * block scales. With the knob on, libr4d's relayout re-encodes each one at load and refuses the
 * whole load if a single element is not exactly an E4M3 code times its block's power of two
 * (r4d_layout.cpp, r4d_bf16f8_layout.h). This runs that same hook over every opted-in entry ahead
 * of time, so a weight it would refuse can be kept bf16 (RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE)
 * instead of failing a GPU window's load.
 *
 *   ds41_dense_fp8_scan MODEL.rad [--census sources.jsonl] [--threads N] [--json OUT.json]
 *                       [--exclude OUT.txt] [--yield NAME] [--limit N]
 *
 * Entries: blk.N.{attn_q_a, attn_q_b, attn_kv, attn_output_a.gK, attn_output_b, ffn_gate_up_shexp,
 * ffn_down_shexp, indexer.attn_q_b, engram_kv}.weight, the call sites deepseek41 opts in. Each is
 * read in bands of 1024 rows (whole 32-row blocks; bounded memory), dropped from the page cache
 * after use, and each band goes through the library's own r4d_rad_layout_bf16f8 and
 * r4d_rad_relayout_bf16f8 as a plain bf16 [rows, K] plane, then back through
 * r4d_rad_unrelayout_bf16f8, which must give the band's bytes. A per-block re-encoding of its own
 * counts refused blocks and names the first failure, and must agree with the hook.
 *
 * Also checked: every entry is plain bf16 with whole 16-row tiles and 32-K blocks, and the
 * tensor-parallel slices deepseek41 cuts at 2, 3 and 4 ranks start and end on 32 rows and columns
 * (rank r's q_b rows, wo_b columns, shared-expert columns of gate|up and down, engram_kv rows), so
 * each rank's 32x32 blocks are the container's. With --census, the conversion's sources.jsonl must
 * say each entry was cast to bf16 from E4M3 pieces with E8M0 scales of one per 32x32 block.
 *
 * Read-only. It lowers its own CPU and I/O priority, and with --yield (default ab_window.sh) waits
 * while a process of that name runs, since the A/B windows read the same drive. Exit 0 when every
 * entry re-encodes exactly, 1 otherwise, 2 on a usage or read error. */
#include "rad_format.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_bf16f8_layout.h"

#include "nlohmann/json.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int64_t kBandRows = 1024;

struct Entry {
    std::string name, cls;
    int layer = -1;
    int64_t rows = 0, cols = 0;
    uint64_t plane_off = 0, plane_bytes = 0;
    uint32_t enc = 0;
    /* results */
    bool plain = false, read_ok = true, hook_ok = true, inverse_ok = true, agree = true;
    int hook_rc = 0;
    int64_t blocks = 0, refused_blocks = 0;
    int64_t fail_n = -1, fail_k = -1;
    int fail_bits = 0, fail_s = 0;
    int64_t subnormal = 0, pos_zero = 0, neg_zero = 0;
    int s_min = 999, s_max = -1;
    int64_t hist[256] = {};
    int64_t stored_bytes = 0;   /* the whole tensor as one stored form */
    double seconds = 0;
};

std::string cls_of(const std::string& name, int* layer) {
    static const std::regex re(
        R"(^blk\.(\d+)\.(attn_q_a|attn_q_b|attn_kv|attn_output_a\.g\d+|attn_output_b|ffn_gate_up_shexp|ffn_down_shexp|indexer\.attn_q_b|engram_kv)\.weight$)");
    std::smatch m;
    if (!std::regex_match(name, m, re)) return "";
    *layer = std::atoi(m[1].str().c_str());
    std::string c = m[2].str();
    if (c.rfind("attn_output_a.g", 0) == 0) c = "attn_output_a.gK";
    return c;
}

bool pread_all(int fd, void* dst, size_t n, uint64_t off) {
    uint8_t* p = (uint8_t*)dst;
    while (n) {
        const ssize_t r = ::pread(fd, p, n, (off_t)off);
        if (r <= 0) return false;
        p += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return true;
}

/* Whether a process whose command line holds `name` is running (not this one). */
bool running(const std::string& name) {
    if (name.empty()) return false;
    DIR* d = opendir("/proc");
    if (!d) return false;
    bool found = false;
    const pid_t self = getpid();
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        if (atoi(e->d_name) == self) continue;
        char path[300];
        std::snprintf(path, sizeof path, "/proc/%s/cmdline", e->d_name);
        FILE* f = std::fopen(path, "rb");
        if (!f) continue;
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        std::fclose(f);
        for (size_t i = 0; i < n; ++i) if (!buf[i]) buf[i] = ' ';
        buf[n] = 0;
        if (std::strstr(buf, name.c_str())) { found = true; break; }
    }
    closedir(d);
    return found;
}

/* Waits while a window runs, checking at most once a second. */
struct Yield {
    std::string name;
    std::mutex mu;
    std::chrono::steady_clock::time_point last{};
    std::atomic<long> waited_s{0};
    void wait() {
        if (name.empty()) return;
        std::lock_guard<std::mutex> g(mu);
        const auto now = std::chrono::steady_clock::now();
        if (now - last < std::chrono::seconds(1)) return;
        bool said = false;
        while (running(name)) {
            if (!said) {
                std::fprintf(stderr, "[scan] %s is running: paused\n", name.c_str());
                said = true;
            }
            std::this_thread::sleep_for(std::chrono::seconds(20));
            waited_s += 20;
        }
        if (said) std::fprintf(stderr, "[scan] resumed\n");
        last = std::chrono::steady_clock::now();
    }
};

struct Hook {
    RadParam p[4];
    RadEncoding enc = rad_enc_plain(RAD_BF16);
    int sel[1] = { 0 };
    RadTensor t{};
    Hook(void* data, int64_t N, int64_t K) {
        p[0] = RAD_INT("M", 1);
        p[1] = RAD_INT("N", N);
        p[2] = RAD_INT("K", K);
        p[3] = RAD_STR("dtype", "bf16");
        t.data = data;
        t.dtype = RAD_BF16;
        t.rank = 2;
        t.shape[0] = N;
        t.shape[1] = K;
        t.stride[0] = K;
        t.stride[1] = 1;
    }
};

/* One band of rows [r0, r0 + rows) of e: the hook, the inverse, and the per-block count. */
void scan_band(Entry& e, const std::vector<uint16_t>& w, int64_t r0, int64_t rows,
               std::vector<uint8_t>& blob, std::vector<uint16_t>& back) {
    const int64_t K = e.cols;
    Hook h(const_cast<uint16_t*>(w.data()), rows, K);
    RadLayout L{};
    int rc = r4d_rad_layout_bf16f8(h.p, 4, 1, &h.enc, h.sel, &h.t, 1, &L);
    bool band_ok = false;
    if (rc != RAD_OK || L.bytes != r4d_bf16f8_bytes(rows, K)) {
        e.hook_ok = false;
        e.hook_rc = rc != RAD_OK ? rc : RAD_E_SHAPE;
    } else {
        blob.resize((size_t)L.bytes);
        rc = r4d_rad_relayout_bf16f8(h.p, 4, 1, &h.enc, h.sel, &h.t, 1, blob.data(), L.bytes);
        if (rc != RAD_OK) {
            e.hook_ok = false;
            if (!e.hook_rc) e.hook_rc = rc;
        } else {
            band_ok = true;
            back.assign((size_t)(rows * K), 0);
            Hook hb(back.data(), rows, K);
            const int urc = r4d_rad_unrelayout_bf16f8(hb.p, 4, 1, &hb.enc, hb.sel, blob.data(),
                                                      L.bytes, &hb.t, 1);
            if (urc != RAD_OK || std::memcmp(back.data(), w.data(), back.size() * 2) != 0)
                e.inverse_ok = false;
            /* What the stored codes hold. */
            for (int64_t i = 0; i < rows * K; ++i) {
                const uint8_t c = blob[(size_t)i];
                if ((c & 0x78) == 0 && (c & 0x07) != 0) ++e.subnormal;
                if (c == 0x00) ++e.pos_zero;
                if (c == 0x80) ++e.neg_zero;
            }
        }
    }
    /* Block by block, independently of the hook's walk. */
    int64_t refused_here = 0;
    for (int64_t n0 = 0; n0 < rows; n0 += 32) {
        const int nr = (int)std::min<int64_t>(32, rows - n0);
        for (int64_t k0 = 0; k0 < K; k0 += 32) {
            ++e.blocks;
            const int s = r4d_bf16f8_block_exp(&w[(size_t)(n0 * K + k0)], K, nr);
            bool ok = s <= R4D_BF16F8_SMAX;
            int64_t fn = n0, fk = k0;
            uint16_t fx = 0;
            for (int i = 0; ok && i < nr; ++i)
                for (int j = 0; ok && j < 32; ++j) {
                    const uint16_t x = w[(size_t)((n0 + i) * K + k0 + j)];
                    if (r4d_bf16f8_encode(x, (uint32_t)s) < 0) { ok = false; fn = n0 + i; fk = k0 + j; fx = x; }
                }
            if (ok) {
                e.hist[s & 0xFF]++;
                e.s_min = std::min(e.s_min, s);
                e.s_max = std::max(e.s_max, s);
                continue;
            }
            ++refused_here;
            ++e.refused_blocks;
            if (e.fail_n < 0) {
                e.fail_n = r0 + fn;
                e.fail_k = fk;
                e.fail_bits = fx;
                e.fail_s = s;
            }
        }
    }
    if (band_ok != (refused_here == 0)) e.agree = false;
}

void scan_entry(int fd, Entry& e, Yield& y) {
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t K = e.cols;
    std::vector<uint16_t> w;
    std::vector<uint8_t> blob;
    std::vector<uint16_t> back;
    for (int64_t r0 = 0; r0 < e.rows; r0 += kBandRows) {
        y.wait();
        const int64_t rows = std::min<int64_t>(kBandRows, e.rows - r0);
        w.resize((size_t)(rows * K));
        const uint64_t off = e.plane_off + (uint64_t)(r0 * K * 2), n = (uint64_t)(rows * K * 2);
        if (!pread_all(fd, w.data(), (size_t)n, off)) {
            e.read_ok = false;
            break;
        }
        (void)posix_fadvise(fd, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
        scan_band(e, w, r0, rows, blob, back);
    }
    e.stored_bytes = r4d_bf16f8_bytes(e.rows, e.cols);
    e.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/* The attention world deepseek41's geometry() takes at `world` ranks: the largest d <= world that
 * divides both the output groups and the heads. */
int attn_world(int world, int o_groups, int n_head) {
    for (int d = world; d >= 1; --d)
        if (o_groups % d == 0 && n_head % d == 0) return d;
    return 1;
}
/* deepseek41's shared_slice at `world` ranks (256-column blocks). */
void shared_slice(int64_t n_ff, int world, int rank, int64_t* lo, int64_t* hi) {
    *lo = *hi = 0;
    const int64_t blk = 256;
    if (world <= 1 || n_ff % blk) return;
    if (n_ff % world == 0 && (n_ff / world) % blk == 0) {
        *lo = rank * (n_ff / world);
        *hi = *lo + n_ff / world;
        return;
    }
    const int64_t nb = n_ff / blk, base = nb / world, extra = nb % world;
    const int64_t b0 = rank * base + std::min<int64_t>(rank, extra);
    *lo = b0 * blk;
    *hi = (b0 + base + (rank < extra ? 1 : 0)) * blk;
}

int64_t meta_i(const std::map<std::string, std::string>& meta, const char* k, int64_t dflt) {
    const auto it = meta.find(k);
    return it == meta.end() ? dflt : std::atoll(it->second.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.rad [--census sources.jsonl] [--threads N] [--json OUT] "
                             "[--exclude OUT] [--yield NAME] [--limit N]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    std::string census, json_out, excl_out;
    int threads = 4;
    long limit = 0;
    Yield y;
    y.name = "ab_window.sh";
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--census") census = val();
        else if (a == "--threads") threads = std::max(1, std::atoi(val().c_str()));
        else if (a == "--json") json_out = val();
        else if (a == "--exclude") excl_out = val();
        else if (a == "--yield") y.name = val();
        else if (a == "--limit") limit = std::atol(val().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    /* Background priority for this process and the threads it starts (both are inherited). */
    (void)setpriority(PRIO_PROCESS, 0, 19);
    (void)syscall(SYS_ioprio_set, 1 /* IOPRIO_WHO_PROCESS */, 0, (3 << 13) /* IOPRIO_CLASS_IDLE */);

    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { std::perror(path); return 2; }
    RadFileHeader H{};
    if (!pread_all(fd, &H, sizeof H, 0) || H.magic != RAD_MAGIC || H.version != RAD_FORMAT_VER) {
        std::fprintf(stderr, "%s: not a RAD%u container\n", path, RAD_FORMAT_VER);
        return 2;
    }
    std::vector<char> strs((size_t)H.str_bytes + 1, 0);
    std::vector<RadFileEntry> dir((size_t)H.dir_count);
    std::vector<RadFilePlane> planes((size_t)H.plane_count);
    std::vector<RadEncoding> encs((size_t)H.enc_count);
    std::vector<RadFileKV> kvs((size_t)H.meta_count);
    if (!pread_all(fd, strs.data(), (size_t)H.str_bytes, H.str_off) ||
        !pread_all(fd, dir.data(), dir.size() * sizeof(RadFileEntry), H.dir_off) ||
        !pread_all(fd, planes.data(), planes.size() * sizeof(RadFilePlane), H.plane_off) ||
        !pread_all(fd, encs.data(), encs.size() * sizeof(RadEncoding), H.enc_off) ||
        !pread_all(fd, kvs.data(), kvs.size() * sizeof(RadFileKV), H.meta_off)) {
        std::fprintf(stderr, "%s: cannot read the directory\n", path);
        return 2;
    }
    auto S = [&](rad_stroff o) -> std::string { return o && o < H.str_bytes ? std::string(&strs[(size_t)o]) : std::string(); };
    std::map<std::string, std::string> meta;
    for (const auto& kv : kvs) {
        if (kv.type == RAD_P_STR) meta[S(kv.key)] = S(kv.v.s);
        else if (kv.type == RAD_P_INT) meta[S(kv.key)] = std::to_string(kv.v.i);
    }

    /* The opted-in entries. */
    std::vector<Entry> ents;
    std::map<std::string, int> per_cls;
    for (const auto& d : dir) {
        Entry e;
        e.name = S(d.name);
        e.cls = cls_of(e.name, &e.layer);
        if (e.cls.empty()) continue;
        e.enc = d.enc;
        if (d.rank == 2) { e.rows = d.shape[0]; e.cols = d.shape[1]; }
        const RadEncoding* en = d.enc < encs.size() ? &encs[d.enc] : nullptr;
        e.plain = en && rad_enc_is(en, "plain") && en->n_planes == 1 && en->plane[0].dtype == RAD_BF16 &&
                  d.n_planes == 1 && d.plane_first < planes.size() && d.rank == 2;
        if (e.plain) {
            e.plane_off = planes[(size_t)d.plane_first].offset;
            e.plane_bytes = planes[(size_t)d.plane_first].bytes;
            if (e.plane_bytes != (uint64_t)(e.rows * e.cols * 2)) e.plain = false;
        }
        per_cls[e.cls]++;
        ents.push_back(e);
    }
    std::sort(ents.begin(), ents.end(), [](const Entry& a, const Entry& b) {
        return a.layer != b.layer ? a.layer < b.layer : a.name < b.name;
    });
    std::printf("%s: %zu opted-in entries", path, ents.size());
    for (const auto& [c, n] : per_cls) std::printf(", %d %s", n, c.c_str());
    if (limit > 0 && (size_t)limit < ents.size()) {
        ents.resize((size_t)limit);
        std::printf("; the first %zu only (--limit)", ents.size());
    }
    std::printf("\n");

    /* Geometry: plain bf16, whole 16-row tiles and 32-K blocks. */
    int not_plain = 0, not_whole = 0;
    double total_bytes = 0;
    for (const auto& e : ents) {
        if (!e.plain) { ++not_plain; std::printf("  NOT PLAIN BF16  %s\n", e.name.c_str()); }
        else if (e.rows % 16 || e.cols % 32) { ++not_whole; std::printf("  NOT WHOLE TILES  %s [%lld, %lld]\n",
                                                                         e.name.c_str(), (long long)e.rows, (long long)e.cols); }
        total_bytes += (double)e.rows * (double)e.cols * 2.0;
    }

    /* Tensor-parallel slice boundaries at 2..4 ranks, as deepseek41 cuts them. */
    const int64_t n_head = meta_i(meta, "n_head", 64), head_dim = meta_i(meta, "head_dim", 512);
    const int64_t n_ff = meta_i(meta, "n_ff", 2304), n_embd = meta_i(meta, "n_embd", 5120);
    int64_t o_groups = 8, o_rank = 1024, ple_heads = 0, ple_dim = 0, hc = 4;
    for (const char* k : { "text_config.o_groups", "deepseek41.attention.output_group_count" })
        o_groups = meta_i(meta, k, o_groups);
    for (const char* k : { "text_config.o_lora_rank", "deepseek41.attention.output_lora_rank" })
        o_rank = meta_i(meta, k, o_rank);
    for (const char* k : { "text_config.hc_mult", "deepseek41.hyper_connection.count" })
        hc = meta_i(meta, k, hc);
    (void)ple_heads; (void)ple_dim;
    int bad_cuts = 0, cuts = 0;
    std::string cut_detail;
    auto cut = [&](const char* what, int world, int64_t v) {
        ++cuts;
        if (v % 32) {
            ++bad_cuts;
            char b[160];
            std::snprintf(b, sizeof b, "%s at %d ranks: %lld; ", what, world, (long long)v);
            cut_detail += b;
        }
    };
    for (int world = 2; world <= 4; ++world) {
        const int aw = attn_world(world, (int)o_groups, (int)n_head);
        const int64_t HD = n_head / aw * head_dim, ok = o_groups / aw * o_rank;
        for (int r = 0; r <= aw; ++r) {
            cut("attn_q_b rows", world, r * HD);
            cut("attn_output_b columns", world, r * ok);
        }
        for (int r = 0; r < world; ++r) {
            int64_t lo, hi;
            shared_slice(n_ff, world, r, &lo, &hi);
            if (hi == 0) { lo = r * (n_ff / world); hi = lo + n_ff / world; }
            cut("shared expert columns (gate|up rows, down columns)", world, lo);
            cut("shared expert columns (gate|up rows, down columns)", world, hi);
            cut("gate|up seam in a rank's rows", world, hi - lo);
        }
        const int64_t en = (hc + 1) * n_embd;
        if (en % world == 0 && (en / world) % 32 == 0)
            for (int r = 0; r <= world; ++r) cut("engram_kv rows (ENGRAM_SPLIT)", world, r * (en / world));
    }
    std::printf("tensor-parallel cuts at 2..4 ranks: %d checked, %d not on 32%s%s\n", cuts, bad_cuts,
                bad_cuts ? ": " : "", cut_detail.c_str());

    /* The census: each entry cast to bf16 from E4M3 pieces with one E8M0 scale per 32x32 block. */
    long census_entries = -1, census_pieces = 0, census_bad = 0, census_missing = 0;
    if (!census.empty()) {
        std::ifstream in(census);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", census.c_str()); return 2; }
        std::set<std::string> want;
        for (const auto& e : ents) want.insert(e.name);
        std::set<std::string> seen;
        census_entries = 0;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            nlohmann::json j = nlohmann::json::parse(line, nullptr, false);
            if (j.is_discarded() || !j.contains("name")) continue;
            const std::string nm = j["name"].get<std::string>();
            if (!want.count(nm)) continue;
            ++census_entries;
            seen.insert(nm);
            bool ok = j.value("stored", "") == "cast dtype=bf16" && j.value("encoding", "") == "bf16";
            const auto& pcs = j["pieces"];
            const auto& scs = j["scales"];
            if (!pcs.is_array() || !scs.is_array() || pcs.size() != scs.size() || pcs.empty()) ok = false;
            else
                for (size_t i = 0; i < pcs.size(); ++i) {
                    ++census_pieces;
                    const auto& p = pcs[i];
                    const auto& s = scs[i];
                    if (p.value("dtype", "") != "fp8_e4m3" || s.is_null() || s.value("dtype", "") != "e8m0") {
                        ok = false;
                        continue;
                    }
                    const auto ps = p["shape"], ss = s["shape"];
                    if (ps.size() != 2 || ss.size() != 2 ||
                        ss[0].get<int64_t>() != (ps[0].get<int64_t>() + 31) / 32 ||
                        ss[1].get<int64_t>() != (ps[1].get<int64_t>() + 31) / 32)
                        ok = false;
                }
            if (!ok) {
                ++census_bad;
                std::printf("  CENSUS  %s is not a bf16 cast of E4M3 pieces with 32x32 E8M0 scales\n", nm.c_str());
            }
        }
        census_missing = (long)(want.size() - seen.size());
        std::printf("census %s: %ld of the entries, %ld fp8_e4m3 pieces, %ld not as expected, %ld missing\n",
                    census.c_str(), census_entries, census_pieces, census_bad, census_missing);
    }

    /* The scan. */
    std::printf("scanning %.2f GB with %d threads (yield to '%s')\n", total_bytes / 1e9, threads, y.name.c_str());
    std::fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0}, done{0};
    std::mutex out_mu;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            for (size_t i; (i = next++) < ents.size();) {
                Entry& e = ents[i];
                if (!e.plain || e.rows % 16 || e.cols % 32) continue;
                scan_entry(fd, e, y);
                const size_t d = ++done;
                if (d % 50 == 0 || e.refused_blocks || !e.hook_ok || !e.inverse_ok || !e.agree || !e.read_ok) {
                    std::lock_guard<std::mutex> g(out_mu);
                    std::printf("  [%zu/%zu] %-40s %s\n", d, ents.size(), e.name.c_str(),
                                !e.read_ok ? "READ ERROR"
                                : (e.hook_ok && !e.refused_blocks ? "exact" : "REFUSED"));
                    std::fflush(stdout);
                }
            }
        });
    for (auto& t : pool) t.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    /* Report. */
    long exact = 0, refused = 0, read_err = 0, inv_bad = 0, disagree = 0;
    int64_t blocks = 0, rblocks = 0, sub = 0, pz = 0, nz = 0;
    int64_t hist[256] = {};
    int s_lo = 999, s_hi = -1;
    std::vector<std::string> excl;
    std::map<std::string, std::pair<long, long>> cls_res;
    for (const auto& e : ents) {
        if (!e.read_ok) { ++read_err; excl.push_back(e.name); continue; }
        const bool ok = e.plain && e.hook_ok && e.refused_blocks == 0 && e.rows % 16 == 0 && e.cols % 32 == 0;
        (ok ? exact : refused)++;
        cls_res[e.cls].first += ok;
        cls_res[e.cls].second += 1;
        if (!ok) excl.push_back(e.name);
        if (!e.inverse_ok) ++inv_bad;
        if (!e.agree) ++disagree;
        blocks += e.blocks;
        rblocks += e.refused_blocks;
        sub += e.subnormal;
        pz += e.pos_zero;
        nz += e.neg_zero;
        for (int i = 0; i < 256; ++i) hist[i] += e.hist[i];
        if (e.s_max >= 0) { s_lo = std::min(s_lo, e.s_min); s_hi = std::max(s_hi, e.s_max); }
        if (!ok)
            std::printf("  REFUSED %s [%lld, %lld]: %lld of %lld blocks; first (%lld, %lld) = 0x%04x at s %d (hook rc %d)\n",
                        e.name.c_str(), (long long)e.rows, (long long)e.cols, (long long)e.refused_blocks,
                        (long long)e.blocks, (long long)e.fail_n, (long long)e.fail_k, e.fail_bits, e.fail_s,
                        e.hook_rc);
    }
    for (const auto& [c, r] : cls_res)
        std::printf("  %-20s %ld / %ld exact\n", c.c_str(), r.first, r.second);
    std::printf("blocks %lld (%lld refused); codes: %lld subnormal, %lld +0, %lld -0; block exponents %d..%d\n",
                (long long)blocks, (long long)rblocks, (long long)sub, (long long)pz, (long long)nz, s_lo, s_hi);
    std::printf("inverse mismatches %ld, hook/per-block disagreements %ld, read errors %ld\n", inv_bad,
                disagree, read_err);
    std::printf("RESULT: %ld / %zu entries re-encode exactly in %.1f s (%ld s paused)%s\n", exact, ents.size(),
                secs, (long)y.waited_s, excl.empty() ? "" : "; see the exclude list");
    std::string ex;
    for (size_t i = 0; i < excl.size(); ++i) ex += (i ? "," : "") + excl[i];
    std::printf("RADIANCE_EXPERIMENTAL_DS41_DENSE_FP8_EXCLUDE=%s\n", ex.c_str());

    if (!excl_out.empty()) {
        std::ofstream o(excl_out);
        o << ex << "\n";
    }
    if (!json_out.empty()) {
        nlohmann::json j;
        j["container"] = path;
        j["entries"] = ents.size();
        j["exact"] = exact;
        j["refused"] = refused;
        j["read_errors"] = read_err;
        j["inverse_mismatches"] = inv_bad;
        j["hook_disagreements"] = disagree;
        j["blocks"] = blocks;
        j["refused_blocks"] = rblocks;
        j["subnormal_codes"] = sub;
        j["pos_zero_codes"] = pz;
        j["neg_zero_codes"] = nz;
        j["s_min"] = s_lo;
        j["s_max"] = s_hi;
        nlohmann::json h = nlohmann::json::object();
        for (int i = 0; i < 256; ++i) if (hist[i]) h[std::to_string(i)] = hist[i];
        j["s_histogram"] = h;
        j["tp_cuts_checked"] = cuts;
        j["tp_cuts_not_on_32"] = bad_cuts;
        j["not_plain_bf16"] = not_plain;
        j["not_whole_tiles"] = not_whole;
        j["census_entries"] = census_entries;
        j["census_pieces"] = census_pieces;
        j["census_not_as_expected"] = census_bad;
        j["census_missing"] = census_missing;
        j["seconds"] = secs;
        j["exclude"] = excl;
        nlohmann::json per = nlohmann::json::array();
        for (const auto& e : ents) {
            nlohmann::json x;
            x["name"] = e.name;
            x["shape"] = { e.rows, e.cols };
            x["exact"] = e.plain && e.hook_ok && e.refused_blocks == 0 && e.read_ok;
            x["blocks"] = e.blocks;
            x["refused_blocks"] = e.refused_blocks;
            if (e.fail_n >= 0) x["first_failure"] = { { "n", e.fail_n }, { "k", e.fail_k }, { "bits", e.fail_bits }, { "s", e.fail_s } };
            x["subnormal_codes"] = e.subnormal;
            x["pos_zero"] = e.pos_zero;
            x["neg_zero"] = e.neg_zero;
            x["s_min"] = e.s_min;
            x["s_max"] = e.s_max;
            x["stored_bytes_whole"] = e.stored_bytes;
            x["inverse_ok"] = e.inverse_ok;
            per.push_back(x);
        }
        j["per_entry"] = per;
        std::ofstream o(json_out);
        o << j.dump(1) << "\n";
    }
    ::close(fd);
    const bool pass = exact == (long)ents.size() && !inv_bad && !disagree && !read_err && !bad_cuts &&
                      !not_plain && !not_whole && census_bad <= 0 && census_missing <= 0;
    return pass ? 0 : 1;
}
