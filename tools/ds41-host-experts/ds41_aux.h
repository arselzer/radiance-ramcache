/* ds41_aux.h -- the aux expert worker (core/place/aux_expert_worker.h): its switches, the pure discovery
 * decision, the slot and workspace layout, the compaction of a pass's sorted order to the worker's
 * experts and the unit set's order. Header-only, plain C++17, no HIP and no core types (the ds41_p1.h
 * pattern): the core, the device check, the emulator test and the host tests share it.
 *
 * ---- what the worker does, in one paragraph ---------------------------------------------------
 *
 * Under RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS=<MiB>, a card of the ranks' ISA beyond the --tp cards
 * (an R9700 beside four 9070 XTs) holds the hottest CPU-path units in its VRAM, in the packed RAM
 * cache's form, and computes their rows for routed layer passes of at most 16 rows: the rank thread
 * copies the pass's token rows, a sorted order compacted to the worker's experts and their offsets
 * into a pinned record mapped to that card only, the card runs libr4d's gemv_vp (gate_up),
 * swiglu_clamped and gemv_vp (down) on them on a queue of its own and copies the rows back into
 * pinned memory, and the rank places them in hx_out_ at their sorted rows before the unchanged copy
 * to its own card's y_expert. The CPU pool computes every other unit as before. A job not done by its
 * deadline is computed by the CPU from the leases kept for it, so hx_out_ holds the CPU path's bytes
 * whatever the timing. Off (the default) nothing changes.
 */
#pragma once
#include <dirent.h>
#include <regex.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ds41aux {

/* ---- the switches -----------------------------------------------------------------------------
 *
 * kEnv: unset, "" or "0" off; N in [kCapMinMib, kCapMaxMib] on, the worker's VRAM on the extra card in
 * MiB (tables, lane workspaces and unit slots in one allocation). With it off no other variable is
 * read. With it on a malformed value of any of them refuses startup.
 *   kDeviceEnv    "" automatic; a BDF ("dddd:bb:dd.f" or "bb:dd.f", any case) or a ROCr UUID ("GPU-"
 *                 and 16 hex digits) restricts discovery to that card.
 *   kPolicyEnv    "" serve, or "verify": the CPU computes every row as without the worker, the worker
 *                 computes its units too, and the rows are compared.
 *   kHeadroomEnv  free VRAM (MiB) required beyond the arena, before and after allocating it.
 *   kDeadlineEnv  us from a job's submit after which the CPU computes its rows instead (see
 *                 core/place/stager.cpp aux_finish for the exact rule).
 *   kBackoffEnv   layer jobs a lane declines after a deadline miss, once the late job completed.
 *   kForceLateEnv 0 off; N: every Nth job of a lane is abandoned as if late (its rows computed by
 *                 the CPU fallback), so a window exercises the fallback on real data. */
constexpr const char* kEnv          = "RADIANCE_EXPERIMENTAL_DS41_AUX_EXPERTS";
constexpr const char* kDeviceEnv    = "RADIANCE_EXPERIMENTAL_DS41_AUX_DEVICE";
constexpr const char* kPolicyEnv    = "RADIANCE_EXPERIMENTAL_DS41_AUX_POLICY";
constexpr const char* kHeadroomEnv  = "RADIANCE_EXPERIMENTAL_DS41_AUX_HEADROOM_MIB";
constexpr const char* kDeadlineEnv  = "RADIANCE_EXPERIMENTAL_DS41_AUX_DEADLINE_US";
constexpr const char* kBackoffEnv   = "RADIANCE_EXPERIMENTAL_DS41_AUX_BACKOFF";
constexpr const char* kForceLateEnv = "RADIANCE_EXPERIMENTAL_DS41_AUX_FORCE_LATE";

/* ---- card sharing and the online unit set (work/aux-dyn/SPEC.md §2) ------------------------------
 *
 * Read only with kEnv on, after the six above. A sub-switch set (non-empty) while its master is off refuses
 * startup naming both, so an arm never runs without what its name says:
 *   kShareEnv          "" or always: today's worker (no watch); exclusive: serve only while no other process
 *                      is a client of the extra card; allow: serve while every other client's comm matches
 *                      kShareAllowEnv and together they hold at most kShareForeignEnv MiB.
 *   kShareAllowEnv, kShareForeignEnv           master: kShareEnv=allow.
 *   kShareMinEnv                                master: kShareEnv exclusive or allow.
 *   kShareFloorEnv, kSharePollEnv, kShareRelPollsEnv, kShareReacqEnv, kShareFillEnv
 *                                               master: a controller (kShareEnv exclusive or allow, or
 *                                               kControlEnv set).
 *   kControlEnv        a path: the manual control file (on / off / auto), its holders directory <path>.d and
 *                      the state file <path>.state (work/aux-dyn/SPEC.md A.4). Its own master.
 *   kOnlineEnv         "", 0 off; 1: the online unit set (part B). Master of every kOnline*Env. */
constexpr const char* kShareEnv         = "RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE";
constexpr const char* kShareAllowEnv    = "RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_ALLOW";
constexpr const char* kShareForeignEnv  = "RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FOREIGN_MIB";
constexpr const char* kShareMinEnv      = "RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_MIN_MIB";
constexpr const char* kShareFloorEnv    = "RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FLOOR_MIB";
constexpr const char* kSharePollEnv     = "RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_POLL_MS";
constexpr const char* kShareRelPollsEnv = "RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_RELEASE_POLLS";
constexpr const char* kShareReacqEnv    = "RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_REACQUIRE_S";
constexpr const char* kShareFillEnv     = "RADIANCE_EXPERIMENTAL_DS41_AUX_SHARE_FILL_MBPS";
constexpr const char* kControlEnv       = "RADIANCE_EXPERIMENTAL_DS41_AUX_CONTROL";
constexpr const char* kOnlineEnv        = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE";
constexpr const char* kOnlineRateEnv    = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_RATE";
constexpr const char* kOnlineMaxEnv     = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_MAX";
constexpr const char* kOnlineEveryEnv   = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_EVERY";
constexpr const char* kOnlineHalfEnv    = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_HALFLIFE";
constexpr const char* kOnlinePriorEnv   = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_PRIOR";
constexpr const char* kOnlineMarginEnv  = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_MARGIN";
constexpr const char* kOnlineSpareEnv   = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_SPARE";
constexpr const char* kOnlineMbpsEnv    = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_MBPS";
constexpr const char* kOnlineCheckEnv   = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_CHECK";
constexpr const char* kOnlineDryEnv     = "RADIANCE_EXPERIMENTAL_DS41_AUX_ONLINE_DRY";

constexpr int64_t kCapMinMib = 1024, kCapMaxMib = 1048576;
constexpr int64_t kHeadroomMin = 256, kHeadroomMax = 65536, kHeadroomDef = 1024;
constexpr int64_t kDeadlineMin = 100, kDeadlineMax = 1000000, kDeadlineDef = 2000;
constexpr int64_t kBackoffMin = 0, kBackoffMax = 65536, kBackoffDef = 64;
constexpr int64_t kForceLateMin = 0, kForceLateMax = 1000000;
/* After the pool's rows are in, a job still pending gets at least this long before it is late. */
constexpr int64_t kGraceUs = 50;

enum Share : int { kShareAlways = 0, kShareExclusive = 1, kShareAllow = 2 };
inline const char* share_name(int s) { return s == kShareExclusive ? "exclusive" : s == kShareAllow ? "allow" : "always"; }

struct Config {
    bool        on = false;
    int64_t     cap_mib = 0;
    std::string device;            /* normalized: "dddd:bb:dd.f" or "GPU-<16 hex>", lower-case hex; "" auto */
    bool        device_uuid = false;
    bool        verify = false;
    int64_t     headroom_mib = kHeadroomDef;
    int64_t     deadline_us = kDeadlineDef;
    int64_t     backoff = kBackoffDef;
    int64_t     force_late = 0;
    /* Card sharing (work/aux-dyn/SPEC.md part A). */
    int         share = kShareAlways;
    std::string allow = "^whisper";
    int64_t     foreign_mib = 6144, min_mib = 1, floor_mib = 1024, poll_ms = 100, release_polls = 2;
    int64_t     reacquire_s = 60, fill_mbps = 3000;
    std::string control;           /* the control file's path, "" none */
    /* The online unit set (part B). */
    bool        online = false;
    int64_t     rate = 200, omax = 16, every = 1, halflife = 512, prior = 256, margin = 100, spare = 16;
    int64_t     mbps = 3000, check = 16;
    bool        dry = false;       /* copies and read-backs run, flips never happen (a diagnostic) */
    /* A controller (its thread, probe, gate, state file) exists: sharing or manual control. */
    bool controller() const { return on && (share != kShareAlways || !control.empty()); }
    /* The control thread runs: a controller, or the online set alone. */
    bool control_thread() const { return controller() || (on && online); }
};

/* A decimal number of digits only (no sign, space or suffix), at most 9 digits. */
inline bool parse_uint(const char* s, int64_t* out) {
    if (!s || !*s) return false;
    int64_t v = 0;
    int n = 0;
    for (const char* p = s; *p; ++p) {
        if (*p < '0' || *p > '9' || ++n > 9) return false;
        v = v * 10 + (*p - '0');
    }
    *out = v;
    return true;
}

inline bool is_hex(char c) { return std::isxdigit((unsigned char)c) != 0; }
inline std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

/* "dddd:bb:dd.f" or "bb:dd.f" (hex, any case; device 00-1f, function 0-7) into "dddd:bb:dd.f" lower-case. */
inline bool parse_bdf(const std::string& s, std::string* norm) {
    std::string t = s;
    if (t.size() == 7) t = "0000:" + t;
    if (t.size() != 12 || t[4] != ':' || t[7] != ':' || t[10] != '.') return false;
    for (int i : { 0, 1, 2, 3, 5, 6, 8, 9 })
        if (!is_hex(t[(size_t)i])) return false;
    if (t[11] < '0' || t[11] > '7') return false;
    const long dev = std::strtol(t.substr(8, 2).c_str(), nullptr, 16);
    if (dev > 0x1f) return false;
    if (norm) *norm = lower(t);
    return true;
}

/* "GPU-" and 16 hex digits (ROCr's UUID: the KFD node's unique_id), hex lower-cased. */
inline bool parse_uuid(const std::string& s, std::string* norm) {
    if (s.size() != 20 || s.compare(0, 4, "GPU-") != 0) return false;
    for (size_t i = 4; i < 20; ++i)
        if (!is_hex(s[i])) return false;
    if (norm) *norm = "GPU-" + lower(s.substr(4));
    return true;
}

inline std::string uuid_of(uint64_t unique_id) {
    char b[32];
    std::snprintf(b, sizeof b, "GPU-%016llx", (unsigned long long)unique_id);
    return b;
}

/* The control file's directory exists and is writable (the state file and the holders' directory go beside it);
 * the path is not a directory. */
inline bool control_path_ok(const std::string& path, std::string* why) {
    if (path.empty() || path.back() == '/') {
        if (why) *why = "a file path, not empty and not ending in /";
        return false;
    }
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
        if (why) *why = "a file path, not a directory";
        return false;
    }
    const size_t k = path.rfind('/');
    const std::string dir = k == std::string::npos ? "." : k == 0 ? "/" : path.substr(0, k);
    if (::stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        if (why) *why = "its directory " + dir + " does not exist";
        return false;
    }
    if (::access(dir.c_str(), W_OK) != 0) {
        if (why) *why = "its directory " + dir + " is not writable (the state file goes there)";
        return false;
    }
    return true;
}

/* The sharing and online switches (work/aux-dyn/SPEC.md §2), after the merged six: 1 on, -1 with *why. */
template <class Get>
int parse_dynamic(Get&& get, Config* c, std::string* why) {
    auto bad = [&](const char* name, const char* val, const std::string& want) {
        if (why) *why = std::string(name) + "=" + val + ": " + want;
        return -1;
    };
    auto set = [&](const char* name) -> const char* {
        const char* v = get(name);
        return v && *v ? v : nullptr;
    };
    if (const char* s = set(kShareEnv)) {
        if (!std::strcmp(s, "always")) c->share = kShareAlways;
        else if (!std::strcmp(s, "exclusive")) c->share = kShareExclusive;
        else if (!std::strcmp(s, "allow")) c->share = kShareAllow;
        else return bad(kShareEnv, s, "empty or always (no watch), exclusive, or allow");
    }
    if (const char* p = set(kControlEnv)) {
        std::string w;
        if (!control_path_ok(p, &w)) return bad(kControlEnv, p, w);
        c->control = p;
    }
    if (const char* o = set(kOnlineEnv)) {
        if (!std::strcmp(o, "1")) c->online = true;
        else if (std::strcmp(o, "0")) return bad(kOnlineEnv, o, "empty or 0 (off), or 1");
    }
    const bool ctl = c->share != kShareAlways || !c->control.empty();
    struct Num { const char* name; int64_t lo, hi; int64_t* dst; int master; };
    /* master: 1 kShareEnv=allow, 2 exclusive or allow, 3 a controller, 4 kOnlineEnv=1 */
    const Num nums[] = { { kShareForeignEnv, 0, 65536, &c->foreign_mib, 1 },
                         { kShareMinEnv, 0, 65536, &c->min_mib, 2 },
                         { kShareFloorEnv, 0, 65536, &c->floor_mib, 3 },
                         { kSharePollEnv, 50, 5000, &c->poll_ms, 3 },
                         { kShareRelPollsEnv, 1, 100, &c->release_polls, 3 },
                         { kShareReacqEnv, 0, 86400, &c->reacquire_s, 3 },
                         { kShareFillEnv, 100, 20000, &c->fill_mbps, 3 },
                         { kOnlineRateEnv, 1, 1600, &c->rate, 4 },
                         { kOnlineMaxEnv, 1, 256, &c->omax, 4 },
                         { kOnlineEveryEnv, 1, 64, &c->every, 4 },
                         { kOnlineHalfEnv, 16, 65536, &c->halflife, 4 },
                         { kOnlinePriorEnv, 0, 65536, &c->prior, 4 },
                         { kOnlineMarginEnv, 100, 400, &c->margin, 4 },
                         { kOnlineSpareEnv, 1, 256, &c->spare, 4 },
                         { kOnlineMbpsEnv, 100, 20000, &c->mbps, 4 },
                         { kOnlineCheckEnv, 0, 65536, &c->check, 4 } };
    auto master_ok = [&](int m) {
        return m == 1 ? c->share == kShareAllow : m == 2 ? c->share != kShareAlways : m == 3 ? ctl : c->online;
    };
    auto master_name = [&](int m) {
        return std::string(m == 1 ? std::string(kShareEnv) + "=allow"
                           : m == 2 ? std::string(kShareEnv) + "=exclusive or allow"
                           : m == 3 ? std::string(kShareEnv) + "=exclusive or allow, or " + kControlEnv
                                    : std::string(kOnlineEnv) + "=1");
    };
    if (const char* a = set(kShareAllowEnv)) {
        if (!master_ok(1)) return bad(kShareAllowEnv, a, "set without " + master_name(1));
        regex_t re;
        if (regcomp(&re, a, REG_EXTENDED | REG_NOSUB) != 0) return bad(kShareAllowEnv, a, "not a POSIX extended regular expression");
        regfree(&re);
        c->allow = a;
    }
    for (const Num& m : nums) {
        const char* s = set(m.name);
        if (!s) continue;
        if (!master_ok(m.master)) return bad(m.name, s, "set without " + master_name(m.master));
        int64_t x = 0;
        if (!parse_uint(s, &x) || x < m.lo || x > m.hi)
            return bad(m.name, s, "a whole number from " + std::to_string(m.lo) + " to " + std::to_string(m.hi));
        *m.dst = x;
    }
    if (const char* d = set(kOnlineDryEnv)) {
        if (!master_ok(4)) return bad(kOnlineDryEnv, d, "set without " + master_name(4));
        if (!std::strcmp(d, "1")) c->dry = true;
        else if (std::strcmp(d, "0")) return bad(kOnlineDryEnv, d, "empty or 0 (off), or 1");
    }
    return 1;
}

/* Reads the switches through `get` (std::getenv's shape). 0: off (nothing else read); 1: on; -1: a value
 * is malformed (*why names the variable and the value). */
template <class Get>
int parse(Get&& get, Config* c, std::string* why) {
    *c = Config{};
    const char* v = get(kEnv);
    if (!v || !*v || !std::strcmp(v, "0")) return 0;
    auto bad = [&](const char* name, const char* val, const char* want) {
        if (why) *why = std::string(name) + "=" + val + ": " + want;
        return -1;
    };
    int64_t n = 0;
    if (!parse_uint(v, &n) || n < kCapMinMib || n > kCapMaxMib)
        return bad(kEnv, v, "unset, empty or 0 (off), or the worker's VRAM in MiB, 1024 to 1048576");
    c->on = true;
    c->cap_mib = n;
    if (const char* d = get(kDeviceEnv); d && *d) {
        std::string norm;
        if (parse_bdf(d, &norm)) {
            c->device = norm;
        } else if (parse_uuid(d, &norm)) {
            c->device = norm;
            c->device_uuid = true;
        } else {
            return bad(kDeviceEnv, d, "empty (automatic), a PCI address dddd:bb:dd.f or bb:dd.f, or GPU- and 16 hex digits");
        }
    }
    if (const char* p = get(kPolicyEnv); p && *p) {
        if (!std::strcmp(p, "verify")) c->verify = true;
        else if (std::strcmp(p, "serve")) return bad(kPolicyEnv, p, "empty or serve (the worker's rows are used), or verify");
    }
    struct Num { const char* name; int64_t lo, hi; int64_t* dst; };
    const Num nums[] = { { kHeadroomEnv, kHeadroomMin, kHeadroomMax, &c->headroom_mib },
                         { kDeadlineEnv, kDeadlineMin, kDeadlineMax, &c->deadline_us },
                         { kBackoffEnv, kBackoffMin, kBackoffMax, &c->backoff },
                         { kForceLateEnv, kForceLateMin, kForceLateMax, &c->force_late } };
    for (const Num& m : nums) {
        const char* s = get(m.name);
        if (!s || !*s) continue;
        int64_t x = 0;
        if (!parse_uint(s, &x) || x < m.lo || x > m.hi) {
            const std::string want = "a whole number from " + std::to_string(m.lo) + " to " + std::to_string(m.hi);
            return bad(m.name, s, want.c_str());
        }
        *m.dst = x;
    }
    return parse_dynamic(get, c, why);
}

/* ---- topology ---------------------------------------------------------------------------------- */

/* The realpath of /sys/bus/pci/devices/<bdf> ("/sys/devices/pci0000:c0/0000:c0:01.1/.../0000:c3:00.0"):
 * the root complex ("pci0000:c0") and the root port, the component after it ("0000:c0:01.1"). False,
 * both empty, when the path has no root complex component. */
inline bool parse_pci_path(const std::string& path, std::string* complex, std::string* port) {
    if (complex) complex->clear();
    if (port) port->clear();
    std::vector<std::string> parts;
    size_t i = 0;
    while (i <= path.size()) {
        const size_t j = path.find('/', i);
        const std::string p = path.substr(i, j == std::string::npos ? std::string::npos : j - i);
        if (!p.empty()) parts.push_back(p);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    for (size_t k = 0; k < parts.size(); ++k) {
        const std::string& p = parts[k];
        if (p.size() == 10 && p.compare(0, 3, "pci") == 0 && p[7] == ':') {
            if (k + 1 >= parts.size() || !parse_bdf(parts[k + 1], nullptr)) return false;
            if (complex) *complex = p;
            if (port) *port = lower(parts[k + 1]);
            return true;
        }
    }
    return false;
}

/* One KFD topology node's properties ("name value" lines). */
struct KfdNode {
    uint64_t unique_id = 0, gfx_target_version = 0, domain = 0, location_id = 0;
    bool     ok = false;
};
inline KfdNode parse_kfd_props(const std::string& text) {
    KfdNode n;
    size_t i = 0;
    while (i < text.size()) {
        size_t j = text.find('\n', i);
        if (j == std::string::npos) j = text.size();
        const std::string line = text.substr(i, j - i);
        char key[64] = {0};
        unsigned long long v = 0;
        if (std::sscanf(line.c_str(), "%63s %llu", key, &v) == 2) {
            if (!std::strcmp(key, "unique_id")) n.unique_id = v;
            else if (!std::strcmp(key, "gfx_target_version")) n.gfx_target_version = v;
            else if (!std::strcmp(key, "domain")) n.domain = v;
            else if (!std::strcmp(key, "location_id")) n.location_id = v;
        }
        i = j + 1;
    }
    n.ok = true;
    return n;
}
inline std::string kfd_bdf(const KfdNode& n) {
    char b[32];
    std::snprintf(b, sizeof b, "%04llx:%02llx:%02llx.%llx", (unsigned long long)n.domain,
                  (unsigned long long)(n.location_id >> 8) & 0xff, (unsigned long long)(n.location_id >> 3) & 0x1f,
                  (unsigned long long)n.location_id & 7);
    return b;
}
inline bool read_text(const std::string& path, std::string* out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    out->clear();
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    std::fclose(f);
    return true;
}
/* A ROCr UUID's PCI address through <sysfs>/class/kfd/kfd/topology/nodes/<n>/properties; false when no GPU
 * node has that unique_id. */
inline bool resolve_uuid(const std::string& sysfs, const std::string& uuid, std::string* bdf) {
    std::string want;
    if (!parse_uuid(uuid, &want)) return false;
    const std::string dir = sysfs + "/class/kfd/kfd/topology/nodes";
    DIR* d = opendir(dir.c_str());
    if (!d) return false;
    bool found = false;
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        std::string text;
        if (!read_text(dir + "/" + e->d_name + "/properties", &text)) continue;
        const KfdNode n = parse_kfd_props(text);
        if (!n.gfx_target_version || uuid_of(n.unique_id) != want) continue;
        if (bdf) *bdf = kfd_bdf(n);
        found = true;
        break;
    }
    closedir(d);
    return found;
}

/* "gfx1201:sramecc-:xnack-" -> "gfx1201". */
inline std::string strip_arch(const std::string& a) { return a.substr(0, a.find(':')); }

/* ---- the decision (core/place/aux_expert_worker.cpp inventory() fills the cards) ------------- */

struct Card {
    int         ord = -1;
    std::string name, arch;        /* arch stripped (strip_arch) */
    bool        integrated = false;
    int64_t     vram_bytes = 0, vram_free = -1;   /* the driver's counters, read at the decision */
    std::string bdf;               /* parse_bdf's form, "" unknown */
    std::string complex, port;     /* parse_pci_path's, "" unknown */
};

struct Decision {
    bool        on = false;
    bool        released = false;  /* with a controller: chosen though short of free VRAM (a released start; why) */
    bool        warn = false;      /* the reason is a WARN (a mixed TP set), not an INFO */
    int         ord = -1;          /* the chosen card */
    std::string why;               /* off: why */
    std::string topology;          /* on: the chosen card's root and every rank sharing it */
};

inline std::string mib(int64_t bytes) { return std::to_string(bytes >> 20) + " MiB"; }

/* §3.3 of work/aux/SPEC.md. `cards`: every visible device by ordinal; `ranks`: the ranks' ordinals (devs[0..tp)),
 * rank 0 first; `sel_bdf`: the selector as a BDF (cfg.device itself, or the UUID resolved), "" for none or for a
 * UUID no node names. Pure. */
inline Decision decide(const std::vector<Card>& cards, const std::vector<int>& ranks, const Config& cfg,
                       const std::string& sel_bdf) {
    Decision d;
    auto card = [&](int ord) -> const Card* {
        for (const Card& c : cards)
            if (c.ord == ord) return &c;
        return nullptr;
    };
    auto is_rank = [&](int ord) { return std::find(ranks.begin(), ranks.end(), ord) != ranks.end(); };
    const Card* r0 = ranks.empty() ? nullptr : card(ranks[0]);
    std::vector<const Card*> cand;
    for (const Card& c : cards)
        if (!is_rank(c.ord)) cand.push_back(&c);
    if (cand.empty() || !r0) {
        d.why = "no visible device beyond the --tp " + std::to_string(ranks.size()) + " ranks' cards";
        return d;
    }
    /* 2. A mixed TP set while a card of the ranks' arch is left over: the production accident where the
     * extra card became rank 0. */
    std::string other;
    for (int o : ranks)
        if (const Card* c = card(o); c && c->name != r0->name) { other = c->name; break; }
    if (!other.empty())
        for (const Card* c : cand)
            if (c->arch == r0->arch) {
                d.warn = true;
                d.why = "the ranks' cards mix " + r0->name + " and " + other + " while " + c->name + " (" + c->arch +
                        ") is left over: list the rank cards first in ROCR_VISIBLE_DEVICES";
                return d;
            }
    /* 3. The selector. */
    if (!cfg.device.empty()) {
        for (size_t i = 0; i < ranks.size(); ++i)
            if (const Card* c = card(ranks[i]); c && !sel_bdf.empty() && c->bdf == sel_bdf) {
                d.why = std::string(kDeviceEnv) + "=" + cfg.device + " is rank " + std::to_string(i) + "'s card";
                return d;
            }
        std::vector<const Card*> keep;
        for (const Card* c : cand)
            if (!sel_bdf.empty() && c->bdf == sel_bdf) keep.push_back(c);
        if (keep.empty()) {
            d.why = std::string(kDeviceEnv) + "=" + cfg.device + " names no visible device beyond the ranks'";
            return d;
        }
        cand.swap(keep);
    }
    /* 4. Filters: the first failing one's reason, per candidate. */
    const int64_t need = (cfg.cap_mib + cfg.headroom_mib) << 20;
    std::vector<const Card*> ok, busy;   /* busy: a controller's candidates short only of free VRAM */
    std::string rejected, busy_why;
    for (const Card* c : cand) {
        std::string why;
        if (c->arch != r0->arch) why = "wrong ISA (" + c->arch + "; the ranks' " + r0->arch + ")";
        else if (c->integrated) why = "integrated graphics";
        else if (c->vram_free < need)
            why = "too little free VRAM: " + mib(c->vram_free < 0 ? 0 : c->vram_free) + " free, cap " +
                  std::to_string(cfg.cap_mib) + " MiB + headroom " + std::to_string(cfg.headroom_mib) + " MiB";
        if (why.empty()) {
            ok.push_back(c);
            continue;
        }
        if (cfg.controller() && c->arch == r0->arch && !c->integrated) {   /* work/aux-dyn/SPEC.md A.5 */
            busy.push_back(c);
            if (busy_why.empty()) busy_why = why;
            continue;
        }
        rejected += (rejected.empty() ? "" : "; ") + std::string("device ") + std::to_string(c->ord) + " (" +
                    (c->bdf.empty() ? "?" : c->bdf) + ", " + c->name + "): " + why;
    }
    if (ok.empty() && !busy.empty()) {   /* the card is busy now: a released start, acquired later */
        ok = busy;
        d.released = true;
        d.why = busy_why;
    }
    if (ok.empty()) {
        d.why = "no usable device beyond the ranks': " + rejected;
        return d;
    }
    /* 5. Choice: a root complex no rank card shares (known roots only), then more free VRAM, then the
     * lowest BDF. */
    auto shared = [&](const Card* c) {
        if (c->complex.empty()) return false;
        for (int o : ranks)
            if (const Card* r = card(o); r && r->complex == c->complex) return true;
        return false;
    };
    std::stable_sort(ok.begin(), ok.end(), [&](const Card* a, const Card* b) {
        const bool ia = !a->complex.empty() && !shared(a), ib = !b->complex.empty() && !shared(b);
        if (ia != ib) return ia;
        if (a->vram_free != b->vram_free) return a->vram_free > b->vram_free;
        return a->bdf < b->bdf;
    });
    const Card* c = ok[0];
    d.on = true;
    d.ord = c->ord;
    /* 6. The report: the chosen card's port and complex, and every rank sharing either. */
    if (c->complex.empty()) {
        d.topology = "PCI root unknown";
    } else {
        d.topology = "PCI root port " + c->port + " (complex " + c->complex + ")";
        for (size_t i = 0; i < ranks.size(); ++i) {
            const Card* r = card(ranks[i]);
            if (!r || r->complex != c->complex) continue;
            d.topology += "; shared with rank " + std::to_string(i) + "'s card " + r->bdf +
                          (r->port == c->port ? " (the same root port)" : " (root port " + r->port + ")");
        }
    }
    return d;
}

/* ---- slots, tables and the lanes' workspaces ------------------------------------------------------ */

constexpr int     kEmbd = 5120, kFf = 2304, kGuN = 2 * kFf;
constexpr int     kTopK = 6, kExperts = 96;
constexpr int     kMaxRows = 16;                       /* passes the worker takes: at most 16 rows */
constexpr int     kMaxSlots = kMaxRows * kTopK;        /* 96: a rank's slots in such a pass */
constexpr int64_t kGuBlockBytes = (int64_t)kGuN * (kEmbd / 32) * 17;   /* 12,533,760 */
constexpr int64_t kDnBlockBytes = (int64_t)kEmbd * (kFf / 32) * 17;    /*  6,266,880 */

constexpr int64_t align_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

/* A unit's slot in the arena: gate_up at 0, down at dn_off; units `stride` apart. The slot sizes are the
 * packed RAM cache's class sizes (a borrow's slot[]); at the default raw budgets 512 / 320 they are
 * 11,999,296 and 6,001,984, so dn_off 11,999,488 and stride 18,001,920. */
struct Layout {
    int64_t gu_slot = 0, dn_slot = 0, dn_off = 0, stride = 0;
};
inline Layout layout(int64_t gu_slot, int64_t dn_slot) {
    Layout l;
    l.gu_slot = gu_slot;
    l.dn_slot = dn_slot;
    l.dn_off = align_up(gu_slot, 256);
    l.stride = align_up(l.dn_off + dn_slot, 4096);
    return l;
}

/* The record the rank fills (pinned, the extra card's only), with the layout of the lane's `in` in VRAM:
 * sorted' at 0, off' at 512, then the rows, so a job's inputs reach the card in one copy of
 * x + AR rows. Then the lane's VRAM outputs, each 4 KiB aligned. */
struct Ws {
    int64_t sorted = 0, off = 0, x = 0, rec = 0;       /* the record (also `in`'s layout) */
    int64_t gu_out = 0, act = 0, dn_out = 0, bytes = 0;  /* the lane's VRAM workspace: in, then these */
    int64_t out = 0;                                    /* the pinned output rows */
};
inline Ws ws() {
    Ws w;
    w.sorted = 0;
    w.off = 512;
    w.x = 1024;
    static_assert(kMaxSlots * 4 <= 512 && (kExperts + 1) * 4 <= 512, "the record's index arrays");
    w.rec = align_up(w.x + (int64_t)kMaxRows * kEmbd * 2, 4096);
    w.gu_out = w.rec;
    w.act = align_up(w.gu_out + (int64_t)kMaxSlots * kGuN * 2, 4096);
    w.dn_out = align_up(w.act + (int64_t)kMaxSlots * kFf * 2, 4096);
    w.bytes = align_up(w.dn_out + (int64_t)kMaxSlots * kEmbd * 2, 4096);
    w.out = (int64_t)kMaxSlots * kEmbd * 2;
    return w;
}
/* The tables: per rank and routed layer a gate_up and a down table of kExperts entries. */
inline int64_t tables_bytes(int ranks, int layers) {
    return align_up((int64_t)ranks * layers * 2 * kExperts * 8, 4096);
}
inline int64_t fixed_bytes(int ranks, int layers) { return tables_bytes(ranks, layers) + (int64_t)ranks * ws().bytes; }
/* Units the arena holds: min(candidates, floor((cap - fixed) / stride)), 0 when nothing fits. */
inline int64_t capacity(int64_t cap_mib, int ranks, int layers, int64_t stride, int64_t candidates) {
    const int64_t room = (cap_mib << 20) - fixed_bytes(ranks, layers);
    if (room <= 0 || stride <= 0) return 0;
    return std::min<int64_t>(candidates, room / stride);
}

/* ---- the unit set ---------------------------------------------------------------------------------
 *
 * The planner's order (core/place/planner.cpp rank_units): profile share desc, expert asc, layer asc; the
 * ranks' lists merge with ties by rank asc. The inputs are rank-invariant, so the order is deterministic. */
struct Cand {
    int     rank = 0;
    int32_t unit = -1;
    float   share = 0.0f;
    int32_t expert = -1, layer = -1;
};
/* What decides whether a file-tier unit may be the worker's (work/aux/SPEC.md §5.2; FileStager::aux_candidates
 * fills it): its layer is an eligible trunk routed layer with the CPU path's decode shape and kernel rows the
 * worker can relaunch; the unit's planes are readable by the host, it is in no slab or ring slot, not lent to
 * prefill Phase 1, not refused by the packed cache, and its two weights are a gate_up and a down of the block
 * sizes. */
struct UnitFacts {
    bool    layer_ok = false, trunk = false;
    bool    planes_eligible = false, in_slab = false, lent = false, refused = false;
    int     n_weights = 0;
    int64_t bytes[2] = { 0, 0 };
    int32_t expert = -1;
};
inline bool candidate(const UnitFacts& f) {
    if (!f.layer_ok || !f.trunk || !f.planes_eligible || f.in_slab || f.lent || f.refused) return false;
    if (f.n_weights != 2 || f.expert < 0 || f.expert >= kExperts) return false;
    return (f.bytes[0] == kGuBlockBytes && f.bytes[1] == kDnBlockBytes) ||
           (f.bytes[0] == kDnBlockBytes && f.bytes[1] == kGuBlockBytes);
}
inline bool before(const Cand& a, const Cand& b) {
    if (a.share != b.share) return a.share > b.share;
    if (a.expert != b.expert) return a.expert < b.expert;
    if (a.layer != b.layer) return a.layer < b.layer;
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.unit < b.unit;
}
/* The global order's first n: each rank's quota. A rank's loader then walks its own list in order and
 * loads up to its quota, a unit it cannot load replaced by its own next candidate, so the final set does
 * not depend on the loaders' timing. */
inline std::vector<int64_t> quotas(std::vector<Cand> all, int64_t n, int ranks) {
    std::sort(all.begin(), all.end(), before);
    std::vector<int64_t> q((size_t)ranks, 0);
    for (int64_t i = 0; i < n && i < (int64_t)all.size(); ++i)
        if (all[(size_t)i].rank >= 0 && all[(size_t)i].rank < ranks) ++q[(size_t)all[(size_t)i].rank];
    return q;
}

/* ---- compaction (work/aux/SPEC.md §9.2) -----------------------------------------------------------
 *
 * The pass's sorted order `sorted` (T = AR * top_k slots) and its ne + 1 offsets, as hx_idx_ holds them.
 * For each expert e the worker holds (worker[e]), ascending, its slots [e == 0 ? 0 : off[e], e == ne - 1 ? T :
 * off[e + 1]) clamped to [0, T] (FileStager::hx_jobs_for's rule, libr4d's mx_range), keeping the live ones
 * (sorted[i] >= 0, sorted[i] / top_k < AR) in order. off'[0] = 0, off'[e + 1] = off'[e] + kept(e): every other
 * expert has an empty range, and T' = off'[ne]. sorted'[j] = sorted[i] (so slot j reads token row sorted[i] /
 * top_k, the row the CPU's job reads) and row_of[j] = i, the sorted row the CPU would write. */
struct Compact {
    int     T = 0;
    int32_t sorted[kMaxSlots] = {};
    int32_t off[kExperts + 1] = {};
    int32_t row_of[kMaxSlots] = {};
};
constexpr int kCompactShape = -1000;
/* 0, kCompactShape for a shape outside the worker's, or -(e + 1) for a held expert whose live slots differ
 * from counts[e] (the scatter's count: the order was not read as the card reads it). */
inline int compact(const int32_t* sorted, const int32_t* off, int AR, int top_k, int ne, const uint8_t* worker,
                   const int32_t* counts, int n_counts, Compact* out) {
    if (!sorted || !off || !worker || !out || AR < 1 || AR > kMaxRows || top_k != kTopK || ne != kExperts)
        return kCompactShape;
    const int T = AR * top_k;
    int k = 0;
    out->off[0] = 0;
    for (int e = 0; e < ne; ++e) {
        int kept = 0;
        if (worker[e]) {
            int lo = e == 0 ? 0 : off[e], hi = e == ne - 1 ? T : off[e + 1];
            lo = std::clamp(lo, 0, T);
            hi = std::clamp(hi, lo, T);
            for (int i = lo; i < hi; ++i) {
                if (sorted[i] < 0 || sorted[i] / top_k >= AR) continue;
                if (k >= kMaxSlots) return kCompactShape;
                out->sorted[k] = sorted[i];
                out->row_of[k] = i;
                ++k;
                ++kept;
            }
            if (!counts || e >= n_counts || kept != counts[e]) return -(e + 1);
        }
        out->off[e + 1] = out->off[e] + kept;
    }
    out->T = k;
    return 0;
}

/* ---- verify counting -------------------------------------------------------------------------- */
/* Values of a row of n bf16 that differ, and the first one's column (-1 none). */
inline int row_diff(const uint16_t* a, const uint16_t* b, int n, int* first) {
    int bad = 0;
    *first = -1;
    for (int i = 0; i < n; ++i)
        if (a[i] != b[i]) {
            if (*first < 0) *first = i;
            ++bad;
        }
    return bad;
}

/* ---- the server's log lines the window keys on (tools/ds41-host-experts/ab_window.sh greps) ----- */
constexpr const char* kLogOff    = "aux experts: off: ";
constexpr const char* kLogDevice = "aux experts: device ";
constexpr const char* kLogOn     = "aux experts: on: ";
constexpr const char* kLogRank   = "aux experts: rank ";
/* work/aux-dyn/SPEC.md A.6 / B.8: the controller's startup line, the released start, the online set's startup
 * line and its running / shutdown lines. */
constexpr const char* kLogSharing  = "aux experts: sharing: ";
constexpr const char* kLogReleased = "aux experts: released at start: ";
constexpr const char* kLogOnlineOn = "aux experts: online set: ";
constexpr const char* kLogOnline   = "aux experts: online: ";

}  /* namespace ds41aux */
