/* ds41_aux_share.h -- card sharing for the aux expert worker (work/aux-dyn/SPEC.md part A): the probe of the extra
 * card's KFD clients and VRAM counters (sysfs and procfs reads only, under roots a test fakes), the verdict, the
 * state machine, the control file, the holders and the state line. Header-only, plain C++17, no HIP and no core
 * types (the ds41_aux.h pattern): core/place/aux_control.cpp and the host tests share it.
 *
 * Nothing here touches a card: the probe reads /sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>, /proc/<pid>/comm and
 * the card's mem_info_vram_{total,used} (pread on a kept descriptor), never pp_*, gpu_busy_percent or anything else
 * that wakes the card. */
#pragma once
#include "ds41_aux.h"

#include <dirent.h>
#include <fcntl.h>
#include <regex.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ds41aux {
namespace share {

constexpr int64_t kMiB = (int64_t)1 << 20;
/* A KFD entry whose process this procfs does not show (hidepid, or one that exited) counts as a client once it
 * has been listed this long: an exited process's entry goes within a few seconds (the harness waits up to 10). */
constexpr int64_t kNoProcClientMs = 10000;
/* The room an acquire wants beyond cap + headroom: the harness's rule for the same decision (ab_window.sh). */
constexpr int64_t kRoomMarginMib = 512;
/* A policy release within this long of a (counted) acquire doubles the wait before the next one, up to kMaxBackoff;
 * serving this long resets it. */
constexpr int64_t kBackoffWindowMs = 600000, kBackoffResetMs = 1800000;
constexpr int     kMaxBackoff = 16;

inline bool read_file(const std::string& path, std::string* out, size_t cap = 4096) {
    FILE* f = std::fopen(path.c_str(), "re");
    if (!f) return false;
    out->clear();
    char buf[512];
    size_t n;
    while (out->size() < cap && (n = std::fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    std::fclose(f);
    if (out->size() > cap) out->resize(cap);
    return true;
}
inline bool read_i64(const std::string& path, int64_t* v) {
    std::string t;
    if (!read_file(path, &t, 64)) return false;
    char* end = nullptr;
    const long long x = std::strtoll(t.c_str(), &end, 10);
    if (end == t.c_str() || x < 0) return false;
    *v = (int64_t)x;
    return true;
}
inline std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

/* ---- the probe (work/aux-dyn/SPEC.md A.2) -------------------------------------------------------- */

struct Client {
    long        pid = 0;
    std::string comm;
    int64_t     bytes = 0;
    bool        no_proc = false;   /* listed by the KFD but not in this procfs for kNoProcClientMs */
};

struct Poll {
    bool ok_clients = false, ok_room = false;   /* false: that part of the verdict is unknown */
    std::string why;                             /* why a part is unknown */
    std::vector<Client> clients;                 /* other processes with at least min bytes on the card */
    std::vector<Client> appeared, gone;          /* since the previous poll */
    int64_t total = -1, used = -1, free_b = -1;  /* the card's counters (every process, this one included) */
    int64_t self = -1;                           /* this process's own vram_<gpu_id>, -1 unknown */
};

class Probe {
public:
    Probe() = default;
    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;
    ~Probe() {
        if (used_fd_ >= 0) ::close(used_fd_);
    }
    /* `sysfs` ("/sys"), `procfs` ("/proc"), the card's PCI address (parse_bdf's form), this process's pid, and
     * whether the clients are watched (the exclusive and allow policies; always reads only the room). 0, or -1
     * with *why when the card's counters cannot be found (the room unknown). A missing KFD entry of this process
     * leaves the clients unknown for good (self_ok() false, why_self()). */
    int init(const std::string& sysfs, const std::string& procfs, const std::string& bdf, long self, bool clients,
             std::string* why) {
        sysfs_ = sysfs;
        procfs_ = procfs;
        self_ = self;
        clients_ = clients;
        const std::string dir = sysfs + "/class/kfd/kfd/topology/nodes";
        if (DIR* d = opendir(dir.c_str())) {
            while (dirent* e = readdir(d)) {
                if (e->d_name[0] == '.') continue;
                std::string text;
                if (!read_file(dir + "/" + e->d_name + "/properties", &text, 65536)) continue;
                const KfdNode n = parse_kfd_props(text);
                if (!n.gfx_target_version || kfd_bdf(n) != bdf) continue;
                int64_t id = 0;
                if (read_i64(dir + "/" + e->d_name + "/gpu_id", &id) && id > 0) gpu_id_ = (uint32_t)id;
                break;
            }
            closedir(d);
        }
        const std::string dev = sysfs + "/bus/pci/devices/" + bdf;
        int64_t t = -1;
        if (read_i64(dev + "/mem_info_vram_total", &t) && t > 0) total_ = t;
        used_fd_ = ::open((dev + "/mem_info_vram_used").c_str(), O_RDONLY | O_CLOEXEC);
        if (clients_) {
            if (!gpu_id_) {
                why_self_ = "no KFD topology node has the card's PCI address " + bdf;
            } else {
                struct stat st{};
                const std::string me = sysfs + "/class/kfd/kfd/proc/" + std::to_string(self);
                if (::stat(me.c_str(), &st) != 0)
                    why_self_ = "this process (pid " + std::to_string(self) +
                                ") has no entry under /sys/class/kfd/kfd/proc (another pid namespace?): the card's "
                                "clients cannot be told apart";
            }
        }
        if (total_ <= 0 || used_fd_ < 0) {
            if (why) *why = "the card's mem_info_vram_total / mem_info_vram_used under " + dev + " are unreadable";
            return -1;
        }
        return 0;
    }
    uint32_t           gpu_id() const { return gpu_id_; }
    bool               self_ok() const { return why_self_.empty(); }
    const std::string& why_self() const { return why_self_; }
    int64_t            total() const { return total_; }

    /* One poll: about 10-20 syscalls. `min_bytes`: a process holding less is not a client (and none holding 0). */
    Poll poll(int64_t now_ms, int64_t min_bytes) {
        Poll p;
        p.total = total_;
        if (used_fd_ >= 0 && total_ > 0) {
            char b[64];
            const ssize_t n = ::pread(used_fd_, b, sizeof b - 1, 0);
            if (n > 0) {
                b[n] = 0;
                char* end = nullptr;
                const long long u = std::strtoll(b, &end, 10);
                if (end != b && u >= 0) {
                    p.used = (int64_t)u;
                    p.free_b = total_ - p.used;
                    p.ok_room = true;
                }
            }
        }
        if (!p.ok_room) p.why = "the card's mem_info_vram_used is unreadable";
        if (gpu_id_) {
            int64_t v = 0;
            if (read_i64(sysfs_ + "/class/kfd/kfd/proc/" + std::to_string(self_) + "/vram_" + std::to_string(gpu_id_), &v))
                p.self = v;
        }
        if (!clients_) return p;
        if (!why_self_.empty()) {
            p.why += (p.why.empty() ? "" : "; ") + why_self_;
            return p;
        }
        const std::string dir = sysfs_ + "/class/kfd/kfd/proc";
        DIR* d = opendir(dir.c_str());
        if (!d) {
            p.why += (p.why.empty() ? "" : "; ") + std::string("/sys/class/kfd/kfd/proc is unreadable");
            return p;
        }
        std::set<long> listed;
        while (dirent* e = readdir(d)) {
            char* end = nullptr;
            const long pid = std::strtol(e->d_name, &end, 10);
            if (end == e->d_name || *end || pid <= 0 || pid == self_) continue;
            int64_t v = 0;
            if (!read_i64(dir + "/" + e->d_name + "/vram_" + std::to_string(gpu_id_), &v)) continue;   /* not this card's */
            if (v <= 0 || v < min_bytes) continue;
            listed.insert(pid);
            Client c;
            c.pid = pid;
            c.bytes = v;
            struct stat st{};
            const std::string pd = procfs_ + "/" + std::to_string(pid);
            if (::stat(pd.c_str(), &st) == 0) {
                std::string comm;
                c.comm = read_file(pd + "/comm", &comm, 64) ? trim(comm) : std::string();
                if (c.comm.empty()) c.comm = "?";   /* unreadable: matches no allow list (the safe side) */
                no_proc_since_.erase(pid);
            } else {
                /* An exited process's entry outlives it for a moment; hidepid hides a live one for good. */
                auto it = no_proc_since_.find(pid);
                if (it == no_proc_since_.end()) it = no_proc_since_.emplace(pid, now_ms).first;
                if (now_ms - it->second < kNoProcClientMs) continue;
                c.comm = "?";
                c.no_proc = true;
            }
            p.clients.push_back(c);
        }
        closedir(d);
        for (auto it = no_proc_since_.begin(); it != no_proc_since_.end();)
            it = listed.count(it->first) ? std::next(it) : no_proc_since_.erase(it);
        std::sort(p.clients.begin(), p.clients.end(), [](const Client& a, const Client& b) { return a.pid < b.pid; });
        std::map<long, Client> now;
        for (const Client& c : p.clients) now[c.pid] = c;
        for (const auto& kv : now)
            if (!last_.count(kv.first)) p.appeared.push_back(kv.second);
        for (const auto& kv : last_)
            if (!now.count(kv.first)) p.gone.push_back(kv.second);
        last_.swap(now);
        p.ok_clients = true;
        return p;
    }

private:
    std::string sysfs_, procfs_, why_self_;
    long        self_ = 0;
    bool        clients_ = false;
    uint32_t    gpu_id_ = 0;
    int64_t     total_ = -1;
    int         used_fd_ = -1;
    std::map<long, Client>  last_;
    std::map<long, int64_t> no_proc_since_;
};

/* ---- the verdict (A.1) ----------------------------------------------------------------------------- */

struct Verdict {
    bool unknown_clients = false, unknown_room = false;
    bool foreign = false;   /* a client the policy refuses: a release at the first poll that sees it */
    bool cap = false;       /* allow: the allowed clients hold more than _FOREIGN_MIB together */
    bool floor = false;     /* serving: the card's free VRAM under _FLOOR_MIB */
    bool no_room = false;   /* released: too little room for the arena */
    std::string why;        /* the first reason, for the log */
    /* Free for an acquire in auto mode: everything known and nothing against it. */
    bool free_for_auto() const { return !unknown_clients && !unknown_room && !foreign && !cap && !no_room; }
    /* Room for an acquire in manual on: the room known and enough. */
    bool room() const { return !unknown_room && !no_room; }
};

/* The allow list's matcher (regcomp REG_EXTENDED | REG_NOSUB on the comm). */
class Allow {
public:
    Allow() = default;
    Allow(const Allow&) = delete;
    Allow& operator=(const Allow&) = delete;
    ~Allow() {
        if (ok_) regfree(&re_);
    }
    bool set(const std::string& pattern) {
        if (ok_) regfree(&re_);
        ok_ = regcomp(&re_, pattern.c_str(), REG_EXTENDED | REG_NOSUB) == 0;
        return ok_;
    }
    bool match(const std::string& comm) const { return ok_ && regexec(&re_, comm.c_str(), 0, nullptr, 0) == 0; }

private:
    regex_t re_{};
    bool    ok_ = false;
};

inline std::string client_str(const Client& c) {
    return c.comm + " (pid " + std::to_string(c.pid) + ") " + std::to_string(c.bytes / kMiB) + " MiB";
}

/* `serving`: the floor applies; otherwise the room: free >= cap + headroom + kRoomMarginMib, and the driver's own
 * figure (`kfd_avail`, AMDKFD_IOC_AVAILABLE_MEMORY; -1 unknown) >= cap + headroom. The client rules apply under
 * exclusive and allow only; the floor under every policy once a controller exists (0 turns it off). */
inline Verdict judge(const Config& c, const Allow& allow, const Poll& p, bool serving, int64_t kfd_avail) {
    Verdict v;
    auto why = [&](const std::string& s) {
        if (v.why.empty()) v.why = s;
    };
    if (c.share != kShareAlways) {
        if (!p.ok_clients) {
            v.unknown_clients = true;
            why("probe: " + (p.why.empty() ? std::string("the clients are unknown") : p.why));
        } else {
            int64_t total = 0;
            for (const Client& x : p.clients) {
                total += x.bytes;
                const bool ok = c.share == kShareAllow && allow.match(x.comm);
                if (!ok && !v.foreign) {
                    v.foreign = true;
                    why(client_str(x));
                }
            }
            if (c.share == kShareAllow && !v.foreign && total > c.foreign_mib * kMiB) {
                v.cap = true;
                why("the allowed clients hold " + std::to_string(total / kMiB) + " MiB, above " +
                    std::to_string(c.foreign_mib) + " MiB");
            }
        }
    }
    if (!p.ok_room) {
        v.unknown_room = true;
        why("probe: " + (p.why.empty() ? std::string("the card's free VRAM is unknown") : p.why));
        return v;
    }
    if (serving) {
        if (c.floor_mib > 0 && p.free_b < c.floor_mib * kMiB) {
            v.floor = true;
            why("free " + std::to_string(p.free_b / kMiB) + " MiB under the floor of " + std::to_string(c.floor_mib) +
                " MiB");
        }
    } else {
        const int64_t need = (c.cap_mib + c.headroom_mib + kRoomMarginMib) * kMiB;
        if (p.free_b < need) {
            v.no_room = true;
            why("free " + std::to_string(p.free_b / kMiB) + " MiB, under cap " + std::to_string(c.cap_mib) +
                " + headroom " + std::to_string(c.headroom_mib) + " + " + std::to_string(kRoomMarginMib) + " MiB");
        } else if (kfd_avail >= 0 && kfd_avail < (c.cap_mib + c.headroom_mib) * kMiB) {
            v.no_room = true;
            why("the driver would give " + std::to_string(kfd_avail / kMiB) + " MiB, under cap " +
                std::to_string(c.cap_mib) + " + headroom " + std::to_string(c.headroom_mib) + " MiB");
        }
    }
    return v;
}

/* ---- the state machine (A.3) ------------------------------------------------------------------------- */

enum class St : int { kStarting = 0, kServing, kDraining, kReleased, kLoading, kDead };
enum class Mode : int { kAuto = 0, kOn, kOff };
enum class Act : int { kNone = 0, kRelease, kAcquire };
/* Why a release: the shutdown line's classes. */
enum Why : int { kWhyForeign = 0, kWhyCap, kWhyFloor, kWhyUnknown, kWhyManual, kWhyN };

inline const char* st_name(St s) {
    switch (s) {
        case St::kStarting: return "starting";
        case St::kServing: return "serving";
        case St::kDraining: return "draining";
        case St::kReleased: return "released";
        case St::kLoading: return "loading";
        default: return "dead";
    }
}
inline const char* mode_name(Mode m) { return m == Mode::kOn ? "on" : m == Mode::kOff ? "off" : "auto"; }
inline const char* why_name(int w) {
    return w == kWhyForeign ? "foreign" : w == kWhyCap ? "foreign cap" : w == kWhyFloor ? "floor"
         : w == kWhyUnknown ? "unknown" : "manual";
}

struct Step {
    Act         act = Act::kNone;
    int         why = kWhyManual;
    std::string reason;
};

/* Pure: a fake clock drives it in the tests. `step` once a poll; `done_release` / `done_acquire` when the control
 * thread finished the transition `step` asked for. Starting (before aux_start attached the worker) only releases. */
class Fsm {
public:
    Fsm(int64_t release_polls, int64_t reacquire_ms) : n_(std::max<int64_t>(1, release_polls)), r_ms_(reacquire_ms) {}
    void start(St s, int64_t now_ms, bool released_at_start) {
        st_ = s;
        since_ = now_ms;
        bad_ = 0;
        free_since_ = -1;
        first_acquire_exempt_ = released_at_start;
        if (s == St::kServing) serving_since_ = now_ms;
    }
    /* aux_start attached the worker: serving (the startup load) or released (started released), the free clock
     * from now. */
    void attach(St s, int64_t now_ms) {
        st_ = s;
        since_ = now_ms;
        bad_ = 0;
        free_since_ = -1;
        if (s == St::kServing) serving_since_ = now_ms;
        if (s == St::kReleased) first_acquire_exempt_ = true;
    }
    Step step(int64_t now_ms, const Verdict& v, Mode m) {
        Step out;
        switch (st_) {
            case St::kStarting:
            case St::kServing:
            case St::kLoading: {
                if (st_ == St::kServing && serving_since_ >= 0 && now_ms - serving_since_ >= kBackoffResetMs)
                    backoff_ = 1;
                if (m == Mode::kOff) return rel(now_ms, kWhyManual, "manual off", &out);
                if (m == Mode::kOn) {
                    /* Manual on: only the floor (another tenant pushed the card to its last MiB). */
                    if (st_ == St::kLoading) { bad_ = 0; return out; }
                    if (v.floor) {
                        if (++bad_ >= n_) return rel(now_ms, kWhyFloor, v.why, &out);
                    } else {
                        bad_ = 0;
                    }
                    return out;
                }
                if (v.foreign) return rel(now_ms, kWhyForeign, v.why, &out);
                const bool held = v.cap || v.floor || v.unknown_clients || v.unknown_room;
                if (!held) {
                    bad_ = 0;
                    return out;
                }
                if (++bad_ < n_) return out;
                return rel(now_ms, v.cap ? kWhyCap : v.floor ? kWhyFloor : kWhyUnknown, v.why, &out);
            }
            case St::kReleased: {
                if (m == Mode::kOff) {
                    free_since_ = -1;
                    return out;
                }
                if (m == Mode::kOn) {
                    free_since_ = -1;
                    if (v.room()) acq(now_ms, "manual on", &out);
                    return out;
                }
                if (!v.free_for_auto()) {
                    free_since_ = -1;
                    return out;
                }
                if (free_since_ < 0) free_since_ = now_ms;
                if (now_ms - free_since_ >= r_ms_ * backoff_)
                    acq(now_ms, "free for " + std::to_string((now_ms - free_since_) / 1000) + " s", &out);
                return out;
            }
            default: return out;   /* draining: continue; dead: stay */
        }
    }
    /* The release finished: released (ok) or dead (a lane or the swap queue that did not drain). */
    void done_release(int64_t now_ms, bool ok) {
        st_ = ok ? St::kReleased : St::kDead;
        since_ = now_ms;
        free_since_ = -1;
        bad_ = 0;
    }
    /* The acquire finished: 1 serving, 0 refused (released, the wait doubled), -1 dead. */
    void done_acquire(int64_t now_ms, int result) {
        since_ = now_ms;
        bad_ = 0;
        free_since_ = -1;
        if (result > 0) {
            st_ = St::kServing;
            serving_since_ = now_ms;
            if (first_acquire_exempt_) first_acquire_exempt_ = false;
            else last_acquire_ = now_ms;
        } else if (result == 0) {
            st_ = St::kReleased;
            backoff_ = std::min(kMaxBackoff, backoff_ * 2);
        } else {
            st_ = St::kDead;
        }
    }
    St      state() const { return st_; }
    int     backoff() const { return backoff_; }
    int64_t since() const { return since_; }
    /* The wait an auto acquire needs now, ms. */
    int64_t reacquire_ms() const { return r_ms_ * backoff_; }

private:
    Step& rel(int64_t now_ms, int why, const std::string& reason, Step* out) {
        out->act = Act::kRelease;
        out->why = why;
        out->reason = reason;
        if (why != kWhyManual && last_acquire_ >= 0 && now_ms - last_acquire_ < kBackoffWindowMs)
            backoff_ = std::min(kMaxBackoff, backoff_ * 2);
        st_ = St::kDraining;
        since_ = now_ms;
        bad_ = 0;
        serving_since_ = -1;
        return *out;
    }
    void acq(int64_t now_ms, const std::string& reason, Step* out) {
        out->act = Act::kAcquire;
        out->reason = reason;
        st_ = St::kLoading;
        since_ = now_ms;
        free_since_ = -1;
    }
    St      st_ = St::kStarting;
    int64_t n_, r_ms_;
    int64_t since_ = 0, serving_since_ = -1, free_since_ = -1, last_acquire_ = -1;
    int64_t bad_ = 0;
    int     backoff_ = 1;
    bool    first_acquire_exempt_ = false;
};

/* ---- manual control (A.4): the control file and the holders ---------------------------------------------- */

/* The control file's first word: 1 with *m for on / off / auto (any case), 0 for an empty file (no change), -1 for
 * anything else. */
inline int parse_mode(const std::string& text, Mode* m) {
    std::string w;
    size_t i = 0;
    while (i < text.size() && std::isspace((unsigned char)text[i])) ++i;
    while (i < text.size() && !std::isspace((unsigned char)text[i])) w += (char)std::tolower((unsigned char)text[i++]);
    if (w.empty()) return 0;
    if (w == "on") *m = Mode::kOn;
    else if (w == "off") *m = Mode::kOff;
    else if (w == "auto") *m = Mode::kAuto;
    else return -1;
    return 1;
}

/* The control file, watched by (inode, size, mtime): a write through a temp file and rename, or in place, is seen
 * at the next poll. Missing: auto. Empty or unreadable: no change (`echo off > f` truncates before it writes).
 * Garbage: ignored, `garbage` set to it once per distinct content (the caller WARNs). */
class ControlFile {
public:
    explicit ControlFile(std::string path = "") : path_(std::move(path)) {}
    /* True when the mode changed. */
    bool poll(Mode* mode, std::string* garbage) {
        if (garbage) garbage->clear();
        struct stat st{};
        if (::stat(path_.c_str(), &st) != 0) {
            const bool was = exists_;
            exists_ = false;
            if (was && mode_ != Mode::kAuto) {
                mode_ = Mode::kAuto;
                *mode = mode_;
                return true;
            }
            *mode = mode_;
            return false;
        }
        const int64_t mt = (int64_t)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
        if (exists_ && st.st_ino == ino_ && (int64_t)st.st_size == size_ && mt == mtime_) {
            *mode = mode_;
            return false;
        }
        std::string text;
        if (!read_file(path_, &text, 64)) {
            *mode = mode_;   /* unreadable: no change, and looked at again next poll */
            return false;
        }
        exists_ = true;
        ino_ = st.st_ino;
        size_ = (int64_t)st.st_size;
        mtime_ = mt;
        Mode m = mode_;
        const int rc = parse_mode(text, &m);
        if (rc < 0) {
            const std::string t = trim(text);
            if (garbage && warned_.insert(t).second) *garbage = t;
        }
        const bool changed = rc > 0 && m != mode_;
        if (rc > 0) mode_ = m;
        *mode = mode_;
        return changed;
    }
    Mode mode() const { return mode_; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    bool        exists_ = false;
    ino_t       ino_ = 0;
    int64_t     size_ = -1, mtime_ = -1;
    Mode        mode_ = Mode::kAuto;
    std::set<std::string> warned_;
};

/* The live holders in `dir` (names not starting with '.'), sorted: a holder (aux_yield.sh) keeps a shared flock on
 * its file for as long as its job runs, and the kernel drops it with the holder, across pid namespaces on a shared
 * path. A file whose lock this process can take exclusively has no holder (the lock is given back at once). The
 * names are labels only; nothing here deletes a file. */
inline std::vector<std::string> live_holders(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const std::string p = dir + "/" + e->d_name;
        const int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
        if (fd < 0) continue;
        struct stat st{};
        if (::fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) {
            if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
                if (errno == EWOULDBLOCK) out.push_back(e->d_name);
            } else {
                (void)::flock(fd, LOCK_UN);
            }
        }
        ::close(fd);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

/* ---- the state file (A.4) ---------------------------------------------------------------------------------- */

/* Field 22 of <procfs>/<pid>/stat (the start time in clock ticks since boot), 0 when unreadable. */
inline unsigned long long proc_starttime(const std::string& procfs, long pid) {
    std::string t;
    if (!read_file(procfs + "/" + std::to_string(pid) + "/stat", &t, 4096)) return 0;
    const size_t rp = t.rfind(')');   /* comm may hold spaces and parentheses */
    if (rp == std::string::npos) return 0;
    const char* s = t.c_str() + rp + 1;
    /* fields 3.. after the comm: state is field 3, starttime field 22 */
    int field = 2;
    while (*s) {
        while (*s == ' ') ++s;
        if (!*s) break;
        ++field;
        if (field == 22) return std::strtoull(s, nullptr, 10);
        while (*s && *s != ' ') ++s;
    }
    return 0;
}

/* `<state> <unix ms> pid <pid> <starttime> mode <m> (<source>) <reason>`; `state` may be two or three words
 * ("dead freed", "dead held 26112"). One line, newline-terminated. */
inline std::string state_line(const std::string& state, int64_t unix_ms, long pid, unsigned long long starttime,
                              Mode m, const std::string& source, const std::string& reason) {
    std::string r = reason;
    for (char& ch : r)
        if (ch == '\n') ch = ' ';
    return state + " " + std::to_string(unix_ms) + " pid " + std::to_string(pid) + " " + std::to_string(starttime) +
           " mode " + mode_name(m) + " (" + source + ")" + (r.empty() ? "" : " " + r) + "\n";
}

/* Writes `text` to `path` through a temp file beside it and rename (a reader sees the old or the new line). */
inline bool write_atomic(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp." + std::to_string((long)::getpid());
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return false;
    size_t done = 0;
    while (done < text.size()) {
        const ssize_t n = ::write(fd, text.data() + done, text.size() - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        done += (size_t)n;
    }
    const bool ok = done == text.size();
    ::close(fd);
    if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        return false;
    }
    return true;
}

}  // namespace share
}  // namespace ds41aux
