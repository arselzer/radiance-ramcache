/* plane_mirror.cpp -- see plane_mirror.h. */
#include "plane_mirror.h"

#include "../format/safetensors.h"
#include "../rad_internal.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace rad {

namespace {

constexpr int64_t kAlign = 4096;

/* This thread's O_DIRECT bounce buffer, grown as asked. */
uint8_t* bounce(int64_t bytes) {
    struct Buf {
        uint8_t* p = nullptr;
        int64_t  cap = 0;
        ~Buf() { std::free(p); }
    };
    thread_local Buf b;
    if (b.cap < bytes) {
        std::free(b.p);
        b.p = (uint8_t*)std::aligned_alloc((size_t)kAlign, (size_t)bytes);
        b.cap = b.p ? bytes : 0;
    }
    return b.p;
}

/* configure()'s options, read once by get()'s first call. */
PlaneMirror::Options& options() {
    static PlaneMirror::Options o;
    return o;
}

}  /* namespace */

void PlaneMirror::configure(const Options& o) { options() = o; }

PlaneMirror* PlaneMirror::get() {
    static PlaneMirror* m = []() -> PlaneMirror* {
        const Options& o = options();
        if (o.dir.empty()) return nullptr;
        auto* p = new PlaneMirror();
        p->share_ = std::clamp(o.share, 0.0, 0.31);
        p->min_rows_ = std::max(1, o.min_rows);
        p->decode_share_ = std::clamp(o.decode_share, 0.0, 0.31);
        if (const char* s = std::getenv("RADIANCE_EXPERIMENTAL_DS41_PLANE_MIRROR_FAULT"); s && *s)
            p->fault_ = std::max(1, std::atoi(s)) - 1;
        if (!p->open(o.dir)) {
            RAD_WARN("plane mirror: %s gives no usable DeepSeek-V4.1 expert planes; reading the "
                     "container only", o.dir.c_str());
            delete p;
            return nullptr;
        }
        return p;
    }();
    return m;
}

PlaneMirror::~PlaneMirror() {
    for (Shard& sh : shards_)
        if (sh.fd >= 0) ::close(sh.fd);
}

void PlaneMirror::turn_off(int rc) {
    if (on_.exchange(false))
        RAD_WARN("plane mirror: a read failed (%d); every plane comes from the container from now on", rc);
}

bool PlaneMirror::open(const std::string& dir) {
    std::vector<std::string> names;
    if (DIR* d = ::opendir(dir.c_str())) {
        while (const dirent* e = ::readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) names.push_back(n);
        }
        ::closedir(d);
    }
    std::sort(names.begin(), names.end());
    for (const std::string& n : names) {
        Shard sh;
        sh.path = dir + "/" + n;
        SafeTensorsFile f;
        if (f.open(sh.path) != RAD_OK) return false;
        sh.fd = ::open(sh.path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        struct stat st{};
        uint64_t hlen = 0;
        const int idx = (int)shards_.size();
        shards_.push_back(sh);              /* the destructor closes it, whatever happens next */
        if (sh.fd < 0 || ::fstat(sh.fd, &st) != 0 || ::pread(sh.fd, bounce(kAlign), kAlign, 0) != kAlign)
            return false;
        std::memcpy(&hlen, bounce(kAlign), 8);
        shards_.back().size = st.st_size;
        shards_.back().mtime_ns = (int64_t)st.st_mtim.tv_sec * 1000000000 + st.st_mtim.tv_nsec;
        shards_.back().ctime_ns = (int64_t)st.st_ctim.tv_sec * 1000000000 + st.st_ctim.tv_nsec;
        for (const StTensor& t : f.tensors()) {
            int layer = -1, expert = -1;
            char tail[32] = {};
            if (std::sscanf(t.name.c_str(), "layers.%d.ffn.experts.%d.%31s", &layer, &expert, tail) != 3 ||
                std::strcmp(tail, "w3.weight") != 0)
                continue;
            /* One copy of each expert: two would leave the choice to the directory's order. */
            if (!index_.emplace(std::to_string(layer) + "." + std::to_string(expert),
                                Ref{ idx, 8 + hlen + t.offset, t.bytes }).second) {
                RAD_WARN("plane mirror: %s holds layer %d expert %d's w3 a second time", sh.path.c_str(),
                         layer, expert);
                return false;
            }
        }
    }
    if (index_.empty()) return false;
    RAD_INFO("plane mirror: %zu shards in %s hold %zu experts' up-projection codes; a missed "
             "expert of a pass of %d+ tokens reads %.0f%% of its bytes from them beside the container",
             shards_.size(), dir.c_str(), index_.size(), min_rows_, 100.0 * share_);
    return true;
}

bool PlaneMirror::up_codes(int layer, int expert, Ref* out) const {
    const auto it = index_.find(std::to_string(layer) + "." + std::to_string(expert));
    if (it == index_.end()) return false;
    *out = it->second;
    return true;
}

bool PlaneMirror::unchanged(int shard) const {
    const Shard& sh = shards_[(size_t)shard];
    struct stat st{};
    return ::fstat(sh.fd, &st) == 0 && st.st_size == sh.size &&
           (int64_t)st.st_mtim.tv_sec * 1000000000 + st.st_mtim.tv_nsec == sh.mtime_ns &&
           (int64_t)st.st_ctim.tv_sec * 1000000000 + st.st_ctim.tv_nsec == sh.ctime_ns;
}

int PlaneMirror::read(const Ref& r, uint64_t from, uint64_t bytes, uint8_t* dst) {
    if (r.shard < 0 || (size_t)r.shard >= shards_.size() || !dst || from + bytes > r.bytes)
        return RAD_E_INVAL;
    if (!on()) return RAD_E_STATE;
    if (fault_.load(std::memory_order_relaxed) >= 0 && fault_.fetch_sub(1) == 0) return RAD_E_IO;
    if (!unchanged(r.shard)) return RAD_E_STATE;
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t off = (int64_t)(r.off + from), lo = off / kAlign * kAlign;
    const int64_t len = (off + (int64_t)bytes + kAlign - 1) / kAlign * kAlign - lo;
    uint8_t* b = bounce(len);
    if (!b) return RAD_E_NOMEM;
    /* The rounding may run past the shard's end: stop once what was asked for has arrived, before
     * a pread at the end of the file (unaligned there) is issued at all. */
    for (int64_t done = 0; done < len && done < off - lo + (int64_t)bytes;) {
        const ssize_t n = ::pread(shards_[(size_t)r.shard].fd, b + done, (size_t)(len - done), lo + done);
        if (n <= 0) return RAD_E_IO;
        done += n;
    }
    std::memcpy(dst, b + (off - lo), (size_t)bytes);
    if (!unchanged(r.shard)) return RAD_E_STATE;
    const uint64_t ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(stats_mu_);
    ++stats_.reads;
    stats_.bytes += bytes;
    stats_.ns += ns;
    return RAD_OK;
}

PlaneMirror::Stats PlaneMirror::stats() const {
    std::lock_guard<std::mutex> lk(stats_mu_);
    return stats_;
}

}  /* namespace rad */
