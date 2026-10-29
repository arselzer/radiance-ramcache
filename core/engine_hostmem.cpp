/* engine_hostmem.cpp -- see engine_hostmem.h. */
#include "engine_hostmem.h"

#include <cstdlib>
#include <fstream>

namespace rad {

bool cgroup_room(const std::string& dir, bool v2, int64_t* room) {
    auto read_ll = [](const std::string& f, long long* v) {
        std::ifstream in(f);
        std::string t;
        if (!(in >> t)) return false;
        char* e = nullptr;
        *v = std::strtoll(t.c_str(), &e, 10);
        return e && *e == '\0';
    };
    auto stat_ll = [](const std::string& f, const char* key, long long* v) {
        std::ifstream in(f);
        std::string k;
        long long x = 0;
        while (in >> k >> x) if (k == key) { *v = x; return true; }
        return false;
    };
    long long lim = 0, used = 0, file = 0;
    if (v2) {
        if (!read_ll(dir + "/memory.max", &lim)) return false;           /* "max" fails to parse */
        if (!read_ll(dir + "/memory.current", &used)) return false;
        (void)stat_ll(dir + "/memory.stat", "file", &file);
    } else {
        if (!read_ll(dir + "/memory.limit_in_bytes", &lim) || lim >= ((long long)1 << 62))
            return false;
        if (!read_ll(dir + "/memory.usage_in_bytes", &used)) return false;
        if (!stat_ll(dir + "/memory.stat", "total_cache", &file))
            (void)stat_ll(dir + "/memory.stat", "cache", &file);
    }
    const long long hard = used > file ? used - file : 0;
    *room = lim > hard ? (int64_t)(lim - hard) : 0;
    return true;
}

void cgroup_levels(const std::string& root, std::string path, bool v2,
                   const std::function<bool(const std::string& dir, int64_t room)>& visit) {
    for (;;) {
        const std::string dir = root + (path == "/" ? std::string() : path);
        int64_t room = 0;
        if (cgroup_room(dir, v2, &room) && !visit(dir, room)) return;
        if (path.empty() || path == "/") break;
        const size_t up = path.find_last_of('/');
        path = up == 0 || up == std::string::npos ? "/" : path.substr(0, up);
    }
}

int64_t host_headroom_total(int64_t headroom_mib, int64_t cache_mib, int64_t backed_bytes) {
    const int64_t MiB = (int64_t)1 << 20;
    /* No cache: the flag alone, exactly as before. */
    if (!cache_mib) return headroom_mib * MiB;
    const int64_t limit = INT64_MAX / MiB;
    if (cache_mib < 0 || headroom_mib < 0 || backed_bytes < 0 || cache_mib > limit ||
        headroom_mib > limit - cache_mib)
        return INT64_MAX;
    const int64_t cache = cache_mib * MiB;
    return headroom_mib * MiB + (cache - (backed_bytes < cache ? backed_bytes : cache));
}

}  /* namespace rad */
