/* Copy only the physical fragments an exact logical prefill hash will read. Growth-mode debug
 * hashing must not D2H-copy an entire max-VA layer stripe through unbacked holes. */
#pragma once
#include "mem/kv_storage.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace rad {
using KVHashRuns = std::vector<std::pair<int64_t, int64_t>>;
inline bool kv_hash_physical_runs(const RadKVGroupDecl& d, int64_t block_size, int64_t layer_bytes,
                                  const KVHashRuns& logical, KVHashRuns* out) {
    if (!out || layer_bytes < 0) return false;
    const bool packed = kv_storage_packed(d);
    if (packed && kv_storage_page_error(d, block_size)) return false;
    const int64_t logical_page = packed ? 4 * block_size * d.head_dim : 0;
    KVHashRuns runs;
    runs.reserve(logical.size());
    for (const auto& r : logical) {
        if (r.first < 0 || r.second < 0) return false;
        if (!r.second) continue;
        int64_t off = r.first, bytes = r.second;
        if (packed) {
            const int64_t page = off / logical_page, within = off % logical_page;
            if (bytes > logical_page - within || layer_bytes < d.storage_block_bytes ||
                page >= layer_bytes / d.storage_block_bytes) return false;
            off = page * d.storage_block_bytes;
            bytes = d.storage_block_bytes;
        }
        if (off > layer_bytes || bytes > layer_bytes - off) return false;
        runs.push_back({ off, bytes });
    }
    std::sort(runs.begin(), runs.end());
    size_t count = 0;
    for (const auto& r : runs) {
        if (count && r.first <= runs[count - 1].first + runs[count - 1].second) {
            auto& previous = runs[count - 1];
            previous.second = std::max(previous.second, r.first + r.second - previous.first);
        } else runs[count++] = r;
    }
    runs.resize(count);
    *out = std::move(runs);
    return true;
}
}  // namespace rad
