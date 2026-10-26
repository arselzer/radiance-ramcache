/* rowprefetch.cpp -- see rowprefetch.h. */
#include "runtime/rowprefetch.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>

namespace rad {

bool row_gather_op(const std::string& op) {
    return op == "embed_lookup" || op == "embed_lookup_q" || op == "embed_lookup_planes" ||
           op == "embed_lookup_ggml";
}

int64_t row_prefetch_page() {
    static const int64_t pg = [] {
        const long v = ::sysconf(_SC_PAGESIZE);
        return v > 0 ? (int64_t)v : (int64_t)4096;
    }();
    return pg;
}

bool row_id_at(const RadTensor& ids, int64_t i, int64_t* out) {
    /* Row-major over the shape, through the strides (elements). */
    int64_t off = i;
    if (ids.rank >= 1) {
        off = 0;
        int64_t r = i;
        for (int d = (int)ids.rank - 1; d >= 0; --d) {
            const int64_t s = ids.shape[d] > 0 ? ids.shape[d] : 1;
            off += (r % s) * ids.stride[d];
            r /= s;
        }
    }
    switch (ids.dtype) {
        case RAD_I32: *out = (int64_t)((const int32_t*)ids.data)[off]; return true;
        case RAD_U32: *out = (int64_t)((const uint32_t*)ids.data)[off]; return true;
        case RAD_I64: *out = ((const int64_t*)ids.data)[off]; return true;
        case RAD_I16: *out = (int64_t)((const int16_t*)ids.data)[off]; return true;
        default:      *out = 0; return false;
    }
}

void row_runs_merge(std::vector<PageRun>* runs) {
    std::vector<PageRun>& v = *runs;
    if (v.size() < 2) return;
    std::sort(v.begin(), v.end());
    size_t w = 0;
    for (size_t r = 1; r < v.size(); ++r) {
        if (v[r].first <= v[w].second) {
            if (v[r].second > v[w].second) v[w].second = v[r].second;
        } else {
            v[++w] = v[r];
        }
    }
    v.resize(w + 1);
}

int64_t row_prefetch(const RowPrefetch& rp, const RadTensor& ids, int64_t m, const RadTensor* t,
                     RowPrefetchScratch* s) {
    return row_page_runs(rp, ids, m, t, row_prefetch_page(), s, [](const PageRun& r) {
        /* The table is read-only and mapped; a failure (a range the kernel will not read ahead)
         * leaves the read to the fault, as without the hint. */
        (void)::madvise((void*)r.first, (size_t)(r.second - r.first), MADV_WILLNEED);
    });
}

}  /* namespace rad */
