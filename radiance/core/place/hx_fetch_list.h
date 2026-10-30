/* hx_fetch_list.h -- a routed layer's host round trip as one ranged copy (FileStager,
 * RADIANCE_EXPERIMENTAL_DS41_FETCH_LIST).
 *
 * At each EP-routed layer's counts sync the file stager copies four things to pinned host memory,
 * queued behind the router on the compute stream: the layer's routing counts and, for the CPU pool,
 * the pass's token rows, sorted order and expert offsets (FileStager::start_ep_routed, hx_fetch).
 * As four copies that is four dispatches between the scatter and the host's wake-up, each waiting
 * behind the one before (live dispatches keep their barrier bit) and each ending in its own
 * system-scope release. rad_memcpy_ranges_async copies them in one dispatch.
 *
 * Every range is a byte offset from one source base and one destination base, so the sources and
 * destinations may sit in different allocations: device memory and every pinned range share one
 * address space (aql.cpp: "a pinned range has the same address on host and device"), and an offset
 * may be negative (the kernel adds a signed 64-bit offset to its base). The source base is the
 * counts row, the destination base its pinned copy.
 */
#pragma once
#include "../device/device.h"

#include <cstdint>

namespace rad {

/* Where one routed layer's round trip reads (on the card) and writes (pinned host memory). */
struct HxFetch {
    const void* counts = nullptr;       /* the card's counts row: the source base */
    int32_t*    counts_host = nullptr;  /* its pinned copy: the destination base */
    int64_t     n_counts = 0;           /* entries */
    const void* x = nullptr;            /* the pass's token rows, bf16, `x_ld` elements apart */
    int64_t     x_ld = 0;
    uint16_t*   x_host = nullptr;       /* their pinned copy, packed (`embd` apart) */
    int64_t     embd = 0;
    int64_t     rows = 0;               /* the pass's tokens */
    const void* sorted = nullptr;       /* the sorted order: `n_sorted` int32 */
    int32_t*    sorted_host = nullptr;
    int64_t     n_sorted = 0;
    const void* off = nullptr;          /* the expert offsets: `n_off` int32 */
    int32_t*    off_host = nullptr;
    int64_t     n_off = 0;
};

/* The ranges a fetch of `rows` token rows takes: the counts, one a token row, the sorted order and
 * the offsets. The kernel gives each range one workgroup, so a pass's rows go one a range rather
 * than as one long run copied by a single workgroup. */
constexpr int hx_fetch_n_ranges(int64_t rows) { return (int)rows + 3; }

/* Fills `out` (room for `cap`) with f's ranges in the order the separate copies run, and returns how
 * many; 0 when the copy kernel cannot take them (rad_memcpy_ranges_async: both bases 16-byte
 * aligned, every offset and length a multiple of 4) or `cap` is short. The caller then makes the
 * separate copies. */
inline int hx_fetch_ranges(const HxFetch& f, RadCopyRange* out, int cap) {
    const uintptr_t sb = (uintptr_t)f.counts, db = (uintptr_t)f.counts_host;
    if (!out || !sb || !db || ((sb | db) & 15) || f.rows < 1 || f.embd < 1 || f.x_ld < f.embd ||
        hx_fetch_n_ranges(f.rows) > cap)
        return 0;
    int n = 0;
    bool ok = true;
    auto add = [&](const void* src, const void* dst, int64_t bytes) {
        const int64_t s = (int64_t)((uintptr_t)src - sb), d = (int64_t)((uintptr_t)dst - db);
        ok = ok && src && dst && bytes > 0 && ((s | d | bytes) & 3) == 0;
        out[n++] = RadCopyRange{ s, d, bytes };
    };
    add(f.counts, f.counts_host, f.n_counts * 4);
    for (int64_t r = 0; r < f.rows; ++r)
        add((const char*)f.x + r * f.x_ld * 2, f.x_host + r * f.embd, f.embd * 2);
    add(f.sorted, f.sorted_host, f.n_sorted * 4);
    add(f.off, f.off_host, f.n_off * 4);
    return ok ? n : 0;
}

}  /* namespace rad */
