/* Pure Phase2 retention inventory. NO caller, allocation, env flag or executor admission.
 *
 * Source: arch/deepseek41/deepseek41.cpp buffers()/run_layer and MoeFP8's BF16 sigmoid
 * router/six-slot gather. This conservative inventory preserves original piece boundaries,
 * including the next layer's fused xn and all shared-gather/mHC/index state. GU/activation,
 * router logits, attention, compressor and index-score scratch remain piece-local. It does
 * not establish the still-open KV/window/checkpoint/cancellation span contracts. */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rad { namespace ds41_phase2 {
constexpr int64_t kMaximumSpanRows = 16384;
constexpr int64_t kMaximumPieceRows = 4096;

struct Geometry {
    int64_t embd = 5120, hc = 4, top_k = 6, index_top_k = 512;
    int64_t local_experts = 96;
    int64_t max_ctx = 8192, candidate_ratio = 1, candidate_block = 8;
};
enum class Buffer : uint8_t {
    Streams, StreamsNext, NextNormed, FFNInput, ExpertDown, SharedOutput, SharedGate,
    PreAttention, PreFFN, Post, Combine, ExpertIDs, ExpertWeights, ExpertSorted,
    IndexSelection, IndexKeep, Count
};
constexpr std::size_t kBuffers = std::size_t(Buffer::Count);
struct Slice {
    int64_t offset = 0, bytes = 0;
    int64_t row_bytes = 0;
};
struct Layout {
    int64_t rows = 0, pieces = 0, alignment = 0;
    int64_t keep_words = 0, row_bytes = 0;
    std::array<Slice, kBuffers> buffers{};
    // Local expert offsets/counts are per ORIGINAL piece, not one segment-wide re-sort.
    Slice route_metadata{};
    int64_t total_bytes = 0;
};

inline bool add(int64_t a, int64_t b, int64_t* out) {
    if (!out || a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b) return false;
    *out = a + b; return true;
}
inline bool multiply(int64_t a, int64_t b, int64_t* out) {
    if (!out || a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b)) return false;
    *out = a * b; return true;
}
inline int64_t divide_up(int64_t value, int64_t divisor) {
    return value / divisor + (value % divisor != 0);
}
inline bool aligned(int64_t value, int64_t alignment, int64_t* out) {
    const int64_t tail = value % alignment;
    return add(value, tail ? alignment - tail : 0, out);
}

/* Scope is the current native BF16 routed path, original per-piece sorted six-slot gathers,
 * ungated shared arm and FP32 routing weights. Different geometries/representations need their
 * own inventory; silently applying the old172KiB estimate to them is unsafe. max_ctx is variable.
 * `pieces` is supplied by the ORIGINAL chunk plan, never inferred here by re-partitioning rows.
 * No output mutation on refusal, no unbounded allocations and no 192GiB/cache-size constant. */
inline bool retention_layout(const Geometry& g, int64_t rows, int64_t pieces,
                              int64_t alignment, Layout* out) {
    if (!out || g.embd != 5120 || g.hc != 4 || g.top_k != 6 || g.index_top_k != 512 ||
        g.local_experts != 96 || g.max_ctx < 1 || g.max_ctx == std::numeric_limits<int64_t>::max() ||
        g.candidate_ratio != 1 || g.candidate_block != 8 || rows < 1 || rows > kMaximumSpanRows || rows > g.max_ctx ||
        pieces < divide_up(rows, kMaximumPieceRows) || pieces > rows || alignment < 1 ||
        alignment > (int64_t(1) << 20) || (alignment & (alignment - 1))) return false;
    Layout p;
    p.rows = rows; p.pieces = pieces; p.alignment = alignment;
    // Same logical candidate width/word rule as keep_words(), with quotient/remainder bounds.
    const int64_t width = g.max_ctx / g.candidate_ratio + 1;
    p.keep_words = divide_up(divide_up(width, g.candidate_block), 32);
    int64_t keep_bytes = 0;
    if (!multiply(p.keep_words, 4, &keep_bytes)) return false;
    const int64_t rw[kBuffers] = {
        2*g.hc*g.embd, 2*g.hc*g.embd, 2*g.embd, 2*g.embd, 2*g.top_k*g.embd,
        2*g.embd, 2, 4*g.hc, 4*g.hc, 4*g.hc, 4*g.hc*g.hc,
        4*g.top_k, 4*g.top_k, 4*g.top_k, 4*g.index_top_k, keep_bytes
    };
    int64_t end = 0;
    for (std::size_t i = 0; i < kBuffers; ++i) {
        auto& s = p.buffers[i]; s.row_bytes = rw[i];
        if (!aligned(end, alignment, &s.offset) || !multiply(rows, rw[i], &s.bytes) ||
            !add(s.offset, s.bytes, &end) || !add(p.row_bytes, rw[i], &p.row_bytes)) return false;
    }
    auto& metadata = p.route_metadata;
    metadata.row_bytes = 4*(2*g.local_experts + 1);
    if (!aligned(end, alignment, &metadata.offset) || !multiply(pieces, metadata.row_bytes, &metadata.bytes) ||
        !add(metadata.offset, metadata.bytes, &end) || !aligned(end, alignment, &p.total_bytes)) return false;
    *out = p;
    return true;
}

struct Budget {
    int64_t device_room = 0, device_headroom = 0;
    // Byte room funded by replacing the OLD piece's retained subset, if separately proven.
    // Zero is the conservative default; this helper never assumes an arena union exists.
    int64_t replaceable_piece_bytes = 0;
    // Dense/ring/KV/final-hidden/metadata/capture/checkpoint costs not included in Layout.
    int64_t additional_device_bytes = 0;
    int64_t host_available = 0, host_headroom = 0;
    int64_t future_expert_cache_bytes = 0, additional_host_bytes = 0;
};
/* An accounting decision only. Host room must preserve BOTH configured future expert-cache
 * growth and system headroom; reducing the cache or changing residency is never implicit. */
inline bool fits_budget(const Layout& p, const Budget& b) {
    if (p.total_bytes < 1 || b.device_room < 0 || b.device_headroom < 0 ||
        b.replaceable_piece_bytes < 0 || b.replaceable_piece_bytes > p.total_bytes ||
        b.additional_device_bytes < 0 || b.host_available < 0 || b.host_headroom < 0 ||
        b.future_expert_cache_bytes < 0 || b.additional_host_bytes < 0) return false;
    int64_t need = p.total_bytes - b.replaceable_piece_bytes;
    if (!add(need, b.additional_device_bytes, &need) || !add(need, b.device_headroom, &need) ||
        need > b.device_room) return false;
    int64_t host = 0;
    return add(b.host_headroom, b.future_expert_cache_bytes, &host) &&
           add(host, b.additional_host_bytes, &host) && host <= b.host_available;
}
}}  // namespace rad::ds41_phase2
