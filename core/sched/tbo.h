/* tbo.h -- two-batch overlap of a concurrent decode step (RADIANCE_EXPERIMENTAL_SCHED_TBO=1).
 *
 * A pure decode step of S >= 2 sequences, one row each, is issued as two micro-batches: A, the
 * first ceil(S / 2) sequences (rows [0, Sa)), and B, the rest (rows [Sa, S)). Each micro-batch's
 * layer is issued in three segments:
 *
 *   1  the layer up to the MoE: the Engram block, the attention connection, the attention, its
 *      all-reduce and the ffn collapse and norm;
 *   2  the MoE up to its gather: the router, the routed and shared GEMMs, the ffn mix, and the next
 *      layer's on-die reads;
 *   3  the MoE gather onward: the CPU rows copied over the card's (the file stager waits here for
 *      the pool), the gather, the all-reduce, and the post-connection (fused with the next layer's
 *      collapse and norm).
 *
 * in the order
 *
 *   1A0 2A0 | 1B0 3A0 2B0 1A1 3B0 2A1 | 1B1 3A1 2B1 1A2 3B1 2A2 | ...
 *
 * so while the issuing thread waits in 3A(l) for micro-batch A's CPU experts (and its misses' drive
 * reads), the card runs 1B(l), B's attention; and while it waits in 3B(l), the card runs 2B(l)'s
 * resident experts and 1A(l+1), A's next attention. Every op is the one the batched step issues,
 * on its micro-batch's rows: the activation buffers at the micro-batch's first row, the
 * per-sequence arrays (block tables, seqused, state slots) at its first sequence, and cu_seqlens
 * as its first Sb + 1 entries (0, 1, ..., Sb: one row a sequence). One queue, so the stream orders
 * everything; the MoE's pass-tall buffers (sorted order, offsets, expert rows) are written by B's
 * router only after A's gather in stream order, and the file stager sees each layer as two passes
 * one after the other (gate_up ... gather, then gate_up ... gather), as a sliced pass. The arena
 * keeps every transient buffer unshared under the switch (core/build/rad_builder.cpp), since the
 * declared op order no longer bounds when a buffer's bytes are live.
 *
 * Exactness: each micro-batch's ops are the ones a step of its rows alone issues (same handles, same
 * operands, step-tall buffers at the micro-batch's rows), and the interleaving never lets one
 * micro-batch read bytes the other wrote (tests/ds41_tbo_arch_test.cpp checks both on the recording
 * builder). A row's result does not depend on the other rows of its step (the replies at 1, 2, 4
 * and 8 sequences are identical), so every row is what the batched step computes.
 *
 * The architecture reads the switch at declare and, when on, declares the split
 * (rad_declare_two_batch, Program::two_batch); the core (file stager, routing witness, buffer plan)
 * follows that declaration, never the variable, so an architecture that does not split is never
 * taken for one that does. Off (unset or 0): nothing changes. This header is shared by the core and
 * the architecture (arch/deepseek41), so both take the same steps and the same split. */
#pragma once
#include "rad_runtime.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace rad {
namespace tbo {

/* RADIANCE_EXPERIMENTAL_SCHED_TBO=1 (the architecture reads it at each declare). */
inline bool env_on() {
    const char* v = std::getenv("RADIANCE_EXPERIMENTAL_SCHED_TBO");
    return v && std::strcmp(v, "1") == 0;
}

/* Whether a step is split: a trunk decode step of two or more sequences, one row each, with no
 * speculation, prefill chunk, media row, consumer cut or large-chunk admission. Host fields only. */
inline bool step_splits(const RadBatch* b) {
    return b && b->phase == RAD_PHASE_DECODE && b->draft_pass == 0 && b->n_spec == 0 &&
           b->n_seq >= 2 && b->n_tok == b->n_seq && b->n_seq_decode == b->n_seq &&
           b->n_tok_decode == b->n_tok && b->n_mm_rows == 0 && b->consumer_full == 0 &&
           b->consumer_row0 == nullptr && b->lcp_admit_row <= 0;
}

/* Micro-batch A's sequences (= rows): the first ceil(S / 2); B has the rest. */
inline int64_t split_a(int64_t S) { return (S + 1) / 2; }

/* Micro-batch k's first row and its rows, of a step of S. */
inline int64_t first_row(int64_t S, int k) { return k ? split_a(S) : 0; }
inline int64_t rows_of(int64_t S, int k) { return k ? S - split_a(S) : split_a(S); }

}  /* namespace tbo */
}  /* namespace rad */
