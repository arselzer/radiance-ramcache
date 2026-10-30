/* draft_conf_head.h -- the DSpark confidence head's host half (RADIANCE_EXPERIMENTAL_DS41_DSPARK_CONF,
 * draft_conf.h; core/engine_dconf.cpp drives it): the head file tools/ds41-host-experts/
 * dspark_conf_extract.py writes, and one row's score.
 *
 * The file: 8 bytes "DSCONF1\0", int32 dim, rank, vocab, 0, then f32 w[dim] (the head's weights on
 * the collapsed row) and f32 mscore[vocab] (its weights on the Markov embedding, folded through the
 * table: one score a token). A row's p is sigmoid(w . x + mscore[token]), x the row's collapsed
 * hidden state in bf16 and token its input.
 */
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rad_types.h"
#include "draft_conf.h"

namespace rad {

struct DconfHead {
    int64_t dim = 0, vocab = 0;
    std::vector<float> w, mscore;
};

/* Reads `path`, whose rows must be `dim` wide. RAD_OK, or an error with the reason in `why`. */
inline int dconf_head_load(const char* path, int64_t dim, DconfHead* out, std::string* why) {
    *out = DconfHead{};
    if (!path || !*path) { *why = "no file named"; return RAD_E_INVAL; }
    FILE* f = std::fopen(path, "rb");
    if (!f) { *why = std::string("cannot open ") + path; return RAD_E_NOTFOUND; }
    char magic[8];
    int32_t hdr[4];
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "DSCONF1\0", 8) == 0 &&
              std::fread(hdr, 4, 4, f) == 4;
    if (!ok) {
        std::fclose(f);
        *why = "not a DSCONF1 file (dspark_conf_extract.py writes one)";
        return RAD_E_FORMAT;
    }
    const int64_t d = hdr[0], rank = hdr[1], vocab = hdr[2];
    if (d != dim || rank <= 0 || vocab <= 0 || vocab > (int64_t)1 << 24) {
        std::fclose(f);
        *why = "its head is " + std::to_string(d) + " wide over " + std::to_string(vocab) +
               " tokens; the drafter's rows are " + std::to_string(dim) + " wide";
        return RAD_E_FORMAT;
    }
    out->w.resize((size_t)d);
    out->mscore.resize((size_t)vocab);
    ok = std::fread(out->w.data(), sizeof(float), (size_t)d, f) == (size_t)d &&
         std::fread(out->mscore.data(), sizeof(float), (size_t)vocab, f) == (size_t)vocab;
    char extra;
    ok = ok && std::fread(&extra, 1, 1, f) == 0;
    std::fclose(f);
    if (!ok) {
        *out = DconfHead{};
        *why = "truncated, or longer than its header says";
        return RAD_E_FORMAT;
    }
    out->dim = d;
    out->vocab = vocab;
    return RAD_OK;
}

/* p for one row: x_bf16 its `dim` collapsed values, token its input; -1 for a token outside the
 * table. */
inline float dconf_score(const DconfHead& h, const uint16_t* x_bf16, int32_t token) {
    if (h.dim <= 0 || token < 0 || (int64_t)token >= h.vocab) return -1.0f;
    double z = (double)h.mscore[(size_t)token];
    for (int64_t i = 0; i < h.dim; ++i) {
        const uint32_t u = (uint32_t)x_bf16[i] << 16;
        float v;
        std::memcpy(&v, &u, 4);
        z += (double)h.w[(size_t)i] * (double)v;
    }
    return draft_conf_sigmoid(z);
}

}  /* namespace rad */
