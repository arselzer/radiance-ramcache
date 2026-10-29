/* GLM noaux_tc router. Correction bias changes selection only; mixture weights
 * come from the uncorrected probabilities -- sigmoid, or the score Config::score names
 * (DeepSeek-V4's sqrt(softplus)). All routing arithmetic is F32. */
#ifndef RAD_BLOCK_GLM_ROUTER_H
#define RAD_BLOCK_GLM_ROUTER_H
#include "rad_glm_tp.h"
#include <cmath>

namespace rad {
namespace arch {
struct GlmRouter {
    struct Config {
        int64_t experts = 288, top_k = 8, rows = 32;
        int norm = 1;
        double route_scale = 2.5;
        /* The score function by name (router_scored_topk: "sqrtsoftplus" for DeepSeek-V4); null is
         * GLM's sigmoid (router_sigmoid_topk). */
        const char* score = nullptr;
        /* The projection read through gemm_nt_f32bf16: a weight whose values are bf16
         * (DeepSeek-V4.1 ships its routers bf16 and the container holds them widened to f32),
         * stored as its bf16 halves, half the bytes for every logit bit unchanged. The declared
         * dtype follows the source: an f32 plane stays declared f32 and the kernel's layout keeps
         * its high halves, refusing the load if any low half is not zero; a bf16 plane is declared
         * bf16 and read as it is. Off (GLM's f32 routers): gemm_nt_f32 over the f32 weight, as
         * before. */
        bool bf16_halves = false;
    };
    struct Src {
        const char *weight = nullptr, *correction = nullptr;
    };
    struct Wire {
        rad_buf input = 0, logits = 0, ids = 0, weights = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    rad_weight projection = 0, correction = 0;
    rad_op op_project = 0, op_select = 0;
    SplitF32Proj split{};

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int layer, const char* name,
                const Config& cfg, const Wire& wire, const Src& src) {
        g = geom;
        c = cfg;
        w = wire;
        if (!b || !name || !src.weight || !src.correction || g.n_embd < 1 || c.experts < 1 ||
            c.experts > 512 || c.top_k < 1 || c.top_k > c.experts || c.rows < 1 ||
            c.rows > g.max_tok || !std::isfinite(c.route_scale) || c.route_scale <= 0 || !w.input ||
            !w.logits || !w.ids || !w.weights || w.logits == w.weights)
            return RAD_E_INVAL;
        const char *pn = nm.f("%s.weight", name), *cn = nm.f("%s.correction", name);
        RAD_ARCH_TRY(map_copy(b, pn, src.weight));
        RAD_ARCH_TRY(map_copy(b, cn, src.correction));
        const char* gemm = "gemm_nt_f32";
        uint32_t wdt = RAD_F32;
        if (c.bf16_halves) {
            /* Refused rather than served as f32: a library without the op would leave the switch
             * silently off. */
            if (!rad_op_declared(b, "gemm_nt_f32bf16")) {
                fprintf(stderr, "radiance: %s: the router's bf16 halves need gemm_nt_f32bf16, "
                                "which no loaded kernel library declares\n", name);
                return RAD_E_UNSUPPORTED;
            }
            gemm = "gemm_nt_f32bf16";
            RadEncoding e{};
            if (weight_enc(b, pn, &e) && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16)
                wdt = RAD_BF16;
        }
        projection = decl_w(b, pn, wdt, { c.experts, g.n_embd }, RAD_ACCESS_PER_TOKEN,
                            RAD_SHARD_NONE, grp_layer(layer));
        correction = decl_w(b, cn, RAD_F32, { c.experts }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(layer));
        if (!projection || !correction) return RAD_E_INVAL;
        op_project = rw(b,
                        RAD_OP(b, gemm,
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", c.experts),
                                          RAD_INT("K", g.n_embd), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(projection)),
                        { w.input }, { w.logits });
        RAD_ARCH_TRY(split.declare(b, nm, g, name, c.rows, c.experts, g.n_embd, projection, w.input,
                                   w.logits, gemm));
        if (c.score)
            op_select =
                rw(b,
                   RAD_OP(b, "router_scored_topk",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.experts),
                                     RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm),
                                     RAD_F64("route_scale", c.route_scale),
                                     RAD_STR("score", c.score), RAD_STR("dtype", "f32")),
                          RAD_WEIGHTS(correction)),
                   { w.logits }, { w.ids, w.weights });
        else
            op_select =
                rw(b,
                   RAD_OP(b, "router_sigmoid_topk",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.experts),
                                     RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm),
                                     RAD_F64("route_scale", c.route_scale), RAD_STR("dtype", "f32")),
                          RAD_WEIGHTS(correction)),
                   { w.logits }, { w.ids, w.weights });
        return op_project && op_select ? RAD_OK : RAD_E_INVAL;
    }
    void project(RadCtx* ctx, int64_t first, int64_t rows, int64_t output_first = 0) const {
        if (split.takes(rows))
            split.step(ctx, rows, brow_slice(w.input, first, rows, g.n_embd), projection,
                       brow_slice(w.logits, output_first, rows, c.experts));
        else
            RAD_ISSUE_N(ctx, op_project, rows, brow_slice(w.input, first, rows, g.n_embd),
                        RAD_W(projection), brow_slice(w.logits, output_first, rows, c.experts));
    }
    void step(RadCtx* ctx, int64_t first, int64_t rows, int64_t output_first = 0) const {
        project(ctx, first, rows, output_first);
        RAD_ISSUE_N(ctx, op_select, rows, brow_slice(w.logits, output_first, rows, c.experts),
                    RAD_W(correction), brow_slice(w.ids, output_first, rows, c.top_k),
                    brow_slice(w.weights, output_first, rows, c.top_k));
    }
};
}
} // namespace rad::arch
#endif
