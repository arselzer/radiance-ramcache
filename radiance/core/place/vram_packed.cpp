/* vram_packed.cpp -- see vram_packed.h. */
#include "vram_packed.h"

#include "ds41_host_experts.h"
#include "format/radfile.h"
#include "planner.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rad {

int vram_pack_config(VramPackConfig* out) {
    *out = VramPackConfig{};
    const char* e = std::getenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED");
    if (!e || !*e || !std::strcmp(e, "0")) return RAD_OK;
    if (std::strcmp(e, "1")) {
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED=%s: 1 holds DeepSeek-V4.1's VRAM-resident experts in the "
                "packed-scale form, 0 or unset leaves them block_mxfp4", e);
        return RAD_E_INVAL;
    }
    out->on = true;
    const char* r = std::getenv("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_RAW");
    if (!r || !*r) return RAD_OK;
    int a = -1, b = -1, used = 0;
    if (std::sscanf(r, "%d,%d%n", &a, &b, &used) != 2 || r[used] || a < 0 || b < 0 || a > (1 << 24) ||
        b > (1 << 24)) {
        RAD_ERR("RADIANCE_EXPERIMENTAL_DS41_VRAM_PACKED_RAW=%s: <gate_up>,<down>, the raw rounds a packed weight "
                "has room for (default %d,%d)", r, out->raw[0], out->raw[1]);
        return RAD_E_INVAL;
    }
    out->raw[0] = a;
    out->raw[1] = b;
    return RAD_OK;
}

void vram_pack_eligible(const Program& prog, const RadFile& file, const VramPackConfig& cfg,
                        std::vector<VramPackWeight>* out, std::string* why) {
    out->assign(prog.weights.size(), VramPackWeight{});
    if (why) why->clear();
    if (!cfg.on) return;
    /* Each weight's readers: 1 only gemv_vp tables, 2 anything else as well. A routed GEMM with a band
     * another row took (a pass over 16 tokens) reads its table as block_mxfp4 there, so none of its
     * weights may be packed. */
    std::vector<uint8_t> state(prog.weights.size(), 0);
    int64_t vp_ops = 0, other_ops = 0;
    std::string other;
    for (const OpInfo& o : prog.ops) {
        if (o.weights.empty()) continue;
        bool vp = o.op == "moe_gemm_ggml" && !o.bands.empty();
        for (const Band& b : o.bands) {
            const Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
            vp = vp && r && r.row->info && r.row->info->name && !std::strcmp(r.row->info->name, kVramPackRow);
        }
        for (size_t k = 0; k < o.weights.size(); ++k) {
            const rad_weight w = o.weights[k];
            if (w == 0 || (size_t)w >= prog.weights.size()) continue;
            /* moe_gemm_ggml's weight table is its operand 1 (`w`). */
            const bool table = k < o.weight_opd.size() && o.weight_opd[k] == 1;
            if (vp && table) { if (!state[w]) state[w] = 1; }
            else state[w] = 2;
        }
        if (o.op != "moe_gemm_ggml") continue;
        if (vp) { ++vp_ops; continue; }
        if (!other_ops++) {
            std::string rows;
            for (const Band& b : o.bands) {
                const Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
                rows += fmt("%s%s: %s", rows.empty() ? "" : ", ", b.span.c_str(),
                            r && r.row->info ? r.row->info->name : "no device kernel");
            }
            other = fmt("op %d (%s)", (int)o.index, rows.c_str());
        }
    }
    if (why && other_ops)
        *why = fmt("%lld routed moe_gemm_ggml op(s) have a band %s does not serve (passes over its 16 tokens, "
                   "or another type), so their experts stay block_mxfp4 in VRAM; the first: %s",
                   (long long)other_ops, kVramPackRow, other.c_str());
    if (why && !vp_ops && !other_ops) *why = "no moe_gemm_ggml op reads a weight table";

    /* Weights the gemv_vp tables read that are still refused, by reason, for the startup log. */
    int64_t read = 0, not_trunk = 0, not_planes = 0, not_whole = 0, not_smaller = 0, in_bank = 0;
    for (size_t w = 1; w < prog.weights.size(); ++w) {
        if (state[w] != 1) continue;
        const WeightInfo& wi = prog.weights[w];
        /* A DSpark bank's weight (DsparkRadBanks::bind_weights: file_id 1, the draft's) keeps its block
         * form and its own GEMV path; `file` is the main container, which holds the trunk's. Counted
         * apart, so the line below reads as it does without a bank. */
        if (wi.file_id != 0) {
            ++in_bank;
            continue;
        }
        ++read;
        const int32_t L = wi.decl.group.layer, E = wi.decl.group.expert, S = wi.decl.group.slot;
        /* The trunk's routed experts: gate_up (slot 0) or down (slot 1) of a layer below n_layers, so a
         * draft head's experts (layers past the trunk) are never repacked. */
        if (L < 0 || E < 0 || (S != 0 && S != 1) || (prog.meta.n_layers > 0 && L >= (int32_t)prog.meta.n_layers)) {
            ++not_trunk;
            continue;
        }
        RadTensor pt[RAD_ENC_MAX_PLANES];
        weight_planes(wi, pt);
        const int64_t N = pt[0].shape[0], K = pt[0].shape[1];
        if (wi.layout_tag != "ggml_mxfp4" || wi.identity || wi.widen || wi.narrow || wi.n_sel != 2 ||
            wi.file_id != 0 || !wi.enc_known || pt[0].dtype != RAD_FP4E2M1 || pt[1].dtype != RAD_E8M0 ||
            N <= 0 || K <= 0 || K % 32 || pt[1].shape[0] != N || pt[1].shape[1] * 32 != K || N > INT_MAX ||
            K > INT_MAX || wi.stored_bytes != N * (K / 32) * 17) {
            ++not_planes;
            continue;
        }
        const RadFileEntry* e = file.find(wi.source);
        /* This rank holds the whole weight: each selected plane is the container's whole plane. */
        if (!e || (int64_t)file.plane(*e, wi.sel[0]).bytes != N * (K / 2) ||
            (int64_t)file.plane(*e, wi.sel[1]).bytes != N * (K / 32)) {
            ++not_whole;
            continue;
        }
        const RadFilePlane& ps = file.plane(*e, wi.sel[1]);
        const int cap = cfg.raw[S];
        const int64_t bytes = (int64_t)ds41hx::packed_bytes((int)N, (int)K, cap);
        if (bytes <= 0 || bytes >= wi.stored_bytes) {
            ++not_smaller;
            continue;
        }
        VramPackWeight& v = (*out)[w];
        v.bytes = bytes;
        v.cap = cap;
        v.N = (int32_t)N;
        v.K = (int32_t)K;
        v.scales_off = ps.offset;
        v.scales_bytes = (int64_t)ps.bytes;
    }
    if (why && (not_trunk || not_planes || not_whole || not_smaller))
        *why += fmt("%s%lld of the %lld weights the gemv_vp tables read stay block_mxfp4: %lld not a trunk "
                    "expert's gate_up or down, %lld not ggml_mxfp4 over fp4 codes and e8m0 planes, %lld not "
                    "whole in the container, %lld no smaller packed",
                    why->empty() ? "" : "; ", (long long)(not_trunk + not_planes + not_whole + not_smaller),
                    (long long)read, (long long)not_trunk, (long long)not_planes, (long long)not_whole,
                    (long long)not_smaller);
    if (why && in_bank)
        *why += fmt("%s%lld more weights the gemv_vp tables read are a DSpark bank's (the draft's) and stay "
                    "block_mxfp4", why->empty() ? "" : "; ", (long long)in_bank);
}

int vram_pack_scan(const RadFile& file, const VramPackWeight& w, std::vector<uint8_t>* buf) {
    if (w.bytes <= 0 || w.N <= 0 || w.K <= 0 || w.scales_bytes != (int64_t)w.N * (w.K / 32)) return -1;
    buf->resize((size_t)w.scales_bytes);
    if (file.read_at(buf->data(), w.scales_bytes, (int64_t)w.scales_off) != RAD_OK) return -1;
    /* Read once here and again, O_DIRECT, by the load: no reason to keep it in the page cache. */
    file.drop_range((int64_t)w.scales_off, w.scales_bytes);
    ds41hx::PackInfo plan;
    if (!ds41hx::pack_plan(buf->data(), w.N, w.K, &plan)) return -1;
    return plan.n_exc;
}

int64_t vram_pack_query(const RadFile& file, const std::vector<VramPackWeight>& table, rad_weight w, bool scan,
                        int32_t* cap, int32_t* raw, std::vector<uint8_t>* buf) {
    if ((size_t)w >= table.size() || table[(size_t)w].bytes <= 0) return 0;
    const VramPackWeight& v = table[(size_t)w];
    if (cap) *cap = v.cap;
    if (!scan) return v.bytes;
    const int n = vram_pack_scan(file, v, buf);
    if (raw) *raw = n;
    /* Unreadable scales (-1) or more raw rounds than the form has room for: the weight stays block_mxfp4. */
    return n >= 0 && n <= v.cap ? v.bytes : 0;
}

int vram_pack_weight(const uint8_t* codes, const uint8_t* scales, int N, int K, int cap, uint8_t* dst,
                     int64_t dst_bytes, int* n_exc) {
    if (n_exc) *n_exc = -1;
    const size_t need = ds41hx::packed_bytes(N, K, cap);
    if (!codes || !scales || !dst || need == 0 || dst_bytes < (int64_t)need) return RAD_E_SHAPE;
    ds41hx::PackInfo plan;
    if (!ds41hx::pack_plan(scales, N, K, &plan)) return RAD_E_SHAPE;
    if (n_exc) *n_exc = plan.n_exc;
    if (plan.n_exc > cap) return RAD_E_FULL;
    if (ds41hx::pack_planes(codes, scales, N, K, cap, plan, dst, (size_t)dst_bytes) != ds41hx::kPackOk)
        return RAD_E_STATE;
    /* What the card will read: every block's e8m0 through the kernels' widening (raw rounds included, each
     * once and in order) against the planes, and the codes plane as it was. */
    ds41hx::Mx v;
    if (!ds41hx::packed_view(dst, (size_t)dst_bytes, N, K, &v) || !ds41hx::packed_scales_equal(v, scales) ||
        std::memcmp(v.codes, codes, (size_t)N * (size_t)(K / 2)) != 0)
        return RAD_E_STATE;
    return RAD_OK;
}

int vram_pack_bank_check(const Program& prog, const Plan& plan, VramPackBankReport* out, std::string* why) {
    *out = VramPackBankReport{};
    if (why) why->clear();
    const int64_t trunk = prog.meta.n_layers;
    auto draft = [&](rad_weight w) {
        if (w == 0 || (size_t)w >= prog.weights.size()) return false;
        const WeightInfo& wi = prog.weights[w];
        return wi.file_id != 0 || (trunk > 0 && wi.decl.group.layer >= trunk);
    };
    for (const OpInfo& o : prog.ops) {
        if (o.op != "moe_gemm_ggml") continue;
        bool d = false;
        for (rad_weight w : o.weights) d = d || draft(w);
        if (!d) continue;
        ++out->draft_ops;
        for (const Band& b : o.bands) {
            const Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
            if (r && r.row->info && r.row->info->name && !std::strcmp(r.row->info->name, kVramPackRow)) {
                if (why) *why = fmt("op %d, over the draft's experts, resolved %s at %s", (int)o.index,
                                    kVramPackRow, b.span.c_str());
                return RAD_E_STATE;
            }
        }
    }
    for (size_t w = 1; w < prog.weights.size(); ++w) {
        const int64_t vb = w < plan.weight.size() ? plan.weight[w].vram_bytes : 0;
        if (prog.weights[w].file_id == 0) {
            out->trunk_packed += vb > 0;
            continue;
        }
        ++out->bank_weights;
        if (w < plan.weight.size() && plan.weight[w].tier == Tier::VRAM) ++out->bank_vram;
        if (vb > 0) {
            if (why) *why = fmt("'%s', a DSpark bank's, is planned packed", prog.weights[w].name.c_str());
            return RAD_E_STATE;
        }
    }
    return RAD_OK;
}

}  /* namespace rad */
