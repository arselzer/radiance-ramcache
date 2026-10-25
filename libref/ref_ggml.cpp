/* ref_ggml.cpp -- gemm_nt_ggml: a linear over llama.cpp's own blocks, read at native density.
 *
 * The weight is stored in the container as the affine planes of its GGUF type
 * (libquant/lq_ggml_planes: every block field in a plane, exactly). This row's layout hook asks for
 * all of the type's planes and relayouts them back into the type's native blocks with pack_of,
 * [N, K / block * block_bytes] bytes, so the card or host holds GGUF density (IQ1_S's planes are a
 * third larger than its blocks); the unrelayout unpacks them again. `type` names the GGUF type as
 * llama.cpp spells it ("iq1_s", "q6_K", ...).
 *
 * The arithmetic is the reference every faster row is checked against: each weight row is
 * dequantised to f32 with llama.cpp's own dequantize_row_<type> (lq_ggml_dequant, equal bit for bit
 * to the core's decode of the planes), dotted with the activation row in f32 in index order, and
 * rounded once to `dtype`. A plain float weight [N, K] (the logical form the op oracle and
 * rad-kbench pass) is read as it is, so both readings go through one dot.
 *
 * Built only when libquant's sources are beside libref's (REF_HAVE_GGML): an out-of-tree libref
 * has no ggml row.
 */
#include "ref_common.h"
#include "ref_ops.h"

#ifdef REF_HAVE_GGML
#include "lq_ggml_dequant.h"
#include "lq_ggml_planes.h"

#include <cstring>
#include <vector>

using namespace ref;

namespace {

const lq::ggml::PlaneKind* kind_named(const char* s) {
    if (!s) return nullptr;
    for (uint32_t t = 0; t < 64; ++t) {
        const lq::ggml::PlaneKind* k = lq::ggml::plane_kind(t);
        if (k && !std::strcmp(k->name, s) && lq::ggml::pack_of(t) && lq::ggml::dequant_of(t)) return k;
    }
    return nullptr;
}

int64_t pint(const RadParam* p, int n_p, const char* key) {
    const RadParam* q = rad_param_find(p, n_p, key);
    if (!q) return 0;
    if (q->kind == RAD_P_RANGE) return (int64_t)q->ihi;
    return q->kind == RAD_P_INT ? (int64_t)q->ival : 0;
}

const lq::ggml::PlaneKind* kind_of(const RadParam* p, int n_p) {
    const RadParam* t = rad_param_find(p, n_p, "type");
    return kind_named(t && t->kind == RAD_P_STR ? t->sval : nullptr);
}

/* The selection must be exactly the type's encoding, every plane in order. */
int ggml_check(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
               int n_planes, const lq::ggml::PlaneKind** out) {
    const lq::ggml::PlaneKind* k = kind_of(p, n_p);
    if (operand != 1 || !k || !enc || !sel) return RAD_E_DTYPE;
    const RadEncoding want = lq::ggml::encoding_of(k->type);
    if (!rad_enc_equal(enc, &want) || n_planes != want.n_planes) return RAD_E_DTYPE;
    for (int i = 0; i < n_planes; ++i)
        if (sel[i] != i) return RAD_E_SHAPE;
    *out = k;
    return RAD_OK;
}

/* Plane i's row r, or null for a TABLE or per-tensor plane (pack and unpack do not touch those). */
uint8_t* plane_row(const RadEncoding* enc, const RadTensor* planes, int i, int64_t r) {
    const RadEncPlane& pl = enc->plane[i];
    if (pl.kind == RAD_PLANE_TABLE || (pl.block[0] == 0 && pl.block[1] == 0)) return nullptr;
    const int64_t cols = planes[i].rank >= 2 ? planes[i].shape[1] : 0;
    return (uint8_t*)planes[i].data + r * rad_enc_row_bytes(pl.dtype, cols);
}

int64_t logical_rows(const RadTensor* planes) { return planes[0].rank >= 1 ? planes[0].shape[0] : 0; }

}  // namespace

extern "C" int ref_ggml_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                               const int* sel, const RadTensor* planes, int n_planes, RadLayout* out) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = ggml_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = pint(p, n_p, "K"), rows = logical_rows(planes);
    if (K <= 0 || K % k->block || rows <= 0) return RAD_E_SHAPE;
    static thread_local char tag[32];
    snprintf(tag, sizeof tag, "ggml_%s", k->name);
    std::memset(out, 0, sizeof *out);
    out->tag = tag;
    out->dtype = RAD_U8;
    out->rank = 2;
    out->shape[0] = rows;
    out->shape[1] = K / k->block * (int64_t)k->block_bytes;
    out->align = 64;
    out->bytes = out->shape[0] * out->shape[1];
    return RAD_OK;
}

extern "C" int ref_ggml_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                 const int* sel, const RadTensor* planes, int n_planes, void* dst,
                                 int64_t dst_bytes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = ggml_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = pint(p, n_p, "K"), rows = logical_rows(planes);
    const int64_t nb = K / k->block, rb = nb * (int64_t)k->block_bytes;
    if (K <= 0 || K % k->block || rows <= 0 || dst_bytes < rows * rb) return RAD_E_SHAPE;
    const lq::ggml::PackFn pack = lq::ggml::pack_of(k->type);
    std::vector<const uint8_t*> row((size_t)n_planes);
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = plane_row(enc, planes, i, r);
        pack(row.data(), nb, (uint8_t*)dst + r * rb);
    }
    return RAD_OK;
}

extern "C" int ref_ggml_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const void* src, int64_t src_bytes,
                                   const RadTensor* planes, int n_planes) {
    if (n_planes <= 1) return RAD_E_UNSUPPORTED;
    const lq::ggml::PlaneKind* k = nullptr;
    const int rc = ggml_check(p, n_p, operand, enc, sel, n_planes, &k);
    if (rc < 0) return rc;
    const int64_t K = pint(p, n_p, "K"), rows = logical_rows(planes);
    const int64_t nb = K / k->block, rb = nb * (int64_t)k->block_bytes;
    if (K <= 0 || K % k->block || rows <= 0 || src_bytes < rows * rb) return RAD_E_SHAPE;
    std::vector<uint8_t*> row((size_t)n_planes);
    std::vector<void*> all((size_t)n_planes);
    for (int i = 0; i < n_planes; ++i) all[(size_t)i] = planes[i].data;
    for (int64_t r = 0; r < rows; ++r) {
        for (int i = 0; i < n_planes; ++i) row[(size_t)i] = plane_row(enc, planes, i, r);
        k->unpack((const uint8_t*)src + r * rb, nb, row.data());
    }
    lq::ggml::write_shared(k->type, *enc, all.data());
    return RAD_OK;
}

/* y[m, n] = sum_k a[m, k] w[n, k]: w native blocks [N, K/block*block_bytes] u8 dequantised a row at
 * a time, or a plain float [N, K]; f32 accumulation in index order, rounded once. */
int ref_gemm_nt_ggml(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    const lq::ggml::PlaneKind* k = kind_of(a->p, a->n_p);
    const int64_t M = band_rows(x.rank >= 1 ? x.shape[0] : 0, p_int(a, "M", 0));
    const int64_t N = p_int(a, "N", 0), K = p_int(a, "K", 0);
    if (!k || M <= 0 || N <= 0 || K <= 0 || K % k->block || x.rank != 2 || x.shape[1] != K ||
        y.rank != 2 || y.shape[1] != N || y.shape[0] < M || w.rank != 2 || w.shape[0] != N)
        return RAD_E_SHAPE;
    const int64_t rb = K / k->block * (int64_t)k->block_bytes;
    const bool native = w.dtype == RAD_U8;
    if (native ? w.shape[1] != rb : w.shape[1] != K) return RAD_E_SHAPE;
    const lq::ggml::DequantFn dq = lq::ggml::dequant_of(k->type);
#pragma omp parallel
    {
        std::vector<float> wr((size_t)K);
#pragma omp for schedule(static)
        for (int64_t n = 0; n < N; ++n) {
            if (native)
                dq((const uint8_t*)w.data + n * w.stride[0], wr.data(), K);
            else
                for (int64_t i = 0; i < K; ++i) wr[(size_t)i] = ldt(&w, rowoff(&w, n, K) + i * laststride(&w));
            for (int64_t m = 0; m < M; ++m) {
                const int64_t xb = rowoff(&x, m, K), xs = laststride(&x);
                float acc = 0.0f;
                for (int64_t i = 0; i < K; ++i) acc += ldt(&x, xb + i * xs) * wr[(size_t)i];
                stt(&y, rowoff(&y, m, N) + n * laststride(&y), acc);
            }
        }
    }
    return RAD_OK;
}

/* Values one block of GGUF type `type` covers, or 0 for a type this row does not read. */
extern "C" int64_t ref_ggml_block(const char* type) {
    const lq::ggml::PlaneKind* k = kind_named(type);
    return k ? k->block : 0;
}

#endif /* REF_HAVE_GGML */
