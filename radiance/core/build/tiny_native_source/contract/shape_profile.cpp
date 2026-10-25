#include "profile.h"
#include <cstring>
namespace glm53::engine_core_contract {
int admit_declaration_shape(const RadModelMeta* m,const RadBuildCtx* c)noexcept{
    if(!m||!c||!m->arch_id||std::strcmp(m->arch_id,arch_id)||m->n_layers!=L||m->n_embd!=H||
       m->n_ff!=F||m->n_expert!=E||m->n_expert_used!=K||m->n_expert_shared||m->n_vocab!=V||
       m->n_head||m->n_head_kv||m->head_dim||m->n_ctx_train!=T||c->rank||c->world_size!=1||
       c->max_tok!=T||c->max_seqs!=1||c->max_ctx!=T||c->max_spec||c->shape_probe||
       c->max_enc_patches||c->max_audio_frames||c->max_out_rows||c->max_mixed_tok||c->cvec||
       c->tp_wire_lossy||c->tp_wire_min_bytes||c->kv_dtype||c->is_draft||(c->scope&&*c->scope)||
       c->kv_quant||c->attn_prefill_fp8||c->tp_attn_rows||c->state_dtype||c->draft_coupling)
        return RAD_E_UNSUPPORTED;
    return RAD_OK;
}
}
