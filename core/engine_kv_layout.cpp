/* Default-off, bounded exact diagnostic receipt. humanb() rounds GiB displays and cannot
 * authorize the physical MAX used by a later growing-KV process. No inference arithmetic. */
#include "engine_priv.h"
#include "mem/kv_layout_receipt.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>

namespace rad {
namespace {
std::string name_hex(const std::string& value) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size()*2);
    for (unsigned char c : value) { out += digits[c >> 4]; out += digits[c & 15]; }
    return out;
}
bool layer_stride(const KVGroupPlan& p, int64_t* out) {
    if (p.n_layers <= 0 || p.bytes_per_block < 0 || p.bytes_per_state < 0 || p.n_slots < 0 ||
        p.bytes_per_block % p.n_layers || p.bytes_per_state % p.n_layers) return false;
    return kv_layout_mul(p.paged() ? p.pool_blocks() : p.n_states,
        (p.paged() ? p.bytes_per_block : p.bytes_per_state) / p.n_layers, out);
}
}
int Engine::kv_layout_receipt() {
    const int flag = kv_layout_flag(std::getenv(kKVLayoutEnv));
    if (flag == 0) return RAD_OK; // flag-off startup emits nothing and does no inventory/device read
    if (flag < 0 || cfg_.tp != 4 || ranks_.size() != 4 || !meta_.arch_id ||
        std::strcmp(meta_.arch_id, "deepseek41") || cfg_.max_seqs != 1 || cfg_.n_spec != 0 ||
        !cfg_.deterministic) return RAD_E_INVAL;
    int64_t payload = 0, bitmap = 0, metadata = 0;
    int64_t late = 0;
    if (!kv_layout_add(late_step_bytes_, late_lazy_bytes_, &late)) return RAD_E_INVAL;
    if (ck_store_.bitmap_capacity_words() > uint64_t(std::numeric_limits<int64_t>::max()) ||
        !kv_layout_mul(ck_store_.slot_bytes(), ck_store_.capacity(), &payload) ||
        !kv_layout_mul((int64_t)ck_store_.bitmap_capacity_words(), sizeof(uint64_t), &bitmap) ||
        !kv_layout_add(bitmap, sizeof(HostArena), &metadata)) return RAD_E_INVAL;
    // Native bounds keep the receipt below ~8K rows; malformed plugin metadata is refused.
    std::array<int64_t, 4> kh_caps{};
    for (size_t rank = 0; rank < ranks_.size(); ++rank) {
        const auto& p = ranks_[rank];
        const auto& r = *p;
        int64_t logits = 0;
        if (r.kv.plans().size() < 2 || r.kv.plans().size() > 64 || r.program.kv_groups.size() != r.kv.plans().size() ||
            r.program.buffers.empty() || r.program.buffers.size() > 2048 || r.program.arena_bytes < 0 ||
            r.program.arena_fixed_bytes < 0 || r.program.host_arena_bytes < 0 || r.program.scratch_bytes < 0 ||
            !kv_layout_mul(r.sampler.config().n_vocab_local, 4, &logits))
            return RAD_E_INVAL;
        kh_caps[rank] = logits;
        for (size_t g = 1; g < r.kv.plans().size(); ++g) {
            const auto& plan = r.kv.plans()[g];
            const auto& group = r.program.kv_groups[g];
            int64_t covered = 0, stride = 0;
            if (group.name.size() > 256 || group.layers.size() > 128 || plan.n_layers < 1 ||
                uint64_t(plan.n_layers) != group.layers.size() || plan.zero || plan.streamed() ||
                plan.n_blocks < 0 || plan.n_states < 0 || plan.block_size < 0 ||
                !kv_layout_extent(plan.pool_offset, plan.pool_bytes, r.pools.kv().bytes()) ||
                !layer_stride(plan, &stride) || !kv_layout_mul(stride, plan.n_layers, &covered) || covered > plan.pool_bytes)
                return RAD_E_INVAL;
            kh_caps[rank] = std::max(kh_caps[rank], stride);
        }
        for (size_t h = 1; h < r.program.buffers.size(); ++h) {
            const auto& b = r.program.buffers[h];
            const int64_t cap = b.decl.domain == RAD_DOMAIN_HOST ? r.program.host_arena_bytes : r.program.arena_bytes;
            if ((b.decl.domain != RAD_DOMAIN_HOST && b.decl.domain != RAD_DOMAIN_DEVICE) ||
                b.name.size() > 256 || b.bytes < 0 || b.arena_offset < -1 ||
                (b.arena_offset >= 0 && !kv_layout_extent(b.arena_offset, b.bytes, cap))) return RAD_E_INVAL;
        }
    }
    RAD_INFO("DS41_KV_SNAPSHOT enabled=%d slots=%lld slot_bytes=%lld payload_bytes=%lld "
             "bitmap_capacity_words=%llu bitmap_payload_bytes=%lld object_bytes=%llu metadata_payload_bytes=%lld "
             "allocator_headers_untracked=1 prefix_map_metadata_untracked=1",
             (int)ck_store_.enabled(), (long long)ck_store_.capacity(), (long long)ck_store_.slot_bytes(),
             (long long)payload, (unsigned long long)ck_store_.bitmap_capacity_words(), (long long)bitmap,
             (unsigned long long)sizeof(HostArena), (long long)metadata);
    for (size_t rank = 0; rank < ranks_.size(); ++rank) {
        const auto& p = ranks_[rank];
        const auto& r = *p;
        RAD_TRY(rad_dev_set(r.device));
        const auto& pool = r.pools.kv();
        const int64_t G = pool.backing_granule_bytes(), reserved = pool.backing_reserved_bytes();
        if (!pool.elastic() || G <= 0 || reserved <= 0 || reserved % G || pool.bytes() <= 0 ||
            pool.bytes() > reserved || pool.used() < 0 || pool.used() > pool.bytes() ||
            pool.committed() < 0 || pool.committed() > reserved || pool.committed() % G) {
            (void)rad_dev_set(ranks_[0]->device); return RAD_E_UNSUPPORTED;
        }
        RadDeviceProps props{};
        const int rc = rad_dev_props(r.device, &props);
        if (rc < 0) { (void)rad_dev_set(ranks_[0]->device); return rc; }
        if (props.is_host_backend || props.vram_bytes <= 0 || props.vram_free < 0) {
            (void)rad_dev_set(ranks_[0]->device); return RAD_E_UNSUPPORTED;
        }
        RAD_INFO("DS41_KV_LAYOUT rank=%d device=%d backend_vmem=1 grow=%d max_ctx=%lld max_tok=%lld max_seqs=%lld "
                 "logical=%lld carved=%lld reserved=%lld granule=%lld committed=%lld base=%p "
                 "arena_device=%lld arena_fixed=%lld arena_host=%lld scratch=%lld checkpoint_shard=%lld "
                 "kh_pinned_capacity_per_rank=%lld group_count=%zu buffer_count=%zu vram_total=%lld vram_free=%lld "
                 "late_step=%lld late_lazy=%lld late_total=%lld hash_logit_bytes=%lld prefix=%d checkpoint_interval=%lld "
                 "capture_driver_bytes_untracked=1",
                 r.index, r.device, (int)bool(kv_growth_), (long long)cfg_.max_ctx, (long long)cfg_.max_tok,
                 (long long)cfg_.max_seqs, (long long)pool.bytes(), (long long)pool.used(), (long long)reserved,
                 (long long)G, (long long)pool.committed(), pool.base(), (long long)r.program.arena_bytes,
                 (long long)r.program.arena_fixed_bytes, (long long)r.program.host_arena_bytes,
                 (long long)r.program.scratch_bytes, (long long)r.kv.checkpoint_bytes(), (long long)kh_caps[rank],
                 r.kv.plans().size()-1, r.program.buffers.size()-1, (long long)props.vram_bytes,
                 (long long)props.vram_free, (long long)late_step_bytes_, (long long)late_lazy_bytes_,
                 (long long)late, (long long)r.sampler.config().n_vocab_local * 4, (int)cfg_.prefix_cache,
                 (long long)geo_.checkpoint_interval);
        for (size_t g = 1; g < r.kv.plans().size(); ++g) {
            const auto& a = r.kv.plans()[g]; const auto& b = r.program.kv_groups[g];
            std::string layers;
            for (int l : b.layers) layers += (layers.empty() ? "" : ",") + std::to_string(l);
            RAD_INFO("DS41_KV_LAYOUT_GROUP rank=%d group=%zu kind=%d format=%u layers=%lld layer_ids=%s "
                     "block=%lld blocks=%lld states=%lld page=%lld state=%lld offset=%lld bytes=%lld stride=%lld name_hex=%s",
                     r.index, g, a.kind, a.storage_format, (long long)a.n_layers, layers.c_str(),
                     (long long)a.block_size, (long long)a.n_blocks, (long long)a.n_states,
                     (long long)a.bytes_per_block, (long long)a.bytes_per_state, (long long)a.pool_offset,
                     (long long)a.pool_bytes, (long long)a.layer_stride(), name_hex(b.name).c_str());
        }
        for (size_t h = 1; h < r.program.buffers.size(); ++h) {
            const auto& b = r.program.buffers[h];
            RAD_INFO("DS41_KV_LAYOUT_BUFFER rank=%d handle=%zu kind=%d domain=%d dtype=%u offset=%lld bytes=%lld name_hex=%s",
                     r.index, h, b.decl.kind, b.decl.domain, b.decl.dtype, (long long)b.arena_offset,
                     (long long)b.bytes, name_hex(b.name).c_str());
        }
    }
    return rad_dev_set(ranks_[0]->device);
}
} // namespace rad
