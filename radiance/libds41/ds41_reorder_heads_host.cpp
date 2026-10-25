#include "ds41_reorder_heads_impl.h"
extern "C" int ds41_ref_reorder_heads(const RadArgs* a,RadStream) noexcept {
    return rad::ds41::reorder_heads::emulate(a);
}
