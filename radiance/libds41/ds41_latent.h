/* Experimental opaque latent cache copies. No attention, causal selection or request commit.
 * Cache geometry is the actual core KV binding [blocks,1,B,2*head_dim]. The packing is
 * contiguous across ratio original-token slots, as state_layout.h specifies. */
#pragma once
#include "rad_abi.h"
#include "../arch/deepseek41/state_layout.h"
namespace rad::ds41::latent {
struct Launch {
    deepseek41::state::Layout layout;
    int64_t rows=0,blocks=0,slots=0,pitch=0;
    uint32_t dtype=0;
};
int prepare(const RadArgs*,bool store,Launch*);
}
extern "C" {
int ds41_latent_store(const RadArgs*,RadStream);
int ds41_latent_gather(const RadArgs*,RadStream);
int ds41_ref_latent_store(const RadArgs*,RadStream);
int ds41_ref_latent_gather(const RadArgs*,RadStream);
}
