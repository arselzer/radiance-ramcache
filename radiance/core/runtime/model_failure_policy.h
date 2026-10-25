/* Shared, device-free admission rules used by the actual Engine and tests.
 * This is not a collective cancellation protocol or a quiescence proof. */
#pragma once
#include "../rad_core.h"
namespace rad {
inline bool checked_model_profile(const Program& p, bool collectors) {
    return collectors || p.requires_permanent_model_failure();
}
inline int model_profile_admission(const Program& p, bool collectors,
                                   int world, int n_spec) {
    if (!checked_model_profile(p, collectors)) return RAD_OK;
    if (world != 1 || n_spec != 0) return RAD_E_UNSUPPORTED;
    if (p.requires_permanent_model_failure() &&
        (p.is_pass_driven() || p.encoder.modalities || p.audio_encoder.modalities))
        return RAD_E_UNSUPPORTED; // Gen/media paths have no verified protocol yet.
    return RAD_OK;
}
} // namespace rad
