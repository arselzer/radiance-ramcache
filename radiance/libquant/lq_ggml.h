/* lq_ggml.h -- the `ggml` quantiser: llama.cpp's legacy, K- and I-quant formats, chosen by
 * llama.cpp's own searches and stored as the canonical affine planes. lq_ggml.cpp says how each
 * format's blocks become planes; lq_ggml_port.h is the search itself.
 */
#pragma once
#include "rad_quant.h"
#include "rad_import.h"

namespace lq {

/* The quantiser's table row, for lq_registry.cpp. Its initialiser is a constant expression, so it
 * is in place before any table that copies it is built. */
extern const RadQuantizerInfo ggml_quantizer;
int ggml_import_count();
const RadCheckpointImporterInfo* ggml_import_at(int i);

}  /* namespace lq */
