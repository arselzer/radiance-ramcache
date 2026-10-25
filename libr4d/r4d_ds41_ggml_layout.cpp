/* r4d_ds41_ggml_layout.cpp -- the native-block GEMMs' weight layout: affine planes (a GGUF import)
 * relayouted into llama.cpp's blocks at load, through libquant's definition of them. Pure host. */
#include "r4d_plugin.h"
