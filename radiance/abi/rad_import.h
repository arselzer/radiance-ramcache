/* rad_import.h -- lossless checkpoint block import, optionally offered by a quantiser plugin.
 *
 * This interface never chooses new numbers: it unpacks an existing checkpoint's codes, scales
 * and codebooks into canonical RadEncoding planes. It needs neither an imatrix nor float weights.
 * A recipe that quantises again remains a separate, explicit operation.
 *
 * The exports are optional, beside rad_quant_count/at. No existing ABI structure changes: older
 * plugins need not export them, and older cores ignore them. Both exports must be present or
 * absent. A (source_format, source_type) pair is unique across the loaded registry.
 */
#ifndef RAD_IMPORT_H
#define RAD_IMPORT_H
#include "rad_encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RadCheckpointImportWeight {
    const char* name;
    const char* source_type;          /* the exact registered spelling, e.g. "iq2_xs" */
    int64_t rows, cols;               /* logical row-major matrix; blocks never span rows */
} RadCheckpointImportWeight;

typedef struct RadCheckpointImporterInfo {
    const char* source_format;        /* "gguf"; private/nonstandard types need another namespace */
    const char* source_type;
    const char* doc;
    int64_t block_values, block_bytes; /* source geometry, checked against the checkpoint reader */
    int (*encoding)(const RadCheckpointImportWeight* w, RadEncoding* out);
    /* Import `rows` consecutive packed rows. The input starts at the requested row, and
     * packed_bytes is exactly rows * (cols / block_values) * block_bytes. TILED planes have
     * block rows 1 and their buffers start at the requested row; all other planes are TABLEs
     * or per-tensor constants and their buffers start at zero. All buffers arrive zeroed.
     * rows=0 and packed=null writes only shared planes, without reading the source payload.
     * Each output pointer must be non-null when that plane has data to write. Reentrant. */
    int (*import_rows)(const RadCheckpointImportWeight* w, const void* packed,
                       int64_t packed_bytes, int64_t rows, void* const* planes);
} RadCheckpointImporterInfo;

int rad_checkpoint_import_count(void);
const RadCheckpointImporterInfo* rad_checkpoint_import_at(int i);

#ifdef __cplusplus
}
#endif
#endif
