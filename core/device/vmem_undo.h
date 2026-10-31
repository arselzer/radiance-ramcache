/* Undo newly mapped granules after access protection failed. A failed driver unmap still owns
 * physical memory: never erase that granule from the actual-map ledger. Kept separate so this
 * error-only path has fault injection without replacing the process-global device backend. */
#pragma once
#include "rad_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rad {
template <class Unmap, class Forget>
int vmem_undo_new(const std::vector<int64_t>& offsets, size_t count, int64_t granule,
                  const Unmap& unmap, const Forget& forget) {
    int status = RAD_OK;
    for (size_t i = 0; i < count; ++i) {
        const int rc = unmap(offsets[i], granule);
        if (rc < 0) {
            if (status >= 0) status = rc;
        } else forget(offsets[i], granule);
    }
    return status;
}
}  // namespace rad
