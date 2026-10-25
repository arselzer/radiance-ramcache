/* One bit-preserving local transpose. Raw launch descriptor is <=128B; no staged Plan,
 * payload failure/status/scratch or host synchronization is needed for a copy. */
#include <hip/hip_runtime.h>
#include "ds41_reorder_heads_impl.h"
namespace {
namespace r=rad::ds41::reorder_heads;
__global__ __launch_bounds__(128) void reorder(r::Launch p) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<p.query_elements;i+=int64_t(blockDim.x)*gridDim.x)
        r::copy_element(p,i);
}
}
extern "C" int ds41_reorder_heads(const RadArgs* a,RadStream stream) {
    r::Launch p;const int rc=r::validate(a,&p);if(rc!=RAD_OK)return rc;
    const auto groups=(p.query_elements+127)/128;
    hipLaunchKernelGGL(reorder,dim3(unsigned(groups<4096?groups:4096)),dim3(128),0,static_cast<hipStream_t>(stream),p);
    return hipGetLastError()==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
