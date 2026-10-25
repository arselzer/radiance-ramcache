/* Host ABI adapters for native GGUF runtime layouts. No kernel is registered by this header.
 * A future packed kernel may use abi_layout/abi_relayout/abi_unrelayout<format> as its hooks, after
 * checking its own operand role/GEMM geometry. All canonical planes, in encoding order, are
 * required. Opaque storage is U8 [logical_rows, native_row_bytes], aligned to 256 bytes.
 *
 * Layout planning never reads payloads: RadTensor.data may be null at declare time. Both copy
 * directions require dense canonical tensors. The ABI supplies geometry, not byte capacities,
 * so those source extents are a loader guarantee. The lower-level bounded View/repack API
 * should be used when the caller has separately sized buffers or pitched planes.
 */
#pragma once
#include "rad_gguf_unrepack.h"
namespace rad::gguf::detail {
inline int abi_plan(Format format,const RadEncoding* encoding,const int* selection,
                      const RadTensor* planes,int n_planes,Layout* layout) {
    if(!encoding||!selection||!planes||!layout||!geometry(format))return RAD_E_INVAL;
    const auto expected=canonical_encoding(format);
    if(!rad_enc_equal(encoding,&expected)||!rad_enc_valid(encoding))return RAD_E_DTYPE;
    if(n_planes!=encoding->n_planes||n_planes<1)return RAD_E_SHAPE;
    for(int p=0;p<n_planes;++p)if(selection[p]!=p)return RAD_E_UNSUPPORTED;
    const auto& first=planes[0];const int64_t values=encoding->plane[0].block[1];
    if(first.rank!=2||first.shape[0]<=0||first.shape[1]<=0||values<=0||first.shape[1]>INT64_MAX/values)
        return RAD_E_SHAPE;
    const int64_t rows=first.shape[0],cols=first.shape[1]*values;
    int status=plan(format,*encoding,rows,cols,{0,rows,0,cols},layout);
    if(status!=RAD_OK)return status;
    for(int p=0;p<n_planes;++p) {
        const auto& tensor=planes[p];int64_t pr=0,pc=0;
        rad_enc_plane_dims(&encoding->plane[p],rows,cols,&pr,&pc);
        if(tensor.dtype!=encoding->plane[p].dtype)return RAD_E_DTYPE;
        if(tensor.rank!=2||tensor.shape[0]!=pr||tensor.shape[1]!=pc||!rad_tensor_is_contiguous(&tensor))
            return RAD_E_SHAPE;
    }
    return RAD_OK;
}
} // namespace rad::gguf::detail
namespace rad::gguf {
template<Format format>
int abi_layout(const RadParam*,int,int,const RadEncoding* encoding,const int* selection,
                const RadTensor* planes,int n_planes,RadLayout* out) {
    if(!out)return RAD_E_INVAL;
    Layout packed;const int status=detail::abi_plan(format,encoding,selection,planes,n_planes,&packed);
    if(status!=RAD_OK)return status;
    RadLayout result{};result.tag=packed.tag;result.dtype=RAD_U8;result.rank=2;
    result.shape[0]=packed.rows;result.shape[1]=packed.row_bytes;result.align=256;result.bytes=packed.bytes;
    *out=result;return RAD_OK;
}
template<Format format>
int abi_relayout(const RadParam*,int,int,const RadEncoding* encoding,const int* selection,
                  const RadTensor* planes,int n_planes,void* dst,int64_t dst_bytes) {
    if(!dst||dst_bytes<0||uint64_t(dst_bytes)>SIZE_MAX)return RAD_E_INVAL;
    Layout packed;int status=detail::abi_plan(format,encoding,selection,planes,n_planes,&packed);
    if(status!=RAD_OK)return status;
    if(dst_bytes<packed.bytes)return RAD_E_SHAPE;
    View view;view.encoding=*encoding;view.rows=packed.rows;view.cols=packed.cols;
    for(int p=0;p<n_planes;++p) {
        if(!planes[p].data)return RAD_E_INVAL;
        const auto bytes=rad_enc_plane_bytes(&encoding->plane[p],packed.rows,packed.cols);
        if(bytes<=0||uint64_t(bytes)>SIZE_MAX)return RAD_E_SHAPE;
        view.planes[p].bytes={static_cast<const uint8_t*>(planes[p].data),size_t(bytes)};
    }
    return repack(format,view,{0,packed.rows,0,packed.cols},
                    {static_cast<uint8_t*>(dst),size_t(dst_bytes)});
}
template<Format format>
int abi_unrelayout(const RadParam*,int,int,const RadEncoding* encoding,const int* selection,
                    const void* src,int64_t src_bytes,const RadTensor* planes,int n_planes) {
    if(!src||src_bytes<0||uint64_t(src_bytes)>SIZE_MAX)return RAD_E_INVAL;
    Layout packed;int status=detail::abi_plan(format,encoding,selection,planes,n_planes,&packed);
    if(status!=RAD_OK)return status;
    if(src_bytes<packed.bytes)return RAD_E_SHAPE;
    MutableView view;view.encoding=*encoding;view.rows=packed.rows;view.cols=packed.cols;
    for(int p=0;p<n_planes;++p) {
        if(!planes[p].data)return RAD_E_INVAL;
        const auto bytes=rad_enc_plane_bytes(&encoding->plane[p],packed.rows,packed.cols);
        if(bytes<=0||uint64_t(bytes)>SIZE_MAX)return RAD_E_SHAPE;
        view.planes[p].bytes={static_cast<uint8_t*>(planes[p].data),size_t(bytes)};
    }
    return unrepack(format,{static_cast<const uint8_t*>(src),size_t(src_bytes)},view);
}
} // namespace rad::gguf
