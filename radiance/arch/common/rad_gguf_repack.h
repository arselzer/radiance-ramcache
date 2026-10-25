/* Host-only lossless bridge to original GGUF blocks for future packed kernels.
 * plan() validates metadata/geometry without needing payloads. repack() validates every input
 * extent, fixed table and selected block before touching output. The model/loader still owns
 * source lifetime and synchronization; immutable source bytes are required during this call.
 * There is no GPU kernel registration here and no unsupported-format fallback. */
#pragma once
#include "rad_gguf_repack_base.h"
#include "rad_gguf_repack_iq.h"
#include "rad_gguf_repack_dense.h"
namespace rad::gguf {
inline int repack(Format format,const View& input,const Region& region,std::span<uint8_t> output,
                    std::string* why=nullptr) {
    Layout layout;int status=plan(format,input.encoding,input.rows,input.cols,region,&layout,why);
    if(status!=RAD_OK)return status;
    if(uint64_t(layout.bytes)>SIZE_MAX||uint64_t(layout.bytes)>output.size()||!output.data())
        return refuse(RAD_E_SHAPE,"packed output buffer is missing or too small",why);
    Reader reader(input);status=reader.init(why);
    if(status!=RAD_OK)return status;
    for(int p=0;p<input.encoding.n_planes;++p)
        if(detail::overlaps(output,input.planes[p].bytes))
            return refuse(RAD_E_INVAL,"packed output overlaps a canonical input plane",why);
    const bool dense=format==Format::q5_k||format==Format::q8_0;
    status=dense?detail::dense_validate(format,reader,region,why):detail::iq_validate(format,reader,region,why);
    if(status!=RAD_OK)return status;
    if(dense)detail::dense_write(format,reader,region,output.data());
    else detail::iq_write(format,reader,region,output.data());
    if(why)why->clear();
    return RAD_OK;
}
} // namespace rad::gguf
