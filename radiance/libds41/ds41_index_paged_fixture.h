#pragma once
#include "ds41_index_fixture.h"
#include "ds41_index_paged_emulate.h"
namespace rad::ds41::paged_index::testing {
using Buffer=index::testing::Buffer;
struct Fixture {
    int64_t rows=3,keys=11,sequences=3,heads=3,dim,ratio,block,blocks_per_sequence,physical_blocks;
    std::array<std::optional<Buffer>,8> buffers;std::array<RadTensor,8> tensors{};
    std::vector<RadParam> params;std::array<int32_t,4> status{911,912,913,914};
    Fixture(uint32_t dtype=RAD_F32,int pitch=0,int64_t compression=2,int64_t block_size=4,int64_t dimension=8,
            int64_t local_heads=3,bool origins=true)
        :heads(local_heads),dim(dimension),ratio(compression),block(block_size) {
        index::testing::Fixture dense(index::Op::scores,dtype,pitch,rows,keys,4,true,dim,heads);
        params=dense.params;params.push_back(RAD_INT("ratio",ratio));params.push_back(RAD_INT("block_size",block));
        for(int i:{0,2,3,4})buffers[i]=dense.buffer(i);
        buffers[7]=dense.buffer(5);
        blocks_per_sequence=(keys*ratio+block-1)/block;physical_blocks=sequences*blocks_per_sequence+1;
        buffers[1].emplace(dtype,std::initializer_list<int64_t>{physical_blocks,1,block,dim/ratio},pitch);
        buffers[5].emplace(RAD_I32,std::initializer_list<int64_t>{sequences,blocks_per_sequence},pitch);
        if(origins){buffers[6].emplace(RAD_I32,std::initializer_list<int64_t>{sequences},pitch);for(int64_t s=0;s<sequences;++s)buffer(6).id(s,0);}
        for(int64_t i=0;i<buffer(1).count();++i)buffer(1).set(i,NAN);
        for(int64_t seq=0;seq<sequences;++seq)for(int64_t page=0;page<blocks_per_sequence;++page)
            buffer(5).id(seq*blocks_per_sequence+page,int32_t(physical_blocks-1-(seq*blocks_per_sequence+page)));
        for(int64_t seq=0;seq<sequences;++seq)for(int64_t key=0;key<keys;++key)for(int64_t d=0;d<dim;++d)
            buffer(1).set(physical_scalar(seq,key,d),dense.buffer(1).get((seq*keys+key)*dim+d));
        refresh();
    }
    int nt()const{return 8;}
    int output()const{return 7;}
    Buffer& buffer(int i){return *buffers[i];}
    const Buffer& buffer(int i)const{return *buffers[i];}
    int64_t physical_scalar(int64_t seq,int64_t key,int64_t d)const{
        const int64_t original=key*ratio,page=original/block,slot=original%block;
        const int64_t physical=buffer(5).id(seq*blocks_per_sequence+page);
        return (physical*block+slot)*(dim/ratio)+d;
    }
    void refresh(){for(int i=0;i<8;++i)if(buffers[i]){buffer(i).t.data=buffer(i).words.data();tensors[i]=buffer(i).t;}else tensors[i]={};}
    RadArgs args(){refresh();RadArgs a{};a.t=tensors.data();a.n_t=8;a.p=params.data();a.n_p=params.size();a.scratch=status.data();a.scratch_bytes=4;return a;}
    int run(){auto a=args();return emulate(&a);}
};
}
