#pragma once
#include "physical_source_bridge.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <array>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <type_traits>

namespace glm53::physical_fixture {
namespace bridge=physical_bridge;
namespace native=rad::expert_store_candidate;
inline uint64_t materializations=0,verified_bytes=0,negative_cases=0,fault_groups=0,authority_refusals=0;
inline bool negatives_done=false;
static_assert(std::is_nothrow_move_assignable_v<bridge::Result>);
inline void need(bool good,const char* why){if(!good)throw std::runtime_error(why);}
inline uint64_t slice_checksum(const std::string& path,uint64_t at,uint64_t count){
    const int fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);need(fd>=0,"physical input generator open");
    struct Close{int n;~Close(){::close(n);}}close{fd};
    uint64_t h=14695981039346656037ull;std::array<std::byte,4096> buffer{};
    for(uint64_t done=0;done<count;){const size_t n=size_t(std::min<uint64_t>(buffer.size(),count-done));size_t got=0;
        while(got<n){const auto k=::pread(fd,buffer.data()+got,n-got,off_t(at+done+got));need(k>0,"physical generator read");got+=size_t(k);}
        for(size_t i=0;i<n;++i){h^=std::to_integer<uint8_t>(buffer[i]);h*=1099511628211ull;}done+=n;
    }return h;
}
inline uint64_t separate_piece(const std::string& path,uint64_t at,uint64_t count,const std::string& destination){
    struct Pair{int a=-1,b=-1;~Pair(){if(a>=0)::close(a);if(b>=0)::close(b);}}fds;
    fds.a=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);fds.b=::open(destination.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);need(fds.a>=0&&fds.b>=0,"separate physical producer files");
    uint64_t h=14695981039346656037ull;std::array<std::byte,4096> buffer{};
    for(uint64_t done=0;done<count;){const auto n=size_t(std::min<uint64_t>(buffer.size(),count-done));size_t read=0,written=0;
        while(read<n){const auto k=::pread(fds.a,buffer.data()+read,n-read,off_t(at+done+read));need(k>0,"separate physical producer read");read+=size_t(k);}
        while(written<n){const auto k=::pwrite(fds.b,buffer.data()+written,n-written,off_t(done+written));need(k>0,"separate physical producer write");written+=size_t(k);}
        for(size_t i=0;i<n;++i){h^=std::to_integer<uint8_t>(buffer[i]);h*=1099511628211ull;}done+=n;
    }need(!::fsync(fds.b),"physical producer fsync");return h;
}
inline bridge::Catalog split(const rad::Program& program,const rad::Plan& plan,const native::SourceCatalog& source,const std::filesystem::path& root){
    bridge::Catalog input;input.source_id=source.source_id;input.revision=source.revision;input.tensors.reserve(source.tensors.size());
    const auto directory=root/"complete-physical-inputs";need(std::filesystem::create_directory(directory),"physical producer directory");size_t physical_index=0;
    for(const auto& tensor:source.tensors){bridge::Tensor t;t.expected=tensor;
        const auto unit=std::find_if(plan.units.begin(),plan.units.end(),[&](const auto& u){return u.expert>=0&&std::any_of(u.weights.begin(),u.weights.end(),[&](auto w){return program.weights[w].source==tensor.logical;});});
        need(unit!=plan.units.end(),"physical tensor has no actual Program/Plan group");t.layer=unit->layer;t.expert=unit->expert;
        const bool gu=tensor.projection==native::Projection::gate_up;const size_t parts=gu?2:1;
        for(size_t p=0;p<2;++p){const auto& plane=tensor.planes[p];
            for(size_t k=0;k<parts;++k){bridge::Piece piece;piece.part=gu?(k?bridge::Part::up:bridge::Part::gate):bridge::Part::down;
                const auto role=gu?(k?"up":"gate"):"down";piece.tensor=tensor.logical+"/"+role+"/"+(p?"scale":"codes");
                piece.path=(directory/("tensor-"+std::to_string(physical_index++)+".bin")).string();piece.offset=0;piece.bytes=plane.bytes/parts;
                piece.rows=uint64_t(tensor.shape[0])/parts;piece.row_bytes=p?4:64;piece.dtype=p?RAD_E8M0:RAD_FP4E2M1;piece.complete=true;
                piece.checksum=separate_piece(plane.path,plane.offset+k*piece.bytes,piece.bytes,piece.path);t.planes[p].push_back(std::move(piece));
            }
        }input.tensors.push_back(std::move(t));
    }return input;
}
inline void compare_planes(const native::SourceCatalog& original,const native::SourceCatalog& canonical){
    need(original.tensors.size()==canonical.tensors.size(),"physical canonical source count");
    std::array<std::byte,4096> a{},b{};
    for(size_t t=0;t<original.tensors.size();++t)for(size_t p=0;p<2;++p){const auto& x=original.tensors[t].planes[p];const auto& y=canonical.tensors[t].planes[p];
        need(original.tensors[t].logical==canonical.tensors[t].logical&&x.bytes==y.bytes&&x.checksum==y.checksum,"physical logical mapping/CRC changed");
        struct Close{int a=-1,b=-1;~Close(){if(a>=0)::close(a);if(b>=0)::close(b);}}close;
        close.a=::open(x.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);close.b=::open(y.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);need(close.a>=0&&close.b>=0,"independent canonical oracle open");const int in=close.a,out=close.b;
        for(uint64_t at=0;at<x.bytes;){const auto n=size_t(std::min<uint64_t>(a.size(),x.bytes-at));
            need(::pread(in,a.data(),n,off_t(x.offset+at))==ssize_t(n)&&::pread(out,b.data(),n,off_t(y.offset+at))==ssize_t(n),"independent full plane reads");
            need(std::equal(a.begin(),a.begin()+n,b.begin()),"physical gate/up/down assembly changed original coordinate bytes");verified_bytes+=n;at+=n;
        }
    }
}
// Defined by the checker, using only intercepted libc I/O/allocation; no bridge hooks.
void test_failures(const rad::Program&,const rad::Plan&,const bridge::Catalog&,const std::filesystem::path&,bridge::Result*);
void test_authority(const rad::Program&,const rad::Plan&,const bridge::Catalog&,const std::filesystem::path&,bridge::Result*);
inline void materialize(const rad::Program& program,const rad::Plan& plan,native::SourceCatalog* catalog,const std::filesystem::path& root){
    auto input=split(program,plan,*catalog,root);bridge::Result result;std::string why;
    const auto destination=(root/"canonical-physical").string();
    const auto status=bridge::materialize(program,plan,input,destination,&result,&why);need(status==bridge::Status::ok,why.c_str());
    need(result.source_bytes==835584&&result.output_bytes==835584&&result.physical_pieces==192&&result.window_bytes==4096&&result.metadata_reservation==8u<<20,"physical source closed receipt");
    compare_planes(*catalog,result.sources);++materializations;
    if(!negatives_done){
        const auto kept=result.sources;
        for(int k=0;k<10;++k){auto bad=input;
            if(k==0)bad.origin="checkpoint_partial_row_samples";
            if(k==1)bad.tensors[0].planes[0][0].complete=false;
            if(k==2)bad.tensors[0].planes[0][0].part=bridge::Part::up;
            if(k==3)bad.tensors[0].planes[1][0].dtype=RAD_F32;
            if(k==4)bad.tensors[0].planes[0][0].bytes=1;
            if(k==5)bad.tensors[0].planes[0][0].checksum^=1;
            if(k==6)bad.tensors.pop_back();
            if(k==8)bad.tensors[0].layer=(bad.tensors[0].layer+1)%8;
            if(k==9)bad.tensors[0].expected.shape[0]+=1;
            const auto failed_destination=k==7?destination:(root/("physical-refusal-"+std::to_string(k))).string();
            const auto failed=bridge::materialize(program,plan,bad,failed_destination,&result,&why);
            need(failed!=bridge::Status::ok&&bridge::same_catalog(result.sources,kept),"bad physical authority replaced source result");
            need(k==7||!std::filesystem::exists(failed_destination),"failed physical bundle became visible");++negative_cases;
        }
        test_failures(program,plan,input,root,&result);test_authority(program,plan,input,root,&result);negatives_done=true;
    }
    *catalog=std::move(result.sources);
}
} // namespace glm53::physical_fixture
