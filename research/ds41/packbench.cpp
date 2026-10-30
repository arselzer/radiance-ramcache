#include <emmintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
static inline void pack(const uint8_t* const* row, int64_t nb, void* dst) {
    const __m128i mask = _mm_set1_epi8(15);
    auto* out = static_cast<uint8_t*>(dst);
    for (int64_t b = 0; b < nb; ++b) {
        const __m128i in = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[0] + b * 16));
        const __m128i low = _mm_and_si128(in, mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(in, 4), mask);
        const __m128i first = _mm_unpacklo_epi8(low, high);
        const __m128i last = _mm_unpackhi_epi8(low, high);
        out[b * 17] = row[1][b];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + b * 17 + 1), _mm_or_si128(first, _mm_slli_epi16(last, 4)));
    }
}
using C=std::chrono::steady_clock;
static double ms(C::time_point a){return std::chrono::duration<double,std::milli>(C::now()-a).count();}
int main(int argc,char**argv){
  struct Shape{const char*n;int64_t rows,nb;} S[2]={{"GU",4608,160},{"DN",5120,72}};
  for(auto s:S){
    const int64_t blocks=s.rows*s.nb, codes=blocks*16, out=blocks*17;
    uint8_t* c=(uint8_t*)aligned_alloc(4096,(codes+4095)/4096*4096); uint8_t* e=(uint8_t*)aligned_alloc(4096,(blocks+4095)/4096*4096);
    uint8_t* d=(uint8_t*)aligned_alloc(4096,(out+4095)/4096*4096); uint8_t* d2=(uint8_t*)aligned_alloc(4096,(out+4095)/4096*4096);
    for(int64_t i=0;i<codes;++i)c[i]=uint8_t(i*131+7); for(int64_t i=0;i<blocks;++i)e[i]=uint8_t(i); memset(d,1,out); memset(d2,1,out);
    double best=1e9,bestc=1e9;
    for(int it=0;it<7;++it){ auto t=C::now();
      for(int64_t r=0;r<s.rows;++r){const uint8_t* row[2]={c+r*s.nb*16,e+r*s.nb}; pack(row,s.nb,d+r*s.nb*17);} best=std::min(best,ms(t));
      t=C::now(); memcpy(d2,d,out); bestc=std::min(bestc,ms(t)); }
    printf("%s %.2f MB: sse2 pack %.2f ms (%.1f GB/s out), memcpy %.2f ms (%.1f GB/s)\n",s.n,out/1e6,best,out/best/1e6,bestc,out/bestc/1e6);
  }
  // O_DIRECT read of a 12.5 MB extent from a scratch file
  const char* path=argc>1?argv[1]:"odirect.bin"; const int64_t N=12533760+4096, A=4096, n=(N+A-1)/A*A;
  { int fd=open(path,O_CREAT|O_WRONLY|O_TRUNC,0600); std::vector<char> z(1<<20,3); for(int64_t w=0;w<8*n;w+=z.size()) if(write(fd,z.data(),z.size())<0) return 1; fsync(fd); close(fd);}
  int fd=open(path,O_RDONLY|O_DIRECT); void* buf=aligned_alloc(A,n); double bestr=1e9;
  for(int it=0;it<6;++it){ auto t=C::now(); if(pread(fd,buf,n,(it%7)*n)!=n){perror("pread");return 1;} bestr=std::min(bestr,ms(t)); }
  printf("O_DIRECT pread %.2f MB: best %.2f ms (%.1f GB/s)\n",n/1e6,bestr,n/bestr/1e6); close(fd); unlink(path);
}
