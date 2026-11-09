// Full-dataset DeXOR check (no timing): new (upstream-faithful) vs previous exact port.
// Input conversion matches BenchmarkData::convert_shifted_to_double (raw / 10^decimals).
#include "DeXOR/DeXOR.hpp"
#define dexor dexor_old
#include "DeXOR_old/DeXOR.hpp"
#undef dexor
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>
int main(int,char**argv){
 std::ifstream in(argv[1],std::ios::binary); uint64_t n; in.read((char*)&n,8);
 in.seekg(0,std::ios::end); size_t sz=in.tellg(); in.seekg(8);
 int64_t dec=0; if(sz==16+8*n) in.read((char*)&dec,8);
 int64_t div=1; for(int64_t d=0;d<dec;d++) div*=10; const double divisor=double(div);
 size_t newB=0,oldB=0,inexact=0,over=0,oldMis=0,blocks=0; double maxrel=0; std::vector<int64_t> raw(1000); std::vector<double> v;
 for(size_t first=0;first<n;first+=1000){size_t c=std::min<size_t>(1000,n-first); in.read((char*)raw.data(),8*c); v.resize(c);
  for(size_t i=0;i<c;i++) v[i]=double(raw[i])/divisor;
  auto a=gef_compressors::dexor::compress(v); newB+=a.size(); auto d=gef_compressors::dexor::decompress(a,c);
  for(size_t i=0;i<c;i++) if(gef_compressors::dexor::detail::bits(d[i])!=gef_compressors::dexor::detail::bits(v[i])){inexact++;double r=std::abs(d[i]-v[i])/std::max(1.0,std::abs(v[i]));if(!(r<=1e-9)){over++;if(over<=3)fprintf(stderr,"%s over1e-9 at %zu: in=%.17g out=%.17g\n",argv[1],first+i,v[i],d[i]);}if(r>maxrel)maxrel=r;}
  auto b=gef_compressors::dexor_old::compress(v); oldB+=b.size(); auto e=gef_compressors::dexor_old::decompress(b,c);
  for(size_t i=0;i<c;i++) oldMis+=gef_compressors::dexor_old::detail::bits(e[i])!=gef_compressors::dexor_old::detail::bits(v[i]);
  blocks++;}
 const char* name=argv[1]; for(const char* p=argv[1];*p;p++) if(*p=='/') name=p+1;
 printf("%s,%llu,%zu,%zu,%zu,%.4f,%.4f,%zu,%zu,%.3g,%zu\n",name,(unsigned long long)n,blocks,newB,oldB,newB*8.0/n,oldB*8.0/n,inexact,over,maxrel,oldMis);
}
