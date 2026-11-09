// Diagnostic only: identical binary64 input, original encoder, different flags.
#include "alp.hpp"
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char** argv) {
 for(int a=1;a<argc;++a) {
  std::ifstream f(argv[a],std::ios::binary|std::ios::ate);
  size_t n=size_t(f.tellg())/8;f.seekg(0);
  std::vector<double> data(n),sample(alp::config::ROWGROUP_SIZE);
  f.read(reinterpret_cast<char*>(data.data()),n*8);
  uint64_t bits=0,exceptions=0,rd_vectors=0;
  n=n/1024*1024;
  for(size_t base=0;base<n;base+=alp::config::ROWGROUP_SIZE) {
   alp::state<double> state;
   alp::encoder<double>::init(data.data(),base,n,sample.data(),state);
   bool rd=state.scheme==alp::Scheme::ALP_RD;
   if(rd)alp::rd_encoder<double>::init(data.data(),base,n,sample.data(),state);
   for(size_t v=base;v<std::min(n,base+alp::config::ROWGROUP_SIZE);v+=1024) {
    alignas(64) int64_t encoded[1024];alignas(64) double exc[1024];
    alignas(64) uint16_t pos[1024],left[1024],rex[1024];
    alignas(64) uint64_t right[1024];uint16_t count=0;uint8_t width=0;int64_t for_base=0;
    if(rd) {alp::rd_encoder<double>::encode(data.data()+v,rex,pos,&count,right,left,state);
     bits+=1024*(state.right_bit_width+state.left_bit_width)+count*32;++rd_vectors;}
    else {alp::encoder<double>::encode(data.data()+v,exc,pos,&count,encoded,state);
     alp::encoder<double>::analyze_ffor(encoded,width,&for_base);bits+=1024*width+count*80;}
    exceptions+=count;
   }
  }
  std::cout<<argv[a]<<","<<n<<","<<bits<<","<<exceptions<<","<<rd_vectors<<"\n";
 }
}
