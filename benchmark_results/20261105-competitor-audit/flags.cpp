#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include "sdsl/dac_vector.hpp"
#include "sdsl/util.hpp"
#include "Buff/CompressorBuff.hpp"
#include "Gorilla/CompressorGorilla.hpp"
#include "Chimp/CompressorChimp.hpp"
#include "Chimp128/CompressorChimp128.hpp"
#include "TSXor/CompressorTSXor.hpp"
#include "Elf/CompressorElf.hpp"
#include "Camel/CompressorCamel.hpp"
#include "Falcon/CompressorFalcon.hpp"
#include <type_traits>
template<class C> uint64_t stream(const std::vector<double>& data,int decimals) {
 uint64_t bits=0;
 for(size_t first=0;first<data.size();first+=1000) {
  const size_t end=std::min(first+1000,data.size());
  std::unique_ptr<C> c;
  if constexpr(std::is_same_v<C,CompressorCamel<double>>)c=std::make_unique<C>(data[first],decimals);
  else if constexpr(std::is_same_v<C,CompressorFalcon<double>>)c=std::make_unique<C>(data[first],end-first,decimals);
  else c=std::make_unique<C>(data[first]);
  for(size_t i=first+1;i<end;++i)c->addValue(data[i]);
  c->close();bits+=c->getSize();
 }
 return bits;
}
int main(int argc,char**argv) {
 for(int a=1;a<argc;++a) {
  std::ifstream f(argv[a],std::ios::binary|std::ios::ate);size_t bytes=size_t(f.tellg());f.seekg(0);
  uint64_t n,dec;f.read(reinterpret_cast<char*>(&n),8);f.read(reinterpret_cast<char*>(&dec),8);
  n=(bytes-16)/16;
  std::vector<uint64_t> ints(n);std::vector<double> data(n);
  f.read(reinterpret_cast<char*>(ints.data()),n*8);f.read(reinterpret_cast<char*>(data.data()),n*8);
  auto emit=[&](const char*name,uint64_t bits){std::cout<<argv[a]<<","<<name<<","<<n<<","<<bits<<std::endl;};
  sdsl::dac_vector_dp<> dac(ints);emit("dac",sdsl::size_in_bytes(dac)*8);
  for(size_t i=0;i<n;++i)if(dac[i]!=ints[i])throw std::runtime_error("DAC mismatch");
  int64_t scale=1;for(size_t i=0;i<dec;++i)scale*=10;
  auto b=buff::compress_block(data.data(),n,scale);emit("buff",b.data.size()*8);
  uint64_t hash=1469598103934665603ULL;for(auto x:b.data){hash^=x;hash*=1099511628211ULL;}emit("buff_payload_hash",hash);
  emit("gorilla",stream<CompressorGorilla<double>>(data,dec));
  emit("chimp",stream<CompressorChimp<double>>(data,dec));
  emit("chimp128",stream<CompressorChimp128<double>>(data,dec));
  emit("tsx",stream<CompressorTSXor<double>>(data,dec));
  emit("elf",stream<CompressorElf<double>>(data,dec));
  emit("camel",stream<CompressorCamel<double>>(data,dec));
  emit("falcon",stream<CompressorFalcon<double>>(data,dec));
 }
}
