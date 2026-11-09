#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include "sdsl/dac_vector.hpp"
#include "sdsl/util.hpp"
#include "Buff/CompressorBuff.hpp"
#include "Gorilla/CompressorGorilla.hpp"
#include "Gorilla/DecompressorGorilla.hpp"
#include "Chimp/CompressorChimp.hpp"
#include "Chimp/DecompressorChimp.hpp"
#include "Chimp128/CompressorChimp128.hpp"
#include "Chimp128/DecompressorChimp128.hpp"
#include "TSXor/CompressorTSXor.hpp"
#include "TSXor/DecompressorTSXor.hpp"
#include "Elf/CompressorElf.hpp"
#include "Elf/DecompressorElf.hpp"
#include "Camel/CompressorCamel.hpp"
#include "Camel/DecompressorCamel.hpp"
#include "Falcon/CompressorFalcon.hpp"
#include "Falcon/DecompressorFalcon.hpp"
#include <type_traits>
#include <sstream>
#include <cstring>
template<class C,class D> uint64_t stream(const std::vector<double>& data,int decimals) {
 uint64_t bits=0,mismatches=0,large_errors=0;double max_error=0;
 for(size_t first=0;first<data.size();first+=1000) {
  const size_t end=std::min(first+1000,data.size());
  std::unique_ptr<C> c;
  if constexpr(std::is_same_v<C,CompressorCamel<double>>)c=std::make_unique<C>(data[first],decimals);
  else if constexpr(std::is_same_v<C,CompressorFalcon<double>>)c=std::make_unique<C>(data[first],end-first,decimals);
  else c=std::make_unique<C>(data[first]);
  for(size_t i=first+1;i<end;++i)c->addValue(data[i]);
  c->close();bits+=c->getSize();
  auto buffer=[&](){
   if constexpr(std::is_same_v<C,CompressorTSXor<double>>)return c->bytes;
   else if constexpr(std::is_same_v<C,CompressorFalcon<double>>)return c->getOut();
   else return c->getBuffer();
  }();
  D decoder(buffer,end-first);
  for(size_t i=first;i<end;++i){
   if(i>first && !decoder.hasNext())throw std::runtime_error("short decode");
   const bool different=std::memcmp(&data[i],&decoder.storedValue,8)!=0;
   if(different && mismatches==0)std::cerr<<typeid(C).name()<<",first_mismatch,"<<i<<","<<std::hexfloat<<data[i]<<","<<decoder.storedValue<<std::defaultfloat<<"\n";
   mismatches+=different;
   const double error=std::abs(data[i]-decoder.storedValue);
   max_error=std::max(max_error,error);large_errors+=error>=1e-9;
  }
 }
 std::cerr<<typeid(C).name()<<",mismatches,"<<mismatches<<"\n";
 std::cerr<<typeid(C).name()<<",max_error,"<<max_error<<",above_1e-9,"<<large_errors<<"\n";
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
  std::cerr<<"DATASET,"<<argv[a]<<"\n";
  sdsl::dac_vector_dp<> dac(ints);emit("dac",sdsl::size_in_bytes(dac)*8);
  for(size_t i=0;i<n;++i)if(dac[i]!=ints[i])throw std::runtime_error("DAC mismatch");
  std::stringstream serialized;auto actual=dac.serialize(serialized);
  if(actual!=sdsl::size_in_bytes(dac) || serialized.str().size()!=actual)throw std::runtime_error("DAC size mismatch");
  sdsl::dac_vector_dp<> reopened;reopened.load(serialized);
  for(size_t i=0;i<n;++i)if(reopened[i]!=ints[i])throw std::runtime_error("DAC reopened mismatch");
  int64_t scale=1;for(size_t i=0;i<dec;++i)scale*=10;
  auto b=buff::compress_block(data.data(),n,scale);emit("buff",b.data.size()*8);
  std::vector<double> decoded(n);buff::decompress_block(b,decoded.data());size_t errors=0;
  for(size_t i=0;i<n;++i)errors+=std::abs(decoded[i]-data[i])>0.5/static_cast<double>(scale);
  if(errors)throw std::runtime_error("BUFF precision-bound violation");
  uint64_t hash=1469598103934665603ULL;for(auto x:b.data){hash^=x;hash*=1099511628211ULL;}emit("buff_payload_hash",hash);
  emit("gorilla",stream<CompressorGorilla<double>,DecompressorGorilla<double>>(data,dec));
  emit("chimp",stream<CompressorChimp<double>,DecompressorChimp<double>>(data,dec));
  emit("chimp128",stream<CompressorChimp128<double>,DecompressorChimp128<double>>(data,dec));
  emit("tsx",stream<CompressorTSXor<double>,DecompressorTSXor<double>>(data,dec));
  emit("elf",stream<CompressorElf<double>,DecompressorElf<double>>(data,dec));
  emit("camel",stream<CompressorCamel<double>,DecompressorCamel<double>>(data,dec));
  emit("falcon",stream<CompressorFalcon<double>,DecompressorFalcon<double>>(data,dec));
 }
}
