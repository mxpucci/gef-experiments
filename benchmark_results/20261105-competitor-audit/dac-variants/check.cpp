#include <sdsl/dac_vector.hpp>
#include <sdsl/rrr_vector.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
int main(int argc,char**argv){
 for(int a=1;a<argc;a++){
  std::ifstream f(argv[a],std::ios::binary|std::ios::ate);size_t n=(size_t(f.tellg())-16)/16;f.seekg(16);std::vector<uint64_t> v(n);f.read((char*)v.data(),n*8);
  sdsl::dac_vector_dp<> plain(v);sdsl::dac_vector_dp<sdsl::rrr_vector<>> rrr(v);
  auto lo=*std::min_element(v.begin(),v.end());for(auto&x:v)x-=lo;
  sdsl::dac_vector_dp<> rebased(v);
  for(size_t i=0;i<n;i++)if(plain[i]!=v[i]+lo || rrr[i]!=v[i]+lo || rebased[i]!=v[i])return 2;
  std::cout<<argv[a]<<","<<n<<","<<sdsl::size_in_bytes(plain)<<","<<sdsl::size_in_bytes(rrr)<<","<<sdsl::size_in_bytes(rebased)+8<<","<<lo<<"\n";
 }
}
