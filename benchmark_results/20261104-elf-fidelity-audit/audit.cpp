#include "/Users/michelangelopucci/Downloads/gef-experiments/benchmark/SElfStar/SElfStar.hpp"
#include <fstream>
#include <iostream>
namespace es=gef_compressors::elfstar;
namespace ss=gef_compressors::selfstar;
int main(int argc,char**argv){
 for(int f=1;f<argc;f++){
  std::ifstream in(argv[f],std::ios::binary); std::vector<double> v;double x;while(in.read((char*)&x,8))v.push_back(x);
  ss::Encoder stream;size_t et=0,st=0,eo=0,so=0;
  std::ofstream ef(std::string(argv[f])+".cpp.elf",std::ios::binary),sf(std::string(argv[f])+".cpp.self",std::ios::binary);
  auto write=[](std::ofstream& o,const std::vector<uint8_t>& b,size_t off){uint32_t n=b.size()-off; for(int k=3;k>=0;k--)o.put((n>>(8*k))&255);o.write((char*)b.data()+off,n);};
  for(size_t p=0;p<v.size();p+=1000){std::vector<double> w(v.begin()+p,v.begin()+std::min(v.size(),p+1000));auto e=es::compress(w),s=stream.compress_block(w);es::detail::Reader r(s);ss::detail::read_checkpoint(r,w.size());size_t offset=r.position/8;write(ef,e,8);write(sf,s,offset);et+=e.size()-8;st+=s.size()-offset;eo+=8;so+=offset;}
  std::cout<<argv[f]<<","<<v.size()<<","<<et<<","<<st<<","<<eo<<","<<so<<"\n";
 }
}
