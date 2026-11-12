// Compare the C++ DeXOR port with stock Java DeXOR output (per-block .dexor files from Audit2.java).
#include "DeXOR/DeXOR.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
using std::vector; using std::string;
static vector<vector<double>> load(const string& f){std::ifstream in(f,std::ios::binary);int32_t nb;in.read((char*)&nb,4);vector<vector<double>> r(nb);
 for(auto& b:r){int32_t n;in.read((char*)&n,4);b.resize(n);in.read((char*)b.data(),8*n);}return r;}
static vector<vector<uint8_t>> loadj(const string& f){std::ifstream in(f,std::ios::binary);vector<vector<uint8_t>> r;unsigned char h[4];
 while(in.read((char*)h,4)){uint32_t n=(h[0]<<24)|(h[1]<<16)|(h[2]<<8)|h[3];vector<uint8_t> b(n);in.read((char*)b.data(),n);r.push_back(b);}return r;}
static vector<uint64_t> loadbe(const string& f){std::ifstream in(f,std::ios::binary);vector<uint64_t> r;unsigned char h[8];
 while(in.read((char*)h,8)){uint64_t v=0;for(int k=0;k<8;k++)v=(v<<8)|h[k];r.push_back(v);}return r;}
int main(int argc,char**argv){
 namespace dx=gef_compressors::dexor;
 string name=argv[1]; auto blocks=load(argv[2]); string jp=argv[3];
 auto jdx=loadj(jp+".dexor"); auto jdec=loadbe(jp+".dexor_dec");
 size_t eq=0,bytes=0,payload=0,inexact=0,decEqJava=0,over=0,vpos=0,n=0; double maxrel=0;
 for(size_t b=0;b<blocks.size();b++){auto& v=blocks[b];
  auto x=dx::compress(v); bytes+=x.size(); payload+=x.size()-8;
  if(vector<uint8_t>(x.begin()+8,x.end())==jdx[b]) eq++; else std::cout<<name<<" block "<<b<<" differs from stock Java\n";
  auto d=dx::decompress(x,v.size());
  for(size_t i=0;i<v.size();i++,n++){
   decEqJava+=dx::detail::bits(d[i])==jdec[vpos+i];
   if(dx::detail::bits(d[i])!=dx::detail::bits(v[i])){inexact++;double rel=std::abs(d[i]-v[i])/std::max(1.0,std::abs(v[i]));maxrel=std::max(maxrel,rel);if(rel>1e-9)over++;}
  }
  vpos+=v.size();}
 std::cout<<name<<",blocks="<<blocks.size()<<",values="<<n<<",eq_java_blocks="<<eq<<",bytes="<<bytes<<",payload="<<payload
  <<",decoded_eq_java_decoder="<<decEqJava<<",inexact="<<inexact<<",over_1e-9="<<over<<",max_rel_err="<<maxrel<<"\n";
}
