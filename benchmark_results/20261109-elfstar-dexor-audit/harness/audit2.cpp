// Diagnostic harness: Serf C++ Elf* (NoHuff) vs Java, our Elf* port vs Java, our DeXOR port vs Java.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#define class struct   // diagnostic only: read DeXOR Writer bit position
#include "DeXOR/DeXOR.hpp"
#undef class
#include "ElfStar/ElfStar.hpp"
#include "elf_star.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
using std::vector; using std::string;
static vector<vector<double>> load(const string& f){std::ifstream in(f,std::ios::binary);int32_t nb;in.read((char*)&nb,4);vector<vector<double>> r(nb);
 for(auto& b:r){int32_t n;in.read((char*)&n,4);b.resize(n);in.read((char*)b.data(),8*n);}return r;}
static vector<vector<uint8_t>> loadj(const string& f){std::ifstream in(f,std::ios::binary);vector<vector<uint8_t>> r;unsigned char h[4];
 while(in.read((char*)h,4)){uint32_t n=(h[0]<<24)|(h[1]<<16)|(h[2]<<8)|h[3];vector<uint8_t> b(n);in.read((char*)b.data(),n);r.push_back(b);}return r;}
template<class T> static vector<T> loadbe(const string& f){std::ifstream in(f,std::ios::binary);vector<T> r;unsigned char h[sizeof(T)];
 while(in.read((char*)h,sizeof(T))){uint64_t v=0;for(size_t k=0;k<sizeof(T);k++)v=(v<<8)|h[k];r.push_back(T(v));}return r;}
static int bit(const vector<uint8_t>& b,size_t i){return i/8<b.size()?(b[i/8]>>(7-i%8))&1:0;}
static uint64_t bitsOf(double d){uint64_t u;memcpy(&u,&d,8);return u;}
int main(int argc,char**argv){
 string name=argv[1],in=argv[2],jp=argv[3];
 auto blocks=load(in);auto jelf=loadj(jp+".elfstar"),jnh=loadj(jp+".nohuff"),jdx=loadj(jp+".dexor");
 auto jex=loadj(jp+".dexor_exact");size_t exEq=0;
 auto jbits=loadbe<uint32_t>(jp+".dexor_bits");auto jdec=loadbe<uint64_t>(jp+".dexor_dec");
 size_t portEq=0,serfPrefixOk=0,serfLossless=0,serfBytes=0,portBytes=0,serfBad=0,dxEq=0,dxBytes=0,dxBlocksExpected=0,dxBlocksOther=0,portMis=0,dxMis=0;
 size_t vpos=0; std::ostringstream notes;
 for(size_t b=0;b<blocks.size();b++){
  auto& v=blocks[b];
  // our Elf* port
  auto p=gef_compressors::elfstar::compress(v);portBytes+=p.size();
  if(vector<uint8_t>(p.begin()+8,p.end())==jelf[b])portEq++;
  auto pd=gef_compressors::elfstar::decompress(p,v.size());for(size_t i=0;i<v.size();i++)portMis+=bitsOf(pd[i])!=bitsOf(v[i]);
  // Serf C++ Elf*
  uint8_t* out=nullptr;ssize_t len=elf_star_encode(v.data(),v.size(),&out);serfBytes+=len;
  vector<uint8_t> sb;for(ssize_t w=4;w+4<=len;w+=4)for(int k=3;k>=0;k--)sb.push_back(out[w+k]); // words -> MSB-first bytes
  size_t d=0,total=sb.size()*8;while(d<total&&bit(sb,d)==bit(jnh[b],d))d++;
  bool zeros=true;for(size_t i=d;i<total;i++)if(bit(sb,i))zeros=false;
  size_t jtotal=jnh[b].size()*8; bool ok=zeros&&(d==total||bit(jnh[b],d)==1)&&jtotal>=d&&jtotal-d<=2+2+12+64+8;
  serfPrefixOk+=ok; if(!ok) notes<<name<<" block "<<b<<": serf/java nohuff first diff at bit "<<d<<"/"<<jtotal<<"\n";
  vector<double> sd(v.size()+1);ssize_t m=elf_star_decode(out,len,sd.data());
  bool ll=m==(ssize_t)v.size();for(size_t i=0;ll&&i<v.size();i++)if(bitsOf(sd[i])!=bitsOf(v[i]))ll=false;
  serfLossless+=ll; if(!ll)serfBad++;
  free(out);
  // DeXOR port vs Java, per value
  auto x=gef_compressors::dexor::compress(v);dxBytes+=x.size();
  vector<uint8_t> xp(x.begin()+8,x.end()); if(xp==jdx[b])dxEq++; if(xp==jex[b])exEq++; else notes<<name<<" block "<<b<<": port != exact-check Java\n";
  auto xd=gef_compressors::dexor::decompress(x,v.size());for(size_t i=0;i<v.size();i++)dxMis+=bitsOf(xd[i])!=bitsOf(v[i]);
  if(xp!=jdx[b]){
   gef_compressors::dexor::detail::Writer w;gef_compressors::dexor::detail::State st;size_t jb=0,cb=0;size_t first=SIZE_MAX;
   for(size_t i=0;i<v.size();i++){
    if(!st.decimal(w,v[i])){w.put(3,2);st.exception(w,bitsOf(v[i]));}
    size_t cend=w.bytes.size()*8-(w.used_?8-w.used_:0), jend=jb+jbits[vpos+i];
    bool same=(cend-cb)==(jend-jb);for(size_t k=0;same&&k<cend-cb;k++)if(bit(w.bytes,cb+k)!=bit(jdx[b],jb+k))same=false;
    if(!same){first=i;break;}
    cb=cend;jb=jend;
   }
   if(first==SIZE_MAX){notes<<name<<" block "<<b<<": DeXOR per-value equal but bytes differ\n";dxBlocksOther++;}
   else{
    bool javaInexact=jdec[vpos+first]!=bitsOf(v[first]);
    if(javaInexact)dxBlocksExpected++;else{dxBlocksOther++;
     char buf[256];snprintf(buf,sizeof buf,"%s block %zu value %zu: DeXOR diverges with exact Java reconstruction: v=%.17g prev=%.17g\n",name.c_str(),b,first,v[first],first?v[first-1]:0.0);notes<<buf;}
   }
  }
  vpos+=v.size();
 }
 std::cout<<name<<","<<blocks.size()<<",port_eq="<<portEq<<",port_bytes="<<portBytes<<",port_mis="<<portMis
  <<",serf_prefix_ok="<<serfPrefixOk<<",serf_lossless="<<serfLossless<<",serf_bytes="<<serfBytes
  <<",dexor_eq="<<dxEq<<",dexor_eq_exactjava="<<exEq<<",dexor_bytes="<<dxBytes<<",dexor_mis="<<dxMis<<",dexor_div_java_inexact="<<dxBlocksExpected<<",dexor_div_other="<<dxBlocksOther<<"\n"<<notes.str();
}
