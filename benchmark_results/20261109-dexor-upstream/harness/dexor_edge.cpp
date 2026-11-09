// Compare the C++ DeXOR port with stock Java on the DeXOREdge.java fixture.
#include "DeXOR/DeXOR.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
using std::vector;
static bool rd(std::ifstream& in,uint64_t& v,int n){unsigned char h[8];if(!in.read((char*)h,n))return false;v=0;for(int k=0;k<n;k++)v=(v<<8)|h[k];return true;}
static double rel(double a,double b){return std::abs(a-b)/std::max(1.0,std::abs(b));}
int main(int,char**argv){
 namespace dx=gef_compressors::dexor; std::ifstream in(argv[1],std::ios::binary); uint64_t t;
 size_t encOnly=0,encOnlyEq=0,vec=0,javaOk=0,bytesEq=0,decEq=0,threw=0,cppDecodeFail=0,cppLossless=0,cppOver=0;
 while(rd(in,t,4)){
  int n=int(t);vector<double> v(n);for(auto& x:v){rd(in,t,8);x=dx::detail::value(t);}
  rd(in,t,4); int32_t len=int32_t(uint32_t(t)); vec++;
  auto x=dx::compress(v); vector<uint8_t> p(x.begin()+8,x.end());
  vector<double> d; bool ok=true;
  try{d=dx::decompress(x,n);}catch(const std::exception& e){ok=false;cppDecodeFail++;std::cout<<"cpp decode fail vector "<<vec-1<<": "<<e.what()<<"\n";}
  double cm=0; if(ok) for(int i=0;i<n;i++) if(!std::isnan(v[i])||!std::isnan(d[i])) cm=std::max(cm,rel(d[i],v[i]));
  if(len==-2){rd(in,t,4);vector<uint8_t> j(t);in.read((char*)j.data(),t);encOnly++;encOnlyEq+=p==j;continue;}
  if(len<0){threw++; if(ok){bool ll=true;for(int i=0;i<n;i++)if(dx::detail::bits(d[i])!=dx::detail::bits(v[i]))ll=false;cppLossless+=ll;} continue;}
  javaOk++; vector<uint8_t> j(len); in.read((char*)j.data(),len); vector<uint64_t> jd(n); for(auto& y:jd) rd(in,y,8);
  bool be=p==j; bytesEq+=be; bool de=ok; for(int i=0;de&&i<n;i++) if(dx::detail::bits(d[i])!=jd[i]) de=false; decEq+=de;
  if(cm>1e-9) cppOver++;
  if(!be||!de){double jm=0;for(int i=0;i<n;i++) jm=std::max(jm,rel(dx::detail::value(jd[i]),v[i]));
   std::cout<<"vector "<<vec-1<<" bytes_eq="<<be<<" dec_eq="<<de<<" java_max_rel_err="<<jm<<" cpp_max_rel_err="<<cm<<"\n";}
 }
 std::cout<<"vectors="<<vec<<" java_ok="<<javaOk<<" bytes_eq="<<bytesEq<<" decoded_eq="<<decEq<<" cpp_over_1e-9_on_java_ok="<<cppOver
  <<" java_encoder_threw="<<threw<<" (cpp lossless on those="<<cppLossless<<") java_decoder_threw="<<encOnly<<" (bytes_eq="<<encOnlyEq
  <<") cpp_decode_fail="<<cppDecodeFail<<"\n";
}
