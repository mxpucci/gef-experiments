#include <sys/types.h>
#include <cstring>
#include "elf_star.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
int main(int c,char**v){std::ifstream in(v[1],std::ios::binary);int nb;in.read((char*)&nb,4);int want=atoi(v[2]);
 for(int b=0;b<nb;b++){int n;in.read((char*)&n,4);std::vector<double> x(n);in.read((char*)x.data(),8*n);if(b!=want)continue;
  uint8_t*o;ssize_t len=elf_star_encode(x.data(),n,&o);
  for(int w=4;w<4+12;w+=4){uint32_t u;memcpy(&u,o+w,4);for(int k=31;k>=0;k--)putchar('0'+((u>>k)&1));} puts("");
  std::ifstream j(std::string(v[3]),std::ios::binary);unsigned char h[4];int bi=0;while(j.read((char*)h,4)){int m=(h[0]<<24)|(h[1]<<16)|(h[2]<<8)|h[3];std::vector<unsigned char> d(m);j.read((char*)d.data(),m);
   if(bi++==want){for(int i=0;i<12;i++)for(int k=7;k>=0;k--)putchar('0'+((d[i]>>k)&1));puts("");}}
  printf("first=%.17g bits=%016llx\n",x[0],*(unsigned long long*)&x[0]);}
}
