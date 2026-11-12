#include <sys/types.h>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "elf_star.h"
static void t(const char* name,std::vector<double> v){uint8_t*o;ssize_t len=elf_star_encode(v.data(),v.size(),&o);std::vector<double> d(v.size());ssize_t n=elf_star_decode(o,len,d.data());
 int bad=0;for(size_t i=0;i<v.size();i++)if(memcmp(&v[i],&d[i],8)){bad++;if(bad<3)printf("  %s[%zu]: in=%g out=%g\n",name,i,v[i],d[i]);}printf("%s: n=%zd mismatches=%d\n",name,n,bad);free(o);}
int main(){t("zero-first",{0.0,1.5,2.25,3.0});t("nan-middle",{1.5,NAN,2.5,3.5});t("negzero",{-0.0,1.0,-0.0});t("inf",{INFINITY,1.0,-INFINITY});}
