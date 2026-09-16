/* Same router matvec reduction in a batched grid, including tensor offsets. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d\n",__LINE__);return 1;}}while(0)
static uint32_t seed=94173;static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int main(void){const unsigned k=5120,n=384,nmax=8;const size_t off=4096,bytes=off+(size_t)k*n*4,ow=(size_t)nmax*n+128;
 void*map;CHECK(!posix_memalign(&map,getpagesize(),bytes));memset(map,0,off);float*w=(float*)((char*)map+off);for(size_t i=0;i<(size_t)k*n;i++)w[i]=((int)(rnd()%20001)-10000)/32768.f;
 CHECK(ds4_gpu_init());ds41_levers_init_from_env();CHECK(ds4_gpu_set_model_map(map,bytes));
 ds4_gpu_tensor*xb=ds4_gpu_tensor_alloc(128+(size_t)nmax*k*4),*ob=ds4_gpu_tensor_alloc(ow*4);CHECK(xb&&ob);
 ds4_gpu_tensor*x=ds4_gpu_tensor_view(xb,128,(size_t)nmax*k*4),*o=ds4_gpu_tensor_view(ob,256,(size_t)nmax*n*4);CHECK(x&&o);
 float*xx=malloc((size_t)nmax*k*4);uint32_t*a=malloc(ow*4),*b=malloc(ow*4);CHECK(xx&&a&&b);
 for(int draw=0;draw<64;draw++){for(size_t i=0;i<(size_t)nmax*k;i++)xx[i]=((int)(rnd()%20001)-10000)/8192.f;CHECK(ds4_gpu_tensor_write(x,0,xx,(size_t)nmax*k*4));
 for(unsigned rows=1;rows<=nmax;rows++){for(int arm=0;arm<2;arm++){CHECK(ds41_levers_set("mtp_f32_rows",arm));for(size_t i=0;i<ow;i++)b[i]=0x7fc12345;CHECK(ds4_gpu_tensor_write(ob,0,b,ow*4));CHECK(ds4_gpu_begin_commands());
 if(arm){CHECK(ds4_gpu_matmul_f32_tensor(o,map,bytes,off,k,n,x,rows));}else for(unsigned j=0;j<rows;j++){ds4_gpu_tensor*vx=ds4_gpu_tensor_view(x,(size_t)j*k*4,k*4),*vo=ds4_gpu_tensor_view(o,(size_t)j*n*4,n*4);CHECK(vx&&vo);CHECK(ds4_gpu_matmul_f32_tensor(vo,map,bytes,off,k,n,vx,1));ds4_gpu_tensor_free(vx);ds4_gpu_tensor_free(vo);}
 CHECK(ds4_gpu_end_commands());CHECK(ds4_gpu_tensor_read(ob,0,arm?b:a,ow*4));}
 for(size_t i=0;i<ow;i++){CHECK(a[i]==b[i]);if(i<64||i>=64+rows*n)CHECK(b[i]==0x7fc12345);else CHECK((b[i]&0x7f800000u)!=0x7f800000u);}
 }}printf("PASS router F32 rows1..8, 64 draws, offsets and guard words exact\n");return 0;}
