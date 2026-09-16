/* Full production projection shape and row-count/tensor-offset controls. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) {fprintf(stderr,"FAIL line %d\n",__LINE__);return 1;} } while(0)
static uint32_t seed=7731;static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int main(void){
 const uint32_t k=6144,n=25600,maxrows=8;const size_t offset=4096,bytes=offset+(size_t)k*n*2;
 void *map=NULL;CHECK(!posix_memalign(&map,getpagesize(),bytes));memset(map,0,offset);
 uint16_t *w=(uint16_t*)((char*)map+offset);for(size_t i=0;i<(size_t)k*n;i++)w[i]=(uint16_t)(0x2000+rnd()%8192)|((rnd()&1)?0x8000:0);
 CHECK(ds4_gpu_init());ds41_levers_init_from_env();CHECK(ds4_gpu_set_model_map(map,bytes));
 ds4_gpu_tensor *xb=ds4_gpu_tensor_alloc(128+(size_t)maxrows*k*4),*ob=ds4_gpu_tensor_alloc(256+((size_t)maxrows*n+64)*4);
 CHECK(xb&&ob);ds4_gpu_tensor *x=ds4_gpu_tensor_view(xb,128,(size_t)maxrows*k*4),*o=ds4_gpu_tensor_view(ob,256,(size_t)maxrows*n*4);CHECK(x&&o);
 float *xx=malloc((size_t)maxrows*k*4);uint32_t *a=malloc(256+((size_t)maxrows*n+64)*4),*b=malloc(256+((size_t)maxrows*n+64)*4);CHECK(xx&&a&&b);size_t checked=0;
 for(unsigned draw=0;draw<20;draw++){
  for(size_t i=0;i<(size_t)maxrows*k;i++)xx[i]=((int)(rnd()%20001)-10000)/8192.f;
  CHECK(ds4_gpu_tensor_write(x,0,xx,(size_t)maxrows*k*4));
  for(unsigned rows=1;rows<=maxrows;rows++){
   for(unsigned arm=0;arm<2;arm++){
    CHECK(ds41_levers_set("mtp_f16_trio",arm));
    for(size_t i=0;i<128+(size_t)maxrows*n;i++)b[i]=0x7fc12345;
    CHECK(ds4_gpu_tensor_write(ob,0,b,256+((size_t)maxrows*n+64)*4));
    CHECK(ds4_gpu_begin_commands());CHECK(ds4_gpu_dsv41_projection_rows(o,map,bytes,offset,k,n,rows,x));CHECK(ds4_gpu_end_commands());
    CHECK(ds4_gpu_tensor_read(ob,0,arm?b:a,256+((size_t)maxrows*n+64)*4));
   }
   for(size_t i=0;i<128+(size_t)maxrows*n;i++){
    CHECK(a[i]==b[i]);
    if(i<64||i>=64+(size_t)rows*n)CHECK(b[i]==0x7fc12345);
    else CHECK(b[i]!=0x7fc12345 && (b[i]&0x7f800000u)!=0x7f800000u);
   }
   checked+=(size_t)rows*n;
  }
 }
 printf("PASS F16 trio production shape rows1..8, tensor offsets/tails, %zu finite exact output words\n",checked);
 ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(o);ds4_gpu_tensor_free(xb);ds4_gpu_tensor_free(ob);ds4_gpu_cleanup();free(xx);free(a);free(b);free(map);return 0;
}
