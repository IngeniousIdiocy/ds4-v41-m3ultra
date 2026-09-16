/* Compare folded BF16 stores against the deployed projection-plus-round chain at all production shapes.
 * Exercise six-row scheduling and unaffected row counts; no model is loaded. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
typedef struct { uint16_t scale; int8_t q[32]; } block;
static uint32_t rng=71;
static uint32_t rnd(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d\n",__LINE__);return 1;}}while(0)
int main(void) {
 const uint32_t shapes[][2]={{5120,1280},{1280,32768},{5120,512},{8192,5120},{5120,2304},{2304,5120}};
 const uint32_t rows[]={6,1,2,4,8,6}; uint64_t words=0;
 CHECK(ds4_gpu_init());ds41_levers_init_from_env();
 g_ds41_levers.mtp_q8_pair6=1;g_ds41_levers.mtp_q8_stream6=1;g_ds41_levers.mtp_q8_stream6_mask=2;g_ds41_levers.mtp_q8_swizzle=1;
 CHECK(!ds41_levers_set("mtp_q8_trio_mask",-1)&&!ds41_levers_set("mtp_q8_trio_mask",128));
 for(unsigned j=0;j<sizeof(shapes)/sizeof(*shapes);j++) {
  uint32_t k=shapes[j][0],n=shapes[j][1];size_t wb=(size_t)k/32*n*sizeof(block),size=wb+256;
  void *map=NULL;CHECK(!posix_memalign(&map,getpagesize(),size));memset(map,0,size);
  block *b=(block *)((char *)map+256);
  for(size_t i=0;i<wb/sizeof(block);i++){b[i].scale=0x2800+(rnd()%1024);for(unsigned z=0;z<32;z++)b[i].q[z]=(int8_t)rnd();}
  float *x=malloc((size_t)8*k*4),*ref=malloc((size_t)8*n*4),*got=malloc((size_t)8*n*4);
  CHECK(x&&ref&&got);for(size_t i=0;i<(size_t)8*k;i++)x[i]=(int32_t)rnd()*0x1p-32f;
  ds4_gpu_tensor *xt=ds4_gpu_tensor_alloc((size_t)8*k*4),*out=ds4_gpu_tensor_alloc((size_t)8*n*4);
  CHECK(xt&&out&&ds4_gpu_set_model_map(map,size)&&ds4_gpu_tensor_write(xt,0,x,(size_t)8*k*4));
  for(unsigned r=0;r<sizeof(rows)/sizeof(*rows);r++) {
   for(unsigned arm=0;arm<2;arm++) {
    CHECK(ds41_levers_set("mtp_q8_trio_mask",80));CHECK(ds4_gpu_begin_commands());
    if(arm) CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_round_tensor(out,map,size,256,k,n,xt,rows[r]));
    else {CHECK(ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out,map,size,256,k,n,xt,rows[r]));CHECK(ds4_gpu_dsv41_quantize(out,n,rows[r],DS4_V41_BF16));}
    CHECK(ds4_gpu_end_commands()&&ds4_gpu_tensor_read(out,0,arm?got:ref,(size_t)rows[r]*n*4));
   }
   CHECK(!memcmp(ref,got,(size_t)rows[r]*n*4));words+=(size_t)rows[r]*n;
  }
  printf("PASS Q8 k=%u n=%u all row controls\n",k,n);fflush(stdout);
  ds4_gpu_tensor_free(xt);ds4_gpu_tensor_free(out);ds4_gpu_cleanup();free(x);free(ref);free(got);free(map);
  if(j+1<sizeof(shapes)/sizeof(*shapes))CHECK(ds4_gpu_init());
 }
 CHECK(ds4_gpu_init());
 { const uint32_t n=8*2304;float *x=malloc(n*4),*a=malloc(n*4),*b=malloc(n*4);CHECK(x&&a&&b);
 ds4_gpu_tensor *g=ds4_gpu_tensor_alloc(n*4),*u=ds4_gpu_tensor_alloc(n*4),*o=ds4_gpu_tensor_alloc(n*4);CHECK(g&&u&&o);
 for(uint32_t i=0;i<n;i++)x[i]=((int)(rnd()%40001)-20000)/1024.f;CHECK(ds4_gpu_tensor_write(g,0,x,n*4));
 for(uint32_t i=0;i<n;i++)x[i]=((int)(rnd()%40001)-20000)/1024.f;CHECK(ds4_gpu_tensor_write(u,0,x,n*4));
 CHECK(ds4_gpu_begin_commands());CHECK(ds4_gpu_swiglu_tensor(o,g,u,n,7,1));CHECK(ds4_gpu_dsv41_quantize(o,2304,8,DS4_V41_BF16));CHECK(ds4_gpu_end_commands());CHECK(ds4_gpu_tensor_read(o,0,a,n*4));
 CHECK(ds4_gpu_begin_commands());CHECK(ds4_gpu_dsv41_swiglu_round_tensor(o,g,u,n,7,1));CHECK(ds4_gpu_end_commands());CHECK(ds4_gpu_tensor_read(o,0,b,n*4));CHECK(!memcmp(a,b,n*4));
 ds4_gpu_tensor_free(g);ds4_gpu_tensor_free(u);ds4_gpu_tensor_free(o);free(x);free(a);free(b);ds4_gpu_cleanup();printf("PASS shared SwiGLU BF16 host path\n");}
 printf("PASS %llu exact Q8 output words\n",(unsigned long long)words);return 0;
}
