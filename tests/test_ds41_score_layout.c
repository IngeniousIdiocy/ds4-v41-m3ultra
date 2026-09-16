/* Exact full-score comparison through the production Metal entry points.
 * No weights/model load. Exercise odd tails, masks, scratch reuse and offsets. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static uint32_t state=731;
static float rnd(void) { state^=state<<13;state^=state>>17;state^=state<<5;return (int32_t)state*0x1p-30f; }
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"FAIL line %d\n",__LINE__);return 1; } } while(0)
int main(void) {
 CHECK(ds4_gpu_init()); ds41_levers_init_from_env();
 const uint32_t lengths[]={1,7,33,511,512,513,8193,16385,32769,131073,300005};
 uint64_t compared=0;
 for(unsigned c=0;c<sizeof(lengths)/sizeof(*lengths);c++) {
  uint32_t n=lengths[c];size_t kb=(size_t)n*128*4,sb=(size_t)n*4,mb=((n+7)/8)*4;
  float q[4096],w[32];float *k=malloc(kb),*m=malloc(mb),*ref=malloc(sb),*got=malloc(sb);
  for(unsigned i=0;i<4096;i++)q[i]=rnd();for(unsigned i=0;i<32;i++)w[i]=rnd();
  for(size_t i=0;i<kb/4;i++) k[i]=rnd();
  ds4_gpu_tensor *base=ds4_gpu_tensor_alloc(kb+256),*kt=ds4_gpu_tensor_view(base,256,kb);
  ds4_gpu_tensor *qt=ds4_gpu_tensor_alloc(sizeof(q)),*wt=ds4_gpu_tensor_alloc(sizeof(w)),*mt=ds4_gpu_tensor_alloc(mb),*out=ds4_gpu_tensor_alloc(sb);
  CHECK(base&&kt&&qt&&wt&&mt&&out);
  CHECK(ds4_gpu_tensor_write(kt,0,k,kb)&&ds4_gpu_tensor_write(qt,0,q,sizeof(q))&&ds4_gpu_tensor_write(wt,0,w,sizeof(w)));
  for(int mode=0;mode<5;mode++) {
   for(size_t i=0;i<mb/4;i++)m[i]=mode==1?0.f:mode==2?1.f:mode==3?(i%3?1.f:0.f):(i%17?1.f:0.f);
   CHECK(ds4_gpu_tensor_write(mt,0,m,mb));
   for(int arm=0;arm<2;arm++) {
    g_ds41_levers.index_score_stream=arm;CHECK(ds4_gpu_begin_commands());
    CHECK(mode==0 ? ds4_gpu_dsv41_indexer_score_one_tensor(out,qt,wt,kt,n,32,128,1.f/64.f,false):ds4_gpu_dsv41_indexer_score_masked(out,qt,wt,kt,mt,n,mode>=2));
    CHECK(ds4_gpu_end_commands());CHECK(ds4_gpu_tensor_read(out,0,arm?got:ref,sb));
   }
   size_t bad=0;for(size_t i=0;i<n;i++)if(memcmp(ref+i,got+i,4)){if(!bad)fprintf(stderr,"n=%u mode=%d row=%zu expected=%a got=%a\n",n,mode,i,ref[i],got[i]);bad++;}
   CHECK(!bad);compared+=n;
   if (mode==0) {
    // GLM keeps its original entry point even with the V4.1 lever enabled.
    CHECK(ds4_gpu_glm_indexer_score_one_tensor(out,qt,wt,kt,n,32,128,1.f/64.f,false));
    CHECK(ds4_gpu_tensor_read(out,0,got,sb));CHECK(!memcmp(ref,got,sb));
   }
  }
  printf("PASS n=%u all five masks/layouts\n",n);fflush(stdout);
  ds4_gpu_tensor_free(out);ds4_gpu_tensor_free(mt);ds4_gpu_tensor_free(wt);ds4_gpu_tensor_free(qt);ds4_gpu_tensor_free(kt);ds4_gpu_tensor_free(base);free(k);free(m);free(ref);free(got);
 }
 printf("PASS %llu exact score words\n",(unsigned long long)compared);ds4_gpu_cleanup();return 0;
}
