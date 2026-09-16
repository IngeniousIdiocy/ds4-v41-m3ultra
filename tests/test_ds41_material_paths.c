/* Exercise production host paths against their runtime-disabled controls. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d\n",__LINE__);return 1;}}while(0)
static uint32_t state=419;static uint32_t rnd(void){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
static float val(void){return ((int)(rnd()%20001)-10000)/4096.f;}
int main(void){
 CHECK(ds4_gpu_init());ds41_levers_init_from_env();
 const size_t sz=4096ull/32*8192*34+16384;void *map=NULL;CHECK(!posix_memalign(&map,getpagesize(),sz));memset(map,0,sz);
 float *bias=(float*)map;for(int i=0;i<384;i++)bias[i]=val();
 CHECK(ds4_gpu_set_model_map(map,sz));
 ds4_gpu_tensor *log=ds4_gpu_tensor_alloc(8*384*4),*sel=ds4_gpu_tensor_alloc(8*6*4),*weights=ds4_gpu_tensor_alloc(8*6*4),*probs=ds4_gpu_tensor_alloc(8*384*4),*tokens=ds4_gpu_tensor_alloc(8*4);
 CHECK(log&&sel&&weights&&probs&&tokens);uint32_t zero[8]={0};CHECK(ds4_gpu_tensor_write(tokens,0,zero,sizeof(zero)));
 float inputs[8*384];uint32_t ref[8*396],got[8*396];unsigned long words=0;
 for(unsigned draw=0;draw<100;draw++){
  for(int i=0;i<8*384;i++)inputs[i]=draw%10==0?0:val();if(draw%10==0)memset(bias,0,384*4);else for(int i=0;i<384;i++)bias[i]=val();
  CHECK(ds4_gpu_tensor_write(log,0,inputs,sizeof(inputs)));
  for(unsigned rows=1;rows<=8;rows++)for(unsigned arm=0;arm<2;arm++){
   CHECK(ds41_levers_set("router_simd",arm));CHECK(ds41_levers_set("router_hier",arm));CHECK(ds4_gpu_begin_commands());
   CHECK(ds4_gpu_router_select_batch_tensor(sel,weights,probs,map,sz,0,0,0,1,0,true,false,log,tokens,384,6,1.5f,rows));CHECK(ds4_gpu_end_commands());
   uint32_t *out=arm?got:ref;CHECK(ds4_gpu_tensor_read(sel,0,out,rows*6*4)&&ds4_gpu_tensor_read(weights,0,out+rows*6,rows*6*4)&&ds4_gpu_tensor_read(probs,0,out+rows*12,rows*384*4));
   if(arm){for(unsigned i=0;i<rows*396;i++)if(ref[i]!=got[i]){fprintf(stderr,"router draw%u rows%u word%u ref%08x got%08x\n",draw,rows,i,ref[i],got[i]);return 2;}words+=rows*396;}
  }
 }
 printf("PASS router 1..8 rows %lu exact words\n",words);fflush(stdout);
 for(size_t i=16384;i+34<=sz;i+=34){uint16_t sc=0x2800+rnd()%1024;memcpy((char*)map+i,&sc,2);for(int j=2;j<34;j++)((char*)map)[i+j]=(char)rnd();}
 ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(8*32768*4),*out=ds4_gpu_tensor_alloc(8*8192*4);float *xx=malloc(8*32768*4),*aa=malloc(8*8192*4),*bb=malloc(8*8192*4);CHECK(x&&out&&xx&&aa&&bb);
 for(int i=0;i<8*32768;i++)xx[i]=val();CHECK(ds4_gpu_tensor_write(x,0,xx,8*32768*4));
 for(int round=0;round<2;round++)for(unsigned rows=2;rows<=6;rows+=4)for(int arm=0;arm<2;arm++){
  CHECK(ds41_levers_set("mtp_q8_trio_mask",arm?8:0));CHECK(ds4_gpu_begin_commands());CHECK(ds4_gpu_dsv41_attention_low_paired_tensor(out,map,sz,16384,4096,1024,8,x,rows,round));CHECK(ds4_gpu_end_commands());CHECK(ds4_gpu_tensor_read(out,0,arm?bb:aa,rows*8192*4));CHECK(!arm||!memcmp(aa,bb,rows*8192*4));
 }
 printf("PASS grouped Q8 2/6 rows, F32 and BF16 outputs\n");fflush(stdout);
 return 0;
}
