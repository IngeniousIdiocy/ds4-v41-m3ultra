/* Compare each layout with the original HC/Q8 mixed grid, including its
 * completion counter and Sinkhorn output. No model file is required. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
static uint32_t rng=93217;
static uint32_t rnd(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static float val(void){return ((int)(rnd()%20001)-10000)/4096.f;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t alignpage(uint64_t x){return (x+16383)&~16383ull;}
int main(void){
 int configured=-1;
 if(!ds41_levers_get("hc_stream_layout", &configured) || configured!=2) return 10;
 if(ds41_levers_set("hc_stream_layout",-1) || ds41_levers_set("hc_stream_layout",4) || ds41_levers_set("hc_stream_layout",1) || ds41_levers_set("hc_stream_layout",3)) return 11;

 const uint64_t hc=0,scale=alignpage(20480*24*2),base=scale+16,norm=base+128;
 const uint64_t q=alignpage(norm+5120*4),qb=32768ull*1280/32*34;
 const uint64_t gate=alignpage(q+qb),gb=2304ull*5120/32*34,up=alignpage(gate+gb),bytes=alignpage(up+gb);
 void *model=NULL;if(posix_memalign(&model,getpagesize(),bytes))return 1;memset(model,0,bytes);
 uint16_t *hw=model;for(unsigned i=0;i<20480*24;i++)hw[i]=0x2000+(rnd()%2048)|(rnd()&1?0x8000:0);
 float *sc=(void*)((char*)model+scale);sc[0]=.01;sc[1]=.02;sc[2]=.03;
 float *ba=(void*)((char*)model+base);for(unsigned i=0;i<24;i++)ba[i]=val();
 float *nw=(void*)((char*)model+norm);for(unsigned i=0;i<5120;i++)nw[i]=val();
 uint64_t offsets[]={q,gate,up},sizes[]={qb,gb,gb};
 for(int w=0;w<3;w++)for(uint64_t j=0;j<sizes[w];j+=34){char *p=(char*)model+offsets[w]+j;uint16_t d=0x1800+(rnd()%4096);memcpy(p,&d,2);for(int k=2;k<34;k++)p[k]=(char)rnd();}
 ds4_gpu_tensor *o=ds4_gpu_tensor_alloc(32768*4),*m=ds4_gpu_tensor_alloc(96),*s=ds4_gpu_tensor_alloc(96),*c=ds4_gpu_tensor_alloc(4),*r=ds4_gpu_tensor_alloc(20480*4),*x=ds4_gpu_tensor_alloc(5120*4),*p=ds4_gpu_tensor_alloc(16),*co=ds4_gpu_tensor_alloc(5120*4),*no=ds4_gpu_tensor_alloc(5120*4);
 if(!o||!m||!s||!c||!r||!x||!p||!co||!no||!ds4_gpu_set_model_map(model,bytes))return 2;
 float res[20480],input[5120],pre[4]={.2,.3,.4,.1},ref[32768+48],out[32768+48];uint32_t count;
 unsigned long checked=0;
 for(int rep=0;rep<12;rep++){
  for(int i=0;i<20480;i++)res[i]=val();for(int i=0;i<5120;i++)input[i]=val();
  if(!ds4_gpu_tensor_write(r,0,res,sizeof(res))||!ds4_gpu_tensor_write(x,0,input,sizeof(input))||!ds4_gpu_tensor_write(p,0,pre,16))return 3;
  for(int shared=0;shared<2;shared++)for(int layout=0;layout<=2;layout+=2){
   if(!ds41_levers_set("hc_stream_layout",layout))return 8;int got=-1;if(!ds41_levers_get("hc_stream_layout",&got)||got!=layout)return 9;count=0;
   if(!ds4_gpu_tensor_write(c,0,&count,4)||!ds4_gpu_dsv41_arch_hc_stream_tensor(o,m,s,c,r,x,model,bytes,hc,scale,base,shared?gate:q,up,shared,20,1e-6f,1e-6f,7.f))return 4;
   size_t n=shared?2304:32768;
   if(!ds4_gpu_tensor_read(o,0,out,n*4)||!ds4_gpu_tensor_read(m,0,out+n,96)||!ds4_gpu_tensor_read(s,0,out+n+24,96)||!ds4_gpu_tensor_read(c,0,&count,4)||count!=12)return 5;
   if(!layout)memcpy(ref,out,(n+48)*4);else{for(size_t j=0;j<n+48;j++)if(memcmp(out+j,ref+j,4)){fprintf(stderr,"mismatch rep=%d shared=%d layout=%d word=%zu expected=%a actual=%a\n",rep,shared,layout,j,ref[j],out[j]);return 6;}checked+=n+48;}
  }
 }
 printf("PASS %lu exact output, HC mix and Sinkhorn words\n",checked);fflush(stdout);
 for(int shared=0;shared<2;shared++)for(int rep=0;rep<3;rep++)for(int it=0;it<2;it++){
  int layout=rep%2?2-2*it:2*it;if(!ds41_levers_set("hc_stream_layout",layout))return 8;int got=-1;if(!ds41_levers_get("hc_stream_layout",&got)||got!=layout)return 9;double t=now();
  if(!ds4_gpu_begin_commands())return 7;
  for(int j=0;j<100;j++){
   if(!ds4_gpu_dsv41_arch_collapse_tensor(co,no,c,r,p,model,bytes,norm,1e-6f)||!ds4_gpu_dsv41_arch_hc_stream_tensor(o,m,s,c,r,x,model,bytes,hc,scale,base,shared?gate:q,up,shared,20,1e-6f,1e-6f,7.f))return 7;
  }
  if(!ds4_gpu_end_commands())return 7;
  printf("shared=%d rep=%d layout=%d us=%.3f\n",shared,rep,layout,(now()-t)*1e4);fflush(stdout);
 }
 ds4_gpu_tensor *all[]={o,m,s,c,r,x,p,co,no};for(unsigned i=0;i<sizeof(all)/sizeof(all[0]);i++)ds4_gpu_tensor_free(all[i]);ds4_gpu_cleanup();free(model);return 0;
}
