/* Row batching changes launch boundaries, not the RoPE/quantization formulas. */
#include "../ds4.h"
#include "../ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do{if(!(x)){fprintf(stderr,"FAIL line%d\n",__LINE__);return 1;}}while(0)
static uint32_t seed=61391;static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int main(void){CK(ds4_gpu_init());const unsigned rows=6,width=512;const size_t maxwords=(size_t)rows*64*width;float*input=malloc(maxwords*4);uint32_t*a=malloc((maxwords+128)*4),*b=malloc((maxwords+128)*4);CK(input&&a&&b);
 unsigned starts[]={0,123,127,128,8189,8192,299997,300000,393209,1048569};
 for(unsigned heads=1;heads<=64;heads*=64){const size_t rowwords=heads*width,words=rows*rowwords,all=words+128;ds4_gpu_tensor*root=ds4_gpu_tensor_alloc(all*4),*x=ds4_gpu_tensor_view(root,256,words*4);CK(root&&x);
 for(unsigned draw=0;draw<64;draw++){unsigned start=starts[draw%10],compressed=(draw/2)%2,inverse=draw%2;for(size_t j=0;j<words;j++)input[j]=((int)(rnd()%20001)-10000)/4096.f;
 for(unsigned arm=0;arm<2;arm++){for(size_t j=0;j<all;j++)b[j]=0x7fc12345;CK(ds4_gpu_tensor_write(root,0,b,all*4));CK(ds4_gpu_tensor_write(x,0,input,words*4));CK(ds4_gpu_begin_commands());
 if(arm){CK(ds4_gpu_dsv41_rope(x,width,heads,rows,start,compressed,inverse));if(heads==1)CK(ds4_gpu_dsv41_quantize(x,width,rows,DS4_V41_FP8_E8M0));}
 else for(unsigned t=0;t<rows;t++){ds4_gpu_tensor*v=ds4_gpu_tensor_view(x,t*rowwords*4,rowwords*4);CK(v);CK(ds4_gpu_dsv41_rope(v,width,heads,1,start+t,compressed,inverse));if(heads==1)CK(ds4_gpu_dsv41_quantize(v,width,1,DS4_V41_FP8_E8M0));ds4_gpu_tensor_free(v);}
 CK(ds4_gpu_end_commands());CK(ds4_gpu_tensor_read(root,0,arm?b:a,all*4));}
 for(size_t j=0;j<all;j++){CK(a[j]==b[j]);if(j<64||j>=words+64)CK(b[j]==0x7fc12345);else CK((b[j]&0x7f800000u)!=0x7f800000u);}
 }ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(root);}
 puts("PASS pointwise batching: 64 draws each heads1/64, both frequencies/directions, wrap/deep positions, offsets and guards exact");return 0;}
