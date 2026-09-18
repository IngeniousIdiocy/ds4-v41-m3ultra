/* Compare large-prefill producers against the unchanged separate operations.
 * Poison both arms independently, compare every output and guard word, and
 * repeat the final input. No model weights or model session required. */
#include "../ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CK(x) do { if(!(x)){fprintf(stderr,"FAIL line %d case %u draw %u\n",__LINE__,which,draw);return 1;} }while(0)
static uint32_t seed=917;static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int main(int argc,char **argv){
 unsigned draws=argc>1?atoi(argv[1]):64,which=0,draw=0;
 CK(ds4_gpu_init());
 const unsigned rows=17,width=5120;const size_t cap=(size_t)rows*width*4;
 float *data=malloc(cap*4),*ref=malloc((cap+128)*4),*got=malloc((cap+128)*4);CK(data&&ref&&got);
 void *model=NULL;const size_t model_size=(size_t)5120*1024*2+4096;CK(!posix_memalign(&model,getpagesize(),model_size));memset(model,0,model_size);
 for(size_t i=0;i<(model_size-4096)/2;i++)((uint16_t*)((char*)model+4096))[i]=0x3000+(rnd()%2048);
 for(unsigned i=0;i<width;i++)((float*)model)[i]=1; /* norm weights; F16 tests use offset 4096 */
 CK(ds4_gpu_set_model_map(model,model_size));
 ds4_gpu_tensor *a=ds4_gpu_tensor_alloc(cap*4),*b=ds4_gpu_tensor_alloc(cap*4),*split=ds4_gpu_tensor_alloc(rows*24*4),*root=ds4_gpu_tensor_alloc((cap+128)*4);
 CK(a&&b&&split&&root);
 for(which=argc>2?atoi(argv[2]):0;which<11;which++){
  unsigned inputwidth=which==6?512:which==7?1280:which==9?1280:which==10?512:width;
  unsigned changed=0;uint32_t previous=0;
  unsigned outwidth=which==2?width*4:which==4?512:which==5?1024:which==6?128:which==7?4096:which==8?32:which>=9?inputwidth:width;
  size_t n=(size_t)rows*outwidth;
  ds4_gpu_tensor *out=ds4_gpu_tensor_view(root,256,n*4);CK(out);
  for(draw=0;draw<draws+1000;draw++){
   if(draw<draws){
    size_t awords=(which==0||which==2)?cap:(size_t)rows*inputwidth;
    for(size_t i=0;i<awords;i++)data[i]=(int32_t)rnd()*0x1p-30f;
    CK(ds4_gpu_tensor_write(a,0,data,awords*4));
    if(which==2||which==3){for(size_t i=0;i<(size_t)rows*width;i++)data[i]=(int32_t)rnd()*0x1p-32f;CK(ds4_gpu_tensor_write(b,0,data,(size_t)rows*width*4));}
    if(which==0||which==2){for(size_t i=0;i<rows*24;i++)data[i]=(int32_t)rnd()*0x1p-32f;CK(ds4_gpu_tensor_write(split,0,data,rows*24*4));}
   }
   for(unsigned arm=draw<draws?0:1;arm<2;arm++){
    for(size_t i=0;i<n+128;i++)((uint32_t*)got)[i]=0x7fc12345;
    CK(ds4_gpu_tensor_write(root,0,got,(n+128)*4));CK(ds4_gpu_begin_commands());
    switch(which){
    case 0: CK(arm?ds4_gpu_hc_weighted_sum_split_bf16_tensor(out,a,split,width,4):ds4_gpu_hc_weighted_sum_split_tensor(out,a,split,width,4));break;
    case 1: case 9: case 10: CK(arm?ds4_gpu_dsv41_norm_bf16_rows(out,a,model,model_size,0,inputwidth,rows,1e-6f):ds4_gpu_rms_norm_weight_rows_tensor(out,a,model,model_size,0,inputwidth,rows,1e-6f));break;
    case 2: CK(arm?ds4_gpu_hc_expand_split_bf16_tensor(out,b,a,split,width,4):ds4_gpu_hc_expand_split_tensor(out,b,a,split,width,4));break;
    case 3: CK(arm?ds4_gpu_dsv41_add_bf16_rows(out,a,b,width,rows):ds4_gpu_add_tensor(out,a,b,rows*width));break;
    default: CK(arm?ds4_gpu_dsv41_projection_rows2(out,model,model_size,4096,inputwidth,outwidth,rows,a):ds4_gpu_dsv41_projection_rows(out,model,model_size,4096,inputwidth,outwidth,rows,a));break;
    }
    if(!arm&&(which<4||which>=9))CK(ds4_gpu_dsv41_quantize(out,outwidth,rows,DS4_V41_BF16));
    CK(ds4_gpu_end_commands());CK(ds4_gpu_tensor_read(root,0,arm?got:ref,(n+128)*4));
    if(arm){
     if(memcmp(ref,got,(n+128)*4)){for(size_t i=0;i<n+128;i++)if(((uint32_t*)ref)[i]!=((uint32_t*)got)[i]){fprintf(stderr,"MISMATCH word %zu %08x %08x\n",i,((uint32_t*)ref)[i],((uint32_t*)got)[i]);break;}CK(0);}
     for(size_t i=64;i<64+n;i++)CK((((uint32_t*)got)[i]&0x7f800000u)!=0x7f800000u);
    }
   }
   if(draw<draws){changed+=previous!=((uint32_t*)got)[64];previous=((uint32_t*)got)[64];}
  }
  CK(changed>draws/10);
  ds4_gpu_tensor_free(out);printf("PASS case %u draws %u repeats 1000 rows %u\n",which,draws,rows);fflush(stdout);
 }
 return 0;
}
