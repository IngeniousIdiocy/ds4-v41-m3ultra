#include "../ds4_gpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CK(x) do{if(!(x)){fprintf(stderr,"FAIL line%d draw%u\n",__LINE__,draw);return 1;}}while(0)
static uint32_t seed=991;static uint32_t rnd(void){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int main(int argc,char **argv){unsigned draw=0,draws=argc>1?atoi(argv[1]):64;CK(ds4_gpu_init());const unsigned width=5120,rows=17,vocab=128;const size_t n=(size_t)width*4*rows,sz=(size_t)vocab*width*2+4096;
 void *model=NULL;CK(!posix_memalign(&model,getpagesize(),sz));memset(model,0,sz);uint16_t *w=(uint16_t*)((char*)model+4096);for(size_t j=0;j<vocab*width;j++)w[j]=(rnd()&0x83ffu)|0x3000;
 CK(ds4_gpu_set_model_map(model,sz));uint32_t *ref=malloc((n+128)*4),*got=malloc((n+128)*4),pre[rows*4+128];int ids[rows];CK(ref&&got);
 ds4_gpu_tensor *root=ds4_gpu_tensor_alloc((n+128)*4),*pr=ds4_gpu_tensor_alloc(sizeof(pre)),*x=ds4_gpu_tensor_alloc(rows*width*4),*tokens=ds4_gpu_tensor_alloc(rows*4);CK(root&&pr&&x&&tokens);ds4_gpu_tensor *out=ds4_gpu_tensor_view(root,256,n*4),*pv=ds4_gpu_tensor_view(pr,256,rows*4*4);CK(out&&pv);
 for(draw=0;draw<draws+1000;draw++){
  if(draw<draws){for(unsigned j=0;j<rows;j++)ids[j]=rnd()%vocab;CK(ds4_gpu_tensor_write(tokens,0,ids,rows*4));}
  for(unsigned arm=draw<draws?0:1;arm<2;arm++){
   for(size_t j=0;j<n+128;j++)got[j]=0x7fc12345;for(size_t j=0;j<rows*4+128;j++)pre[j]=0x7fc12345;
   CK(ds4_gpu_tensor_write(root,0,got,(n+128)*4)&&ds4_gpu_tensor_write(pr,0,pre,sizeof(pre))&&ds4_gpu_begin_commands());
   if(arm)CK(ds4_gpu_dsv41_embed_init_rows(out,pv,x,tokens,model,sz,4096,vocab,rows,width));
   else for(unsigned t=0;t<rows;t++){ds4_gpu_tensor *v=ds4_gpu_tensor_view(out,(uint64_t)t*width*4*4,width*4*4);CK(v&&ds4_gpu_embed_token_hc_tensor(v,model,sz,4096,vocab,ids[t],width,4));ds4_gpu_tensor_free(v);}
   CK(ds4_gpu_end_commands()&&ds4_gpu_tensor_read(root,0,arm?got:ref,(n+128)*4));
   if(arm){CK(!memcmp(ref,got,(n+128)*4));CK(ds4_gpu_tensor_read(pr,0,pre,sizeof(pre)));for(unsigned j=0;j<rows*4+128;j++)CK(pre[j]==(j<64||j>=64+rows*4?0x7fc12345:j%4?0:0x3f800000));}
  }
 }
 printf("PASS batch embedding rows17 width5120 draws%u repeats1000 output/pre/guards exact\n",draws);return 0;
}
