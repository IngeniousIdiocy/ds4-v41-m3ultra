/* Public admission must see the complete post-Markov rows, with exactly the
 * same argmax and arithmetic as the original chain. Run fresh logits and
 * conditioning tokens, poisoned outputs, guard regions and repeated inputs. */
#include "../ds4_gpu.h"
#include "../ds4_ds41_dspark_adaptive.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d draw %u arm %u\n",__LINE__,draw,arm); return 1; } } while(0)
static uint32_t seed=3919;
static uint32_t random_u32(void) { seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed; }
static float confidence(const float *r, unsigned n) {
    float top=-1e30f;for(unsigned i=0;i<n;i++)if(r[i]>top)top=r[i];
    const float cut=top-30.0f;double sum=0.0;
    for(unsigned i=0;i<n;i++)if(r[i]>cut)sum+=exp((double)r[i]-(double)top);
    return sum>0.0?(float)(1.0/sum):0.0f;
}
int main(int argc,char **argv) {
    unsigned draw=0,arm=0,draws=argc>1?(unsigned)atoi(argv[1]):64;
    const unsigned vocab=argc>2?(unsigned)atoi(argv[2]):129280,rank=256,dim=5120,block=5;
    const unsigned parts=vocab<8192?32:1024,words=block*vocab;
    const size_t table=(size_t)vocab*rank*2,model_size=4096+2*table;
    CHECK(ds4_gpu_init());void *model=NULL;CHECK(!posix_memalign(&model,getpagesize(),model_size));memset(model,0,model_size);
    uint16_t *w=(uint16_t*)((char*)model+4096);
    for(size_t i=0;i<table;i++)w[i]=(random_u32()&0x83ffu)|0x2000u;
    CHECK(ds4_gpu_set_model_map(model,model_size));
    float *inputs=malloc((size_t)(words+dim+rank)*4),*reference=malloc(words*4),*got=malloc(words*4);
    uint32_t *poison=malloc(((size_t)words+128)*4);CHECK(inputs&&reference&&got&&poison);
    for(size_t i=0;i<(size_t)words+128;i++)poison[i]=0x7fc12345;
    ds4_gpu_tensor *lg=ds4_gpu_tensor_alloc(words*4),*x=ds4_gpu_tensor_alloc(block*dim*4),*proj=ds4_gpu_tensor_alloc((dim+rank)*4);
    ds4_gpu_tensor *tok=ds4_gpu_tensor_alloc((block+1)*4),*conf=ds4_gpu_tensor_alloc(block*4),*pt=ds4_gpu_tensor_alloc(parts*8);
    ds4_gpu_tensor *root=ds4_gpu_tensor_alloc(((size_t)words+128)*4);
    CHECK(lg&&x&&proj&&tok&&conf&&pt&&root);
    ds4_gpu_tensor *post=ds4_gpu_tensor_view(root,256,words*4);CHECK(post);
    for(unsigned i=0;i<dim+rank;i++)inputs[i]=(int32_t)random_u32()*0x1p-32f;
    CHECK(ds4_gpu_tensor_write(proj,0,inputs,(dim+rank)*4));
    uint32_t rawref[6+5+2048],rawgot[6+5+2048],init[6],guard[64];
    uint64_t changed=0;float previous=0.0f;
    for(draw=0;draw<draws+1000;draw++) {
        if(draw<draws) {
            for(unsigned i=0;i<words;i++)inputs[i]=(int32_t)random_u32()*0x1p-28f;
            CHECK(ds4_gpu_tensor_write(lg,0,inputs,words*4));
            for(unsigned i=0;i<block*dim;i++)inputs[i]=(int32_t)random_u32()*0x1p-30f;
            CHECK(ds4_gpu_tensor_write(x,0,inputs,block*dim*4));
            for(unsigned i=0;i<6;i++)init[i]=0x7fc12345;
            init[0]=draw%4==0?draw%64:draw%4==1?(draw-1)%64:random_u32()%vocab;
        }
        /* Original chain, complete post rows, and an identical-input repeat. */
        for(arm=draw<draws?0:2;arm<3;arm++) {
            CHECK(ds4_gpu_tensor_write(root,0,poison,((size_t)words+128)*4));
            CHECK(ds4_gpu_tensor_write(tok,0,init,sizeof(init))&&ds4_gpu_tensor_write(conf,0,poison,block*4)&&ds4_gpu_tensor_write(pt,0,poison,parts*8));
            CHECK(ds4_gpu_begin_commands());
            CHECK(arm?ds4_gpu_dsv41_markov_chain_post(block,vocab,rank,dim,lg,x,model,model_size,4096,4096+table,1,proj,tok,conf,pt,parts,post):ds4_gpu_dsv41_markov_chain(block,vocab,rank,dim,lg,x,model,model_size,4096,4096+table,1,proj,tok,conf,pt,parts));
            CHECK(ds4_gpu_end_commands());
            uint32_t *raw=arm?rawgot:rawref;
            CHECK(ds4_gpu_tensor_read(tok,0,raw,24)&&ds4_gpu_tensor_read(conf,0,raw+6,20)&&ds4_gpu_tensor_read(pt,0,raw+11,parts*8));
            if(arm)CHECK(!memcmp(rawref,rawgot,(11+parts*2)*4));
            if(arm) {
                float *out=arm==1?reference:got;CHECK(ds4_gpu_tensor_read(post,0,out,words*4));
                if(arm>1)CHECK(!memcmp(reference,got,words*4));
                for(unsigned r=0;r<block;r++) {
                    float top=-INFINITY;unsigned best=0;
                    for(unsigned v=0;v<vocab;v++) {
                        const float z=out[(size_t)r*vocab+v];uint32_t bits;memcpy(&bits,&z,4);
                        CHECK((bits&0x7f800000u)!=0x7f800000u);
                        if(z>top){top=z;best=v;}
                    }
                    CHECK(best==raw[r+1]);
                    if (draw < 64 || draw % 1000 == 0) {
                        float p=confidence(out+(size_t)r*vocab,vocab);CHECK(ds41_adapt_probability(p)&&p>0);
                    }
                }
            }
            CHECK(ds4_gpu_tensor_read(root,0,guard,256)&&!memcmp(guard,poison,256));
            CHECK(ds4_gpu_tensor_read(root,((size_t)words+64)*4,guard,256)&&!memcmp(guard,poison,256));
        }
        if(draw<draws){changed+=reference[0]!=previous;previous=reference[0];}
        if((draw+1)%1000==0){printf("draw %u/%u post rows/tokens/confidence/parts exact\n",draw+1,draws);fflush(stdout);}
    }
    CHECK(changed>draws/5);printf("PASS full post-Markov rows vocab %u draws %u repeats 1000 changed %llu\n",vocab,draws,(unsigned long long)changed);
    return 0;
}
