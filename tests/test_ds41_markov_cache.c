/* Uncached production Markov chain vs memoization: complete logits and tokens,
 * alternating independent support maps, eviction, bypass, and poisoned outputs. */
#include "../ds4.c"
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d\n",__LINE__);exit(1);}}while(0)
static uint32_t test_rng=7281;static uint32_t rnd(void){test_rng^=test_rng<<13;test_rng^=test_rng>>17;test_rng^=test_rng<<5;return test_rng;}
int main(void){
 g_ds4_shape.n_vocab=257;const unsigned v=257,k=256,b=5;const size_t wb=(size_t)v*k/32*34;
 ds4_tensor w1={.type=DS4_TENSOR_Q8_0,.ndim=2,.dim={256,257}},w2=w1;w2.abs_offset=wb;
 ds4_dspark_weights dw={.n_stages=1,.block_size=5,.markov_rank=256};dw.stage[0].markov_w1=&w1;dw.stage[0].markov_w2=&w2;
 ds4_model models[2]={{0},{0}};ds41_dspark d[2]={{0},{0}};
 CHECK(ds4_gpu_init());ds41_levers_init_from_env();
 for(int m=0;m<2;m++){
  models[m].map=malloc(wb*2);models[m].size=wb*2;CHECK(models[m].map);
  for(size_t i=0;i<wb*2;i+=34){uint16_t sc=0x2000+rnd()%1024;memcpy((char*)models[m].map+i,&sc,2);for(int z=2;z<34;z++)((char*)models[m].map)[i+z]=(char)rnd();}
  d[m].model=&models[m];d[m].dw=&dw;d[m].vocab=v;d[m].markov_rank=k;d[m].block=b;d[m].ready=true;d[m].logits=ds4_gpu_tensor_alloc(v*b*4);CHECK(d[m].logits);
 }
 float base[1285],ref[1285],got[1285],state[256],bias[257];int32_t a[DS4_DSPARK_MAX_BLOCK_SIZE],c[DS4_DSPARK_MAX_BLOCK_SIZE];unsigned long words=0,plants=0;
 for(unsigned draw=0;draw<50000;draw++){
  unsigned m=draw%2,prev=rnd()%v;for(unsigned i=0;i<v*b;i++)base[i]=((int)(rnd()%20001)-10000)/4096.f;
  /* Force repeated common tokens and a >64-key churn set, with varying logits. */
  for(unsigned j=0;j<b;j++)base[j*v+(draw%3==0?rnd()%17:rnd()%v)]+=1000.f;
  memcpy(ref,base,sizeof(base));for(unsigned i=0;i<v*b;i++)((uint32_t*)got)[i]=0x7fc12345;
  uint32_t len=0;CHECK(dspark_apply_markov_greedy_probe(ref,&models[m],&dw,prev,0,state,bias,a,&len)&&len==b);
  CHECK(ds4_gpu_tensor_write(d[m].logits,0,base,sizeof(base)));g_ds41_levers.markov_cache=draw%19!=0;
  CHECK(ds41_dspark_markov_greedy(&d[m],prev,got,c));CHECK(!memcmp(ref,got,sizeof(got))&&!memcmp(a,c,b*4));
  for(unsigned i=0;i<v*b;i++)CHECK((((uint32_t*)got)[i]&0x7f800000u)!=0x7f800000u&&((uint32_t*)got)[i]!=0x7fc12345);
  ((uint32_t*)got)[draw%(v*b)]^=1;CHECK(memcmp(ref,got,sizeof(got)));plants++;words+=v*b;
  if(draw%5000==0){printf("draw%u exact words%lu\n",draw,words);fflush(stdout);}
 }
 for(unsigned rep=0;rep<1000;rep++){
  memcpy(ref,base,sizeof(base));uint32_t len=0;CHECK(dspark_apply_markov_greedy_probe(ref,&models[0],&dw,23,0,state,bias,a,&len));
  CHECK(ds4_gpu_tensor_write(d[0].logits,0,base,sizeof(base)));g_ds41_levers.markov_cache=1;memset(got,0xff,sizeof(got));CHECK(ds41_dspark_markov_greedy(&d[0],23,got,c));CHECK(!memcmp(ref,got,sizeof(got))&&!memcmp(a,c,b*4));
 }
 printf("PASS determinism1000/1000\n");
 printf("PASS draws50000 words%lu controls%lu independent models, cache hit/eviction/bypass\n",words,plants);
 for(int m=0;m<2;m++){ds41_dspark_free(&d[m]);CHECK(!d[m].markov_bias_cache&&!d[m].markov_cache_count);free((void*)models[m].map);}ds4_gpu_cleanup();return 0;
}
