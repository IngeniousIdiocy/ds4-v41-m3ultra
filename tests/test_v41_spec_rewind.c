/* Private UAT gate: compare an interior rewind with direct acceptance of the
 * same verified rows, including replacement steps and window wraparound. */
#include "../ds4.c"

static void require(bool ok, const char *what) {
    if (!ok) { fprintf(stderr, "REWIND_FAIL %s\n", what); exit(1); }
}
static void snapshot(ds4_session *s, ds4_session_snapshot *p) {
    char err[256]={0}; require(!ds4_session_save_snapshot(s,p,err,sizeof(err)),err);
}
static void restore(ds4_session *s,const ds4_session_snapshot *p) {
    char err[256]={0}; require(!ds4_session_load_snapshot(s,p,err,sizeof(err)),err);
}
static void verify_commit(ds4_session *s,const int *tokens,uint32_t n,uint32_t kept,int replacement) {
    ds41_gpu_graph *g=&s->ds41_graph;ds41_draft *d=g->draft;
    float *rows=malloc((size_t)n*DS4_N_VOCAB*sizeof(float)); require(rows!=NULL,"logit allocation");
    ds4_engram_history history=g->history;
    require(ds41_draft_verify(g,&s->engine->model,&s->engine->weights,tokens,n,rows),"verify");
    memcpy(s->logits,rows+(kept-1u)*DS4_N_VOCAB,DS4_N_VOCAB*sizeof(float));
    for(uint32_t i=0;i<kept;i++)token_vec_push(&s->checkpoint,tokens[i]);
    require(ds41_draft_rollback(g,tokens,n,kept,history),"commit");
    if(replacement>=0) {char err[256]={0};require(!ds4_session_eval(s,replacement,err,sizeof(err)),err);}
    d->rewind_valid=true;d->rewind_end=g->pos; free(rows);
}
int main(int argc,char **argv) {
    require(argc==6,"MODEL DRAFT CORPUS CACHE8K CACHE62K");
    ds4_engine_options opt={.model_path=argv[1],.mtp_path=argv[2],.dspark=true,
        .backend=DS4_BACKEND_METAL,.context_size=65665,.prefill_chunk=8192,.n_threads=4,.warm_weights=true};
    ds4_engine *e=NULL;require(!ds4_engine_open(&e,&opt),"engine");
    ds4_session *ref=NULL,*got=NULL;require(!ds4_session_create(&ref,e,65665)&&!ds4_session_create(&got,e,65665),"sessions");
    unsigned cases=0;uint64_t words=0;char err[256]={0};
    const unsigned offsets[]={0,1,125,127},counts[]={2,3,6};
    for(int depth=0;depth<2;depth++) {
        FILE *f=fopen(argv[4+depth],"rb");require(f!=NULL,"cache open");fseek(f,0,SEEK_END);long len=ftell(f);rewind(f);
        ds4_session_snapshot base={.ptr=malloc(len),.len=len,.cap=len};require(base.ptr&&fread(base.ptr,1,len,f)==(size_t)len,"cache read");fclose(f);
        restore(ref,&base);int start=ds4_session_pos(ref);
        for(unsigned oi=0;oi<4;oi++) {
            restore(ref,&base);
            for(unsigned i=0;i<offsets[oi];i++){int t=ds4_session_argmax_ignoring_eos(ref,DS4_THINK_NONE);require(!ds4_session_eval(ref,t,err,sizeof(err)),err);}
            ds4_session_snapshot frontier={0};snapshot(ref,&frontier);int pos=ds4_session_pos(ref);
            int tokens[6];for(int i=0;i<6;i++){tokens[i]=ds4_session_argmax_ignoring_eos(ref,DS4_THINK_NONE);require(!ds4_session_eval(ref,tokens[i],err,sizeof(err)),err);}
            for(unsigned ni=0;ni<3;ni++)for(unsigned replacement=0;replacement<2;replacement++)for(unsigned keep=1;keep<counts[ni];keep++) {
                uint32_t n=counts[ni];restore(ref,&frontier);restore(got,&frontier);
                verify_commit(ref,tokens,n,keep,-1);
                verify_commit(got,tokens,n,replacement?n-1:n,replacement?tokens[n-1]:-1);
                require(ds4_session_rewind_speculative(got,pos+(int)keep),"fast rewind");
                require(!memcmp(ref->logits,got->logits,DS4_N_VOCAB*sizeof(float)),"frontier logits");
                ds4_session_snapshot a={0},b={0};snapshot(ref,&a);snapshot(got,&b);
                if(a.len!=b.len||memcmp(a.ptr,b.ptr,a.len)) {
                    size_t first=0;while(first<a.len&&first<b.len&&a.ptr[first]==b.ptr[first])first++;
                    fprintf(stderr,"state ctx=%d offset=%u n=%u keep=%u replacement=%u first=%zu len=%llu/%llu\n",start,offsets[oi],n,keep,replacement,first,(unsigned long long)a.len,(unsigned long long)b.len);
                    require(false,"persistent state");
                }
                ds4_session_snapshot_free(&a);ds4_session_snapshot_free(&b);
                require(!ds4_session_rewind_speculative(got,pos+(int)keep-1),"expired frontier");
                for(int step=0;step<4;step++) {
                    int t=ds4_session_argmax_ignoring_eos(ref,DS4_THINK_NONE);
                    require(t==ds4_session_argmax_ignoring_eos(got,DS4_THINK_NONE),"continuation token");
                    require(!ds4_session_eval(ref,t,err,sizeof(err))&&!ds4_session_eval(got,t,err,sizeof(err)),err);
                    require(!memcmp(ref->logits,got->logits,DS4_N_VOCAB*sizeof(float)),"continuation logits");
                    words+=DS4_N_VOCAB;
                }
                cases++;printf("REWIND_PASS ctx=%d offset=%u rows=%u keep=%u replacement=%u\n",start,offsets[oi],n,keep,replacement);fflush(stdout);
            }
            ds4_session_snapshot_free(&frontier);
        }
        ds4_session_snapshot_free(&base);
    }
    printf("REWIND_GATE_PASS cases=%u continuation_logit_words=%llu\n",cases,(unsigned long long)words);
    ds4_session_free(ref);ds4_session_free(got);ds4_engine_close(e);return 0;
}
