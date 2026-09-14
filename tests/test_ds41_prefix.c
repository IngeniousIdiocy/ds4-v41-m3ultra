/* Production verifier/undo against a fresh serial oracle. One loaded engine,
 * one target session at a time; snapshots on host. Does not measure speed.
 * Usage: test_ds41_prefix MAIN SUPPORT PROMPT_FILE POSITION ROWS [PAUSE [TOLERANCE]]
 * ROWS=2..6; every keep and unchanged/whole/single-position suffix variants.
 * PAUSE=129 also exercises a serial interval exceeding the capture ring.
 * TOLERANCE is an integer ULP ceiling or absrel:ATOL:RTOL; default ULP0.
 * Choose and record tolerances before gating.
 * Cross-path target-logit failures are deferred; live state remains exact.
 */
#include "../ds4.c"
#include <assert.h>

static char error[512];
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s (%s)\n",__LINE__,#x,error); exit(1); } } while (0)
static void equal_bytes(const char *name, const void *a, const void *b, size_t n) {
    const unsigned char *x=a,*y=b;
    for (size_t i=0;i<n;i++) if (x[i]!=y[i]) {
        fprintf(stderr,"DIFF %s byte=%zu got=%02x expected=%02x\n",name,i,x[i],y[i]); exit(1);
    }
}
/* Only scalar-vs-batch final target logits use this ceiling. It is not a
 * tolerance for intermediate activations, persistent state or native proposals.
 * Read via memcpy: canonical payload words need not be float-aligned. */
static uint32_t max_logit_ulp;
static bool logit_absrel;
static double logit_atol,logit_rtol;
static bool logit_tolerance_parse(const char *text) {
    char *end=NULL;
    if (!strncmp(text,"absrel:",7)) {
        const char *a=text+7;
        if (!((*a>='0' && *a<='9') || *a=='.')) return false;
        errno=0; double atol=strtod(a,&end);
        if (errno || end==a || *end!=':' || !isfinite(atol) || atol<0) return false;
        const char *r=end+1;
        if (!((*r>='0' && *r<='9') || *r=='.')) return false;
        errno=0; double rtol=strtod(r,&end);
        if (errno || end==r || *end || !isfinite(rtol) || rtol<0) return false;
        logit_absrel=true;logit_atol=atol;logit_rtol=rtol;
    } else {
        if (*text<'0' || *text>'9') return false;
        errno=0; unsigned long ulp=strtoul(text,&end,10);
        if (errno || end==text || *end || ulp>UINT32_MAX) return false;
        logit_absrel=false;max_logit_ulp=(uint32_t)ulp;
    }
    return true;
}
static uint64_t logit_rows_checked, logit_rows_failed;
static uint32_t logit_max_observed;
static uint32_t float_order(uint32_t bits) {
    /* Map both signed zeros to 0x80000000, adjacent +/- subnormals one
     * step away. Negative floats then increase in numerical order too. */
    return bits & 0x80000000u ? 0x80000000u - (bits & 0x7fffffffu)
                              : 0x80000000u + bits;
}
static void target_logits(const char *name, const void *actual,
                          const void *expected, unsigned rows) {
    const unsigned char *a=actual,*b=expected;
    for (unsigned row=0;row<rows;row++) {
        uint32_t peak=0, peak_at=0, over=0, nonfinite=0, different=0;
        uint32_t first_at=UINT32_MAX, first_a=0, first_b=0;
        unsigned arg_a=0,arg_b=0;
        float best_a=-INFINITY,best_b=-INFINITY;
        double abs_peak=0;
        for (unsigned i=0;i<DS4_N_VOCAB;i++) {
            size_t off=((size_t)row*DS4_N_VOCAB+i)*sizeof(float);
            uint32_t ua,ub; float fa,fb;
            memcpy(&ua,a+off,4); memcpy(&ub,b+off,4);
            memcpy(&fa,&ua,4); memcpy(&fb,&ub,4);
            if (fa>best_a) { best_a=fa;arg_a=i; }
            if (fb>best_b) { best_b=fb;arg_b=i; }
            if (ua!=ub) different++;
            bool finite=(ua & 0x7f800000u)!=0x7f800000u &&
                        (ub & 0x7f800000u)!=0x7f800000u;
            uint32_t distance=0;
            bool violation=!finite;
            if (finite) {
                uint32_t oa=float_order(ua),ob=float_order(ub);
                distance=oa>ob?oa-ob:ob-oa;
                double delta=fabs((double)fa-(double)fb);
                if (delta>abs_peak) abs_peak=delta;
                if (distance>peak) { peak=distance;peak_at=i; }
                double limit=logit_atol+logit_rtol*fabs((double)fb);
                violation=logit_absrel ? (!isfinite(limit) || delta>limit)
                                      : distance>max_logit_ulp;
                if (violation) over++;
            } else nonfinite++;
            if (violation && first_at==UINT32_MAX) {
                first_at=i;first_a=ua;first_b=ub;
            }
        }
        bool failed=over || nonfinite || arg_a!=arg_b;
        logit_rows_checked++; logit_rows_failed+=failed;
        if (peak>logit_max_observed) logit_max_observed=peak;
        fprintf(stderr,"%s %s row=%u argmax=%u/%u mode=%s ulp_bound=%u atol=%.9g rtol=%.9g max_ulp=%u at=%u "
                "max_abs=%.9g different=%u over=%u nonfinite=%u\n",
                failed?"FAIL-DEFERRED":"OK",name,row,arg_a,arg_b,logit_absrel?"absrel":"ulp",
                max_logit_ulp,logit_atol,logit_rtol,peak,peak_at,abs_peak,different,over,nonfinite);
        if (first_at!=UINT32_MAX)
            fprintf(stderr," first violation id=%u bits=%08x/%08x\n",first_at,first_a,first_b);
    }
}
/* A host-only mode for validating the comparator without opening an engine.
 * Expected failures below must increment the deferred failure counter. */
static void logit_comparator_self_test(void) {
    CHECK(float_order(0)==float_order(0x80000000u));
    CHECK(float_order(1)-float_order(0x80000001u)==2);
    CHECK(float_order(0xbf800000u)-float_order(0xbf800001u)==1);
    float *a=malloc((size_t)DS4_N_VOCAB*4),*b=malloc((size_t)DS4_N_VOCAB*4);
    CHECK(a && b);
    for (unsigned i=0;i<DS4_N_VOCAB;i++) a[i]=b[i]=-2.0f;
    a[0]=b[0]=1.0f;
    uint32_t bits=0xbf800001u; memcpy(a+1,&bits,4); b[1]=-1.0f;
    max_logit_ulp=1;
    target_logits("self-test negative adjacent floats",a,b,1);
    CHECK(logit_rows_failed==0);
    max_logit_ulp=0;
    target_logits("self-test expected ULP failure",a,b,1);
    CHECK(logit_rows_failed==1);
    bits=0x80000000u; memcpy(a+1,&bits,4); b[1]=0;
    target_logits("self-test signed zero",a,b,1);
    CHECK(logit_rows_failed==1);
    max_logit_ulp=UINT32_MAX; a[2]=2.0f;
    target_logits("self-test expected argmax failure",a,b,1);
    CHECK(logit_rows_failed==2);
    a[2]=b[2]; bits=0x7fc12345u;
    memcpy(a+1,&bits,4);memcpy(b+1,&bits,4);
    target_logits("self-test expected identical NaN failure",a,b,1);
    CHECK(logit_rows_failed==3);
    bits=0x7f800000u;memcpy(a+1,&bits,4);memcpy(b+1,&bits,4);
    target_logits("self-test expected identical infinity failure",a,b,1);
    CHECK(logit_rows_failed==4);
    CHECK(!logit_tolerance_parse("absrel:nan:0"));
    CHECK(!logit_tolerance_parse("absrel:0:inf"));
    CHECK(!logit_tolerance_parse("absrel:-1:0"));
    CHECK(!logit_tolerance_parse("absrel:0:0junk"));
    CHECK(!logit_tolerance_parse("absrel:0:"));
    CHECK(!logit_tolerance_parse("4294967296"));
    CHECK(logit_tolerance_parse("absrel:0.00000095367431640625:0")); // 2^-20
    a[1]=0x1p-20f;b[1]=0;
    target_logits("self-test abs floor at zero boundary",a,b,1);
    CHECK(logit_rows_failed==4);
    a[1]=0x1p-19f;
    target_logits("self-test expected abs floor failure",a,b,1);
    CHECK(logit_rows_failed==5);
    CHECK(logit_tolerance_parse("absrel:0:0.5"));
    a[1]=-1.5f;b[1]=-1.0f;
    target_logits("self-test relative boundary negative reference",a,b,1);
    CHECK(logit_rows_failed==5);
    a[1]=-1.75f;
    target_logits("self-test expected relative failure",a,b,1);
    CHECK(logit_rows_failed==6);
    CHECK(logit_tolerance_parse("absrel:100:1"));
    a[1]=b[1];a[2]=2.0f;
    target_logits("self-test absrel expected argmax failure",a,b,1);
    CHECK(logit_rows_failed==7);
    a[2]=b[2];bits=0x7fc12345u;memcpy(a+1,&bits,4);memcpy(b+1,&bits,4);
    target_logits("self-test absrel expected NaN failure",a,b,1);
    CHECK(logit_rows_failed==8);
    CHECK(logit_tolerance_parse("4") && !logit_absrel && max_logit_ulp==4);
    free(a);free(b);
    puts("logit comparator self-test: PASS (eight intentional failures detected)");
}
static void save(ds4_session *s, ds4_session_snapshot *snap) {
    CHECK(!ds4_session_save_snapshot(s,snap,error,sizeof(error)));
}
static void restore(ds4_session *s, const ds4_session_snapshot *snap) {
    CHECK(!ds4_session_load_snapshot(s,snap,error,sizeof(error)));
}
enum { LOGITS_EXACT, LOGITS_BATCH_ACTUAL, LOGITS_BATCH_SNAPSHOT };
static void canonical(ds4_session *s, const ds4_session_snapshot *oracle,
                       const ds4_engram_history *history, int logit_mode) {
    ds41_gpu_graph *g=&s->ds41_graph;
    CHECK(s->checkpoint_valid && g->valid && g->pos==(unsigned)s->checkpoint.len);
    CHECK(g->dspark_hidden_end==g->pos && g->dspark_hidden_len==(g->pos<128?g->pos:128));
    for (unsigned i=0;i<2;i++) CHECK(!g->fetch[i].inflight && !g->engram_carry.armed[i]);
    equal_bytes("Engram history",&g->history,history,sizeof(*history));
    ds4_session_snapshot actual={0}; save(s,&actual);
    CHECK(actual.len==oracle->len);
    /* Flavor3 contains tokens, logits, all live window/capture slots, published
     * owner KV/index caches and unfinished ratio2 carry. Epoch IDs are absent. */
    size_t logits_at=((size_t)DS4_SESSION_PAYLOAD_U32_FIELDS+2u+g->pos)*4u;
    size_t logits_end=logits_at+(size_t)DS4_N_VOCAB*4u;
    CHECK(g->dspark_capture && g_ds41_levers.mtp_state_fix && logits_end<=actual.len);
    equal_bytes("canonical header/capture metadata/tokens",actual.ptr,oracle->ptr,logits_at);
    if (logit_mode==LOGITS_BATCH_ACTUAL)
        target_logits("canonical final target logits",actual.ptr+logits_at,oracle->ptr+logits_at,1);
    else if (logit_mode==LOGITS_BATCH_SNAPSHOT)
        target_logits("canonical final target logits",oracle->ptr+logits_at,actual.ptr+logits_at,1);
    else equal_bytes("same-path canonical logits",actual.ptr+logits_at,oracle->ptr+logits_at,
                     logits_end-logits_at);
    equal_bytes("canonical live state",actual.ptr+logits_end,oracle->ptr+logits_end,
                (size_t)actual.len-logits_end);
    ds4_session_snapshot_free(&actual);
}
static void poison(ds41_gpu_graph *g) {
    /* Only transient scalar scratch. Every live cache and previous-pair carry
     * is outside DS41_SCRATCH. Poison actual view bytes (including compact
     * formats), not an assumed F32 extent. Future logits are compared below. */
    uint32_t words[1024]; for(unsigned i=0;i<1024;i++) words[i]=0x7fc12345u;
#define POISON(name) do { \
    uint64_t nb=ds4_gpu_tensor_bytes(g->name); CHECK(nb); \
    for(uint64_t off=0;off<nb;off+=sizeof(words)) { \
        uint64_t n=nb-off<sizeof(words)?nb-off:sizeof(words); \
        CHECK(ds4_gpu_tensor_write(g->name,off,words,n)); \
    } \
} while(0);
    POISON(residual)
    POISON(after_attn)
    POISON(flat_norm)
    POISON(hc_counter) POISON(mix) POISON(attn_split) POISON(ffn_split) POISON(pre)
    POISON(x) POISON(norm) POISON(block)
    POISON(qr) POISON(q)
    POISON(kv) POISON(latent)
    POISON(pool_kv) POISON(pool_score)
    POISON(index_q)
    POISON(index_k) POISON(index_weights)
    POISON(selected_comp) POISON(selected_kv)
    POISON(block_mask)
    POISON(heads) POISON(low)
    POISON(route_logits) POISON(route_probs)
    POISON(selected) POISON(route_weights)
    POISON(gate) POISON(up)
    POISON(mid) POISON(experts)
    POISON(shared_gate) POISON(shared_up)
    POISON(shared_mid) POISON(shared) POISON(routed)
    POISON(logits)
#undef POISON
}
static void ring_dump(ds41_dspark_decode *dd, ds41_gpu_graph *g, float *out) {
    CHECK(ds41_dspark_seed_to(dd,g,g->pos-1));
    CHECK(dd->seeded_end==g->pos && dd->seeded_generation==g->state_generation);
    unsigned from=g->pos>128?g->pos-128:0;
    for(unsigned st=0;st<3;st++) for(unsigned p=from;p<g->pos;p++) {
        CHECK(ds4_gpu_tensor_read(dd->drafter.ring[st],(uint64_t)(p%192)*512*4,
                                  out,512*4)); out+=512;
    }
    CHECK(ds4_gpu_tensor_read(dd->drafter.main_x,0,out,DS4_N_EMBD*4u));
}
static void propose(ds41_dspark_decode *dd,ds4_session *s,float *base,int32_t ids[5]) {
    const int anchor=ds4_session_argmax(s);
    CHECK(ds41_dspark_seed_to(dd,&s->ds41_graph,s->ds41_graph.pos-1));
    CHECK(ds41_dspark_stage_input(dd,&s->engine->model,&s->engine->weights,anchor));
    CHECK(ds41_dspark_forward(&dd->drafter,s->ds41_graph.pos-1,&s->engine->model,s->engine->weights.output));
    CHECK(ds41_dspark_markov_greedy(&dd->drafter,anchor,base,ids));
}
static void mirror(ds4_session *s,const int *inputs,unsigned keep,const float *logits) {
    for(unsigned i=0;i<keep;i++) token_vec_push(&s->checkpoint,inputs[i]);
    if (s->logits!=logits) memcpy(s->logits,logits,(size_t)DS4_N_VOCAB*4);
}
static void ordinary_cycle(ds4_session *s,ds41_dspark_decode *dd) {
    ds4_session_snapshot start={0},candidate={0}; save(s,&start);
    int token=ds4_session_argmax(s),anchor=token,out[8],fed[8]; unsigned n=0;
    bool declined=false;
    CHECK(ds41_dspark_decode_cycle(&s->ds41_graph,dd,&s->engine->model,&s->engine->weights,
          &token,out,&n,s->logits,-1,-1,6,&declined,NULL,NULL));
    CHECK(!declined && n>=1 && n<=6);
    fed[0]=anchor; for(unsigned i=1;i<n;i++) fed[i]=out[i-1];
    mirror(s,fed,n,s->logits);
    ds4_engram_history history=s->ds41_graph.history; save(s,&candidate);
    restore(s,&start);
    for(unsigned i=0;i<n;i++) {
        CHECK(!ds4_session_eval(s,fed[i],error,sizeof(error)));
        CHECK(ds4_session_argmax(s)==out[i]);
    }
    canonical(s,&candidate,&history,LOGITS_BATCH_SNAPSHOT);
    ds4_session_snapshot_free(&start); ds4_session_snapshot_free(&candidate);
}
int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"--self-test-logits")) {
        logit_comparator_self_test(); return 0;
    }
    if(argc<6 || argc>8) { fprintf(stderr,"usage: %s MAIN SUPPORT PROMPT POSITION ROWS [PAUSE [TOLERANCE]]\n",argv[0]);return 2; }
    unsigned P=(unsigned)strtoul(argv[4],NULL,10),r=(unsigned)strtoul(argv[5],NULL,10);
    unsigned pause=argc>=7?(unsigned)strtoul(argv[6],NULL,10):0;
    if (argc==8) CHECK(logit_tolerance_parse(argv[7]));
    fprintf(stderr,"final target logits: greedy equality, mode=%s max_ulp=%u "
            "atol=%.9g rtol=%.9g; failures deferred until state checks finish\n",
            logit_absrel?"absrel":"ulp",max_logit_ulp,logit_atol,logit_rtol);
    CHECK(P>=2 && P<=131072 && r>=2 && r<=6 && pause<=256);
    ds41_levers_init_from_env();
    // This harness compares state exactly. A changed-arithmetic verifier must
    // use its matching scalar replay, never silently compare with another tree.
    CHECK(g_ds41_levers.mtp_gu_mk6 == g_ds41_levers.mtp_gu_mk6_oracle);
    if (g_ds41_levers.mtp_gu_mk6) {
        CHECK(r == 6);
        fprintf(stderr,"GU mk6: six-row / same-arithmetic scalar replay; exact state checks unchanged\n");
    }
    CHECK(ds41_levers_set("dspark_adaptive",0));
    CHECK(ds41_levers_set("dspark_verify_rows",(int)r));
    CHECK(ds41_levers_set("mtp_state_fix",1));
    CHECK(ds41_levers_set("dspark_capture",1));
    const unsigned ctx=P+pause+64;
    ds4_engine_options opt={.model_path=argv[1],.mtp_path=argv[2],.backend=DS4_BACKEND_METAL,
        .context_size=(int)ctx,.power_percent=100,.dspark=true};
    ds4_engine *e=NULL; CHECK(!ds4_engine_open(&e,&opt));
    FILE *fp=fopen(argv[3],"rb"); CHECK(fp && !fseek(fp,0,SEEK_END));
    long bytes=ftell(fp); CHECK(bytes>=0 && bytes<64*1024*1024); rewind(fp);
    char *text=malloc((size_t)bytes+1); CHECK(text && fread(text,1,(size_t)bytes,fp)==(size_t)bytes);
    text[bytes]=0; fclose(fp);
    ds4_tokens prompt={0}; tokenize_rendered_chat_vocab(&e->vocab,text,&prompt); free(text);
    CHECK(prompt.len>=(int)(P+r+pause));
    ds4_session *s=NULL; CHECK(!ds4_session_create(&s,e,(int)ctx));
    int full_len=prompt.len; prompt.len=(int)P;
    CHECK(!ds4_session_sync(s,&prompt,error,sizeof(error))); prompt.len=full_len;
    ds4_session_snapshot prefix={0}; save(s,&prefix);
    fprintf(stderr,"prefix P=%u r=%u pause=%u snapshot=%llu bytes; levers:\n",P,r,pause,(unsigned long long)prefix.len);
    for(size_t i=0;i<ds41_levers_count();i++) {
        const char *name=g_ds41_lever_map[i].name; int value=0; CHECK(ds41_levers_get(name,&value));
        fprintf(stderr," %s=%d",name,value);
    }
    fprintf(stderr,"\n");
    ds41_verify_ctx vc={0}; CHECK(ds41_verify_alloc(&vc,r));
    ds41_dspark_decode dd={0}; CHECK(ds41_dspark_decode_alloc(&dd,&e->mtp_model,&e->dspark_weights));
    size_t logbytes=(size_t)DS4_N_VOCAB*4,ringbytes=((size_t)3*128*512+DS4_N_EMBD)*4;
    float *batch_reference=malloc(r*logbytes);
    float *rowlog=malloc(r*logbytes),*rings=malloc(ringbytes),*gotrings=malloc(ringbytes);
    float *proposal=malloc(5*logbytes),*gotproposal=malloc(5*logbytes);
    CHECK(batch_reference && rowlog && rings && gotrings && proposal && gotproposal);
    int inputs[8]={0}; inputs[0]=ds4_session_argmax(s);
    for(unsigned i=1;i<r;i++) inputs[i]=prompt.v[P+i-1];
    for(unsigned keep=1;keep<=r;keep++) {
        restore(s,&prefix);
        for(unsigned i=0;i<keep;i++) {
            CHECK(!ds4_session_eval(s,inputs[i],error,sizeof(error)));
            memcpy(rowlog+(size_t)i*DS4_N_VOCAB,s->logits,logbytes);
        }
        ds4_session_snapshot oracle={0},after={0}; save(s,&oracle);
        ds4_engram_history history=s->ds41_graph.history;
        int32_t ids[5],gotids[5];
        ring_dump(&dd,&s->ds41_graph,rings); propose(&dd,s,proposal,ids);
        int next=ds4_session_argmax(s);
        CHECK(!ds4_session_eval(s,next,error,sizeof(error))); save(s,&after);
        ds4_engram_history after_history=s->ds41_graph.history;
        /* mode0 unchanged, mode1 entire suffix, mode2+ individual suffix slots. */
        unsigned variants=keep==r?1:2+r-keep;
        for(unsigned mode=0;mode<variants;mode++) {
            restore(s,&prefix);
            int tokens[8]; memcpy(tokens,inputs,sizeof(tokens));
            for(unsigned i=keep;i<r;i++) if(mode==1 || (mode>=2 && i==keep+mode-2)) {
                /* Actual valid corpus tokens, deterministic alternate; force a
                 * change even when the corpus happens to repeat this token. */
                int alt=prompt.v[(P+r+17*i+mode)%prompt.len];
                if(alt==tokens[i]) alt=(alt+1)%(int)DS4_N_VOCAB;
                tokens[i]=alt;
            }
            fprintf(stderr,"case P=%u r=%u keep=%u mutation=%u inputs=",P,r,keep,mode);
            for(unsigned i=0;i<r;i++) fprintf(stderr,"%s%d",i?",":"",tokens[i]);
            fprintf(stderr,"\n");
            ds41_gpu_graph *graphs[8]; for(unsigned i=0;i<r;i++) graphs[i]=&s->ds41_graph;
            CHECK(ds4_gpu_begin_commands());
            bool ok=ds41_graph_step_batch(graphs,tokens,(int)r,r,&e->model,&e->weights,&vc);
            if(!ds4_gpu_end_commands()) ok=false; CHECK(ok);
            CHECK(ds4_gpu_tensor_read(vc.logits,0,vc.row_logits,r*logbytes));
            if (!mode) memcpy(batch_reference,vc.row_logits,keep*logbytes);
            else equal_bytes("same-batch retained logits across suffix mutation",
                              vc.row_logits,batch_reference,keep*logbytes);
            target_logits("retained target logits batch/serial",vc.row_logits,rowlog,keep);
            CHECK(ds41_verify_commit(&s->ds41_graph,&vc,keep));
            mirror(s,tokens,keep,vc.row_logits+(size_t)(keep-1)*DS4_N_VOCAB);
            canonical(s,&oracle,&history,LOGITS_BATCH_ACTUAL);
            ring_dump(&dd,&s->ds41_graph,gotrings);
            size_t live=(size_t)(P+keep<128?P+keep:128);
            equal_bytes("live drafter rings + main_x",gotrings,rings,(3*live*512+DS4_N_EMBD)*4);
            propose(&dd,s,gotproposal,gotids);
            equal_bytes("native proposal logits",gotproposal,proposal,5*logbytes);
            equal_bytes("native proposal IDs",gotids,ids,sizeof(ids));
            poison(&s->ds41_graph);
            CHECK(!ds4_session_eval(s,next,error,sizeof(error)));
            canonical(s,&after,&after_history,LOGITS_EXACT);
            ordinary_cycle(s,&dd);
        }
        /* Restore a different branch at the same and then a longer frontier
         * into an already seeded drafter. The epoch relationship must repair it. */
        for (unsigned shorter=0;shorter<2;shorter++) {
            restore(s,&prefix);
            for (unsigned i=0;i<keep-shorter;i++) {
                int alt=(inputs[i]+1)%(int)DS4_N_VOCAB;
                CHECK(!ds4_session_eval(s,alt,error,sizeof(error)));
            }
            ring_dump(&dd,&s->ds41_graph,gotrings);
            restore(s,&oracle);
            ring_dump(&dd,&s->ds41_graph,gotrings);
            size_t branch_live=P+keep<128?P+keep:128;
            equal_bytes("same/longer branch reseed",gotrings,rings,(3*branch_live*512+DS4_N_EMBD)*4);
        }
        /* Flavor3 fresh-session restore, fresh drafter, same retained frontier. */
        ds41_dspark_decode_free(&dd); ds4_session_free(s); s=NULL;
        CHECK(!ds4_session_create(&s,e,(int)ctx)); restore(s,&oracle);
        CHECK(ds41_dspark_decode_alloc(&dd,&e->mtp_model,&e->dspark_weights));
        ring_dump(&dd,&s->ds41_graph,gotrings);
        size_t live=(size_t)(P+keep<128?P+keep:128);
        equal_bytes("fresh restore drafter rings",gotrings,rings,(3*live*512+DS4_N_EMBD)*4);
        propose(&dd,s,gotproposal,gotids);
        equal_bytes("fresh restore proposal logits",gotproposal,proposal,5*logbytes);
        equal_bytes("fresh restore proposal IDs",gotids,ids,sizeof(ids));
        for(unsigned i=0;i<pause;i++) CHECK(!ds4_session_eval(s,prompt.v[P+r+i],error,sizeof(error)));
        if(pause) {
            ring_dump(&dd,&s->ds41_graph,gotrings);
            /* Rebuild drafter independently from the current capture window. */
            ds41_dspark_decode_free(&dd);
            CHECK(ds41_dspark_decode_alloc(&dd,&e->mtp_model,&e->dspark_weights));
            ring_dump(&dd,&s->ds41_graph,rings);
            size_t pause_live=s->ds41_graph.pos<128?s->ds41_graph.pos:128;
            equal_bytes("pause incremental vs fresh seeds",gotrings,rings,(3*pause_live*512+DS4_N_EMBD)*4);
        }
        ordinary_cycle(s,&dd);
        ds4_session_snapshot_free(&oracle); ds4_session_snapshot_free(&after);
    }
    /* Negative capture metadata/legacy flavor gates; no invented valid rows. */
    restore(s,&prefix);
    uint32_t saved_word;
    memcpy(&saved_word,prefix.ptr+3*4,4);
    for (uint32_t flavor=1;flavor<=2;flavor++) {
        memcpy(prefix.ptr+3*4,&flavor,4);
        CHECK(ds4_session_load_snapshot(s,&prefix,error,sizeof(error))!=0);
    }
    memcpy(prefix.ptr+3*4,&saved_word,4); restore(s,&prefix);
    uint32_t capture_at=DS4_SESSION_PAYLOAD_U32_FIELDS*4;
    memcpy(&saved_word,prefix.ptr+capture_at,4);
    uint32_t bad_end=P+1; memcpy(prefix.ptr+capture_at,&bad_end,4);
    CHECK(ds4_session_load_snapshot(s,&prefix,error,sizeof(error))!=0);
    memcpy(prefix.ptr+capture_at,&saved_word,4); restore(s,&prefix);
    s->ds41_graph.dspark_hidden_len=0;
    dd.seeded_end=0;
    CHECK(!ds41_dspark_seed_to(&dd,&s->ds41_graph,P-1));
    restore(s,&prefix);
    CHECK(!ds41_levers_set("dspark_verify_rows",1));
    CHECK(!ds41_levers_set("dspark_verify_rows",7));
    /* Short-request state must end at precisely P + emitted - 1. */
    for (int limit=1;limit<=7;limit++) {
        restore(s,&prefix);
        int output[8],n=0; ds4_dspark_decode_stats stats;
        CHECK(!ds4_session_dspark_generate(s,limit,-1,output,&n,&stats,error,sizeof(error)));
        CHECK(n==limit && s->ds41_graph.pos==P+(unsigned)limit-1);
        ds4_session_snapshot candidate={0}; save(s,&candidate);
        ds4_engram_history ch=s->ds41_graph.history;
        restore(s,&prefix);
        CHECK(ds4_session_argmax(s)==output[0]);
        for (int i=1;i<limit;i++) {
            CHECK(!ds4_session_eval(s,output[i-1],error,sizeof(error)));
            CHECK(ds4_session_argmax(s)==output[i]);
        }
        canonical(s,&candidate,&ch,LOGITS_BATCH_SNAPSHOT); ds4_session_snapshot_free(&candidate);
    }
    restore(s,&prefix);
    CHECK(ds41_levers_set("dspark_excl_eos",0));
    int stop=ds4_session_argmax(s),output[8],n=0;
    CHECK(!ds4_session_dspark_generate(s,8,stop,output,&n,NULL,error,sizeof(error)));
    CHECK(n==1 && output[0]==stop && s->ds41_graph.pos==P);
    ds41_dspark_decode_free(&dd); ds41_verify_free(&vc);
    ds4_session_free(s); ds4_engine_close(e); ds4_tokens_free(&prompt);
    ds4_session_snapshot_free(&prefix);
    free(batch_reference);free(rowlog);free(rings);free(gotrings);free(proposal);free(gotproposal);
    puts("exact state/suffix/native continuation/fresh restore checks: PASS");
    fprintf(stderr,"target-logit rows=%llu failed=%llu max_ulp=%u mode=%s ulp_bound=%u atol=%.9g rtol=%.9g\n",
            (unsigned long long)logit_rows_checked,(unsigned long long)logit_rows_failed,
            logit_max_observed,logit_absrel?"absrel":"ulp",max_logit_ulp,logit_atol,logit_rtol);
    if (logit_rows_failed) {
        fputs("prefix gate: FAIL (deferred target-logit checks)\n",stderr); return 1;
    }
    puts("prefix gate: PASS"); return 0;
}
