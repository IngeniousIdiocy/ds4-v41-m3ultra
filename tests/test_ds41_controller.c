/* Host policy tests; no model, GPU or timing measurement. */
#include "../ds4_dspark_controller.h"
#include <assert.h>
#include <limits.h>
#include <unistd.h>

static int select_row(void *ctx, unsigned r) { return ((int *)ctx)[r]; }
static bool stop_boundary(void *ctx, int token) { return token == *(int *)ctx; }
int main(void) {
    int policy[6]={10,11,12,13,14,15},next=-1;
    int32_t draft[5]={10,11,12,13,14};
    for (unsigned r=2;r<=6;r++) for (unsigned cap=1;cap<=r;cap++) {
        assert(ds41_ctl_accept(r,cap,-1,false,select_row,policy,draft,NULL,NULL,&next)==cap);
        assert(next==policy[cap-1]);
        for (unsigned stop=0;stop<cap;stop++) {
            assert(ds41_ctl_accept(r,cap,policy[stop],false,select_row,policy,draft,NULL,NULL,&next)==stop+1);
            assert(next==policy[stop]);
        }
        assert(ds41_ctl_accept(r,cap,-1,true,select_row,policy,draft,NULL,NULL,&next)==1);
    }
    int boundary=12;
    assert(ds41_ctl_accept(6,6,-1,false,select_row,policy,draft,stop_boundary,&boundary,&next)==3);
    draft[1]=99;
    assert(ds41_ctl_accept(6,6,-1,false,select_row,policy,draft,NULL,NULL,&next)==2 && next==11);

    ds41_ctl_config c = {0}; ds41_ctl_state s = {0};
    c.min_prefix = 1; c.survival = .5; c.margin = .01;
    c.nc = 2;
    c.costs[0] = (ds41_ctl_cost){100,200,2,100,32,10,40};
    c.costs[1] = (ds41_ctl_cost){100,200,6,100,32,10,110};
    for (unsigned i = 0; i < 5; i++) {
        c.bins[c.nb++] = (ds41_ctl_bin){100,200,i,90,100,5};
        c.bins[c.nb++] = (ds41_ctl_bin){100,200,i,90,100,-1};
    }
    float conf[5] = {0};
    assert(ds41_ctl_sigmoid(1000) == 1 && ds41_ctl_sigmoid(-1000) == 0);
    assert(!ds41_ctl_finite(NAN) && !ds41_ctl_finite(INFINITY));
    assert(ds41_ctl_probability(&c, 150, 0, NAN, true) < 0);
    assert(ds41_ctl_choose(&c,150,6,6,conf,true,false,10) == 6);
    assert(ds41_ctl_choose(&c,150,6,6,conf,false,true,10) == 6);
    c.costs[1].tail_ms = 200;
    assert(ds41_ctl_choose(&c,150,6,6,conf,true,false,10) == 0);
    assert(ds41_ctl_choose(&c,150,6,6,conf,true,true,10) == 2);
    assert(ds41_ctl_choose(&c,200,6,6,conf,true,true,10) == 0);
    assert(!ds41_ctl_cost_at(&c,150,3)); /* no interpolation */
    assert(ds41_ctl_choose(&c,150,6,1,conf,true,true,10) == 0);
    /* A sunk draft must not force an even more expensive paid decline. */
    assert(ds41_ctl_choose(&c,150,6,6,conf,true,true,1000) == 2);
    for (unsigned episode = 0; episode < 5; episode++) {
        for (unsigned i = 0; i < 3; i++) ds41_ctl_record(&c,&s,42,1,32,true);
        unsigned skip = 16u << (episode < 3 ? episode : 3);
        assert(s.cooldown == skip);
        for (unsigned i = 0; i < skip; i++) assert(!ds41_ctl_preflight(&c,&s,150,6,6,true));
        assert(s.cooldown == 0 && ds41_ctl_preflight(&c,&s,150,6,6,true));
    }
    assert(s.attempts == 15 && s.declines == 15 && s.paid_ms == 630);
    for (unsigned i = 0; i < 3; i++) ds41_ctl_record(&c,&s,30,2,32,false);
    assert(s.level == 0);
    ds41_ctl_text(&s,"<thi",4); assert(!s.thinking);
    ds41_ctl_text(&s,"nk>",3); assert(s.thinking);
    assert(!ds41_ctl_preflight(&c,&s,150,6,6,true));
    s.cooldown=128; s.level=3; s.count=3;
    ds41_ctl_text(&s,"</th",4); assert(s.thinking);
    ds41_ctl_text(&s,"ink>",4);
    assert(!s.thinking && !s.cooldown && !s.count && !s.level);
    c.fixed_probe=8;
    for (unsigned i = 0; i < 3; i++) ds41_ctl_record(&c,&s,42,1,32,true);
    assert(s.cooldown==8);
    char path[]="/tmp/ds41-controller-test-XXXXXX";
    int fd=mkstemp(path); assert(fd>=0);
    FILE *f=fdopen(fd,"w"); assert(f);
    fputs("ds41-controller-v1\npolicy 1 0.5 0.01 0 0 1.96\n"
          "cost 100 200 2 32 10 40 100\nbin 100 200 0 5 90 100\n",f);
    assert(!fclose(f));
    ds41_ctl_config loaded; assert(ds41_ctl_load(&loaded,path));
    assert(loaded.nc==1 && loaded.nb==1);
    f=fopen(path,"a"); assert(f); fputs("cost 100 200 2 32 10 40 100\n",f);
    assert(!fclose(f)); assert(!ds41_ctl_load(&loaded,path)); unlink(path);
    puts("controller policy: PASS");
    return 0;
}
