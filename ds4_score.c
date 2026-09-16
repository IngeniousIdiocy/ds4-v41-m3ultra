#include "ds4_score.h"
#include <math.h>

/* Same traversal, tie rule and FP64 logsum as score_official.c. Keep this
 * translation unit out of the inference engine's -ffast-math build: scorer
 * finite checks and reduction order are part of the measurement contract. */
int ds4_score_logits(const float *logits, int vocab, int token,
                     double *nll, int *greedy) {
    if (!logits || !nll || !greedy || vocab <= 0 || token < 0 || token >= vocab) return 0;
    float maximum = -INFINITY;
    int best = -1;
    for (int i = 0; i < vocab; ++i) {
        if (!isfinite(logits[i])) return 0;
        if (best < 0 || logits[i] > maximum) { maximum = logits[i]; best = i; }
    }
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) sum += exp((double)logits[i] - (double)maximum);
    double logsum = (double)maximum + log(sum);
    if (!isfinite(logsum)) return 0;
    *nll += -((double)logits[token] - logsum);
    *greedy = best;
    return 1;
}
