#ifndef DS4_SCORE_H
#define DS4_SCORE_H

/* Accumulate teacher-forced NLL; compiled without fast-math. */
int ds4_score_logits(const float *logits, int vocab, int token,
                     double *nll, int *greedy);

#endif
