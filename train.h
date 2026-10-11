#ifndef TRAIN_H_INCLUDED
#define TRAIN_H_INCLUDED

#include "kernel.h"

void generate_horizontal(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c, int seed);

void generate_vertical(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c, int seed);

int pos_max_output(float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1]);

void generate_control(float control[MAX_COLUMNS*MAX_ROWS], int label, int size_output);

#endif // TRAIN_H_INCLUDED
