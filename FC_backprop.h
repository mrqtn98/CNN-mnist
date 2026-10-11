#ifndef FC_BACKPROP_H_INCLUDED
#define FC_BACKPROP_H_INCLUDED

#include "convolu.h"
#include "kernel.h"
#include "nn.h"

float cost_function(float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], float control[MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1]);

void backprop(float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float control[MAX_ROWS*MAX_COLUMNS], float grad_w[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float grad_b[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS]);

float deriv_activ(float a);

float deriv_cost(int l, float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], int j, float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS]);

#endif // FC_BACKPROP_H_INCLUDED
