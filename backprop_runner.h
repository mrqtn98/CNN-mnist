#ifndef BACKPROP_RUNNER_H_INCLUDED
#define BACKPROP_RUNNER_H_INCLUDED

#include "convolu.h"
#include "kernel.h"
#include "nn.h"

#define BETA 0.9999f // decreases the learning rate at each iteration if != 1.0f (lr_iter = lr * BETA^iter)


void run_backprop(float CNN_activs[NB_LAYERS * 2 + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD],
    float CNN_weights[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    float CNN_biases[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int CNN_sizes[NB_LAYERS+1][2],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], int kernel_size, int pool, float lr, int padding);

void run_stochastic_backprop(float CNN_activs[NB_LAYERS * 2 + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD],
    float CNN_weights[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    float CNN_biases[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int CNN_sizes[NB_LAYERS+1][2],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], int kernel_size, int pool, float lr, int padding,
    float grad_w_FC_base[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float grad_b_FC_base[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float grad_w_CNN_base[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS],
    float grad_b_CNN_base[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS], int z, int size_avg, int iter);

void init_grad_w_CNN(float mat[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS]);

void init_grad_b_CNN(float mat[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS]);

void init_grad_w_FC(float mat[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS]);

void init_grad_b_FC(float mat[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS]);

#endif // BACKPROP_RUNNER_H_INCLUDED
