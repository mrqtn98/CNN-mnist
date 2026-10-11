#ifndef CNN_BACKPROP_H_INCLUDED
#define CNN_BACKPROP_H_INCLUDED

#include "convolu.h"
#include "kernel.h"
#include "nn.h"

void CNN_backprop(float CNN_activs[NB_LAYERS * 2 + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD],
    float CNN_weights[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    float CNN_biases[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int CNN_sizes[NB_LAYERS+1][2],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], int kernel_size, int pool, float grad_w[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS],
    float grad_b[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS], int actipad,
    float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float deriv_activs_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS]);

float deriv_cost_cnn_conv(int l, float CNN_activs[NB_LAYERS + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD], int i,int j, float CNN_weights[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    float CNN_biases[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int CNN_sizes[NB_LAYERS+1][2],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], int pool, int kernel_size, int padding, int actipad,
    float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float deriv_activs_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS]);

float deriv_cost_cnn_pool(int l, float CNN_activs[NB_LAYERS + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD], int ipol,int jpol, float CNN_weights[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    float CNN_biases[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int CNN_sizes[NB_LAYERS+1][2],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], int pool, int kernel_size, int padding, int actipad,
    float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float deriv_activs_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS]);

void init_grad_w(float mat[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size);

void find_max_pixel(float mat[MAX_ROWS + PAD][MAX_COLUMNS + PAD],int stride, int i, int j, int * p_i_max, int * p_j_max);


#endif
