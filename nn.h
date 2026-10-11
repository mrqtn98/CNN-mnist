#ifndef NN_H_INCLUDED
#define NN_H_INCLUDED

#define NB_LAYERS 1 // Number of CNN layers
#define NB_FC_LAYERS 2 // Number of FC layers
#define NOM 1 // Type of activation function used : RELU == 0, SIGMOID == 1; TANH == 2
#define COST 1 // Type of cost function used : MSE == 0, BCE == 1


#include "convolu.h"
#include "kernel.h"



void activation(float mat[MAX_ROWS][MAX_COLUMNS], float bias[MAX_ROWS][MAX_COLUMNS],int size_r, int size_c);

void normalise_image(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c);

void NN_forward(float z[MAX_ROWS][MAX_COLUMNS] ,float filter_list[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    int * p_size_r, int * p_size_c, int kernel_size, int padding,
    float biases_list[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int pool,
    float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS],
    int FC_sizes[NB_FC_LAYERS+1], float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float CNN_activs[NB_LAYERS * 2 + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD], int CNN_sizes[NB_LAYERS+1][2]);

void init_biases(float biases[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c);
void flattening(float z[MAX_ROWS][MAX_COLUMNS],int size_r, int size_c, float z_flat[MAX_COLUMNS*MAX_ROWS]);
void W_x_product(float W[MAX_COLUMNS*MAX_ROWS][MAX_ROWS*MAX_COLUMNS],float x[MAX_ROWS*MAX_COLUMNS], int input_size, int output_size);
void activationFC(float flattened[MAX_ROWS*MAX_COLUMNS], float FC_bias[MAX_COLUMNS*MAX_ROWS],int input_size,int output_size);
void fully_connected_forward(float x[MAX_ROWS*MAX_COLUMNS],int input_size,
    float W[MAX_COLUMNS*MAX_ROWS][MAX_ROWS*MAX_COLUMNS], int output_size,float FC_bias[MAX_COLUMNS*MAX_ROWS]);

void init_FC_biases(float FC_biases[MAX_COLUMNS*MAX_ROWS], int output_size);

void const_init_FC_weights(float FC_weights[MAX_COLUMNS*MAX_ROWS][MAX_ROWS*MAX_COLUMNS], int input_size,
    int output_size, float cst);

void random_init_FC_weights(float FC_weights[MAX_COLUMNS*MAX_ROWS][MAX_ROWS*MAX_COLUMNS], int input_size,
    int output_size, int seed);

#endif // NN_H_INCLUDED
