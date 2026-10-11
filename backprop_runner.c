#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "FC_backprop.h"
#include "CNN_backprop.h"
#include "backprop_runner.h"



void run_backprop(float CNN_activs[NB_LAYERS * 2 + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD],
    float CNN_weights[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    float CNN_biases[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int CNN_sizes[NB_LAYERS+1][2],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], int kernel_size, int pool, float lr, int padding){

    static float grad_w_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS];
    static float grad_b_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS];

    static float grad_w_CNN[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS];
    static float grad_b_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS];

    static float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS];
    static float deriv_activs_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS];

    backprop(FC_weights, FC_biases, FC_sizes, FC_activs, control, grad_w_FC, grad_b_FC, deriv_activs_FC);

    CNN_backprop(CNN_activs, CNN_weights, CNN_biases, CNN_sizes, FC_activs, FC_weights, FC_biases, FC_sizes,
    control, kernel_size, pool, grad_w_CNN, grad_b_CNN, padding, deriv_activs_FC, deriv_activs_CNN);


    for (int l = 1; l < NB_FC_LAYERS + 1; l ++){
        int output_size = FC_sizes[l];
        int input_size = FC_sizes[l-1];

        for (int j = 0; j < output_size; j++){
            for (int i = 0; i < input_size; i++){
                FC_weights[l-1][j][i] -= lr *  grad_w_FC[l][j][i];
            }

            FC_biases[l-1][j] -= lr * grad_b_FC[l][j];
        }
    }

    for (int l = 1; l < NB_LAYERS + 1; l ++){
        int size_r = CNN_sizes[l-1][0] - (kernel_size/2) * 2;
        int size_c = CNN_sizes[l-1][1] - (kernel_size/2) * 2;

        for (int i = 0; i < size_r; i++){
            for (int j = 0; j < size_c; j++){
                CNN_biases[l-1][i][j] -= lr * grad_b_CNN[l][i][j];
            }
        }

        for (int r = 0; r < kernel_size; r++){
            for (int c = 0; c < kernel_size; c++){
                CNN_weights[l-1][r][c] -= lr * grad_w_CNN[l][r][c];
            }
        }


    }
}


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
    float grad_b_CNN_base[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS], int z, int size_avg, int iter){

    static float grad_w_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS];
    static float grad_b_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS];

    static float grad_w_CNN[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS];
    static float grad_b_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS];

    static float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS];
    static float deriv_activs_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS];

    backprop(FC_weights, FC_biases, FC_sizes, FC_activs, control, grad_w_FC, grad_b_FC, deriv_activs_FC);

    CNN_backprop(CNN_activs, CNN_weights, CNN_biases, CNN_sizes, FC_activs, FC_weights, FC_biases, FC_sizes,
    control, kernel_size, pool, grad_w_CNN, grad_b_CNN, padding, deriv_activs_FC, deriv_activs_CNN);


    for (int l = 1; l < NB_FC_LAYERS + 1; l ++){
        int output_size = FC_sizes[l];
        int input_size = FC_sizes[l-1];

        for (int j = 0; j < output_size; j++){
            for (int i = 0; i < input_size; i++){
                grad_w_FC_base[l][j][i] +=  (float)grad_w_FC[l][j][i]/size_avg;
            }

            grad_b_FC_base[l][j] += (float)grad_b_FC[l][j]/size_avg;
        }
    }

    for (int l = 1; l < NB_LAYERS + 1; l ++){
        int size_r = CNN_sizes[l-1][0] - (kernel_size/2) * 2;
        int size_c = CNN_sizes[l-1][1] - (kernel_size/2) * 2;

        for (int i = 0; i < size_r; i++){
            for (int j = 0; j < size_c; j++){
                grad_b_CNN_base[l][i][j] += (float)grad_b_CNN[l][i][j]/size_avg;
            }
        }

        for (int r = 0; r < kernel_size; r++){
            for (int c = 0; c < kernel_size; c++){
                grad_w_CNN_base[l][r][c] += (float)grad_w_CNN[l][r][c]/size_avg;
            }
        }
    }


    if (z == size_avg - 1) {
        for (int l = 1; l < NB_FC_LAYERS + 1; l ++){
            int output_size = FC_sizes[l];
            int input_size = FC_sizes[l-1];

            for (int j = 0; j < output_size; j++){
                for (int i = 0; i < input_size; i++){
                    FC_weights[l-1][j][i] -= lr * pow(BETA, iter) * grad_w_FC_base[l][j][i];
                }

                FC_biases[l-1][j] -= lr * pow(BETA, iter) * grad_b_FC_base[l][j];
            }
        }

        for (int l = 1; l < NB_LAYERS + 1; l ++){
            int size_r = CNN_sizes[l-1][0] - (kernel_size/2)*2;
            int size_c = CNN_sizes[l-1][1] - (kernel_size/2)*2;

            for (int i = 0; i < size_r; i++){
                for (int j = 0; j < size_c; j++){
                    CNN_biases[l-1][i][j] -= lr * pow(BETA, iter) * grad_b_CNN_base[l][i][j];
                }
            }

            for (int r = 0; r < kernel_size; r++){
                for (int c = 0; c < kernel_size; c++){
                    CNN_weights[l-1][r][c] -= lr * pow(BETA, iter) * grad_w_CNN_base[l][r][c];
                }
            }
        }
    }
}


void init_grad_w_CNN(float mat[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS]){
    for (int l = 0; l < NB_LAYERS + 1; l ++){
        for(int r = 0; r < MAX_F_ROWS; r ++){
            for (int c = 0; c < MAX_F_COLUMNS; c ++){
                mat[l][r][c] = 0;
            }
        }
    }
}

void init_grad_b_CNN(float mat[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS]){
    for (int l = 0; l < NB_LAYERS + 1; l ++){
        for(int r = 0; r < MAX_ROWS; r ++){
            for (int c = 0; c < MAX_COLUMNS; c ++){
                mat[l][r][c] = 0;
            }
        }
    }
}

void init_grad_w_FC(float mat[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS]){
    for (int l = 0; l < NB_FC_LAYERS + 1; l ++){
        for(int r = 0; r < MAX_COLUMNS*MAX_ROWS; r ++){
            for (int c = 0; c < MAX_COLUMNS*MAX_ROWS; c ++){
                mat[l][r][c] = 0;
            }
        }
    }
}

void init_grad_b_FC(float mat[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS]){
    for (int l = 0; l < NB_FC_LAYERS + 1; l ++){
        for (int r = 0; r < MAX_COLUMNS*MAX_ROWS; r ++){
            mat[l][r] = 0;
        }
    }
}





