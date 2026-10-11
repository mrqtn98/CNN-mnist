#include <stdio.h>
#include <stdlib.h>

#include "FC_backprop.h"
#include "CNN_backprop.h"

void CNN_backprop(float CNN_activs[NB_LAYERS * 2 + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD],
    float CNN_weights[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    float CNN_biases[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int CNN_sizes[NB_LAYERS+1][2],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], int kernel_size, int pool, float grad_w[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS],
    float grad_b[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS], int actipad,
    float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float deriv_activs_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS]){


    init_grad_w(grad_w,kernel_size);

    int padding = kernel_size/2;

    for (int l = NB_LAYERS; l > 0; l --){
        int size_r = CNN_sizes[l-1][0];
        int size_c = CNN_sizes[l-1][1];


        for (int r = 0; r < kernel_size; r++){
            for (int c = 0; c < kernel_size; c++){
                for (int i = kernel_size/2; i < size_r - kernel_size/2; i++){
                    for (int j = kernel_size/2; j < size_c - kernel_size/2; j++){
                        grad_w[l][r][c] += CNN_activs[2*(l - 1)][i+r-kernel_size/2][j+c-kernel_size/2]*
                        deriv_activ(CNN_activs[2*l - 1][i - kernel_size/2][j - kernel_size/2])*
                        deriv_cost_cnn_conv(l, CNN_activs, i, j, CNN_weights, CNN_biases, CNN_sizes, FC_activs, FC_weights,
                        FC_biases, FC_sizes, control, pool, kernel_size, padding, actipad, deriv_activs_FC,
                        deriv_activs_CNN);
                    }
                }
            }
        }



        for (int i = kernel_size/2; i < size_r - kernel_size/2; i++){
            for (int j = kernel_size/2; j < size_c - kernel_size/2; j++){
                //printf("=======\n");
                //printf("\nentree : %d, %d\n",i - padding,j - padding);
                grad_b[l][i - kernel_size/2][j - kernel_size/2] =
                deriv_activ(CNN_activs[2*l - 1][i - kernel_size/2][j - kernel_size/2])*
                deriv_cost_cnn_conv(l, CNN_activs, i, j, CNN_weights, CNN_biases, CNN_sizes, FC_activs, FC_weights,
                FC_biases, FC_sizes, control, pool, kernel_size, padding, actipad, deriv_activs_FC,
                deriv_activs_CNN);
                //printf("%.5f\n", CNN_activs[2*l - 1][i - kernel_size/2][j - kernel_size/2]);
                //printf("sortie: %.10f\n",grad_b[l][i - padding][j - padding] );

            }
        }
    }
}



float deriv_cost_cnn_conv(int l, float CNN_activs[NB_LAYERS + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD], int i,int j, float CNN_weights[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    float CNN_biases[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int CNN_sizes[NB_LAYERS+1][2],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], int pool, int kernel_size, int padding, int actipad,
    float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float deriv_activs_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS]){

    float d = 1;

    int corr = 1;

    int stride = POOL_SIZE;

    i = i - padding;
    j = j - padding;

    if (l == NB_LAYERS){
        corr = 0;
    }

    if (pool == 0){
        int i_max = 0;
        int j_max = 0;


        if (i/stride < CNN_sizes[l][0] - 2 * padding * corr * actipad && j/stride < CNN_sizes[l][1] - 2 * padding * corr
        * actipad){

            find_max_pixel(CNN_activs[2*l - 1], stride, i, j, &i_max, &j_max);

            if (i==i_max && j==j_max){

                d *= deriv_cost_cnn_pool(l,CNN_activs, i/stride,j/stride,CNN_weights,CNN_biases,CNN_sizes,FC_activs,FC_weights,FC_biases,
                FC_sizes,control,pool, kernel_size, padding, actipad, deriv_activs_FC,
                deriv_activs_CNN);
            }

            else d = 0;
        }

        else d = 0;
    }

    else {
        if (i/stride < CNN_sizes[l][0] - 2 * padding * corr * actipad && j/stride < CNN_sizes[l][1] - 2 * padding * corr
        * actipad){

            d *= (float)deriv_cost_cnn_pool(l,CNN_activs,i/stride,j/stride,CNN_weights,CNN_biases,CNN_sizes,FC_activs,FC_weights,FC_biases,
            FC_sizes,control,pool, kernel_size, padding, actipad, deriv_activs_FC,
            deriv_activs_CNN)/stride;

            //printf("%d, %d, %.10f\n", i,j , d );

        }

        else d =  0;
    }

    deriv_activs_CNN[l][i][j] = d;

    return d;
}

float deriv_cost_cnn_pool(int l, float CNN_activs[NB_LAYERS + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD], int ipol,int jpol, float CNN_weights[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    float CNN_biases[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int CNN_sizes[NB_LAYERS+1][2],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], int pool, int kernel_size, int padding, int actipad,
    float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float deriv_activs_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS]){

    float d = 0;

    int size_r = CNN_sizes[l][0];
    int size_c = CNN_sizes[l][1];

    if (l == NB_LAYERS){
            d = deriv_cost(0, FC_activs, ipol * size_c + jpol, FC_weights, FC_biases, FC_sizes,control,
            deriv_activs_FC);
            //printf("%d, %d, %.6f\n ", ipol - padding,jpol - padding, d);
            return d;
    }

    for (int i = kernel_size/2; i < size_r - kernel_size/2; i++){
        for (int j = kernel_size/2; j < size_c - kernel_size/2; j++){
            for (int r = 0; r < kernel_size; r++){
                for (int c = 0; c < kernel_size; c++){

                    if (i + r - kernel_size/2 == ipol + padding * actipad && j + c - kernel_size/2 == jpol + padding * actipad){
                        d +=  CNN_weights[l][r][c] *  deriv_activ(CNN_activs[2*l + 2][i - padding][j - padding]) *
                        deriv_activs_CNN[l + 1][i - padding][j - padding];
                    }
                }
            }
        }
    }

    return d;
}


void init_grad_w(float mat[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size){

    for (int l = 0; l < NB_LAYERS + 1; l ++){
        for(int r = 0; r < kernel_size; r ++){
            for (int c = 0; c < kernel_size; c ++){
            mat[l][r][c] = 0;
            }
        }
    }

}

void find_max_pixel(float mat[MAX_ROWS + PAD][MAX_COLUMNS + PAD],int stride, int i, int j, int * p_i_max, int * p_j_max){

    float max = mat[i][j];
    *p_i_max = i;
    *p_j_max = j;

    for (int l = 0; l < stride; l ++){
        for (int c = 0; c < stride; c ++){

            if (mat[(i/stride) * stride + l ][(j/stride) * stride + c ] >= max){
                *p_i_max = (i/stride) * stride + l;
                *p_j_max = (j/stride) * stride + c;
                max = mat[(i/stride) * stride + l ][(j/stride) * stride + c ];

            }
        }
    }

    /*
    if (*p_i_max == i && *p_j_max == j){
        printf("%d, %d, %.3f\n", i, j, mat[*p_i_max][*p_j_max]);
    }*/
}

