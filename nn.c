#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#include "nn.h"
#include "convolu.h"
#include "kernel.h"


void activation(float mat[MAX_ROWS][MAX_COLUMNS], float bias[MAX_ROWS][MAX_COLUMNS],int size_r, int size_c){

    switch(NOM) {
        case 0 :    // RELU
            for (int i = 0; i < size_r ; i++){
                for (int j = 0; j < size_c ; j++){
                    if (mat[i][j] + bias[i][j]>0){
                        mat[i][j] = mat[i][j] + bias[i][j];
                    }
                    else{
                        mat[i][j] = 0;
                    }
                }
            }
            break;

        case 1 :   // SIGMOID
            for (int i = 0; i < size_r ; i++){
                for (int j = 0; j < size_c ; j++){
                    mat[i][j] = 1/(1 + expf(-(mat[i][j] + bias[i][j])));
                }
            }
            break;

        case 2 : // TANH
            for (int i = 0; i < size_r ; i++){
                for (int j = 0; j < size_c ; j++){
                    mat[i][j] = tanhf(mat[i][j] + bias[i][j]);
                }
            }
            break;

        default :
            printf("Ceci n'est pas une fonction valide\n" );
    }
}

void NN_forward(float z[MAX_ROWS][MAX_COLUMNS] ,float filter_list[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS],
    int * p_size_r, int * p_size_c, int kernel_size, int padding,
    float biases_list[NB_LAYERS][MAX_ROWS][MAX_COLUMNS], int pool,
    float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS],
    int FC_sizes[NB_FC_LAYERS+1], float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float CNN_activs[NB_LAYERS * 2 + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD], int CNN_sizes[NB_LAYERS+1][2]){

    normalise_image(z,*p_size_r,*p_size_c);
    //printf("\nmat after normalisation :\n");
    //print_mat(z,*p_size_r,*p_size_c);

    /*
    int corr = 0;

    if(padding == 0){
        corr = 1;
    }
    */


    for (int n = 0; n < NB_LAYERS; n++){

        /*
        printf("\nConv Layer number %d :\n", n+1);
        printf("biases of this layer :\n");
        print_mat(biases_list[n], *p_size_r - 2*(kernel_size/2)*corr, *p_size_c - 2*(kernel_size/2)*corr);

        printf("filter of this layer :\n");
        print_w(filter_list[n], kernel_size);
        */



        if(padding!=0){

            padding = 0.5*(kernel_size - 1);

            for (int i = 0; i < *p_size_r + 2*padding; i ++){
                for(int j = 0; j < *p_size_c + 2*padding; j ++){
                    CNN_activs[2*n][i][j] = 0;
                }
            }

            for (int i = 0; i < *p_size_r; i++){
                for (int j = 0; j < *p_size_c; j++){
                    CNN_activs[2*n][i + padding][j + padding] = z[i][j];
                }
            }

            CNN_sizes[n][0] = *p_size_r + 2*padding;
            CNN_sizes[n][1] = *p_size_c + 2*padding;
        }

        else{
            for (int i = 0; i < *p_size_r; i++){
                for (int j = 0; j < *p_size_c; j++){
                    CNN_activs[2*n][i][j] = z[i][j];
                }
            }

            CNN_sizes[n][0] = *p_size_r;
            CNN_sizes[n][1] = *p_size_c;
        }

        convol(z, p_size_r, p_size_c, kernel_size, filter_list[n], padding);
        activation(z, biases_list[n], *p_size_r, *p_size_c);

        for (int i = 0; i < *p_size_r; i++){
            for (int j = 0; j < *p_size_c; j++){
                CNN_activs[2*n + 1][i][j] = z[i][j];
            }
        }

        /*
        printf("\n");
        print_activs(CNN_activs[2*n + 1], *p_size_r, *p_size_c);
        printf("\n");
        */


        if (pool == 0) max_pool(z, p_size_r, p_size_c);
        else avg_pool(z, p_size_r, p_size_c);

        //printf("Result layer %d :\n", n+1);
        //print_mat(z, *p_size_r, *p_size_c);
    }

    CNN_sizes[NB_LAYERS][0] = *p_size_r;
    CNN_sizes[NB_LAYERS][1] = *p_size_c;


    int size_r = *p_size_r;
    int size_c = *p_size_c;
    FC_sizes[0] = size_c*size_r;

    float z_flat[MAX_COLUMNS*MAX_ROWS];
    flattening(z,size_r, size_c, z_flat);

    for (int i = 0; i < FC_sizes[0]; i++ ){
        FC_activs[0][i] = z_flat[i];
    }

    for (int m = 0; m < NB_FC_LAYERS; m++){

        /*
        printf("\nFC Layer number %d :\n", m+1);

        printf("entry of this layer:\n");
        print_mat_flat(z_flat, FC_sizes[m]);

        printf("weights of this layer :\n");
        print_weights(FC_weights[m], FC_sizes[m + 1], FC_sizes[m]);

        printf("biases of this layer :\n");
        print_mat_flat(FC_biases[m], FC_sizes[m + 1]);
        */

        fully_connected_forward(z_flat,FC_sizes[m],FC_weights[m],FC_sizes[m+1],FC_biases[m]);

        //printf("FC output of this layer:\n");
        //print_mat_flat(z_flat, FC_sizes[m+1]);

        for (int i = 0; i < FC_sizes[m + 1]; i++ ){
            FC_activs[m + 1][i] = z_flat[i];
        }
    }
}

void normalise_image(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c){

    for (int i = 0 ; i < size_r; i++){
        for (int j = 0 ; j < size_c; j++){
            //mat[i][j] = mat[i][j]/127.5f - 1.0f;
            mat[i][j] = mat[i][j]/255.0f;
        }
    }
}

void init_biases(float biases[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c){

    for (int i = 0; i < size_r; i ++){
        for(int j = 0; j < size_c; j ++){
            biases[i][j] = 0.0;
        }
    }
}

void const_init_FC_weights(float FC_weights[MAX_COLUMNS*MAX_ROWS][MAX_ROWS*MAX_COLUMNS], int input_size,
    int output_size, float cst){

    for (int i = 0; i < output_size; i ++){
        for(int j = 0; j < input_size; j ++){
            FC_weights[i][j] = cst;
        }
    }
}

void random_init_FC_weights(float FC_weights[MAX_COLUMNS*MAX_ROWS][MAX_ROWS*MAX_COLUMNS], int input_size,
    int output_size, int seed){
    srand(seed);

    for (int i = 0; i < output_size; i ++){
        for(int j = 0; j < input_size; j ++){
            FC_weights[i][j] = ((float)rand()/(float)(RAND_MAX))*2 - 1;
            //FC_weights[i][j] = ((float)rand()/(float)(RAND_MAX));
        }
    }
}

void init_FC_biases(float FC_biases[MAX_COLUMNS*MAX_ROWS], int output_size){

    for (int i = 0; i < output_size; i ++){
        FC_biases[i] = 0.0;
    }
}

void fully_connected_forward(float x[MAX_ROWS*MAX_COLUMNS],int input_size,
    float W[MAX_COLUMNS*MAX_ROWS][MAX_ROWS*MAX_COLUMNS], int output_size,float FC_bias[MAX_COLUMNS*MAX_ROWS]){

    W_x_product(W, x, input_size, output_size);

    activationFC(x, FC_bias,input_size, output_size);


}

void activationFC(float flattened[MAX_ROWS*MAX_COLUMNS], float FC_bias[MAX_COLUMNS*MAX_ROWS],int input_size,int output_size){

    switch(NOM) {
        case 0 :    // RELU
            for (int i = 0; i < output_size ; i++){
                if (flattened[i] + FC_bias[i] > 0){
                    flattened[i] = flattened[i] + FC_bias[i];
                }
                else{
                    flattened[i] = 0;
                }
            }
            break;

        case 1 :   // SIGMOID
            for (int i = 0; i < output_size ; i++){
                flattened[i] = 1/(1 + expf(-(flattened[i] + FC_bias[i])));
            }
            break;

        case 2 : // TANH
            for (int i = 0; i < output_size ; i++){
                flattened[i] = tanhf(flattened[i] + FC_bias[i]);
            }
            break;

        default :
            printf("Ceci n'est pas une fonction valide\n" );
    }
}

void W_x_product(float W[MAX_COLUMNS*MAX_ROWS][MAX_ROWS*MAX_COLUMNS],float x[MAX_ROWS*MAX_COLUMNS], int input_size, int output_size){
    float sum = 0;
    float y[MAX_ROWS*MAX_COLUMNS];

    for (int c = 0; c < output_size; c++) {
        for (int k = 0; k < input_size; k++) {
          sum += W[c][k]*x[k];
        }
        y[c] = sum;
        sum = 0;
    }
    for (int c = 0; c < output_size ; c++) x[c] = y[c];
}

void flattening(float z[MAX_ROWS][MAX_COLUMNS],int size_r, int size_c, float z_flat[MAX_COLUMNS*MAX_ROWS]){

    for (int i=0 ; i < size_r ; i++){
        for (int j=0 ; j < size_c ; j++){

            z_flat[i*size_c + j] = z[i][j];
        }
    }
}
