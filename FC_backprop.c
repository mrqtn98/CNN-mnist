#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "FC_backprop.h"

float cost_function(float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], float control[MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1]){

    int n = FC_sizes[NB_FC_LAYERS];
    float c = 0.0f;

    for (int j = 0; j < n; j++){
        if(COST == 0){
            c += (float)1/n * (FC_activs[NB_FC_LAYERS][j] - control[j]) * (FC_activs[NB_FC_LAYERS][j] - control[j]);
        }

        else{
            c += -(control[j] * logf(FC_activs[NB_FC_LAYERS][j]) + (1 - control[j]) * logf(1 - FC_activs[NB_FC_LAYERS][j]));
        }
    }

    return c;

}


void backprop(float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float control[MAX_ROWS*MAX_COLUMNS], float grad_w[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float grad_b[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS],
    float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS]){


    for (int l = NB_FC_LAYERS; l > 0; l --){
        int output_size = FC_sizes[l];
        int input_size = FC_sizes[l-1];

        for (int j = 0; j < output_size; j++){
            for (int i = 0; i < input_size; i++){
                grad_w[l][j][i] = FC_activs[l-1][i] * deriv_activ(FC_activs[l][j])
                * deriv_cost(l, FC_activs, j, FC_weights, FC_biases, FC_sizes,control, deriv_activs_FC);
            }

            grad_b[l][j] = deriv_activ(FC_activs[l][j]) *
            deriv_cost(l, FC_activs, j, FC_weights, FC_biases, FC_sizes,control, deriv_activs_FC);
        }
    }
}


float deriv_activ(float a){
    float d = 0;
    switch(NOM) {
        case 0 :    // RELU
            if (a > 0){
                d = 1;
            }
            else{
                d = 0;
            }

            break;

        case 1 :   // SIGMOID
            d = a * (1 - a);

            break;

        case 2 : // TANH
            d = 1 - a * a;

            break;

        default :
            printf("Ceci n'est pas une fonction valide\n" );
    }

    return d;
}


float deriv_cost(int l, float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], int j, float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS],
    float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1],
    float control[MAX_ROWS*MAX_COLUMNS], float deriv_activs_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS]){

    float d = 0;

    if (l == NB_FC_LAYERS){
        if (COST == 0){
            d = (float)1/FC_sizes[l] * 2 * (FC_activs[l][j] - control[j]);
        }

        else{
            d = - (control[j] * (float)1/FC_activs[l][j] -
            (1 - control[j]) * (float)1/(1 - FC_activs[l][j]));
        }

        deriv_activs_FC[l][j] = d;
        return d;
    }

    float output_size = FC_sizes[l + 1];
    for (int p = 0; p < output_size; p++){
        d += FC_weights[l][p][j] * deriv_activ(FC_activs[l + 1][p])
        * deriv_activs_FC[l + 1][p];
    }

    deriv_activs_FC[l][j] = d;

    return d;
}
