#include <stdio.h>
#include <stdlib.h>

#include "train.h"

void generate_horizontal(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c, int seed){

    srand(seed);

    for (int i = 0; i < size_r; i ++){
        for(int j = 0; j < size_c; j ++){

            if (i%2 == 0){
                mat[i][j] = 0;
            }

            else mat[i][j] = rand()%122 + 122;
        }
    }
}

void generate_vertical(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c, int seed){

    srand(seed);

    for (int i = 0; i < size_r; i ++){
        for(int j = 0; j < size_c; j ++){
            if (j%2 == 0){
                mat[i][j] = 0;
            }

            else mat[i][j] = rand()%122 + 122;
        }

    }
}


int pos_max_output(float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS], int FC_sizes[NB_FC_LAYERS+1]){

    int n = FC_sizes[NB_FC_LAYERS];
    int pos = 0;
    float max = FC_activs[NB_FC_LAYERS][0];

    for (int j = 1; j < n; j++){
        if (FC_activs[NB_FC_LAYERS][j] >= max){
            pos = j;
            max = FC_activs[NB_FC_LAYERS][j];
        }

    }

    return pos;

}

void generate_control(float control[MAX_COLUMNS*MAX_ROWS], int label, int size_output){

    for (int j = 0; j < size_output; j++){
        if(label == j){
            control[j] = 1;
        }

        else control[j] = 0;
    }
}

