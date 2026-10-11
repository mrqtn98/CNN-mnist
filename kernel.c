#include <stdio.h>
#include <stdlib.h>

#include "kernel.h"


void identity_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size){

    for (int i = 0; i < kernel_size; i ++){
        for (int j = 0; j < kernel_size; j ++){
            w[i][j] = 0;
        }
    }

    w[kernel_size/2][kernel_size/2] = 1;
}

void random_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size, int seed){
    srand(seed);

    for (int i = 0; i < kernel_size; i ++){
        for (int j = 0; j < kernel_size; j ++){
            w[i][j] = ((float)rand()/(float)(RAND_MAX))*2 - 1;
            //w[i][j] = ((float)rand()/(float)(RAND_MAX));

        }
    }
}

void const_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size, float cst){

    for (int i = 0; i < kernel_size; i ++){
        for (int j = 0; j < kernel_size; j ++){
            w[i][j] = cst;
        }
    }
}


void opposite_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size){

    for (int i = 0; i < kernel_size; i ++){
        for (int j = 0; j < kernel_size; j ++){
            w[i][j] = 0;
        }
    }

    w[kernel_size/2][kernel_size/2] = -1;
}

void init_filters(float filter_list[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size){

    for (int n = 0 ; n < NB_LAYERS ; n++){
        for (int i = 0 ; i < kernel_size ; i++){
            for (int j = 0 ; j < kernel_size ; j++){
                float random_weight= ((float)rand()/(float)(RAND_MAX))*2 - 1;
                filter_list[n][i][j] = random_weight;
            }
        }
    }
}

void print_w(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size){
    for (int i = 0; i < kernel_size; i ++){
        for(int j = 0; j < kernel_size; j ++){
            printf("%.1f  ", w[i][j]);
        }
        printf("\n");
    }
}


/*
void vertical_edges_detector_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size){
    for (int i = 0; i < 3; i ++){
        w[i][0] = 1;
        w[i][1] = 0;
        w[i][2] = -1;
    }
}

void horizontal_edges_detector_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size){
    for (int j = 0; j < 3; j ++){
        w[0][j] = 1;
        w[1][j] = 0;
        w[2][j] = -1;
    }
}

void gaussian_blur_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size){
    for (int i = 0; i < 2; i += 2){
        w[i][0] = 1;
        w[i][1] = 2;
        w[i][2] = 1;
    }
    for (int j = 0; j < 3; j ++){
        w[1][j] = 2*w[0][j];
    }
}
*/
