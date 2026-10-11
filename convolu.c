#include <stdio.h>
#include <stdlib.h>

#include "convolu.h"
#include "kernel.h"


void convol(float mat[MAX_ROWS][MAX_COLUMNS], int * p_size_r, int * p_size_c,int kernel_size, float w[MAX_F_ROWS][MAX_F_COLUMNS], int padding){

    int size_r = *p_size_r;
    int size_c = *p_size_c;

    if (padding!=0){
        padding = 0.5*(kernel_size - 1);
        //printf("Padding activated\n");

    }

    //else printf("Padding desactivated\n");

    float tab[MAX_ROWS + PAD][MAX_COLUMNS + PAD];

    if(padding!=0){

        for (int i = 0; i < size_r + 2*padding; i ++){
            for(int j = 0; j < size_c + 2*padding; j ++){
                tab[i][j] = 0;
            }
        }

        for (int i = padding; i < size_r + padding ; i ++){
            for(int j = padding; j < size_c + padding ; j ++){
                tab[i][j] = mat[i-padding][j-padding];
            }
        }

        //calcul convolution
        for (int i = padding; i < size_r + padding ; i ++){
            for(int j = padding; j < size_c + padding ; j ++){
                mat[i - padding][j - padding] = filter(tab, kernel_size , w, i, j);
            }
        }

    }

    else{ // Convolution classique si on n'utilise pas de remplissage

        for (int i = 0; i < size_r; i ++){
            for(int j = 0 ; j < size_c; j ++){
                tab[i][j] = mat[i][j];
            }
        }

        for (int i = kernel_size/2; i < size_r - kernel_size/2; i ++){
            for(int j = kernel_size/2; j < size_c - kernel_size/2; j ++){
                mat[i - kernel_size/2][j - kernel_size/2] = filter(tab, kernel_size , w, i, j);
            }
        }


        *p_size_r = *p_size_r - kernel_size + 1;
        *p_size_c = *p_size_c - kernel_size + 1;
    }
}

void max_pool(float mat[MAX_ROWS][MAX_COLUMNS], int * p_size_r, int * p_size_c){
    // "Origine" de chaque carré (de taille paire ici) : en haut à gauche
    int size_r = *p_size_r;
    int size_c = *p_size_c;

    int stride = POOL_SIZE; // size_pool
    float tab[MAX_ROWS][MAX_COLUMNS];

    for (int i = 0; i < size_r/stride; i ++){
        for(int j = 0; j < size_c/stride; j ++){
            tab[i][j] = max_pool_filter(mat, i*stride, j*stride);
        }
    }

    *p_size_r = *p_size_r/stride;
    *p_size_c = *p_size_c/stride;
    size_r = *p_size_r;
    size_c = *p_size_c;

    for (int i = 0; i < size_r; i += 1){
        for(int j = 0; j < size_c; j += 1){
            mat[i][j] = tab[i][j];
        }
    }
}


void avg_pool(float mat[MAX_ROWS][MAX_COLUMNS], int * p_size_r, int * p_size_c){

    int size_r = *p_size_r;
    int size_c = *p_size_c;

    int stride = POOL_SIZE; // size_pool
    float tab[MAX_ROWS][MAX_COLUMNS];

    for (int i = 0; i < size_r/stride; i ++){
        for(int j = 0; j < size_c/stride; j ++){
            tab[i][j] = avg_pool_filter(mat, i*stride, j*stride);
        }
    }

    *p_size_r = *p_size_r/stride;
    *p_size_c = *p_size_c/stride;

    size_r = *p_size_r;
    size_c = *p_size_c;

    for (int i = 0; i < size_r; i += 1){
        for(int j = 0; j < size_c; j += 1){
            mat[i][j] = tab[i][j];
        }
    }
}

float filter(float tab[MAX_ROWS + PAD][MAX_COLUMNS + PAD],int kernel_size, float w[MAX_F_ROWS][MAX_F_COLUMNS], int i, int j){
    float sum = 0;
    for (int l = 0; l < kernel_size; l ++){ // +1 suivant la definition de a/b entiers ?
        for (int c = 0; c < kernel_size; c ++){ //idem
            sum += tab[i - kernel_size/2 + l][j - kernel_size/2 + c] * w[l][c];
        }
    }

    return sum;
}


float max_pool_filter(float mat[MAX_ROWS][MAX_COLUMNS], int i, int j){
    // "Origine" de chaque carré (de taille paire ici) : en haut à gauche
    float max = mat[i][j];
    for (int l = 0; l < POOL_SIZE; l ++){
        for (int c = 0; c < POOL_SIZE; c ++){
            if (mat[i + l][j + c]>=max){
                max=mat[i + l][j + c];
            }
        }
    }
    return max;
}

float avg_pool_filter(float mat[MAX_ROWS][MAX_COLUMNS], int i, int j){
    float sum = 0;
    for (int l = 0; l < POOL_SIZE; l ++){
        for (int c = 0; c < POOL_SIZE; c ++){
            sum += mat[i + l][j + c];
        }
    }

    return (float)sum/(POOL_SIZE*POOL_SIZE);
}

void print_mat(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c){
    for (int i = 0; i < size_r; i ++){
        for(int j = 0; j < size_c; j ++){
            printf("%.3f  ", mat[i][j]);
        }
        printf("\n");
    }
}

void print_activs(float mat[MAX_ROWS + PAD][MAX_COLUMNS + PAD], int size_r, int size_c){
    for (int i = 0; i < size_r; i ++){
        for(int j = 0; j < size_c; j ++){
            printf("%.3f  ", mat[i][j]);
        }
        printf("\n");
    }
}


void print_mat_flat(float flat[MAX_ROWS*MAX_COLUMNS], int size){
    for (int i = 0; i < size; i ++){
        printf("%.3f  ", flat[i]);
        printf("\n");
    }
}

void print_weights(float mat[MAX_ROWS*MAX_COLUMNS][MAX_ROWS*MAX_COLUMNS], int size_r, int size_c){
    for (int i = 0; i < size_r; i ++){
        for(int j = 0; j < size_c; j ++){
            printf("%.3f  ", mat[i][j]);
        }
        printf("\n");
    }
}

void init_mat(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c){

    for (int i = 0; i < size_r; i ++){
        for(int j = 0; j < size_c; j ++){
            mat[i][j] = 255/(size_r + size_c - 2) * (i + j);
        }
    }
}


