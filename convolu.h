#ifndef CONVOLU_H_INCLUDED
#define CONVOLU_H_INCLUDED

#define MAX_ROWS 40
#define MAX_COLUMNS 40

#include "kernel.h"

void print_mat(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c);

float filter(float tab[MAX_ROWS + PAD][MAX_COLUMNS + PAD],int kernel_size, float w[MAX_F_ROWS][MAX_F_COLUMNS], int i, int j);

float max_pool_filter(float mat[MAX_ROWS][MAX_COLUMNS], int i, int j);

float avg_pool_filter(float mat[MAX_ROWS][MAX_COLUMNS], int i, int j);

void convol(float mat[MAX_ROWS][MAX_COLUMNS], int * p_size_r, int * p_size_c,int kernel_size, float w[MAX_F_ROWS][MAX_F_COLUMNS], int padding_same);

void max_pool(float mat[MAX_ROWS][MAX_COLUMNS], int * p_size_r, int * p_size_c);

void avg_pool(float mat[MAX_ROWS][MAX_COLUMNS], int * p_size_r, int * p_size_c);

void init_mat(float mat[MAX_ROWS][MAX_COLUMNS], int size_r, int size_c);

void print_mat_flat(float flat[MAX_ROWS*MAX_COLUMNS], int size0);

void print_weights(float mat[MAX_ROWS*MAX_COLUMNS][MAX_ROWS*MAX_COLUMNS], int size_r, int size_c);

void print_activs(float mat[MAX_ROWS + PAD][MAX_COLUMNS + PAD], int size_r, int size_c);

#endif // CONVOLU_H_INCLUDED
