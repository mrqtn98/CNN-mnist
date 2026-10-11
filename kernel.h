#ifndef KERNEL_H_INCLUDED
#define KERNEL_H_INCLUDED

#define MAX_F_ROWS 10 //kernel_size_max
#define MAX_F_COLUMNS 10 //kernel_size_max
#define PAD 2 //kernel_size - 1
#define POOL_SIZE 2 // Pooling : POOL_SIZExPOOL_SIZE

#include "convolu.h"
#include "nn.h"

void identity_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size);

void opposite_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size);

void init_filters(float filter_list[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size);

void print_w(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size);

void random_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size, int seed);

void const_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size, float cst);

/*

void vertical_edges_detector_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size);

void horizontal_edges_detector_kernel(float w[MAX_F_ROWS][MAX_F_COLUMNS], int kernel_size);

*/


#endif // CONVOLU_H_INCLUDED
