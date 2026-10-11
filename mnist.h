#ifndef MNIST_H_INCLUDED
#define MNIST_H_INCLUDED

#include "convolu.h"

void mnist_loader(int train_image[60000][28][28], int train_label[60000][1],
    int test_image[10000][28][28], int test_label[10000][1]);

void init_mat_mnist(float mat[MAX_ROWS][MAX_COLUMNS], int image[28][28]);

#endif // MNIST_H_INCLUDED
