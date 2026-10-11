#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "convolu.h"
#include "kernel.h"
#include "nn.h"
#include "FC_backprop.h"
#include "CNN_backprop.h"
#include "train.h"
#include "backprop_runner.h"
#include "mnist.h"


int main(){

    /////////////////////////////////////// CNN size and characteristics ////////////////////////////////////////////////

    /*
    Change the constants NB_FC_LAYERS, NB_LAYERS and NOM if you want in the "nn.h"
    */

    int base_size_r = 28; //number of rows of the input
    int base_size_c = 28; //number of columns of the input
    int kernel_size = 3; // change PAD const if you change kernel_size !! (cf kernel.h)
    int padding = 0; // activation of padding or not (padding = 0 padding deactivated, padding = 1 padding activated)
    int pool = 1; // pool == 0 for max_pool and pool == 1 for avg_pool
    static int FC_sizes[NB_FC_LAYERS+1] = {0, 128, 10};
    // {0 becomes size of the output of conv layers in the code, size_FC_layer1, size_FC_layer2, ..., size_FC_output_layer}

    ///////////////////////////////////// Learning and test variables /////////////////////////////////////

    float lr = 1.0f; // initial learning rate (it decays by a factor BETA at each iteration, cf backprop_runner.h)
    int nb_iter_max = 25000;
    int nb_test = 250;
    float seek_accur = 1.0f; // early stop if the accuracy on the nb_test images reaches this value (1.0f : no early stop)
    int size_avg = 8; // batch_size

    ///////////////////////////////////////////// DO NOT TOUCH THIS PART ////////////////////////////////////////

    static int train_image[60000][28][28];
    static int train_label[60000][1];

    static int test_image[10000][28][28];
    static int test_label[10000][1];

    mnist_loader(train_image, train_label, test_image, test_label);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    char str[200];
    sprintf(str,"data/ACCUR__lr %.4f,nb_test %d,nb_iter_max %d,size_avg %d,size_r %d,size_c %d,NB_FC_LAYERS %d,NB_CNN_LAYERS %d,kernel_size %d,pool %d,size_pool %d,padding %d",
    lr, nb_test, nb_iter_max, size_avg,base_size_r,base_size_c,NB_FC_LAYERS,NB_LAYERS,kernel_size,pool, POOL_SIZE, padding);


    FILE *fp;

    char* filename;
    filename = strcat(str,".csv");

    printf("%s\n\n", filename);

    fp = fopen(filename,"w+");
    if(fp == NULL){
       perror("fopen error: ");
       exit(1);
    }
    fprintf(fp,"num_iter,cost,accur,time");



    static float filter_list[NB_LAYERS][MAX_F_ROWS][MAX_F_COLUMNS];
    static float biases_list[NB_LAYERS][MAX_ROWS][MAX_COLUMNS];
    static float FC_weights[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS];
    static float FC_biases[NB_FC_LAYERS][MAX_COLUMNS*MAX_ROWS];
    static float FC_activs[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS];
    static float CNN_activs[NB_LAYERS * 2 + 1][MAX_ROWS + PAD][MAX_COLUMNS + PAD];
    static int CNN_sizes[NB_LAYERS+1][2];
    static float control[MAX_COLUMNS*MAX_ROWS];

    static float mat[MAX_ROWS][MAX_COLUMNS];

    for (int n = 0; n < NB_LAYERS; n++){
        int seed = 1000 + n;
        random_kernel(filter_list[n], kernel_size, seed);
        //const_kernel(filter_list[n], kernel_size, 0.9);
        init_biases(biases_list[n], MAX_ROWS, MAX_COLUMNS);

    }

    for (int m = 0; m < NB_FC_LAYERS; m++){
        int seed = 2000 + m;
        random_init_FC_weights(FC_weights[m], MAX_COLUMNS*MAX_ROWS, MAX_COLUMNS*MAX_ROWS, seed);
        //const_init_FC_weights(FC_weights[m], MAX_COLUMNS*MAX_ROWS, MAX_COLUMNS*MAX_ROWS, 0.9);
        init_FC_biases(FC_biases[m], MAX_COLUMNS*MAX_ROWS);
    }

    /////////////////////////////////////////////////////////  Train CNN ////////////////////////////////////////////////:

    float cost = 10000;
    int i = 0;
    float max_accur = 0;
    float time = 0;

    static float grad_w_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS][MAX_COLUMNS*MAX_ROWS];
    static float grad_b_FC[NB_FC_LAYERS + 1][MAX_COLUMNS*MAX_ROWS];

    static float grad_w_CNN[NB_LAYERS + 1][MAX_F_ROWS][MAX_F_COLUMNS];
    static float grad_b_CNN[NB_LAYERS + 1][MAX_ROWS][MAX_COLUMNS];

    while ((float)max_accur/nb_test < seek_accur){

        init_grad_w_CNN(grad_w_CNN);
        init_grad_b_CNN(grad_b_CNN);
        init_grad_w_FC(grad_w_FC);
        init_grad_b_FC(grad_b_FC);
        float cost_avg = 0;

        for(int z = 0; z < size_avg; z ++){

            int size_r = base_size_r;
            int size_c = base_size_c;
            int seed = 1000 + i * 1000 + z;
            int label = 0;
            int accur = 0;

            if(z == size_avg - 1){

                for (int x = 0; x < nb_test; x++){

                    int size_r = base_size_r;
                    int size_c = base_size_c;
                    int seed = 30000 + x;
                    int label = 0;

                    srand(seed);

                    int s = rand()%10000;

                    init_mat_mnist(mat, test_image[s]);
                    label = test_label[s][0];


                    NN_forward(mat, filter_list, &size_r, &size_c, kernel_size, padding, biases_list, pool, FC_weights, FC_biases,
                    FC_sizes, FC_activs, CNN_activs, CNN_sizes);

                    if (pos_max_output(FC_activs, FC_sizes) == label){
                        accur++;
                    }

                }
            }

            srand(seed);

            int s = rand()%60000;

            init_mat_mnist(mat, train_image[s]);
            label = train_label[s][0];

            generate_control(control, label, FC_sizes[NB_FC_LAYERS]);

            clock_t startf = clock();

            NN_forward(mat, filter_list, &size_r, &size_c, kernel_size, padding, biases_list, pool, FC_weights, FC_biases,
            FC_sizes, FC_activs, CNN_activs, CNN_sizes);

            clock_t endf = clock();

            time += (float) (endf - startf) / CLOCKS_PER_SEC;

            cost_avg += (float)cost_function(FC_activs, control, FC_sizes)/size_avg;

            if(z == size_avg - 1) {
                cost = cost_avg;

                printf("cost of iter_nb %d : %.5f\n", i, cost);
                printf("Accuracy of the CNN after %d tests : %.2f %%\n\n", nb_test,(float)accur/nb_test * 100);

                //fprintf(fp ,"\n%d,%.5f,%.2f,%.3f", i, cost, (float)accur/nb_test, time);

                if (max_accur < accur){
                    fprintf(fp ,"\n%d,%.5f,%.2f,%.3f", i, cost, (float)accur/nb_test, time);
                    max_accur = accur;
                }

                if ((float)max_accur/nb_test >= seek_accur){
                    break;
                }

                if(i == nb_iter_max){
                    break;
                }
            }

            clock_t startb = clock();

            run_stochastic_backprop(CNN_activs, filter_list, biases_list, CNN_sizes, FC_activs, FC_weights, FC_biases, FC_sizes, control,
            kernel_size, pool, lr, padding, grad_w_FC, grad_b_FC, grad_w_CNN, grad_b_CNN, z, size_avg, i);

            clock_t endb = clock();

            time += (float) (endb - startb) / CLOCKS_PER_SEC;
        }

        if ((float)max_accur/nb_test >= seek_accur){
            break;
        }

        if(i == nb_iter_max){
            printf("\nNombre d'iteration max atteint !!\n");
            break;
        }

        i++;
    }

    ////////////////////////////////////////////////////// Test CNN /////////////////////////////////////////////////

    int accur = 0;

    for (int x = 0; x < 10000; x++){

        int size_r = base_size_r;
        int size_c = base_size_c;
        int label = 0;

        init_mat_mnist(mat, test_image[x]);
        label = test_label[x][0];

        NN_forward(mat, filter_list, &size_r, &size_c, kernel_size, padding, biases_list, pool, FC_weights, FC_biases,
        FC_sizes, FC_activs, CNN_activs, CNN_sizes);

        if (pos_max_output(FC_activs, FC_sizes) == label){
            accur++;
        }

    }

    printf("\nAccuracy of the CNN after %d tests : %.2f %%\n", 10000,(float)accur/10000 * 100);

    printf("\ntime : %.3f", time);

    fclose(fp);

    return 0;
}




