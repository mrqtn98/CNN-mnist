#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "convolu.h"

void mnist_loader(int train_image[60000][28][28], int train_label[60000][1],
    int test_image[10000][28][28], int test_label[10000][1]){

    FILE *fp = NULL;

	char *line,*record;

	char buffer[100000];

	char train_file[100];

    sprintf(train_file, "data/mnist/mnist_train.csv");

	if ((fp = fopen(train_file, "r")) != NULL) {

		int j = 0;
		int i = 0;

        while ((line = fgets(buffer, sizeof(buffer), fp))!=NULL){ //The loop continues when the end of the file is not read

			record = strtok(line, ",");
			train_label[i][0] = atoi(record);
			record = strtok(NULL, ",");

            while (record != NULL) { //Read the data of each row

                train_image[i][j/28][j%28] = atoi(record);

                record = strtok(NULL, ",");
                j++;
			}

			j = 0;
			i ++;

		}

		fclose(fp);
		fp = NULL;
	}
	else {
		perror(train_file);
		fprintf(stderr, "MNIST CSV file not found: see README.md (put mnist_train.csv and mnist_test.csv in data/mnist/)\n");
		exit(1);
	}

	char test_file[100];

	sprintf(test_file, "data/mnist/mnist_test.csv");

	if ((fp = fopen(test_file, "r")) != NULL) {

		int j = 0;
		int i = 0;

        while ((line = fgets(buffer, sizeof(buffer), fp))!=NULL){ //The loop continues when the end of the file is not read

			record = strtok(line, ",");
			test_label[i][0] = atoi(record);
			record = strtok(NULL, ",");

            while (record != NULL) { //Read the data of each row

                test_image[i][j/28][j%28] = atoi(record);

                record = strtok(NULL, ",");
                j++;
			}

			j = 0;
			i ++;

		}

		fclose(fp);
		fp = NULL;
	}
	else {
		perror(test_file);
		fprintf(stderr, "MNIST CSV file not found: see README.md (put mnist_train.csv and mnist_test.csv in data/mnist/)\n");
		exit(1);
	}

}

void init_mat_mnist(float mat[MAX_ROWS][MAX_COLUMNS], int image[28][28]){

    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            mat[i][j] = (float) image[i][j];
        }
    }
}



