# Compilation du CNN : C99, aucune dependance hormis la bibliotheque mathematique (-lm)
CC     = gcc
CFLAGS = -std=c99 -Wall -O3
LDLIBS = -lm

SRC = main.c convolu.c kernel.c nn.c CNN_backprop.c FC_backprop.c backprop_runner.c train.c mnist.c
OBJ = $(SRC:.c=.o)
HDR = $(wildcard *.h)

.PHONY: all run clean

all: cnn

cnn: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# une modification d'un .h (nn.h, kernel.h, ...) recompile tout
%.o: %.c $(HDR)
	$(CC) $(CFLAGS) -c $<

run: cnn
	./cnn

clean:
	rm -f $(OBJ) cnn
