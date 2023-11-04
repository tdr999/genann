#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "genann.h"

/* This example is to illustrate how to use GENANN.
 * It is NOT an example of good machine learning techniques.
 */

const char *iris_data = "example/iris.data";

double *input, *class;
int samples;
const char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

#define ADD_WEIGHT_CHANCE 0.3
#define DELETE_WEIGHT_CHANCE 0.2
#define CHANGE_WEIGHT_CHANCE 0.5
#define ADD_NEURON 0.2

void load_data() {
    /* Load the iris data-set. */
    FILE *in = fopen("example/iris.data", "r");
    if (!in) {
        printf("Could not open file: %s\n", iris_data);
        exit(1);
    }

    /* Loop through the data to get a count. */
    char line[1024];
    while (!feof(in) && fgets(line, 1024, in)) {
        ++samples;
    }
    fseek(in, 0, SEEK_SET);

    printf("Loading %d data points from %s\n", samples, iris_data);

    /* Allocate memory for input and output data. */
    input = malloc(sizeof(double) * samples * 4);
    class = malloc(sizeof(double) * samples * 3);

    /* Read the file into our arrays. */
    int i, j;
    for (i = 0; i < samples; ++i) {
        double *p = input + i * 4;
        double *c = class + i * 3;
        c[0] = c[1] = c[2] = 0.0;

        if (fgets(line, 1024, in) == NULL) {
            perror("fgets");
            exit(1);
        }

        char *split = strtok(line, ",");
        for (j = 0; j < 4; ++j) {
            p[j] = atof(split);
            split = strtok(0, ",");
        }

        split[strlen(split)-1] = 0;
        if (strcmp(split, class_names[0]) == 0) {c[0] = 1.0;}
        else if (strcmp(split, class_names[1]) == 0) {c[1] = 1.0;}
        else if (strcmp(split, class_names[2]) == 0) {c[2] = 1.0;}
        else {
            printf("Unknown class %s.\n", split);
            exit(1);
        }

        /* printf("Data point %d is %f %f %f %f  ->   %f %f %f\n", i, p[0], p[1], p[2], p[3], c[0], c[1], c[2]); */
    }

    fclose(in);
}

typedef struct genome_t {
    genann *ann;
    double acc;
} genome_t;

#define POP_SIZE 15

//objective is to make this into kind of a genetic algorithm
int main(int argc, char *argv[])
{

    srand(time(0));

    /* Load the data from file. */
    load_data();

    /* 4 inputs.
     * 1 hidden layer(s) of 4 neurons.
     * 3 outputs (1 per class)
     */
    int i, j;
    // genann *ann = genann_init(4, 1, 4, 3);


    double err;
    // double last_err = 1000;
    // int generations = 30;
    int gen_counter = 0;
    // how often a cataclysm appears
    int cataclysm_counter = 500;
    // int correct = 0;
    genome_t *best_global_genome = (genome_t*)malloc(sizeof(genome_t));
    memset(best_global_genome, 0, sizeof(genome_t));
    best_global_genome->acc = 1000;

    genome_t population[POP_SIZE];
    // init
    for (i = 0; i < POP_SIZE; i++)
    {
        population[i].acc = 0;
        population[i].ann = genann_init(4, 2, 3, 3);
        // genann_init_weights(population[i].ann);
        // genann_init_weights(population[i].ann);
        // genann_randomize(population[i].ann);
    }
    genome_t *best_genome = (genome_t *)malloc(sizeof(genome_t));
    memset(best_genome, 0, sizeof(genome_t));
    do
    {
        if (gen_counter % cataclysm_counter == 0)
        {
            for (i = 0; i < POP_SIZE; i++)
            {
                population[i].acc = 0;
                // population[i].ann = genann_init(4, 2, 4, 3);
                // genann_init_weights(population[i].ann);
                // genann_init_weights(population[i].ann);
                genann_randomize(population[i].ann);
            }
        }
        for (i = 0; i < POP_SIZE; i++)
        {
            genann *ann = population[i].ann;

            err = 0;
            // evaluate genome code
            for (j = 0; j < samples; ++j)
            {
                /* See how we did. */
                const double *guess = genann_run(ann, input + j * 4);
                err += pow(guess[0] - class[j * 3 + 0], 2.0);
                err += pow(guess[1] - class[j * 3 + 1], 2.0);
                err += pow(guess[2] - class[j * 3 + 2], 2.0);
            }
            population[i].acc = err;
        }

        // sort by acc, but for now, we just find min acc, as in, min error
        float max_acc = 1000;
        float_t avg_gen_acc = 0.0f;
        for (i = 0; i < POP_SIZE; i++)
        {
            genome_t *g = &population[i];
            if (g->acc < max_acc)
            {
                max_acc = g->acc;
                // best_genome = g;
                memcpy(best_genome, g, sizeof(genome_t));
                memset(g, 0, sizeof(genome_t));
            }
            avg_gen_acc += g->acc / POP_SIZE;
        }

        // // for the first gen, initialize the best global genome
        // if (gen_counter == 0)
        // {
        //     memcpy(best_global_genome, best_genome, sizeof(genome_t));
        // }
        memset(population, 0, sizeof(genome_t) * POP_SIZE);

        printf("best_glob %f best genom acc %f avg gen acc %f in gen %d\n", best_global_genome->acc, best_genome->acc, avg_gen_acc, gen_counter);

        // create new pop, in this case, copy best genome
        // if global genome is better than current, use global
        for (i = 0; i < POP_SIZE; i++)
        {
            genome_t *g = &population[i];
            if (best_genome->acc < best_global_genome->acc)
            {
                // memset(g, 0, sizeof(genome_t));
                // best_global_genome = best_genome;
                memcpy(best_global_genome, best_genome, sizeof(genome_t));
                memcpy(g, best_genome, sizeof(genome_t));
            }
            else
            {
                memcpy(g, best_global_genome, sizeof(genome_t));
            }
        }

        // mutate, for now
        for (i = 0; i < POP_SIZE; i++)
        {
            genann *g = population[i].ann;
            genann_randomize(g);

            float_t mutation = GENANN_RANDOM();
            if (mutation < ADD_WEIGHT_CHANCE)
            {
                // with the mention that adding a neuron is equivalent to adding 2 weights
                // but we can leave it at here, because of how neat works
                add_weight(g);
                // printf("ADD WEIGHT\n");

            }
            if (mutation > ADD_WEIGHT_CHANCE && mutation < CHANGE_WEIGHT_CHANCE + ADD_WEIGHT_CHANCE)
            {
                genann_mutate_weight(g);
                // genann_randomize(g);
                // genann_mutate_weight2(g);
                // printf("MUTATE WEIGHT\n");
            }
            // if (mutation > ADD_WEIGHT_CHANCE + CHANGE_WEIGHT_CHANCE && mutation < ADD_WEIGHT_CHANCE + CHANGE_WEIGHT_CHANCE + DELETE_WEIGHT_CHANCE)
            // {
            //     // printf("DELETE WEIGHT\n");
            //     delete_weight(g);
            // }

            // genann_mutate_weight2(g);
        }

        gen_counter++;
        // } while (gen_counter < generations);
        } while (best_global_genome->acc > 40.0f);
        printf("Best global acc %f\n", best_global_genome->acc );

        // evaluate code

        // get best genome
        genann *ann = best_global_genome->ann;

        int correct = 0;
        for (j = 0; j < samples; ++j)
        {
            const double *guess = genann_run(ann, input + j * 4);
            if (class[j * 3 + 0] == 1.0)
            {
                if (guess[0] > guess[1] && guess[0] > guess[2])
                    ++correct;
            }
            else if (class[j * 3 + 1] == 1.0)
            {
                if (guess[1] > guess[0] && guess[1] > guess[2])
                    ++correct;
            }
            else if (class[j * 3 + 2] == 1.0)
            {
                if (guess[2] > guess[0] && guess[2] > guess[1])
                    ++correct;
            }
            else
            {
                printf("Logic error.\n");
                exit(1);
            }
            // printf(" random guess was %f %f %f\n", guess[0], guess[1], guess[2]);
        }

        printf("%d/%d correct (%0.1f%%).\n", correct, samples, (double)correct / samples * 100.0);
        printf(" Best genome error %f\n", best_global_genome->acc);
        free(best_global_genome);

    // genann_free(ann);
    free(input);
    free(class);

    return 0;
}
