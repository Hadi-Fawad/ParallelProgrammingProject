#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Define names if not defined already
#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME baseline_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME baseline_collect
#endif  

#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME baseline_allocate
#endif

#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME baseline_free
#endif

void COMPUTE_NAME(int m0, int k0,
                  float *input_distributed,
                  float *weights_distributed,
                  float *output_distributed)
{
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Computing on all ranks
    for (int i0 = 0; i0 < m0; ++i0) {
        float res = 0.0f;
        for (int p0 = 0; p0 < k0; ++p0) {
            res += input_distributed[(p0 + i0) % m0] * weights_distributed[p0];
        }
        output_distributed[i0] = res;
    }
}

void DISTRIBUTED_ALLOCATE_NAME(int m0, int k0, float **input_distributed, float **weights_distributed, float **output_distributed)
{
    *input_distributed = (float *)malloc(sizeof(float) * m0);
    *output_distributed = (float *)malloc(sizeof(float) * m0);
    *weights_distributed = (float *)malloc(sizeof(float) * k0);
}

void DISTRIBUTE_DATA_NAME(int m0, int k0, float *input_sequential, float *weights_sequential, float *input_distributed, float *weights_distributed) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    MPI_Bcast(weights_sequential, k0, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rid == 0) {
        for (int i = 0; i < k0; ++i) {
            weights_distributed[i] = weights_sequential[i];
        }
    }

    // Handle the possibility that m0 is not divisible by num_ranks
    int portion = m0 / num_ranks; // Basic portion size
    int extra = m0 % num_ranks;   // Extra elements
    int *sendcounts = malloc(num_ranks * sizeof(int));
    int *displs = malloc(num_ranks * sizeof(int));
    int cum_sum = 0;
    
    for (int i = 0; i < num_ranks; i++) {
        sendcounts[i] = (i < extra ? portion + 1 : portion) * k0;
        displs[i] = cum_sum * k0;
        cum_sum += sendcounts[i] / k0;
    }

    int local_m0 = (rid < extra ? portion + 1 : portion) * k0;

    MPI_Scatterv(input_sequential, sendcounts, displs, MPI_FLOAT, input_distributed, local_m0, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(sendcounts);
    free(displs);
}


void COLLECT_DATA_NAME(int m0, int k0, float *output_distributed, float *output_sequential) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int portion = m0 / num_ranks;
    int extra = m0 % num_ranks;
    int *recvcounts = malloc(num_ranks * sizeof(int));
    int *displs = malloc(num_ranks * sizeof(int));
    int cum_sum = 0;
    
    for (int i = 0; i < num_ranks; i++) {
        recvcounts[i] = (i < extra ? portion + 1 : portion) * k0;
        displs[i] = cum_sum * k0;
        cum_sum += recvcounts[i] / k0;
    }

    int local_m0 = (rid < extra ? portion + 1 : portion) * k0;

    MPI_Gatherv(output_distributed, local_m0, MPI_FLOAT, output_sequential, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(recvcounts);
    free(displs);
}


void DISTRIBUTED_FREE_NAME(int m0, int k0, float *input_distributed, float *weights_distributed, float *output_distributed)
{
    free(input_distributed);
    free(weights_distributed);
    free(output_distributed);
}
