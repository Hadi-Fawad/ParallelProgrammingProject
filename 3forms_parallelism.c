#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <emmintrin.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME parallel_baseline
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME parallel_baseline_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME parallel_baseline_collect
#endif

#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME parallel_baseline_allocate
#endif

#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME parallel_baseline_free
#endif

void COMPUTE_NAME(int m0, int k0, float *input_distributed, float *weights_distributed, float *output_distributed) {
    int rid;
    int num_ranks;
    int tag = 0;
    MPI_Status status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Shared Memory Parallelism
    #pragma omp parallel for
    for (int i0 = 0; i0 < m0; ++i0) {
        float res = 0.0f;
        for (int p0 = 0; p0 < k0; ++p0) {
            // ILP Parallelism
            res += input_distributed[(p0 + i0) % m0] * weights_distributed[p0];
        }
        output_distributed[i0] = res;
    }
}




void DISTRIBUTED_ALLOCATE_NAME(int m0, int k0, float **input_distributed, float **weights_distributed, float **output_distributed) {
    int rid;
    int num_ranks;
    int tag = 0;
    MPI_Status status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Calculate the local range for each rank
    int local_start = rid * (m0 / num_ranks);
    int local_end = (rid == num_ranks - 1) ? m0 : (rid + 1) * (m0 / num_ranks);

    if (rid == root_rid) {
        // This block will only run on the node that matches root_rid.
        *input_distributed = (float *)malloc(sizeof(float) * (local_end - local_start));
        *output_distributed = (float *)malloc(sizeof(float) * (local_end - local_start));
        *weights_distributed = (float *)malloc(sizeof(float) * k0);
    } else {
        // This will run on all other nodes whose rid is not root_rid.
    }
}

void DISTRIBUTE_DATA_NAME(int m0, int k0, float *input_sequential, float *weights_sequential, float *input_distributed, float *weights_distributed) {
    int rid;
    int num_ranks;
    int tag = 0;
    MPI_Status status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Calculate the local range for each rank
    int local_start = rid * (m0 / num_ranks);
    int local_end = (rid == num_ranks - 1) ? m0 : (rid + 1) * (m0 / num_ranks);

    if (rid == root_rid) {
        // This block will only run on the node that matches root_rid.
        // Distribute the inputs
        for (int i0 = local_start; i0 < local_end; ++i0)
            input_distributed[i0 - local_start] = input_sequential[i0];

        // Distribute the weights
        for (int p0 = 0; p0 < k0; ++p0)
            weights_distributed[p0] = weights_sequential[p0];
    } else {
        // This will run on all other nodes whose rid is not root_rid.
    }
}

void COLLECT_DATA_NAME(int m0, int k0, float *output_distributed, float *output_sequential) {
    int rid;
    int num_ranks;
    int tag = 0;
    MPI_Status status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Calculate the local range for each rank
    int local_start = rid * (m0 / num_ranks);
    int local_end = (rid == num_ranks - 1) ? m0 : (rid + 1) * (m0 / num_ranks);

    if (rid == root_rid) {
        // This block will only run on the node that matches root_rid.
        // Collect the output
        for (int i0 = local_start; i0 < local_end; ++i0)
            output_sequential[i0] = output_distributed[i0 - local_start];
    } else {
        // This will run on all other nodes whose rid is not root_rid.
    }
}

void DISTRIBUTED_FREE_NAME(int m0, int k0, float *input_distributed, float *weights_distributed, float *output_distributed) {
    int rid;
    int num_ranks;
    int tag = 0;
    MPI_Status status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (rid == root_rid) {
        // This block will only run on the node that matches root_rid.
        free(input_distributed);
        free(weights_distributed);
        free(output_distributed);
    } else {
        // This will run on all other nodes whose rid is not root_rid.
    }
}

