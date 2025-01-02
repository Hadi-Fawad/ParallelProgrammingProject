/*
  This is the baseline implementation of a 1D Stencil operation.

  Parameters:

  m0 > 0: dimension of the original input and output vector(array) size
  k0 > 0: dimesnion of the original weights vector(array)

  float* input_sequential: pointer to original input data
  float* input_distributed: pointer to the input data that you have distributed across
  the system

  float* output_sequential:  pointer to original output data
  float* output_distributed: pointer to the output data that you have distributed across
  the system

  float* weights_sequential:  pointer to original weights data
  float* weights_distributed: pointer to the weights data that you have distributed across
  the system

  Functions: Modify these however you please.

  DISTRIBUTED_ALLOCATE_NAME(...): Allocate the distributed buffers.
  DISTRIBUTE_DATA_NAME(...): takes the sequential data and distributes it across the system.
  COMPUTE_NAME(...): Performs the stencil computation.
  COLLECT_DATA_NAME(...): Collect the distributed output and combine it back to the sequential
  one for testing.
  DISTRIBUTED_FREE_NAME(...): Free the distributed buffers that were allocated


  - richard.m.veras@ou.edu

*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME baseline_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define  COLLECT_DATA_NAME baseline_collect
#endif  


#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME baseline_allocate
#endif


#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME baseline_free
#endif





void COMPUTE_NAME(int m0, int k0, float *input_distributed, float *weights_distributed, float *output_distributed) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int chunk_size = m0 / num_ranks;
    int remainder = m0 % num_ranks;
    int local_chunk_size = chunk_size + (rid == num_ranks - 1 ? remainder : 0);

    // intialize halos
    float left_halo = 0.0, right_halo = 0.0;
    MPI_Status status;

    // exchange boundary data with neighboring processes
    if (num_ranks > 1) {
        // rightmost value to right neighbor and receive left halo from left neighbor
        MPI_Sendrecv(&input_distributed[local_chunk_size - 1], 1, MPI_FLOAT, (rid + 1) % num_ranks, 0,
                     &left_halo, 1, MPI_FLOAT, (rid - 1 + num_ranks) % num_ranks, 0,
                     MPI_COMM_WORLD, &status);

        // leftmost value to left neighbor and receive right halo from right neighbor
        MPI_Sendrecv(&input_distributed[0], 1, MPI_FLOAT, (rid - 1 + num_ranks) % num_ranks, 0,
                     &right_halo, 1, MPI_FLOAT, (rid + 1) % num_ranks, 0,
                     MPI_COMM_WORLD, &status);
    }

    // compute stencil operation for each element in the local chunk
    for (int i0 = 0; i0 < local_chunk_size; ++i0) {
        float res = 0.0f;
        for (int p0 = 0; p0 < k0; ++p0) {
            int index = i0 + p0 - k0 / 2; 

            float val;
            if (index < 0) {
                // if index is negative, use the halo value
                val = (num_ranks > 1 && rid == 0) ? left_halo : input_distributed[index + chunk_size];
                // until we hit the chunk cap
            } else if (index >= local_chunk_size) {
                val = (num_ranks > 1 && rid == num_ranks - 1) ? right_halo : input_distributed[index - chunk_size];
            } else {
                // local chunk val
                val = input_distributed[index];
            }
            res += val * weights_distributed[p0];
        }
        output_distributed[i0] = res;
    }
}

void DISTRIBUTED_ALLOCATE_NAME(int m0, int k0,
                               float **input_distributed,
                               float **weights_distributed,
                               float **output_distributed) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int chunk_size = m0 / num_ranks; 
    // assuming weights are duplicated across ranks
    *input_distributed = (float *)malloc(sizeof(float) * chunk_size);
    *output_distributed = (float *)malloc(sizeof(float) * chunk_size);
    if (rid == 0) {
        // only root rank needs the complete weights array
        *weights_distributed = (float *)malloc(sizeof(float) * k0);
    } else {
        *weights_distributed = (float *)malloc(sizeof(float) * k0);
    }
}

void DISTRIBUTE_DATA_NAME(int m0, int k0,
                          float *input_sequential,
                          float *weights_sequential,
                          float *input_distributed,
                          float *weights_distributed) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int chunk_size = m0 / num_ranks;  // divide into chunks instead of allocating m0/k0 otherwise segfault from overwriting mem
    int start = rid * chunk_size;
    int end = start + chunk_size;

    // give each rank the full weight vector
    memcpy(weights_distributed, weights_sequential, sizeof(float) * k0);

    if (rid == 0) {
        // root will send to all other chunks 
        for (int i = 1; i < num_ranks; ++i) {
            MPI_Send(input_sequential + i * chunk_size, chunk_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        memcpy(input_distributed, input_sequential, sizeof(float) * chunk_size);
    } else {
        // recieve the data from the root
        MPI_Recv(input_distributed, chunk_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}




void COLLECT_DATA_NAME(int m0, int k0,
                       float *output_distributed,
                       float *output_sequential) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int chunk_size = m0 / num_ranks; 

    if (rid == 0) {
        // root rank collects data from each rank
        memcpy(output_sequential, output_distributed, sizeof(float) * chunk_size);
        for (int i = 1; i < num_ranks; ++i) {
            MPI_Recv(output_sequential + i * chunk_size, chunk_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // other ranks send their chunk to the root rank
        MPI_Send(output_distributed, chunk_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
}


void DISTRIBUTED_FREE_NAME( int m0, int k0,
			    float *input_distributed,
			    float *weights_distributed,
			    float *output_distributed )
{
    
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  free(input_distributed);
  free(weights_distributed);
  free(output_distributed);

}


