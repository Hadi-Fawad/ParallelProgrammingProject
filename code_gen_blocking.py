
# the code generator should sweep through unrolling factors to result in a "ideal" file with
# no loops


# print the imports and function definitions

def main():
    code = f"""

    #include <mpi.h>
    #include <stdio.h>
    #include <stdlib.h>

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

    void COMPUTE_NAME( int m0, int k0,
            float *input_distributed,
            float *weights_distributed,
            float *output_distributed )
    {{
    int rid;
    int num_ranks;
    int tag = 0;
    int root_rid = 0;
    float res = 0.0f;
    """


    # do the blocking here
    # block size: variable
    block_size = 8

    code += f"""
    int block_size = {block_size};
    for (int bi = 0; bi < m0; bi += block_size)
    {{
        for (int i0 = bi; i0 < bi + block_size && i0 < m0; ++i0)
        {{
            float res = 0.0f;
            for (int p0 = 0; p0 < k0; ++p0)
            {{
            res += input_distributed[(p0 + i0) % m0] * weights_distributed[p0];
            }}
            output_distributed[i0] = res;
        }}
    }}


    """

    code += f""" }}


    // Create the buffers on each node
    void DISTRIBUTED_ALLOCATE_NAME( int m0, int k0,
                    float **input_distributed,
                    float **weights_distributed,
                    float **output_distributed )
    {{

    int rid;
    int num_ranks;
    int tag = 0;
    MPI_Status  status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if(rid == root_rid )
        {{
        /* This block will only run on the node that matches root_rid .*/

        *input_distributed=(float *)malloc(sizeof(float)*m0);
        *output_distributed=(float *)malloc(sizeof(float)*m0);
        *weights_distributed=(float *)malloc(sizeof(float)*k0);
        }}
    else
        {{
        /* This will run on all other nodes whose rid is not root_rid. */
        }}
    }}

    void DISTRIBUTE_DATA_NAME( int m0, int k0,
                float *input_sequential,
                float *weights_sequential,
                float *input_distributed,
                float *weights_distributed )
    {{


    int rid;
    int num_ranks;
    int tag = 0;
    MPI_Status  status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if(rid == root_rid )
        {{
        /* This block will only run on the node that matches root_rid .*/

        // Distribute the inputs
        for( int i0 = 0; i0 < m0; ++i0 )
        input_distributed[i0] = input_sequential[i0];
    
        // Distribute the weights
        for( int p0 = 0; p0 < k0; ++p0 )
        weights_distributed[p0] = weights_sequential[p0];
        }}
    else
        {{
        /* This will run on all other nodes whose rid is not root_rid. */      
        }}

    }}



    void COLLECT_DATA_NAME( int m0, int k0,
                float *output_distributed,
                float *output_sequential )
    {{

        int rid;
        int num_ranks;
        int tag = 0;
        MPI_Status  status;
        int root_rid = 0;

        MPI_Comm_rank(MPI_COMM_WORLD, &rid);
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

        if(rid == root_rid )
        {{
        /* This block will only run on the node that matches root_rid .*/
    
        // Collect the output
        for( int i0 = 0; i0 < m0; ++i0 )
        output_sequential[i0] = output_distributed[i0];
        }}
        else
        {{
        /* This will run on all other nodes whose rid is not root_rid. */      
        }}
    }}


    void DISTRIBUTED_FREE_NAME( int m0, int k0,
                    float *input_distributed,
                    float *weights_distributed,
                    float *output_distributed )
    {{
        
    int rid;
    int num_ranks;
    int tag = 0;
    MPI_Status  status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if(rid == root_rid )
        {{
        /* This block will only run on the node that matches root_rid .*/

        free(input_distributed);
        free(weights_distributed);
        free(output_distributed);

        }}
    else
        {{
        /* This will run on all other nodes whose rid is not root_rid. */  
        }}
    }}
    """

    filename ="generated.c"
    with open(filename, "w") as f:
        f.write(code)


if __name__ == "__main__":
    main()