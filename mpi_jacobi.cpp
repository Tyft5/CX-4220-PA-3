/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm){
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(comm, &rank);
    int q = sqrt(p);
    int extra = n%q;
    int vecSize = 0;
    int index = 0;

    if(rank == 0){
        for(int i = 0; i < ceil(n/q); i++){
            local_vector[i] = &input_vector[i];
        }
        index = ceil(n/q) + 1;
        for(int i = 1; i < q; i++){
            if(i*q < extra){
                vecSize = ceil(n/q);
            } else{
                vecSize = floor(n/q);
            }
            int* newVector = (int *) malloc(vecSize * sizeof(int));
            for(int j = index; j < index + vecSize; j++){
                newVector[j-index] = input_vector[j];
            }
            index += vecSize;
            MPI_Send(&newVector, vecSize, MPI_INT, i*q, 111, comm );
        }
    } else if(rank%q == 0){
        MPI_Status stat;
        MPI_Recv(&local_vector, vecSize, MPI_INT, 0, MPI_ANY_TAG, comm, &stat);
    }
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm){
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(comm, &rank);
    int vecSize;
    int q = sqrt(p);
    int extra = n%q;
    int index = 0;

    if(rank == 0){
        for(int i = 0; i < ceil(n/q); i++){
            output_vector[i] = local_vector[i];
        }
        index += ceil(n/q) + 1;
        for(int i = 1; i < q; i++){
            if(i*q < extra){
                vecSize = ceil(n/q);
            } else{
                vecSize = floor(n/q);
            }
            int* newVector = (int *) malloc(vecSize * sizeof(int));
            MPI_Status stat;
            MPI_Recv(&newVector, vecSize, MPI_INT, i*q, i*q, comm, &stat);
            for(int j = index; j < index + vecSize; j++){
                output_vector[j] = newVector[j-index];
            }
            index += vecSize;
        }

    } else if(rank%q == 0){
        if(rank < extra){
            vecSize = ceil(n/q);
        } else{
            vecSize = floor(n/q);
        }
        MPI_Send(&local_vector, vecSize, MPI_INT, 0, rank, comm );
    }
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // TODO
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // TODO
    //Do this
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // TODO
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // TODO
}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
