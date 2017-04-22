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
    MPI_Comm_size(comm, &p);
    // MPI_Comm newcomm;
    // const int period = 1;
    int q = sqrt(p);
    // MPI_Cart_create(comm, 2, &q, &period, 1, &newcomm);
    int coordinates[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coordinates); 

    
    int extra = n%q;
    int vecSize = 0;
    int index = 0;
    int destination_coords[2] = {0, 0};
    int destination_rank;

    int rank0;
    MPI_Cart_rank(comm, destination_coords, &rank0);

    if(rank == rank0){
        for(int i = 0; i < ceil(n/q); i++){
            local_vector[i] = &input_vector[i];
        }
        index = ceil(n/q) + 1;
        for(int i = 1; i < q; i++){
            if(i < extra){
                vecSize = ceil(n/q);
            } else{
                vecSize = floor(n/q);
            }
            double* newVector = (double*) malloc(vecSize * sizeof(double));
            for(int j = index; j < index + vecSize; j++){
                newVector[j-index] = input_vector[j];
            }
            index += vecSize;
            destination_coords[0] = i;
            MPI_Cart_rank(comm, destination_coords, &destination_rank);
            MPI_Send(&newVector, vecSize, MPI_DOUBLE, destination_rank, 111, comm );
        }
    } else if(coordinates[1] == 0){
        if(coordinates[0] < extra){
            vecSize = ceil(n/q);
        } else{
            vecSize = floor(n/q);
        }
        MPI_Status stat;
        MPI_Recv(local_vector, vecSize, MPI_DOUBLE, rank0, MPI_ANY_TAG, comm, &stat);
    }
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm){
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int vecSize;
    int q = sqrt(p);
    int extra = n%q;
    int index = 0;

    // MPI_Comm newcomm;
    // const int period = 1;
    // MPI_Cart_create(comm, 2, &q, &period, 1, &newcomm);
    int coordinates[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coordinates);

    int rec_coords[2] = {0, 0};
    int rank0;
    MPI_Cart_rank(comm, rec_coords, &rank0);
    int destination_rank;

    if(rank == rank0){
        for(int i = 0; i < ceil(n/q); i++){
            output_vector[i] = local_vector[i];
        }
        index += ceil(n/q) + 1;
        for(int i = 1; i < q; i++){
            if(i < extra){
                vecSize = ceil(n/q);
            } else{
                vecSize = floor(n/q);
            }
            double* newVector = (double *) malloc(vecSize * sizeof(double));
            MPI_Status stat;
            rec_coords[0] = i;
            MPI_Cart_rank(comm, rec_coords, &destination_rank);
            MPI_Recv(&newVector, vecSize, MPI_DOUBLE, rank0, 111, comm, &stat);
            for(int j = index; j < index + vecSize; j++){
                output_vector[j] = newVector[j-index];
            }
            index += vecSize;
        }

    } else if(coordinates[1] == 0){
        if(rank < extra){
            vecSize = ceil(n/q);
        } else{
            vecSize = floor(n/q);
        }
        MPI_Send(&local_vector, vecSize, MPI_DOUBLE, rank0, 111, comm );
    }
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // TODO
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm){
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(comm, &rank);
    int vecSize = 0;
    int q = sqrt(p);
    int extra = n%q;

    // MPI_Comm newcomm;
    // const int period = 1;
    // MPI_Cart_create(comm, 2, &q, &period, 1, &newcomm);
    int coordinates[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coordinates);

    int rec_coords[2] = {0, 0};
    int rank0;
    MPI_Cart_rank(comm, rec_coords, &rank0);
    int destination_rank;
    int receive_rank;

    if(coordinates[0] == 0 && rank != rank0){
        if(rank < extra){
            vecSize = ceil(n/q);
        } else{
            vecSize = floor(n/q);
        }
        int dest_coords[2] = {coordinates[0],coordinates[0]};
        MPI_Cart_rank(comm, dest_coords, &destination_rank);
        MPI_Send(&col_vector, vecSize, MPI_DOUBLE, destination_rank, 111, comm);
    } else if(coordinates[0] == coordinates[1] && rank > 0){
        if((rank - rank/q)/q < extra){
            vecSize = ceil(n/q);
        } else{
            vecSize = floor(n/q);
        }
        MPI_Status stat;
        int receive_coords[2] = {0,coordinates[1]};
        MPI_Cart_rank(comm, receive_coords, &receive_rank);
        MPI_Recv(&row_vector, vecSize, MPI_DOUBLE, receive_rank, MPI_ANY_TAG, comm, &stat);
    }

    if(coordinates[0] == coordinates[1]){
        rec_coords[1] = coordinates[1];
        for(int i = 0; i < q; i++){
            if(rank != i*(rank%q)){
                if(i < extra){
                    vecSize = ceil(n/q);
                } else{
                    vecSize = floor(n/q);
                }
                rec_coords[0] = i;
                MPI_Cart_rank(comm, rec_coords, &destination_rank);
                MPI_Send(&row_vector, vecSize, MPI_DOUBLE, destination_rank, 111, comm);
            }
        }
    } else{
        if(rank/q < extra){
            vecSize = ceil(n/q);
        } else{
            vecSize = floor(n/q);
        }
        MPI_Status stat;
        rec_coords[1] = coordinates[1];
        rec_coords[0] = coordinates[0];
        MPI_Cart_rank(comm, rec_coords, &receive_rank);
        MPI_Recv(&row_vector, vecSize, MPI_DOUBLE, receive_rank, MPI_ANY_TAG, comm, &stat);
    }
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
