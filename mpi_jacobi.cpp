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
#include <time.h>

/*
 * TODO: Implement your solutions here
 */

//distribute a vector from processor 0 to all processors in the first column
void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm){
    //get rank and coordinates from cartesian topology
    int p, rank;
    MPI_Comm_size(comm, &p);
    int q = sqrt(p);
    int coordinates[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coordinates); 

    //initialize variables
    int extra = n%q;
    int my_vecSize = 0, vecSize = 0;
    int index = 0;
    int destination_coords[2] = {0, 0};
    int destination_rank;
    double* newVector;

    //get the coordinates of processor zero
    int rank0;
    MPI_Cart_rank(comm, destination_coords, &rank0);
    
    //determine the correct vector size for each processor and allocate memory
    if(coordinates[1] < extra){
        my_vecSize = ceil(((double)n)/q);
    } else{
        my_vecSize = floor(((double)n)/q);
    }
    double *tmp_vec = (double *) malloc(my_vecSize * sizeof(double));
    
    //if it's the top left processor
    if(rank == rank0){
        //get the local values to remain
        for(int i = 0; i < ceil(((double)n)/q); i++){
            tmp_vec[i] = input_vector[i];
        }
        //index the input vector
        index = ceil(((double)n)/q);
        //create the vectors that need to be sent to each first column processor
        for(int i = 1; i < q; i++){
            if(i < extra){
                vecSize = ceil(((double)n)/q);
            } else{
                vecSize = floor(((double)n)/q);
            }
            newVector = (double*) malloc(vecSize * sizeof(double));
            for(int j = index; j < index + vecSize; j++){
                newVector[j-index] = input_vector[j];
            }
            //increment the index
            index += vecSize;
            //get the destination coordinates
            destination_coords[1] = i;
            MPI_Cart_rank(comm, destination_coords, &destination_rank);
            //send the vector
            MPI_Send(newVector, vecSize, MPI_DOUBLE, destination_rank, 111, comm );
        }
    }
    //if it's a processor in the first column
    else if(coordinates[0] == 0){
        //get the vector size
        if(coordinates[1] < extra){
            vecSize = ceil(((double)n)/q);
        } else{
            vecSize = floor(((double)n)/q);
        }
        //recieve the new vector
        MPI_Status stat;
        MPI_Recv(tmp_vec, vecSize, MPI_DOUBLE, rank0, MPI_ANY_TAG, comm, &stat);
    }

    //allocate space and store the new vector for the output
    *local_vector = (double *) malloc(my_vecSize * sizeof(double));
    for (int i = 0; i < my_vecSize; i++) {
        (*local_vector)[i] = tmp_vec[i];
    }
}


// gather the local vector distributed among (0,i) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm){
    //get the rank and coordinates from cartesian topology, initialize variables
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int vecSize;
    int q = sqrt(p);
    int extra = n%q;
    int index = 0;
    int coordinates[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coordinates);
    int rec_coords[2] = {0, 0};
    int rank0;
    MPI_Cart_rank(comm, rec_coords, &rank0);
    int destination_rank;
    double* newVector;

    //if it's processor 0
    if(rank == rank0){
        //store the local vector as the first part of the output vector
        for(int i = 0; i < ceil(((double)n)/q); i++){
            output_vector[i] = local_vector[i];
        }
        index += ceil(((double)n)/q);
        //for each processor in column 0
        for(int i = 1; i < q; i++){
            //get vector size
            if(i < extra){
                vecSize = ceil(((double)n)/q);
            } else{
                vecSize = floor(((double)n)/q);
            }
            //allocate memory and receive the vector
            newVector = (double *) malloc(vecSize * sizeof(double));
            MPI_Status stat;
            rec_coords[1] = i;
            MPI_Cart_rank(comm, rec_coords, &destination_rank);
            MPI_Recv(newVector, vecSize, MPI_DOUBLE, destination_rank, 111, comm, &stat);
            //store in the output vector
            for(int j = index; j < index + vecSize; j++){
                output_vector[j] = newVector[j-index];
            }
            //increment th eindex and free the temporary vector
            index += vecSize;
            free(newVector);
        }
    }
    //if it's in the first column
    else if(coordinates[0] == 0){
        //get the vector size
        if(coordinates[1] < extra){
            vecSize = ceil(((double)n)/q);
        } else{
            vecSize = floor(((double)n)/q);
        }
        //send to processor 0
        MPI_Send(local_vector, vecSize, MPI_DOUBLE, rank0, 111, comm );
    }
}

//Equally distributes a matrix stored on processor (0,0) onto the whole grid (i,j).
void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm){
    //initialize variables and get rank/ coordinates
    int i, p, rank, rowsize, colsize, send_r= 0, send_c = 0, send_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(comm, &rank);
    int q = (int) sqrt(p);
    int extra = n % q;
    int coords[2];
    MPI_Cart_coords(comm, rank, 2, coords);
    int rank0;
    int zero_coords[2] = {0, 0};
    MPI_Cart_rank(comm, zero_coords, &rank0);

    // Find size of local matrix
    int fl = floor(((double)n) / q);
    int cl = ceil(((double)n) / q);
    if (coords[0] < extra) {
        rowsize = cl;
    } else {
        rowsize = fl;
    }
    if (coords[1] < extra) {
        colsize = cl;
    } else {
        colsize = fl;
    }

    //allocate memory for arrays to send
    double *loc_mat = (double *) malloc(rowsize * colsize * sizeof(double));
    double *send_mat = NULL;

    //if in the first column
    int row_offset = rowsize, col_offset = 0, index = 0;
    if (coords[0] == 0 && coords[1] == 0) {
        //store values in the local matrix
        for (i = 0; i < colsize; i++) {
            for (int j = 0; j < rowsize; j++) {
                loc_mat[index++] = input_matrix[i*n + j];
            }
        }

        // Send matrices to the rest of the processors
        for (int v = 0; v < q; v++) {
            for (int w = 0; w < q; w++) {
                if (v == 0 && w == 0) continue;
                int send_coords[2] = {v, w};
                MPI_Cart_rank(comm, send_coords, &send_rank);

                // Find size of send matrix
                if (v < extra) { send_r = cl; }
                else { send_r = fl; }
                if (w < extra) { send_c = cl; }
                else { send_c = fl; } 

                // Allocate send matrix
                send_mat = (double *) malloc(send_r * send_c * sizeof(double));
                for (i = 0; i < send_c; i++) {
                    for (int j = 0; j < send_r; j++) {
                        send_mat[i * send_r + j] = input_matrix[row_offset * n + col_offset + i * n + j];
                    }
                }
                //send the matrix
                MPI_Send(send_mat, send_r * send_c, MPI_DOUBLE, send_rank, 222, comm);
                // Add to index offset for input_matrix
                row_offset += send_c;
            }
            // Add to index offset for input_matrix
            col_offset += send_r;
            row_offset = 0;
        }
        //free the temporary matrix
        free(send_mat);
    } else {
        // Receive from 0,0
        MPI_Status stat;
        MPI_Recv(loc_mat, rowsize * colsize, MPI_DOUBLE, rank0, MPI_ANY_TAG, comm, &stat);
    }
    // Make output pointer point to local matrix
    *local_matrix = loc_mat;
}

//Given a vector distributed among the first column, this function transposes 
//it to be distributed among rows.
void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm){
    //get cartesian coordinates and initialize variables
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(comm, &rank);
    int vecSize = 0;
    int q = sqrt(p);
    int extra = n%q;
    int coordinates[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coordinates);
    int rec_coords[2] = {0, 0};
    int rank0;
    MPI_Cart_rank(comm, rec_coords, &rank0);
    int destination_rank;
    int receive_rank;

    //send to the pivot if in first column
    if(coordinates[0] == 0 && rank != rank0){
        if(coordinates[1] < extra){
            vecSize = ceil(((double)n)/q);
        } else{
            vecSize = floor(((double)n)/q);
        }
        int dest_coords[2] = {coordinates[1],coordinates[1]};
        MPI_Cart_rank(comm, dest_coords, &destination_rank);
        MPI_Send(col_vector, vecSize, MPI_DOUBLE, destination_rank, 111, comm);
    }
    //receive from first column if on diagnol
    else if(coordinates[0] == coordinates[1] && coordinates[0] > 0){
        if(coordinates[1] < extra){
            vecSize = ceil(((double)n)/q);
        } else{
            vecSize = floor(((double)n)/q);
        }
        MPI_Status stat;
        int receive_coords[2] = {0, coordinates[1]};
        MPI_Cart_rank(comm, receive_coords, &receive_rank);
        MPI_Recv(row_vector, vecSize, MPI_DOUBLE, receive_rank, MPI_ANY_TAG, comm, &stat);
    }

    //store the column vector into the row vector
    if (coordinates[0] == 0 && coordinates[1] == 0) {
        for (int i = 0; i < ceil(((double)n) / q); i++) {
            row_vector[i] = col_vector[i];
        }
    }

    //if it's on the diagnol send to each processor in column
    if(coordinates[0] == coordinates[1]){
        rec_coords[0] = coordinates[0];
        if(coordinates[1] < extra){
            vecSize = ceil(((double)n)/q);
        } else{
            vecSize = floor(((double)n)/q);
        }
        for(int i = 0; i < q; i++){
            if(i != coordinates[1]){
                rec_coords[1] = i;
                MPI_Cart_rank(comm, rec_coords, &destination_rank);
                MPI_Send(row_vector, vecSize, MPI_DOUBLE, destination_rank, 111, comm);
            }
        }
    }
    //otherwise receive from the diagnol
    else{
        if(coordinates[0] < extra){
            vecSize = ceil(((double)n)/q);
        } else{
            vecSize = floor(((double)n)/q);
        }
        MPI_Status stat;
        rec_coords[1] = coordinates[0];
        rec_coords[0] = coordinates[0];
        MPI_Cart_rank(comm, rec_coords, &receive_rank);
        MPI_Recv(row_vector, vecSize, MPI_DOUBLE, receive_rank, MPI_ANY_TAG, comm, &stat);
    }
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm){
    //initialize variables and get cartesian coordinates
    int p, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int coordinates[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coordinates);
    int q = sqrt(p);
    int extra = n % q;
    int vecSize = ceil(((double)n)/q);

    //transpose the local x vector
    double* new_x = (double*) malloc(vecSize * sizeof(double));
    transpose_bcast_vector(n, local_x, new_x, comm);

    //get the correct row and column sizes for each processor
    int rowsize, colsize;
    int fl = floor(((double)n) / q);
    int cl = ceil(((double)n) / q);
    if (coordinates[0] < extra) {
        rowsize = cl;
    } else {
        rowsize = fl;
    }
    if (coordinates[1] < extra) {
        colsize = cl;
    } else {
        colsize = fl;
    }


    //initialize y to zero and sum local values
    for (int i = 0; i < colsize; i++) {
        local_y[i] = 0;
    }
    for(int i = 0; i < colsize; i++){
        for(int j = 0; j < rowsize; j++){
            local_y[i] += local_A[i*rowsize + j] * new_x[j];
        }
    }

    //reduction among processors in the same row
    if(coordinates[1] < extra){
        vecSize = ceil(((double)n)/q);
    } else{
        vecSize = floor(((double)n)/q);
    }
    double* temp_vector = (double*) malloc(vecSize * sizeof(double));
    int rec_coords[2] = {0, coordinates[1]};
    int receive_rank, send_rank;
    //if it's in the first column receive values
    if(coordinates[0] == 0){
        for(int i = 1; i < q; i++){
            rec_coords[0] = i;
            MPI_Cart_rank(comm, rec_coords, &receive_rank);
            MPI_Status stat;
            MPI_Recv(temp_vector, vecSize, MPI_DOUBLE, receive_rank, MPI_ANY_TAG, comm, &stat);
            for(int j = 0; j < vecSize; j++){
                local_y[j] += temp_vector[j];
            }
        }
    }
    //otherwise send them
    else{
        int send_coords[2] = {0, coordinates[1]};
        MPI_Cart_rank(comm, send_coords, &send_rank);
        MPI_Send(local_y, vecSize, MPI_DOUBLE, send_rank, 111, comm);
    }
    free(temp_vector);
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination){
    //get cartesian coordinates and initialize variables
    int p, rank, rowsize = 0, colsize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);
    int q = sqrt(p);
    int extra = n % q;
    int zero_coords[2] = {0, 0};
    int rank0;
    MPI_Cart_rank(comm, zero_coords, &rank0);

    // Find the local matrix dimensions
    if (coords[0] < extra) {
        rowsize = ceil(((double)n) / q);
    } else {
        rowsize = floor(((double)n) / q);
    }
    if (coords[1] < extra) {
        colsize = ceil(((double)n) / q);
    } else {
        colsize = floor(((double)n) / q);
    }

    //malloc the variables
    double *w = (double *) malloc(colsize * sizeof(double));
    double *squared = (double *) malloc(colsize * sizeof(double));
    double *all_squared = (double *) malloc(n * sizeof(double));
    double *diag = NULL;

    // Copy A into R
    double *local_R = (double *) malloc(rowsize * colsize * sizeof(double));
    for (int i = 0; i < colsize; i++) {
        for (int j = 0; j < rowsize; j++) {
            local_R[i * rowsize + j] = local_A[i * rowsize + j];
        }
    }

    // Find the diagonal and send it to the first column
    if (coords[0] == coords[1]) {
        diag = (double *) malloc(rowsize * sizeof(double));
        for (int i = 0; i < rowsize; i++) {
            diag[i] = local_A[i * rowsize + i];
        }

        //send to the first column
        int send_coords[2] = {0, coords[1]};
        int send_rank;
        MPI_Cart_rank(comm, send_coords, &send_rank);
        if (coords[1] != 0) {
            MPI_Send(diag, rowsize, MPI_DOUBLE, send_rank, 333, comm);
        }

        // set diagonal of R to zero
        for (int i = 0; i < rowsize; i++) {
            local_R[i * rowsize + i] = 0;
        }
    } 
    //receive from the diagnol
    else if (coords[0] == 0 && coords[1] > 0) {
        int rec_size = colsize;
        int rec_coords[2] = {coords[1], coords[1]};
        int rec_rank;
        MPI_Cart_rank(comm, rec_coords, &rec_rank);
        diag = (double *) malloc(rec_size * sizeof(double));
        MPI_Status stat;
        MPI_Recv(diag, rec_size, MPI_DOUBLE, rec_rank, MPI_ANY_TAG, comm, &stat);
    }

    //set all local x values to 0
    for (int i = 0; i < rowsize; i++) {
        local_x[i] = 0.0;
    }

    // Computation loop for Jacobi's method
    int cont = 1;
    for (int iter = 0; iter < max_iter; iter++) {
        //distribute the local matrix
        distributed_matrix_vector_mult(n, local_R, local_x, w, comm);
        //get the local x values for processors in the first column
        if (coords[0] == 0) {
            for (int i = 0; i < colsize; i++) {
                local_x[i] = (local_b[i] - w[i]) / diag[i];
            }
        }
        //distribute the larger matrix
        distributed_matrix_vector_mult(n, local_A, local_x, w, comm);
        // Find local l2 norm
        if (coords[0] == 0) {
            for (int i = 0; i < colsize; i++) {
                squared[i] = pow(w[i] - local_b[i], 2);
            }
            gather_vector(n, squared, all_squared, comm);
            if (coords[1] == 0) {
                double l2, sum = 0.;
                for (int i = 0; i < n; i++) {
                    sum += all_squared[i];
                }
                l2 = sqrt(sum);
                if (l2 <= l2_termination) {
                    cont = 0;
                }
            }
        }
        //broadcast if the process should continue
        MPI_Bcast(&cont, 1, MPI_INT, rank0, comm);
        if (!cont) break;
    }

    //free the variables
    free(local_R);
    free(w);
    free(diag);
    free(squared);
    free(all_squared);
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

    int rank;
    MPI_Comm_rank(comm, &rank);

    // unsigned int t = time(0);
    // while(time(0) < t + rank);
    // printf("Here\n");
    // MPI_Barrier(comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
