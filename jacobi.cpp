/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    // TODO
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    // TODO
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    // Initialize x, find the diagonal of A, copy A to R with diagonal set to 0
    double *diag = (double *) malloc(n * sizeof(double));
    double *R = (double *) malloc(n * n * sizeof(double));
    double *w = (double *) malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        x[i] = 0;
        diag[i] = A[i * n + i];
        for (int j = 0; j < n; j++) {
            if (i == j) { R[i * n + j] = 0; }
            else { R[i * n + j] = A[i * n + j]; }
        }
    }

    int iter = 0;
    double l2, sum = 0;
    while (iter < max_iter) {
        // Perform Jacobi's method
        matrix_vector_mult(n, R, x, w);
        for (int i = 0; i < n; i++) {
            x[i] = (b[i] - w[i]) / diag[i];
        }

        // Check for convergence
        matrix_vector_mult(n, A, x, w);
        for (int i = 0; i < n; i++) {
            sum += pow(w[i] - b[i], 2);
        }
        l2 = sqrt(sum);
        if (l2 <= l2_termination) break;

        iter++
    }

    free(R);
    free(diag);
    free(w);
}
