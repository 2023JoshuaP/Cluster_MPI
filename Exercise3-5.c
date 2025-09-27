#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void distribute(double* A, double* local_A, int n, int local_cols, int my_rank, int comm_sz) {
    if (my_rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < local_cols; j++) {
                local_A[i * local_cols + j] = A[i * n + j];
            }
        }
        for (int p = 1; p < comm_sz; p++) {
            for (int i = 0; i < n; i++) {
                MPI_Send(&A[i * n + p * local_cols], local_cols, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
    }
    else {
        for (int i = 0; i < n; i++) {
            MPI_Recv(&local_A[i * local_cols], local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void local_multiply(double* local_A, double* x, double* local_result, int n, int local_cols, int my_rank) {
    for (int i = 0; i < n; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < local_cols; j++) {
            local_result[i] += local_A[i * local_cols + j] * x[my_rank * local_cols + j];
        }
    }
}

void results(double* local_result, double* result, int n, int comm_sz, int my_rank) {
    int *counts = malloc(comm_sz * sizeof(int));
    for (int i = 0; i < comm_sz; i++) {
        counts[i] = n / comm_sz;
    }
    
    double *final_part = malloc((n / comm_sz) * sizeof(double));
    MPI_Reduce_scatter(local_result, final_part, counts, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        for (int i = 0; i < n / comm_sz; i++) result[i] = final_part[i];
        for (int p = 1; p < comm_sz; p++) {
            MPI_Recv(&result[p * (n / comm_sz)], n / comm_sz, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else {
        MPI_Send(final_part, n / comm_sz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    free(counts);
    free(final_part);

    /* can also be used */

    // MPI_Reduce(local_result, result, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz, n;
    double* A = NULL, *x, *local_A, *local_result, *result = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0) {
        printf("Tamanio matriz (n): ");
        fflush(stdout);
        scanf("%d", &n);

        A = malloc(n * n * sizeof(double));
        x = malloc(n * sizeof(double));
        result = malloc(n * sizeof(double));

        for (int i = 0; i < n; i++) {
            x[i] = i + 1;
            for (int j = 0; j < n; j++) {
                A[i * n + j] = (i + 1) * 10 + (j + 1);
            }
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank != 0) {
        x = malloc(n * sizeof(double));
    }
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_cols = n / comm_sz;
    local_A = malloc(n * local_cols * sizeof(double));
    local_result = malloc(n * sizeof(double));

    distribute(A, local_A, n, local_cols, my_rank, comm_sz);
    local_multiply(local_A, x, local_result, n, local_cols, my_rank);
    results(local_result, result, n, comm_sz, my_rank);

    if (my_rank == 0) {
        printf("Resultado: ");
        for (int i = 0; i < n; i++) {
            printf("%.1f ", result[i]);
        }
        printf("\n");
        free(A);
    }

    free(x);
    free(local_A);
    free(local_result);
    free(result);

    MPI_Finalize();
    return 0;
}