#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void distribute(double* A, double* local_A, int n, int local_cols, int my_rank, int comm_sz) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    if (my_rank == 0) {
        printf("Proceso 0 en %s distribuyendo matriz...\n", hostname);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < local_cols; j++) {
                local_A[i * local_cols + j] = A[i * n + j];
            }
        }
        
        for (int p = 1; p < comm_sz; p++) {
            printf("[ENVIANDO] Proceso 0 en %s enviando columnas %d-%d al proceso %d\n", 
                   hostname, p * local_cols, (p + 1) * local_cols - 1, p);
            for (int i = 0; i < n; i++) {
                MPI_Send(&A[i * n + p * local_cols], local_cols, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
    }
    else {
        printf("[RECIBIENDO] Proceso %d en %s recibiendo columnas %d-%d\n", 
               my_rank, hostname, my_rank * local_cols, (my_rank + 1) * local_cols - 1);
        for (int i = 0; i < n; i++) {
            MPI_Recv(&local_A[i * local_cols], local_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    printf("Proceso %d en %s recibió su bloque de columnas\n", my_rank, hostname);
    fflush(stdout);
}

void local_multiply(double* local_A, double* x, double* local_result, int n, int local_cols, int my_rank) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    printf("Proceso %d en %s calculando A * x local...\n", my_rank, hostname);
    
    for (int i = 0; i < n; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < local_cols; j++) {
            local_result[i] += local_A[i * local_cols + j] * x[my_rank * local_cols + j];
        }
    }
    
    printf("Proceso %d en %s terminó multiplicación local\n", my_rank, hostname);
    fflush(stdout);
}

void results(double* local_result, double* result, int n, int comm_sz, int my_rank) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    printf("Proceso %d en %s participando en MPI_Reduce_scatter\n", my_rank, hostname);
    
    int *counts = malloc(comm_sz * sizeof(int));
    for (int i = 0; i < comm_sz; i++) {
        counts[i] = n / comm_sz;
    }
    
    double *final_part = malloc((n / comm_sz) * sizeof(double));
    MPI_Reduce_scatter(local_result, final_part, counts, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    printf("[REDUCE_SCATTER] Proceso %d en %s completó reduce_scatter\n", my_rank, hostname);
    
    if (my_rank == 0) {
        printf("[RECOPILANDO] Proceso 0 en %s recopilando resultado final...\n", hostname);
        
        for (int i = 0; i < n / comm_sz; i++) {
            result[i] = final_part[i];
        }
        
        for (int p = 1; p < comm_sz; p++) {
            MPI_Recv(&result[p * (n / comm_sz)], n / comm_sz, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Proceso 0 en %s recibió parte del proceso %d\n", hostname, p);
        }
    } else {
        printf("Proceso %d en %s enviando resultado al proceso 0\n", my_rank, hostname);
        MPI_Send(final_part, n / comm_sz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    free(counts);
    free(final_part);
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz, n;
    double* A = NULL, *x, *local_A, *local_result, *result = NULL;
    char hostname[256];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    gethostname(hostname, sizeof(hostname));
    
    if (my_rank == 0) {
        printf("Procesos: %d\n", comm_sz);
        printf("Tamaño matriz (n): ");
        fflush(stdout);
        scanf("%d", &n);
        
        if (n % comm_sz != 0) {
            printf("ERROR: n debe ser divisible por comm_sz\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        A = malloc(n * n * sizeof(double));
        x = malloc(n * sizeof(double));
        result = malloc(n * sizeof(double));
        
        printf("Proceso 0 en %s inicializando datos...\n", hostname);
        for (int i = 0; i < n; i++) {
            x[i] = i + 1;
            for (int j = 0; j < n; j++) {
                A[i * n + j] = (i + 1) * 10 + (j + 1);
            }
        }
        
        printf("Matriz A y vector x inicializados\n");
        printf("Columnas por proceso: %d\n\n", n / comm_sz);
    }
    
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (my_rank != 0) {
        x = malloc(n * sizeof(double));
    }
    
    printf("Proceso %d ejecutándose en %s\n", my_rank, hostname);
    fflush(stdout);
    
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("[BROADCAST] Proceso %d en %s recibió vector x\n", my_rank, hostname);
    
    int local_cols = n / comm_sz;
    local_A = malloc(n * local_cols * sizeof(double));
    local_result = malloc(n * sizeof(double));
    
    MPI_Barrier(MPI_COMM_WORLD);
    distribute(A, local_A, n, local_cols, my_rank, comm_sz);
    MPI_Barrier(MPI_COMM_WORLD);
    local_multiply(local_A, x, local_result, n, local_cols, my_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    results(local_result, result, n, comm_sz, my_rank);
    
    if (my_rank == 0) {
        printf("Vector resultado A*x: ");
        for (int i = 0; i < n; i++) {
            printf("%.1f ", result[i]);
        }
        printf("\n");
        free(A);
    }
    
    printf("Proceso %d en %s terminando\n", my_rank, hostname);
    fflush(stdout);
    
    free(x);
    free(local_A);
    free(local_result);
    if (result) {
        free(result);
    }
    
    MPI_Finalize();
    return 0;
}