#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void multiply(double* local_A, double* local_x, double* local_y, int block_size) {
    for (int i = 0; i < block_size; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < block_size; j++) {
            local_y[i] += local_A[i * block_size + j] * local_x[j];
        }
    }
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int order;
    int block_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if (my_rank == 0) {
        printf("Orden de la matriz: ");
        fflush(stdout);
        scanf("%d", &order);
    }
    MPI_Bcast(&order, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int squared = (int)sqrt(comm_sz);
    if (squared * squared != comm_sz || order % squared != 0) {
        if (my_rank == 0) {
            printf("Error: comm_sz debe ser cuadrado perfecto y order divisible por sqrt(comm_sz)\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    block_size = order / squared;
    
    double *A = NULL, *x = NULL;
    double* local_A = (double*)malloc(block_size * block_size * sizeof(double));
    double* local_x = (double*)malloc(block_size * sizeof(double));
    double* local_y = (double*)malloc(block_size * sizeof(double));
    
    if (my_rank == 0) {
        A = (double*)malloc(order * order * sizeof(double));
        x = (double*)malloc(order * sizeof(double));
        
        for (int i = 0; i < order; i++) {
            x[i] = i + 1;
            for (int j = 0; j < order; j++) {
                A[i * order + j] = i * order + j + 1;
            }
        }
    }
    
    if (my_rank == 0) {
        for (int i = 0; i < squared; i++) {
            for (int j = 0; j < squared; j++) {
                int rank_dest = i * squared + j;
                if (rank_dest == 0) {
                    for (int k = 0; k < block_size; k++) {
                        for (int l = 0; l < block_size; l++) {
                            local_A[k * block_size + l] = A[k * order + l];
                        }
                    }
                }
                else {
                    MPI_Send(&A[(i * block_size * order) + j * block_size], block_size * block_size, MPI_DOUBLE, rank_dest, 0, MPI_COMM_WORLD);
                }
            }
        }
    }
    else {
        MPI_Recv(local_A, block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if (my_rank % (squared + 1) == 0) {
        if (my_rank == 0) {
            for (int i = 1; i < squared; i++) {
                MPI_Send(&x[i * block_size], block_size, MPI_DOUBLE, i * (squared + 1), 0, MPI_COMM_WORLD);
            }
            for (int i = 0; i < block_size; i++) {
                local_x[i] = x[i];
            }
        }
        else {
            MPI_Recv(local_x, block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    multiply(local_A, local_x, local_y, block_size);
    
    double* y = NULL;
    if (my_rank == 0) {
        y = (double*)malloc(order * sizeof(double));
    }
    MPI_Gather(local_y, block_size, MPI_DOUBLE, y, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        printf("Resultado: ");
        for (int i = 0; i < order; i++) {
            printf("%.1f ", y[i]);
        }
        printf("\n");
        
        free(A);
        free(x);
        free(y);
    }
    
    free(local_A);
    free(local_x);
    free(local_y);
    
    MPI_Finalize();
    return 0;
}