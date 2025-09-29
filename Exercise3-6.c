#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

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
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Get_processor_name(processor_name, &name_len);
    
    printf("=== Proceso %d ejecutándose en MÁQUINA: %s ===\n", my_rank, processor_name);
    fflush(stdout);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        printf("Total de procesos: %d\n", comm_sz);
        
        printf("Orden de la matriz: ");
        fflush(stdout);
        scanf("%d", &order);
    }
    MPI_Bcast(&order, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int squared = (int)sqrt(comm_sz);
    if (squared * squared != comm_sz || order % squared != 0) {
        if (my_rank == 0) {
            printf("comm_sz debe ser cuadrado perfecto y order divisible por sqrt(comm_sz)\n");
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
        printf("Proceso %d de la máquina %s inicializando matriz y vector...\n", my_rank, processor_name);
        
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
        double* temp_block = (double*)malloc(block_size * block_size * sizeof(double));
        
        for (int i = 0; i < squared; i++) {
            for (int j = 0; j < squared; j++) {
                int rank_dest = i * squared + j;
                
                for (int k = 0; k < block_size; k++) {
                    for (int l = 0; l < block_size; l++) {
                        temp_block[k * block_size + l] = A[(i * block_size + k) * order + (j * block_size + l)];
                    }
                }
                
                if (rank_dest == 0) {
                    for (int k = 0; k < block_size * block_size; k++) {
                        local_A[k] = temp_block[k];
                    }
                    printf("Proceso %d manteniendo su bloque localmente\n", rank_dest);
                }
                else {
                    printf("Enviando bloque al proceso %d\n", rank_dest);
                    MPI_Send(temp_block, block_size * block_size, MPI_DOUBLE, rank_dest, 0, MPI_COMM_WORLD);
                }
            }
        }
        
        free(temp_block);
    }
    else {
        MPI_Recv(local_A, block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Proceso %d de la máquina %s recibió su bloque de matriz\n", my_rank, processor_name);
        fflush(stdout);
    }
    
    if (my_rank % (squared + 1) == 0) {
        if (my_rank == 0) {
            printf("=== DISTRIBUYENDO VECTOR ===\n");
            for (int i = 1; i < squared; i++) {
                int dest_rank = i * (squared + 1);
                printf("Enviando parte del vector al proceso diagonal %d\n", dest_rank);
                MPI_Send(&x[i * block_size], block_size, MPI_DOUBLE, dest_rank, 0, MPI_COMM_WORLD);
            }
            for (int i = 0; i < block_size; i++) {
                local_x[i] = x[i];
            }
            printf("Proceso %d manteniendo su parte del vector\n", my_rank);
        }
        else {
            MPI_Recv(local_x, block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Proceso diagonal %d de la máquina %s recibió su parte del vector\n", my_rank, processor_name);
            fflush(stdout);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("Proceso %d de la máquina %s realizando multiplicación local...\n", my_rank, processor_name);
    fflush(stdout);
    
    multiply(local_A, local_x, local_y, block_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    double* y = NULL;
    if (my_rank == 0) {
        y = (double*)malloc(order * sizeof(double));
        printf("Proceso %d de la máquina %s recolectando resultados...\n", my_rank, processor_name);
    }
    
    MPI_Gather(local_y, block_size, MPI_DOUBLE, y, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        printf("Resultado: ");
        for (int i = 0; i < order; i++) {
            printf("%.1f ", y[i]);
        }
        
        free(A);
        free(x);
        free(y);
    }
    
    printf("Proceso %d de la máquina %s finalizando...\n", my_rank, processor_name);
    fflush(stdout);
    
    free(local_A);
    free(local_x);
    free(local_y);
    
    MPI_Finalize();
    return 0;
}