#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    int value, partner_rank, addition = 0;
    char hostname[256];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    gethostname(hostname, sizeof(hostname));
    
    if ((comm_sz & (comm_sz - 1)) != 0) {
        if (my_rank == 0) {
            printf("comm_sz (%d) debe ser una potencia de 2.\n", comm_sz);
        }
        MPI_Finalize();
        return 1;
    }
    
    printf("[INICIO] Proceso %d ejecutándose en %s: valor inicial = %d\n", my_rank, hostname, my_rank);
    fflush(stdout);
    
    value = my_rank;
    addition = value;

    for (int step = 1, level = 1; step < comm_sz; step *= 2, level++) {
        if (my_rank % (2 * step) == 0) {
            partner_rank = my_rank + step;
            if (partner_rank < comm_sz) {
                int received_value;
                printf("[Nivel %d] Proceso %d en %s esperando del proceso %d...\n", 
                       level, my_rank, hostname, partner_rank);
                fflush(stdout);
                
                MPI_Recv(&received_value, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                addition += received_value;
                
                printf("[Nivel %d] Proceso %d en %s recibió %d, suma acumulada: %d\n", 
                       level, my_rank, hostname, received_value, addition);
                fflush(stdout);
            }
        }
        else if (my_rank % (2 * step) == step) {
            partner_rank = my_rank - step;
            printf("[Nivel %d] Proceso %d en %s enviando %d al proceso %d\n", level, my_rank, hostname, addition, partner_rank);
            fflush(stdout);
            
            MPI_Send(&addition, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
            
            printf("[Nivel %d] Proceso %d en %s terminó su participación\n", level, my_rank, hostname);
            fflush(stdout);
            break;
        }
    }
    
    if (my_rank == 0) {
        printf("Suma calculada: %d\n", addition);
    }

    printf("Proceso %d en %s terminando\n", my_rank, hostname);
    fflush(stdout);
    
    MPI_Finalize();
    return 0;
}