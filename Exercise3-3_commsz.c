#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    int value, partner_rank, addition = 0;
    char hostname[256];
    char *all_hostnames = NULL;
    int virtual_comm_sz;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    gethostname(hostname, sizeof(hostname));
    
    if (my_rank == 0) {
        all_hostnames = malloc(comm_sz * 256 * sizeof(char));
        printf("Número total de procesos: %d\n", comm_sz);
    }
    
    virtual_comm_sz = 1;
    while (virtual_comm_sz < comm_sz) {
        virtual_comm_sz *= 2;
    }
    
    MPI_Gather(hostname, 256, MPI_CHAR, all_hostnames, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        int is_power_of_2 = (comm_sz > 0) && ((comm_sz & (comm_sz - 1)) == 0);
        if (is_power_of_2) {
            printf("comm_sz = %d es una potencia de 2\n", comm_sz);
        }
        else {
            printf("comm_sz = %d NO es una potencia de 2\n", comm_sz);
        }
        printf("Niveles del árbol virtual: %d\n", (int)log2(virtual_comm_sz));

        for (int i = 0; i < comm_sz; i++) {
            char *host = all_hostnames + i * 256;
            if (i == 0) {
                printf("Proceso %d: %s (RAÍZ - Resultado final)\n", i, host);
            } else {
                printf("Proceso %d: %s (NODO - Valor inicial: %d)\n", i, host, i);
            }
        }
        
        if (virtual_comm_sz > comm_sz) {
            for (int i = comm_sz; i < virtual_comm_sz; i++) {
                printf("Proceso %d: VIRTUAL (no existe)\n", i);
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    value = my_rank;
    addition = value;
    
    printf("Proceso %d en %s: valor inicial = %d\n", my_rank, hostname, value);
    fflush(stdout);
    
    MPI_Barrier(MPI_COMM_WORLD);

    for (int step = 1, level = 1; step < virtual_comm_sz; step *= 2, level++) {
        if (my_rank % (2 * step) == 0) {
            partner_rank = my_rank + step;
            if (partner_rank < comm_sz) {
                int received_value;
                printf("[NIVEL %d] Proceso %d en %s esperando datos del proceso %d...\n", level, my_rank, hostname, partner_rank);
                fflush(stdout);
                
                MPI_Recv(&received_value, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                addition += received_value;
                
                printf("[NIVEL %d] Proceso %d en %s recibió %d, suma acumulada: %d\n", level, my_rank, hostname, received_value, addition);
                fflush(stdout);
            }
            else {
                printf("[NIVEL %d] Proceso %d en %s: partner %d es virtual (no existe)\n", level, my_rank, hostname, partner_rank);
                fflush(stdout);
            }
        }
        else if (my_rank % (2 * step) == step) {
            partner_rank = my_rank - step;
            if (partner_rank >= 0 && partner_rank < comm_sz) {
                printf("[NIVEL %d] Proceso %d en %s enviando %d al proceso %d\n", level, my_rank, hostname, addition, partner_rank);
                fflush(stdout);
                
                MPI_Send(&addition, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
                
                printf("[NIVEL %d] Proceso %d en %s terminó su participación\n", level, my_rank, hostname);
                fflush(stdout);
            }
            break;
        }
    }
    
    if (my_rank == 0) {
        printf("Suma calculada: %d\n", addition);
        free(all_hostnames);
    }
    
    printf("Proceso %d en %s terminando\n", my_rank, hostname);
    fflush(stdout);
    
    MPI_Finalize();
    return 0;
}