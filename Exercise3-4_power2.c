#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    int value, partner_rank, received;
    char hostname[256];
    MPI_Status status;
    
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
    
    value = my_rank;
    
    if (my_rank == 0) {
        printf("Procesos: %d\n", comm_sz);
    }
    
    printf("[INICIO] Proceso %d en %s: valor inicial = %d\n", my_rank, hostname, value);
    fflush(stdout);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int i = 1, step = 1; i < comm_sz; i *= 2, step++) {
        partner_rank = my_rank ^ i;
        
        printf("[PASO %d] Proceso %d en %s intercambiando con proceso %d\n", step, my_rank, hostname, partner_rank);
        fflush(stdout);
        
        MPI_Sendrecv(&value, 1, MPI_INT, partner_rank, 0, &received, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, &status);
        
        value += received;
        
        printf("[PASO %d] Proceso %d en %s: recibió %d, suma acumulada = %d\n", step, my_rank, hostname, received, value);
        fflush(stdout);
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (my_rank == 0) {
        printf("Suma calculada: %d\n", value);
    }
    
    printf("Proceso %d en %s terminando con suma = %d\n", my_rank, hostname, value);
    fflush(stdout);
    
    MPI_Finalize();
    return 0;
}