#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    int value, partner_rank, received;
    char hostname[256];
    int virtual_comm_sz;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    gethostname(hostname, sizeof(hostname));
    
    virtual_comm_sz = 1;
    while (virtual_comm_sz < comm_sz) {
        virtual_comm_sz *= 2;
    }
    
    value = my_rank;
    
    if (my_rank == 0) {
        printf("Procesos reales: %d\n", comm_sz);
        int is_power_of_2 = (comm_sz > 0) && ((comm_sz & (comm_sz - 1)) == 0);
        if (is_power_of_2) {
            printf("comm_sz es potencia de 2\n");
        } else {
            printf("comm_sz NO es potencia de 2\n");
            printf("Tamaño virtual del butterfly: %d\n", virtual_comm_sz);
        }
    }
    
    printf("Proceso %d en %s: valor inicial = %d\n", my_rank, hostname, value);
    fflush(stdout);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int i = 1, step = 1; i < virtual_comm_sz; i *= 2, step++) {
        partner_rank = my_rank ^ i;
        if (partner_rank < comm_sz) {
            printf("[PASO %d] Proceso %d en %s intercambiando con proceso %d\n", step, my_rank, hostname, partner_rank);
            fflush(stdout);
            
            MPI_Sendrecv(&value, 1, MPI_INT, partner_rank, 0, &received, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, &status);
            
            value += received;
            
            printf("[PASO %d] Proceso %d en %s: recibió %d, suma acumulada = %d\n", step, my_rank, hostname, received, value);
            fflush(stdout);
        } else {
            printf("[PASO %d] Proceso %d en %s: partner %d es virtual (no existe)\n", step, my_rank, hostname, partner_rank);
            fflush(stdout);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    int final_sum;
    if (my_rank == 0) {
        final_sum = value;
    }
    MPI_Bcast(&final_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        printf("Suma calculada: %d\n", final_sum);
    }
    
    printf("Proceso %d en %s terminando con suma = %d\n", my_rank, hostname, final_sum);
    fflush(stdout);
    
    MPI_Finalize();
    return 0;
}