#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    int value, partner_rank, received;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0) {
        int power_2 = (comm_sz > 0) && ((comm_sz & (comm_sz - 1)) == 0);
        if (power_2) {
            printf("El commsz es potencia de 2\n");
        }
        else {
            printf("El comm_sz no es potencia de 2\n");
        }
    }

    value = my_rank;

    int virtual_commsz = 1;
    while (virtual_commsz < comm_sz) {
        virtual_commsz *= 2;
    }

    if (my_rank == 0) {
        printf("Butterfly virtual de tamanio: %d\n", virtual_commsz);
    }

    for (int i = 1; i < virtual_commsz; i *= 2) {
        partner_rank = my_rank ^ i;
        if (partner_rank < comm_sz) {
            MPI_Sendrecv(&value, 1, MPI_INT, partner_rank, 0, &received, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, &status);
            value += received;
        }
    }

    int addition;
    if (my_rank == 0) {
        addition = value;
    }
    
    MPI_Bcast(&addition, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Proceso %d con suma total = %d\n", my_rank, addition);

    MPI_Finalize();
    return 0;
}