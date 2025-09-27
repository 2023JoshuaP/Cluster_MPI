#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    int value, partner_rank, addition = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0) {
        int power_2 = (comm_sz > 0) && ((comm_sz & (comm_sz - 1)) == 0);
        if (power_2) {
            printf("El commsz es potencia de 2.\n");
        }
        else {
            printf("El commsz no es potencia de 2.\n");
        }
    }

    value = my_rank;
    addition = value;

    int virtual_commsz = 1;
    while (virtual_commsz < comm_sz) {
        virtual_commsz *= 2;
    }

    if (my_rank == 0) {
        printf("Arbol virtual de tamanio: %d\n", virtual_commsz);
    }

    for (int i = 1; i < virtual_commsz; i *= 2) {
        if (my_rank % (2 * i) == 0) {
            partner_rank = my_rank + i;
            if (partner_rank < comm_sz) {
                int result;
                MPI_Recv(&result, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                addition += result;
            }
        }
        else if (my_rank % (2 * i) == i) {
            partner_rank = my_rank - i;
            if (partner_rank >= 0 && partner_rank < comm_sz) {
                MPI_Send(&addition, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
            }
            break;
        }
    }

    if (my_rank == 0) {
        printf("Suma global: %d", addition);
    }

    MPI_Finalize();
    return 0;
}