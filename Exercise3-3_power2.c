#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    int value, partner_rank, addition = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ((comm_sz & (comm_sz - 1)) != 0) {
        printf("El comm_sz debe ser una potencia de 2.\n");
        MPI_Finalize();
        return 1;
    }

    value = my_rank;
    addition = value;

    for (int i = 1; i < comm_sz; i *= 2) {
        if (my_rank % (2 * i) == 0) {
            partner_rank = my_rank + i;
            int result;
            MPI_Recv(&result, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            addition += result;
        }
        else if (my_rank % (2 * i) == i) {
            partner_rank = my_rank - i;
            MPI_Send(&addition, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
            break;
        }
    }

    if (my_rank == 0) {
        printf("Suma global: %d", addition);
    }

    MPI_Finalize();
    return 0;
}