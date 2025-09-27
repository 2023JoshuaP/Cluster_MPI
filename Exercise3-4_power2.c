#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    int value, partner_rank, received;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ((comm_sz & (comm_sz - 1)) != 0) {
        printf("El comm_sz debe ser una potencia de 2.\n");
        MPI_Finalize();
        return 1;
    }

    value = my_rank;

    if (my_rank == 0) {
        printf("Suma de Butterfly con %d procesos\n", comm_sz);
    }

    for (int i = 1; i < comm_sz; i *= 2) {
        partner_rank = my_rank ^ i;

        MPI_Sendrecv(&value, 1, MPI_INT, partner_rank, 0, &received, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, &status);

        value += received;
    }

    printf("Proceso %d con suma total = %d\n", my_rank, value);

    MPI_Finalize();
    return 0;
}