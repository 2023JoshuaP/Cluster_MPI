#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define lli long long int

int main(int argc, char* argv[]) {
    int comm_sz, my_rank;
    lli tosses_process, total_tosses, number_circle_local = 0, circle_total = 0;
    double coordinate_x, coordinate_y, distance_squared, phi_estimate;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0) {
        printf("Ingresa el total de lanzamientos: ");
        fflush(stdout);
        scanf("%lld", &total_tosses);
    }

    MPI_Bcast(&total_tosses, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    tosses_process = total_tosses / comm_sz;
    srand(time(NULL) + my_rank);

    for (lli i = 0; i < tosses_process; i++) {
        coordinate_x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        coordinate_y = (double)rand() / RAND_MAX * 2.0 - 1.0;

        distance_squared = coordinate_x * coordinate_x + coordinate_y * coordinate_y;

        if (distance_squared <= 1) {
            number_circle_local++;
        }
    }

    MPI_Reduce(&number_circle_local, &circle_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        phi_estimate = 4.0 * circle_total / ((double) total_tosses);
        printf("Phi estimado: %f\n", phi_estimate);
    }

    MPI_Finalize();
    return 0;
}