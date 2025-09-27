#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char* argv[]) {
    int comm_sz;
    int my_rank;
    long long int number_tosses;
    long long int local_tosses;
    long long int number_circle = 0;
    long long int total_circle;
    double phi_estimate;
    double coordinate_x, coordinate_y;
    double distance_squared;
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0) {
        printf("Ingresa el total de lanzamientos de dardos: ");
        fflush(stdout);
        scanf("%lld", &number_tosses);
    }

    MPI_Bcast(&number_tosses, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    local_tosses = number_tosses / comm_sz;

    srand(time(NULL) + my_rank);

    for (i = 0; i < local_tosses; i++) {
        coordinate_x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        coordinate_y = (double)rand() / RAND_MAX * 2.0 - 1.0;

        distance_squared = coordinate_x * coordinate_x + coordinate_y * coordinate_y;

        if (distance_squared <= 1.0) {
            number_circle++;
        }
    }

    MPI_Reduce(&number_circle, &total_circle, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        phi_estimate = 4.0 * total_circle / ((double) number_tosses);
        printf("Total de lanzamientos: %lld\n", number_tosses);
        printf("Procesos utiizados: %d\n", comm_sz);
        printf("Dardos dentro del circulo: %lld\n", total_circle);
        printf("Estimacion de phi: %.10f\n", phi_estimate);
        printf("Valor real de phi: %.10f\n", M_PI);
        printf("Error absoluto: %.10f\n", fabs(phi_estimate - M_PI));
        printf("Error relativo: %.6f%%\n", fabs(phi_estimate - M_PI) / M_PI * 100);
    }

    MPI_Finalize();
    return 0;
}