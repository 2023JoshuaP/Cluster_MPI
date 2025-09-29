#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int iterations = 1000;
    int message = 42;
    double cpu_time, start_mpi, end_mpi, mpi_time;
    clock_t start, end;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Get_processor_name(processor_name, &name_len);

    printf("Proceso %d ejecutándose en máquina: %s\n", my_rank, processor_name);
    fflush(stdout);

    if (comm_sz != 2) {
        if (my_rank == 0) {
            printf("Se necesita 2 procesos para correr el ping pong.\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Proceso 0 (máquina %s) <-> Proceso 1 (enviando datos...)\n", processor_name);
        printf("Medicion con clock()\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    start = clock();
    for (int i = 0; i < iterations; i++) {
        if (my_rank == 0) {
            MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    end = clock();

    cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    if (my_rank == 0) {
        printf("Tiempo con el metodo clock: %f seg.\n", cpu_time);
        printf("Tiempo por ping pong: %f mic.\n", (cpu_time * 1000000) / iterations);
        printf("CLOCKS_PER_SEC = %ld.\n", CLOCKS_PER_SEC);
        printf("Diferencia de ticks: %ld.\n", (long)(end - start));
    }

    if (my_rank == 0) {
        printf("Medicion con MPI_Wtime()\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_mpi = MPI_Wtime();

    for (int i = 0; i < iterations; i++) {
        if (my_rank == 0) {
            MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Recv(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    end_mpi = MPI_Wtime();
    mpi_time = end_mpi - start_mpi;

    if (my_rank == 0) {
        printf("Tiempo con Wtime(): %f seg.\n", mpi_time);
        printf("Tiempo por ping pong: %f mic.\n", (mpi_time * 1000000) / iterations);
        printf("Resolucion de MPI_Wtime(): %e seg.\n", MPI_Wtick());
    }

    if (my_rank == 0) {
        int test_iterations = 1;
        double test_time = 0.0;

        while (test_time == 0.0 && test_iterations <= 1000000) {
            MPI_Barrier(MPI_COMM_WORLD);
            clock_t test_start = clock();

            for (int i = 0; i < test_iterations; i++) {
                MPI_Send(&message, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
                MPI_Recv(&message, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            clock_t test_end = clock();
            test_time = ((double) (test_end - test_start)) / CLOCKS_PER_SEC;

            printf("%d iteraciones: clock() = %f seg, ticks = %ld.\n", test_iterations, test_time, (long)(test_end - test_start));

            if (test_time == 0.0) {
                test_iterations *= 2;
            }
        }

        if (test_time > 0.0) {
            printf("Minimo para tiempo no-cero: %d iteraciones.\n", test_iterations);
        }
        else {
            printf("No se encontro un no-cero hasta %d iteraciones.\n", test_iterations);
        }
    }
    else {
        int test_iterations = 1;

        while (test_iterations <= 1000000) {
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 0; i < test_iterations; i++) {
                MPI_Recv(&message, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&message, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            }
            test_iterations *= 2;

            if (test_iterations > 50000) {
                break;
            }
        }
    }

    if (my_rank == 0) {
        printf("clock(): %f seg.\n", cpu_time);
        printf("MPI_Wtime(): %f seg.\n", mpi_time);
        printf("Diferencia: %f seg.\n", fabs(cpu_time - mpi_time));
        printf("Ratio: %.2fx.\n", cpu_time > 0 ? mpi_time / cpu_time : 0);
        printf("Resoluciones: clock(): 1/%ld = %e seg.\n", CLOCKS_PER_SEC, 1.0 / CLOCKS_PER_SEC);
        printf("MPI_Wtime(): %e seg.\n", MPI_Wtick());
    }

    printf("Proceso %d (máquina %s) finalizando...\n", my_rank, processor_name);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}