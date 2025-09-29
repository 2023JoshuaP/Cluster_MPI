#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

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
    double pi_estimate;
    double coordinate_x, coordinate_y;
    double distance_squared;
    char hostname[256];
    char *all_hostnames = NULL;
    double start_time, end_time, total_time;
    int i;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    gethostname(hostname, sizeof(hostname));
    
    if (my_rank == 0) {
        all_hostnames = malloc(comm_sz * 256 * sizeof(char));
        printf("Número total de procesos: %d\n", comm_sz);
    }
    
    MPI_Gather(hostname, 256, MPI_CHAR, all_hostnames, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        for (int i = 0; i < comm_sz; i++) {
            char *host = all_hostnames + i * 256;
            if (i == 0) {
                printf("Proceso %d: %s (MÁQUINA PRINCIPAL - Master)\n", i, host);
            }
            else {
                printf("Proceso %d: %s (MÁQUINA REMOTA - Worker %d)\n", i, host, i);
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int i = 0; i < comm_sz; i++) {
        if (my_rank == i) {
            if (i == 0) {
                printf("Proceso %d ejecutándose en %s (PRINCIPAL)\n", my_rank, hostname);
            }
            else {
                printf("Proceso %d ejecutándose en %s (REMOTA)\n", my_rank, hostname);
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (my_rank == 0) {
        printf("Ingresa el total de lanzamientos de dardos: ");
        fflush(stdout);
        scanf("%lld", &number_tosses);
    }
    
    MPI_Bcast(&number_tosses, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    
    local_tosses = number_tosses / comm_sz;
    long long int remainder = number_tosses % comm_sz;
    if (my_rank < remainder) {
        local_tosses++;
    }
    
    printf("Proceso %d en %s procesará %lld lanzamientos\n", my_rank, hostname, local_tosses);
    fflush(stdout);
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    srand(time(NULL) + my_rank * 1000);
    printf("Proceso %d en %s iniciando lanzamientos...\n", my_rank, hostname);
    fflush(stdout);
    
    for (i = 0; i < local_tosses; i++) {
        coordinate_x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        coordinate_y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        distance_squared = coordinate_x * coordinate_x + coordinate_y * coordinate_y;
        
        if (distance_squared <= 1.0) {
            number_circle++;
        }
        
        if (local_tosses > 1000000 && i % (local_tosses / 10) == 0 && i > 0) {
            printf("Proceso %d: %d%% completado\n", my_rank, (int)(100.0 * i / local_tosses));
            fflush(stdout);
        }
    }
    
    printf("Proceso %d en %s terminó: %lld dardos dentro del círculo\n", my_rank, hostname, number_circle);
    fflush(stdout);    
    MPI_Reduce(&number_circle, &total_circle, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        end_time = MPI_Wtime();
        total_time = end_time - start_time;
        pi_estimate = 4.0 * total_circle / ((double) number_tosses);

        printf("Total de lanzamientos: %lld\n", number_tosses);
        printf("Procesos utilizados: %d\n", comm_sz);
        printf("Dardos dentro del círculo: %lld\n", total_circle);
        printf("Dardos fuera del círculo: %lld\n", number_tosses - total_circle);
        printf("Ratio dentro/total: %.6f\n", (double)total_circle / number_tosses);
        printf("Estimación de π: %.10f\n", pi_estimate);
        printf("Valor real de π: %.10f\n", M_PI);
        printf("Error absoluto: %.10f\n", fabs(pi_estimate - M_PI));
        printf("Error relativo: %.6f%%\n", fabs(pi_estimate - M_PI) / M_PI * 100);
        printf("Tiempo total: %.4f segundos\n", total_time);

        if (fabs(pi_estimate - M_PI) < 0.1) {
            printf("Estimación de π dentro del rango aceptable\n");
        }
        else {
            printf("Estimación de π fuera del rango esperado (necesitas más lanzamientos)\n");
        }
        
        free(all_hostnames);
    }
    
    printf("Proceso %d en %s terminando ejecución\n", my_rank, hostname);
    fflush(stdout);
    
    MPI_Finalize();
    return 0;
}