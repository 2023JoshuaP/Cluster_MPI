#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

int find_bind(float value, float min_meas, float max_meas, int bin_count) {
    float bin_width = (max_meas - min_meas) / bin_count;
    int bin;
    bin = (int)((value - min_meas) / bin_width);
    if (bin == bin_count) {
        bin = bin_count - 1;
    }
    return bin;
}

void print_histogram(int bin_counts[], int bin_count, float min_meas, float max_meas, int data_count, int my_rank) {
    float bin_width = (max_meas - min_meas) / bin_count;
    printf("\n=== HISTOGRAMA FINAL (Proceso %d) ===\n", my_rank);
    printf("Range\t\t\tCount\n");
    printf("------\t\t\t------\n");
    for (int i = 0; i < bin_count; i++) {
        float bin_max = min_meas + bin_width * (i + 1);
        float bin_min = min_meas + bin_width * i;
        printf("[%.2f, %.2f)\t\t%d\n", bin_min, bin_max, bin_counts[i]);
    }
    printf("\nTotal de puntos de datos procesados: %d\n", data_count);
}

int main(int argc, char* argv[]) {
    int my_rank, comm_sz;
    int data_count, bin_count;
    float min_meas, max_meas;
    float *data = NULL;
    float *local_data;
    int *local_bin_counts;
    int *bin_counts = NULL;
    int local_data_count;
    char hostname[256];
    char *all_hostnames = NULL;
    double start_time, end_time, elapsed_time;
    
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
                printf("Proceso %d: %s\n", i, host);
            } else {
                printf("Proceso %d: %s\n", i, host);
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int i = 0; i < comm_sz; i++) {
        if (my_rank == i) {
            if (i == 0) {
                printf("Proceso %d ejecutándose en %s (PRINCIPAL)\n", my_rank, hostname);
            } else {
                printf("Proceso %d ejecutándose en %s (REMOTA)\n", my_rank, hostname);
            }
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (my_rank == 0) {
        printf("Enter the number of data points: ");
        fflush(stdout);
        scanf("%d", &data_count);
        
        printf("Enter the minimum measurement: ");
        fflush(stdout);
        scanf("%f", &min_meas);
        
        printf("Enter the maximum measurement: ");
        fflush(stdout);
        scanf("%f", &max_meas);
        
        printf("Enter the number of bins: ");
        fflush(stdout);
        scanf("%d", &bin_count);
        
        data = malloc(data_count * sizeof(float));
        bin_counts = malloc(bin_count * sizeof(int));
        
        for (int i = 0; i < bin_count; i++) {
            bin_counts[i] = 0;
        }
        
        srand(time(NULL));
        printf("\nGenerando %d datos aleatorios entre %.2f y %.2f...\n", 
               data_count, min_meas, max_meas);
        for (int i = 0; i < data_count; i++) {
            data[i] = min_meas + ((float)rand() / RAND_MAX) * (max_meas - min_meas);
        }
        printf("Datos generados exitosamente.\n\n");
        
        start_time = MPI_Wtime();
    }
    
    MPI_Bcast(&data_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&min_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bin_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    local_data_count = data_count / comm_sz;
    int remainder = data_count % comm_sz;
    if (my_rank < remainder) {
        local_data_count++;
    }
    
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (my_rank == 0) {
        sendcounts = malloc(comm_sz * sizeof(int));
        displs = malloc(comm_sz * sizeof(int));
        int offset = 0;
        for (int i = 0; i < comm_sz; i++) {
            sendcounts[i] = data_count / comm_sz;
            if (i < remainder) {
                sendcounts[i]++;
            }
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    local_data = malloc(local_data_count * sizeof(float));
    local_bin_counts = malloc(bin_count * sizeof(int));
    
    for (int b = 0; b < bin_count; b++) {
        local_bin_counts[b] = 0;
    }
    
    MPI_Scatterv(data, sendcounts, displs, MPI_FLOAT, local_data, local_data_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    printf("[PROCESANDO] Proceso %d en %s procesando %d datos...\n", my_rank, hostname, local_data_count);
    fflush(stdout);
    
    for (int i = 0; i < local_data_count; i++) {
        int bin = find_bind(local_data[i], min_meas, max_meas, bin_count);
        local_bin_counts[bin]++;
    }
    
    printf("Proceso %d en %s terminó el procesamiento local\n", my_rank, hostname);
    fflush(stdout);
    
    if (my_rank == 0) {
        MPI_Reduce(local_bin_counts, bin_counts, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Reduce(local_bin_counts, NULL, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    if (my_rank == 0) {
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;

        print_histogram(bin_counts, bin_count, min_meas, max_meas, data_count, my_rank);
        printf("Datos distribuidos entre %d procesos\n", comm_sz);
        printf("Procesamiento paralelo completado\n");
        printf("Resultados consolidados exitosamente\n");
        printf("Tiempo de ejecución: %.6f segundos\n", elapsed_time);
        printf("Tiempo de ejecución: %.3f milisegundos\n", elapsed_time * 1000);
        
        free(data);
        free(bin_counts);
        free(sendcounts);
        free(displs);
        free(all_hostnames);
    }
    
    printf("Proceso %d en %s terminando ejecución\n", my_rank, hostname);
    fflush(stdout);
    
    free(local_data);
    free(local_bin_counts);
    
    MPI_Finalize();
    return 0;
}