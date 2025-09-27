#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void initialize_block_vector(int *vector, int local_size, int rank, int total_size) {
    int sizes_per_proc[16];
    int displs[16];
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    int offset = 0;
    for (int i = 0; i < nprocs; i++) {
        sizes_per_proc[i] = total_size / nprocs + (i < total_size % nprocs ? 1 : 0);
        displs[i] = offset;
        offset += sizes_per_proc[i];
    }
    
    int global_start = displs[rank];
    for (int i = 0; i < local_size; i++) {
        vector[i] = global_start + i;
    }
}

void print_distribution(int *local_data, int local_size, int rank, int size, const char *dist_type) {
    printf("Proceso %d (%s): [", rank, dist_type);
    for (int i = 0; i < local_size && i < 10; i++) {
        printf("%d", local_data[i]);
        if (i < local_size - 1 && i < 9) printf(", ");
    }
    if (local_size > 10) printf(", ...");
    printf("] (tamaño: %d)\n", local_size);
    fflush(stdout);
}

double block_to_cyclic(int *block_data, int block_size, int **cyclic_data, int *cyclic_size, int total_size, int rank, int size) {
    double start_time = MPI_Wtime();
    
    *cyclic_size = total_size / size + (rank < total_size % size ? 1 : 0);
    *cyclic_data = (int*)calloc(*cyclic_size, sizeof(int));
    
    int *all_data = (int*)malloc(total_size * sizeof(int));
    int *recvcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        recvcounts[i] = total_size / size + (i < total_size % size ? 1 : 0);
        displs[i] = offset;
        offset += recvcounts[i];
    }
    
    MPI_Allgatherv(block_data, block_size, MPI_INT, all_data, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);
    
    int idx = 0;
    for (int i = rank; i < total_size; i += size) {
        if (idx < *cyclic_size) {
            (*cyclic_data)[idx++] = all_data[i];
        }
    }
    
    free(all_data);
    free(recvcounts);
    free(displs);
    
    double end_time = MPI_Wtime();
    return end_time - start_time;
}

double cyclic_to_block(int *cyclic_data, int cyclic_size, int **block_data, int *block_size,int total_size, int rank, int size) {
    double start_time = MPI_Wtime();
    
    *block_size = total_size / size + (rank < total_size % size ? 1 : 0);
    *block_data = (int*)calloc(*block_size, sizeof(int));
    
    int *send_buffer = (int*)calloc(total_size, sizeof(int));
    
    for (int i = 0; i < cyclic_size; i++) {
        int global_pos = rank + i * size;
        if (global_pos < total_size) {
            send_buffer[global_pos] = cyclic_data[i];
        }
    }
    
    int *all_data = (int*)malloc(total_size * sizeof(int));
    MPI_Allreduce(send_buffer, all_data, total_size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    int block_start = 0;
    for (int i = 0; i < rank; i++) {
        block_start += total_size / size + (i < total_size % size ? 1 : 0);
    }
    
    for (int i = 0; i < *block_size; i++) {
        (*block_data)[i] = all_data[block_start + i];
    }
    
    free(send_buffer);
    free(all_data);
    
    double end_time = MPI_Wtime();
    return end_time - start_time;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int total_size = 1000000;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc > 1) {
        total_size = atoi(argv[1]);
    }
    
    if (rank == 0) {
        printf("=== Análisis de Costos de Redistribución de Datos MPI ===\n");
        printf("Tamaño total del vector: %d elementos\n", total_size);
        printf("Número de procesos: %d\n", size);
        printf("Tamaño del elemento: %zu bytes\n\n", sizeof(int));
    }
    
    int block_size = total_size / size + (rank < total_size % size ? 1 : 0);
    int *initial_block = (int*)malloc(block_size * sizeof(int));
    initialize_block_vector(initial_block, block_size, rank, total_size);
    
    if (rank == 0) {
        printf("Distribución inicial por bloques:\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            print_distribution(initial_block, block_size, rank, size, "Block");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    int *cyclic_data;
    int cyclic_size;
    
    MPI_Barrier(MPI_COMM_WORLD);
    double block_to_cyclic_time = block_to_cyclic(initial_block, block_size, &cyclic_data, &cyclic_size, total_size, rank, size);
    
    double max_block_to_cyclic_time;
    MPI_Reduce(&block_to_cyclic_time, &max_block_to_cyclic_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\nDespués de redistribución Bloque a Cíclica:\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            print_distribution(cyclic_data, cyclic_size, rank, size, "Cyclic");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    int *final_block;
    int final_block_size;
    
    MPI_Barrier(MPI_COMM_WORLD);
    double cyclic_to_block_time = cyclic_to_block(cyclic_data, cyclic_size, &final_block, &final_block_size, total_size, rank, size);
    
    double max_cyclic_to_block_time;
    MPI_Reduce(&cyclic_to_block_time, &max_cyclic_to_block_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\nDespués de redistribución Cíclica a Bloque:\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            print_distribution(final_block, final_block_size, rank, size, "Block");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        printf("\n=== Resultados de Rendimiento ===\n");
        printf("Tiempo redistribución Bloque a Cíclica: %.6f segundos\n", max_block_to_cyclic_time);
        printf("Tiempo redistribución Cíclica a Bloque: %.6f segundos\n", max_cyclic_to_block_time);
        
        double data_volume_mb = (total_size * sizeof(int)) / (1024.0 * 1024.0);
        printf("Volumen de datos: %.2f MB\n", data_volume_mb);
        
        if (max_block_to_cyclic_time > 0) {
            printf("Rendimiento Bloque a Cíclica: %.2f MB/s\n", data_volume_mb / max_block_to_cyclic_time);
        }
        
        if (max_cyclic_to_block_time > 0) {
            printf("Rendimiento Cíclica a Bloque: %.2f MB/s\n", data_volume_mb / max_cyclic_to_block_time);
        }
        
        printf("Relación (Cíclica->Bloque / Bloque->Cíclica): %.2f\n", max_cyclic_to_block_time / max_block_to_cyclic_time);
    }
    
    int errors = 0;
    if (final_block_size == block_size) {
        for (int i = 0; i < block_size; i++) {
            if (initial_block[i] != final_block[i]) {
                errors++;
                if (errors <= 5) {
                    printf("Proceso %d: Error en posición %d: %d != %d\n", rank, i, initial_block[i], final_block[i]);
                }
            }
        }
    }
    else {
        printf("Proceso %d: Error de tamaño: %d != %d\n", rank, block_size, final_block_size);
        errors = 1;
    }
    
    int total_errors;
    MPI_Reduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (total_errors == 0) {
            printf("\nVerificación de datos: CORRECTO - Todos los datos se preservaron correctamente\n");
        }
        else {
            printf("\nVerificación de datos: ERROR - %d errores encontrados\n", total_errors);
        }
    }
    
    free(initial_block);
    free(cyclic_data);
    free(final_block);
    
    MPI_Finalize();
    return 0;
}