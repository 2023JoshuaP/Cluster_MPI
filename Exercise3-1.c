#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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

    printf("\nHistograma:\n");
    printf("Range\t\t\tCount\n");
    printf("------\t\t\t------\n");

    for (int i = 0; i < bin_count; i++) {
        float bin_max = min_meas + bin_width * (i + 1);
        float bin_min = min_meas + bin_width * i;
        printf("[%.2f, %.2f)\t\t%d\n", bin_min, bin_max, bin_counts[i]);
    }

    printf("\nTotal de puntos de datos: %d\n", data_count);
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
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
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
        
        printf("Enter the data:\n");
        for (int i = 0; i < data_count; i++) {
            scanf("%f", &data[i]);
        }
    }
    
    MPI_Bcast(&data_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&min_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_meas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bin_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    local_data_count = data_count / comm_sz;
    local_data = malloc(local_data_count * sizeof(float));
    local_bin_counts = malloc(bin_count * sizeof(int));
    
    for (int b = 0; b < bin_count; b++) {
        local_bin_counts[b] = 0;
    }
    
    MPI_Scatter(data, local_data_count, MPI_FLOAT, local_data, local_data_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < local_data_count; i++) {
        int bin = find_bind(local_data[i], min_meas, max_meas, bin_count);
        local_bin_counts[bin]++;
    }
    
    if (my_rank == 0) {
        MPI_Reduce(local_bin_counts, bin_counts, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Reduce(local_bin_counts, NULL, bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    if (my_rank == 0) {
        print_histogram(bin_counts, bin_count, min_meas, max_meas, data_count, my_rank);
        free(data);
        free(bin_counts);
    }
    
    free(local_data);
    free(local_bin_counts);
    
    MPI_Finalize();
    return 0;
}