#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int* merge(int* first, int f_size, int* second, int s_size) {
    int fi = 0, si = 0, mi = 0;
    int* merged;
    int m_size = f_size + s_size;
    merged = (int*)malloc(m_size * sizeof(int));

    while ((fi < f_size) && (si < s_size)) {
        if (first[fi] <= second[si]) {
            merged[mi++] = first[fi++];
        } else {
            merged[mi++] = second[si++];
        }
    }

    while (fi < f_size) merged[mi++] = first[fi++];
    while (si < s_size) merged[mi++] = second[si++];

    return merged;
}

void sort(int* array, int start, int end) {
    if (start >= end) return;

    int mid = (start + end) / 2;
    sort(array, start, mid);
    sort(array, mid + 1, end);

    int left_count = mid - start + 1;
    int right_count = end - mid;
    int* merged = merge(array + start, left_count, array + mid + 1, right_count);

    for (int i = 0; i < left_count + right_count; i++)
        array[start + i] = merged[i];

    free(merged);
}

void print_elements(int my_rank, const char* host, int* t, int n) {
    printf("[Proceso %d en %s] -> ", my_rank, host);
    for (int i = 0; i < n; i++) printf("%d ", t[i]);
    printf("\n");
}

int main(int argc, char* argv[]) {
    int *local_data, *array_sec;
    int n, my_rank, comm_sz, local_n, step, q;
    double start, stop;
    MPI_Status status;
    char host[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Get_processor_name(host, &name_len);

    printf("[DEBUG] Proceso %d corriendo en máquina: %s\n", my_rank, host);
    fflush(stdout);

    if (my_rank == 0) {
        printf("Total de elementos (n): ");
        fflush(stdout);
        scanf("%d", &n);

        if (n % comm_sz != 0) {
            printf("El tamanio no es divisible por comm_sz, se ajustara.\n");
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_n = n / comm_sz;

    local_data = (int*)malloc(local_n * sizeof(int));
    srand(time(NULL) + my_rank);

    for (int i = 0; i < local_n; i++) {
        local_data[i] = rand() % 1000;
    }

    sort(local_data, 0, local_n - 1);
    MPI_Barrier(MPI_COMM_WORLD);
    print_elements(my_rank, host, local_data, local_n);

    if (my_rank == 0) {
        array_sec = (int*)malloc(local_n * sizeof(int));
        for (int i = 1; i < comm_sz; i++) {
            MPI_Recv(array_sec, local_n, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            print_elements(i, host, array_sec, local_n);
        }
        free(array_sec);
    }
    else {
        MPI_Send(local_data, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    step = 1;
    while (step < comm_sz) {
        if (my_rank % (2 * step) == 0) {
            if (my_rank + step < comm_sz) {
                MPI_Recv(&q, 1, MPI_INT, my_rank + step, 0, MPI_COMM_WORLD, &status);
                array_sec = (int*)malloc(q * sizeof(int));
                MPI_Recv(array_sec, q, MPI_INT, my_rank + step, 0, MPI_COMM_WORLD, &status);

                printf("[DEBUG] Proceso %d en %s recibe %d elementos de %d\n", my_rank, host, q, my_rank + step);

                int* merged = merge(local_data, local_n, array_sec, q);
                free(local_data);
                free(array_sec);
                local_data = merged;
                local_n += q;
            }
        } else {
            int dest = my_rank - step;
            MPI_Send(&local_n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(local_data, local_n, MPI_INT, dest, 0, MPI_COMM_WORLD);

            printf("[DEBUG] Proceso %d en %s envia %d elementos a %d\n", my_rank, host, local_n, dest);

            break;
        }
        step *= 2;
    }

    stop = MPI_Wtime();

    if (my_rank == 0) {
        print_elements(my_rank, host, local_data, local_n);
        printf("Tiempo total: %f seg con %d procesos.\n", stop - start, comm_sz);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}