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
            merged[mi] = first[fi];
            mi++;
            fi++;
        }
        else {
            merged[mi] = second[si];
            mi++;
            si++;
        }
    }

    if (fi >= f_size) {
        for (int i = mi; i < m_size; i++, si++) {
            merged[i] = second[si];
        }
    }
    else if (si >= s_size) {
        for (int i = mi; i < m_size; i++, fi++) {
            merged[i] = first[fi];
        }
    }

    for (int i = 0; i < f_size; i++) {
        first[i] = merged[i];
    }
    for (int i = 0; i < s_size; i++) {
        second[i] = merged[f_size + i];
    }

    return merged;
}

void sort(int* array, int start, int end) {
    int* array_sort;
    int mid = (start + end) / 2;
    int left_count = mid - start + 1;
    int right_count = end - mid;

    if (end == start) {
        return;
    }
    else {
        sort(array, start, mid);
        sort(array, mid + 1, end);
        array_sort = merge(array + start, left_count, array + mid + 1, right_count);
        free(array_sort);
    }
}

void print_elements(int my_rank, int* t, int n) {
    printf("Proceso %d: ", my_rank);
    for (int i = 0; i < n; i++) {
        printf("%d ", t[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    int* local_data;
    int* array_sec;
    int q, n;
    int my_rank, comm_sz;
    int local_n = 0;
    int step;
    double start, stop;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == 0) {
        printf("Total de elementos (n): ");
        fflush(stdout);
        scanf("%d", &n);

        if (n % comm_sz != 0) {
            printf("El tamanio no es divisible por el comm_sz, se ajustara.\n");
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_n = n / comm_sz;

    start = MPI_Wtime();

    local_data = (int*)malloc(local_n * sizeof(int));
    srand(time(NULL) + my_rank);

    for (int i = 0; i < local_n; i++) {
        local_data[i] = rand() % 1000;
    }

    sort(local_data, 0, local_n - 1);

    if (my_rank == 0) {
        print_elements(my_rank, local_data, local_n);

        int* array_sec = (int*)malloc(local_n * sizeof(int));
        for (int i = 1; i < comm_sz; i++) {
            MPI_Recv(array_sec, local_n, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            print_elements(i, array_sec, local_n);
        }

        free(array_sec);
        printf("\n");
    }
    else {
        MPI_Send(local_data, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    step = 1;

    while (step < comm_sz) {
        if (my_rank % (2 * step) == 0) {
            if (my_rank + step < comm_sz) {
                MPI_Recv(&q, 1, MPI_INT, my_rank + step, 0, MPI_COMM_WORLD, &status);
                array_sec = (int*)malloc(q * sizeof(int));
                MPI_Recv(array_sec, q, MPI_INT, my_rank + step, 0, MPI_COMM_WORLD, &status);

                int* merged_result = merge(local_data, local_n, array_sec, q);
                free(local_data);
                free(array_sec);
                local_data = merged_result;
                local_n = local_n + q;
            }
        }
        else {
            int dest = my_rank - step;
            if (dest >= 0 && (dest % (2 * step)) == 0) {
                MPI_Send(&local_n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Send(local_data, local_n, MPI_INT, dest, 0, MPI_COMM_WORLD);
                break;
            }
        }
        step = step * 2;
    }

    stop = MPI_Wtime();

    if (my_rank == 0) {
        printf("Arreglo ordenado (%d elementos): ", local_n);
        print_elements(my_rank, local_data, local_n);
        printf("\nTiempo de ejecucion: %f seg con %d procesos.\n", stop - start, comm_sz);

        int sorted = 1;
        for (int i = 1; i < local_n; i++) {
            if (local_data[i] < local_data[i - 1]) {
                sorted = 0;
                break;
            }
        }
        printf("%s\n", sorted ? "Arreglo ordenado" : "Arreglo no ordenado");
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}