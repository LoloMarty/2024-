#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000000 // Size of the vectors

// Host function for vector addition using CPU
void vectorAddCPU(int *a, int *b, int *c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition using GPU
__global__ void vectorAddGPU(int *a, int *b, int *c, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int *h_a, *h_b, *h_c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    // Allocate memory for host vectors
    h_a = (int *)malloc(N * sizeof(int));
    h_b = (int *)malloc(N * sizeof(int));
    h_c = (int *)malloc(N * sizeof(int));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Allocate memory for device vectors
    cudaMalloc((void **)&d_a, N * sizeof(int));
    cudaMalloc((void **)&d_b, N * sizeof(int));
    cudaMalloc((void **)&d_c, N * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // CPU vector addition and performance measurement
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    vectorAddCPU(h_a, h_b, h_c, N);
    cpu_end = clock();
    printf("CPU Time: %f ms\n", ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000);

    // GPU vector addition and performance measurement
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    dim3 blockDim(256); // 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    cudaEventRecord(gpu_start);
    vectorAddGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_end);
    printf("GPU Time: %f ms\n", gpu_time);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the correctness of the results
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            fprintf(stderr, "Error: CPU and GPU results do not match at index %d\n", i);
            break;
        }
    }

    // Free allocated memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
