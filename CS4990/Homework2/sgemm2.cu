#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// CPU Timer
double myCPUTimer() {
    return clock() / (double)CLOCKS_PER_SEC;
}

// Host function for CPU-only matrix multiplication
void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A_h[i * k + l] * B_h[l * n + j];
            }
            C_h[i * n + j] = sum;
        }
    }
}

// CUDA kernel where each thread computes one output matrix element
__global__ void matrixMulKernel_1thread1element(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A_d[row * k + i] * B_d[i * n + col];
        }
        C_d[row * n + col] = sum;
    }
}

// CUDA kernel where tiled version of matrix multiplication is implemented
__global__ void matrixMulKernel_tiled(int m, int k, int n, const float *A_d, const float *B_d, float* C_d, unsigned Adz_sz, unsigned Bdz_sz) {
    extern __shared__ float As[];
    extern __shared__ float Bs[];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Cvalue = 0.0f;

    for (int t = 0; t < (k + blockDim.x - 1) / blockDim.x; ++t) {
        if (Row < m && t * blockDim.x + tx < k)
            As[ty * blockDim.x + tx] = A_d[Row * k + t * blockDim.x + tx];
        else
            As[ty * blockDim.x + tx] = 0.0f;

        if (t * blockDim.y + ty < k && Col < n)
            Bs[ty * blockDim.x + tx] = B_d[(t * blockDim.y + ty) * n + Col];
        else
            Bs[ty * blockDim.x + tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < blockDim.x; ++i)
            Cvalue += As[ty * blockDim.x + i] * Bs[i * blockDim.x + tx];

        __syncthreads();
    }

    if (Row < m && Col < n)
        C_d[Row * n + Col] = Cvalue;
}

// Host function for handling device memory allocation and free, data copy,
// and calling the specific CUDA kernel matrixMulKernel_1thread1element
void basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float* C_h, double* timing) {
    int size_A = m * k * sizeof(float);
    int size_B = k * n * sizeof(float);
    int size_C = m * n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CHECK(cudaMalloc((void**)&A_d, size_A));
    CHECK(cudaMalloc((void**)&B_d, size_B));
    CHECK(cudaMalloc((void**)&C_d, size_C));

    cudaEventRecord(start);
    CHECK(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Memory Allocation A and B: %.6f seconds\n", milliseconds / 1000.0f);

    cudaEventRecord(start);
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulKernel_1thread1element<<<blocksPerGrid, threadsPerBlock>>>(m, k, n, A_d, B_d, C_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("vecAddKernel<<<(%d, %d, 1), (%d,%d,%d)>>> %.6f seconds\n", (n + 15) / 16, (m + 15) / 16, 16, 16, 1, milliseconds / 1000.0f);

    cudaEventRecord(start);
    CHECK(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Memory Copy C: %.6f seconds\n", milliseconds / 1000.0f);

    *timing = milliseconds / 1000.0f;

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

// Host function for handling device memory allocation and copy, device query,
// and dynamically configuring the amount of shared memory and calling the specific
// CUDA kernel matrixMulKernel_tiled
void basicSgemm_d_tiled(int m, int k, int n, const float *A_h, const float *B_h, float* C_h, double* timing) {
    int size_A = m * k * sizeof(float);
    int size_B = k * n * sizeof(float);
    int size_C = m * n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CHECK(cudaMalloc((void**)&A_d, size_A));
    CHECK(cudaMalloc((void**)&B_d, size_B));
    CHECK(cudaMalloc((void**)&C_d, size_C));

    cudaEventRecord(start);
    CHECK(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Memory Allocation A and B: %.6f seconds\n", milliseconds / 1000.0f);

    cudaEventRecord(start);
    int devId;
    cudaDeviceProp devProp;
    CHECK(cudaGetDevice(&devId));
    CHECK(cudaGetDeviceProperties(&devProp, devId));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    unsigned int Adz_sz = threadsPerBlock.x * threadsPerBlock.y * sizeof(float);
    unsigned int Bdz_sz = threadsPerBlock.x * threadsPerBlock.y * sizeof(float);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("vecAddKernel<<<(%d, %d, 1), (%d,%d,%d)>>> %.6f seconds\n", (n + 15) / 16, (m + 15) / 16, 16, 16, 1, milliseconds / 1000.0f);

    cudaEventRecord(start);
    matrixMulKernel_tiled<<<blocksPerGrid, threadsPerBlock, Adz_sz + Bdz_sz>>>(m, k, n, A_d, B_d, C_d, Adz_sz, Bdz_sz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("vecAddKernel<<<(%d, %d, 1), (%d,%d,%d)>>> %.6f seconds\n", (n + 15) / 16, (m + 15) / 16, 16, 16, 1, milliseconds / 1000.0f);

    cudaEventRecord(start);
    CHECK(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Memory Copy C: %.6f seconds\n", milliseconds / 1000.0f);

    *timing = milliseconds / 1000.0f;

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

// Function to validate if computed matrices using CPU and GPU match
bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {
    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            if (fabs(CPU_Answer[i * nCols + j] - GPU_Answer[i * nCols + j]) > 1e-5)
                return false;
        }
    }
    return true;
}

// Main function
int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: ./sgemm2 <m> <k> <n>\n");
        return 1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    // Allocate memory for matrices on the host
    float *A_h = (float*)malloc(m * k * sizeof(float));
    float *B_h = (float*)malloc(k * n * sizeof(float));
    float *C_h_cpu = (float*)malloc(m * n * sizeof(float));
    float *C_h_gpu_1thread1element = (float*)malloc(m * n *sizeof(float));
    float *C_h_gpu_tiled = (float*)malloc(m * n * sizeof(float));
    double cpu_time, gpu_1thread1element_time, gpu_tiled_time;

    // Fill matrices with random values
    srand(time(NULL));
    for (int i = 0; i < m * k; ++i) {
        A_h[i] = rand() % 100 / 100.0f;
    }
    for (int i = 0; i < k * n; ++i) {
        B_h[i] = rand() % 100 / 100.0f;
    }

    // Compute matrix multiplication on CPU
    double start_cpu = myCPUTimer();
    basicSgemm_h(m, k, n, A_h, B_h, C_h_cpu);
    double end_cpu = myCPUTimer();
    cpu_time = end_cpu - start_cpu;
    printf("Vector Size %d\n", m * n);
    printf("Basic SGEMM on CPU: %.6f seconds\n", cpu_time);

    // Compute matrix multiplication using basic CUDA kernel (one thread per element)
    double gpu_1thread1element_timing;
    printf("\nGPU (1 Thread per Element):\n");
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, C_h_gpu_1thread1element, &gpu_1thread1element_timing);
    gpu_1thread1element_time = gpu_1thread1element_timing;

    // Compute matrix multiplication using tiled CUDA kernel
    double gpu_tiled_timing;
    printf("\nGPU (Tiled):\n");
    basicSgemm_d_tiled(m, k, n, A_h, B_h, C_h_gpu_tiled, &gpu_tiled_timing);
    gpu_tiled_time = gpu_tiled_timing;

    // Verify results
    bool cpu_vs_gpu_1thread1element = verify(C_h_cpu, C_h_gpu_1thread1element, m, n);
    bool cpu_vs_gpu_tiled = verify(C_h_cpu, C_h_gpu_tiled, m, n);

    if (cpu_vs_gpu_1thread1element)
        printf("\nResult from CPU and GPU (1 Thread per Element) matches!\n");
    else
        printf("\nResult from CPU and GPU (1 Thread per Element) differs!\n");

    if (cpu_vs_gpu_tiled)
        printf("Result from CPU and GPU (Tiled) matches!\n");
    else
        printf("Result from CPU and GPU (Tiled) differs!\n");

    // Print GPU execution times
    printf("\nGPU (1 Thread per Element) Time: %.6f seconds\n", gpu_1thread1element_time);
    printf("GPU (Tiled) Time: %.6f seconds\n", gpu_tiled_time);

    // Free host memory
    free(A_h);
    free(B_h);
    free(C_h_cpu);
    free(C_h_gpu_1thread1element);
    free(C_h_gpu_tiled);

    return 0;
}
