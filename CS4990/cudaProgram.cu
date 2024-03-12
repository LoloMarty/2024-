#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>

// CUDA kernel for vector addition
__global__ void vectorAdditionGPU(float *a, float *b, float *c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

// CPU function for vector addition
void vectorAdditionCPU(float *a, float *b, float *c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int size = 1000000;  // Adjust the size as needed

    // Allocate memory for vectors on CPU
    float *h_a = new float[size];
    float *h_b = new float[size];
    float *h_c_cpu = new float[size];
    float *h_c_gpu = new float[size];

    // Initialize vectors with random values
    for (int i = 0; i < size; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory for vectors on GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size * sizeof(float));
    cudaMalloc((void **)&d_b, size * sizeof(float));
    cudaMalloc((void **)&d_c, size * sizeof(float));

    // Copy vectors from CPU to GPU
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);

    // Perform vector addition on GPU and measure the time
    auto start_gpu = std::chrono::high_resolution_clock::now();
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdditionGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Copy result from GPU to CPU
    cudaMemcpy(h_c_gpu, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform vector addition on CPU and measure the time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAdditionCPU(h_a, h_b, h_c_cpu, size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // Verify the results
    for (int i = 0; i < size; ++i) {
        if (std::fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            std::cerr << "Result verification failed at index " << i << std::endl;
            break;
        }
    }

    // Output performance results
    std::cout << "\nVector Addition of size " << size << std::endl;
    std::cout << "CPU Time: " << cpu_time.count() << " seconds" << std::endl;
    std::cout << "GPU Time: " << gpu_time.count() << " seconds" << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}