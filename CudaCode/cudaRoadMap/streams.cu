#include <iostream>

__global__ void vecAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20; // 1M elements
    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) { 
        h_a[i] = 1.0f; 
        h_b[i] = 2.0f; 
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream);

    dim3 threads(256);
    dim3 blocks((n + threads.x - 1) / threads.x);

    cudaEventRecord(start, stream);
    vecAdd<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop, stream);

    cudaStreamSynchronize(stream);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

    cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);
    free(h_a); 
    free(h_b); 
    free(h_c);
    return 0;
}
