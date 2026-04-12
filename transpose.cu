#include <iostream>
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )

__global__ void smem_cuda_transpose(int m, float* a, float* c){
    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y+1];
    const int tileCol = blockDim.x * blockIdx.x;
    const int tileRow = blockDim.y * blockIdx.y;

    smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileRow + threadIdx.y, tileCol + threadIdx.x, m )];
    __syncthreads();

    c[INDX( tileCol + threadIdx.x, tileRow + threadIdx.y, m )] = smemArray[threadIdx.x][threadIdx.y];
    return;
}


int main(){
    int m = 4096;
    size_t bytes = m * m * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int row = 0; row < m; row++)
        for (int col = 0; col < m; col++)
            h_a[INDX(row, col, m)] = (float)(row * m + col);

    float *d_a{};
    float *d_c{};

    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_c,bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 blocks(m / THREADS_PER_BLOCK_X, m / THREADS_PER_BLOCK_Y);

    smem_cuda_transpose<<<blocks, threads>>>(m, d_a, d_c);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            float expected = h_a[INDX(col, row, m)];
            float got      = h_c[INDX(row, col, m)];
            if (expected != got) {
                printf("MISMATCH at C[%d][%d]: expected %.0f got %.0f\n",
                       row, col, expected, got);
                passed = false;
            }
        }
    }
    if (passed){
        std::cout << "Tranpose worked" << "\n";
        free(h_a);
        free(h_c);
        cudaFree(d_a);
        cudaFree(d_c);
        return 0;
    }
    std::cout << "Transpose failed" << "\n";
    free(h_a);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_c);
    return 0;
}
