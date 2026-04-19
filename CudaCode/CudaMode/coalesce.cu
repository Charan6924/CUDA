#include <iostream>
#include <cuda_runtime.h>

__global__ void copyDataNonCoalesced(float* in, float* out, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index<n){
        out[index] = in[(index*2)%n];  
    }
}

__global__ void copyDataCoalesced(float* in, float* out, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index<n){
        out[index] = in[index];  
    }
}

void initializeArray(float* arr, int n){
    for (int i=0;i<n;i++){
        arr[i] = static_cast<float>(i);
    }
}

int main(){
    const int n = 1<<24;
    float* in;
    float* out;

    cudaMallocManaged(&in, n*sizeof(float));
    cudaMallocManaged(&out, n*sizeof(float));

    initializeArray(in, n);

    int blockSize = 128;
    int numBlocks = (n + blockSize - 1)/blockSize;

    copyDataNonCoalesced<<<numBlocks,blockSize>>>(in,out,n);
    cudaDeviceSynchronize();

    copyDataCoalesced<<<numBlocks,blockSize>>>(in,out,n);
    cudaDeviceSynchronize();

    cudaFree(in);
    cudaFree(out);
    return 0;

}
