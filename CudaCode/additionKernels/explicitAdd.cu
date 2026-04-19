#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength){
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < vectorLength){
        C[workIndex] = B[workIndex] + A[workIndex];
    }
}

void initArray(float* A, int length){
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] - B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

#define CUDA_CHECK(expr_to_check) do {                          \
    cudaError_t result  = expr_to_check;                        \
    if(result != cudaSuccess)                                   \
    {                                                           \
        fprintf(stderr,                                         \
                "CUDA Runtime Error: %s:%i:%d = %s\n",         \
                __FILE__,                                       \
                __LINE__,                                       \
                result,                                         \
                cudaGetErrorString(result));                    \
    }                                                           \
} while(0)

void explicitAdd(int vectorLength){
    float* A{};
    float* B{};
    float* C{};
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    float* devA {};
    float* devB {};
    float* devC {};

    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    initArray(A, vectorLength);
    initArray(B,vectorLength);

    CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));

    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));

    int threads = 256;
    int blocks = (vectorLength + threads - 1)/threads;
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    cudaDeviceSynchronize();

    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);
    serialVecAdd(A, B, comparisonResult, vectorLength);
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        std::cout << "Unified Memory: CPU and GPU answers match\n";
    }
    else
    {
        std::cout << "Unified Memory: Error - CPU and GPU answers do not match\n";
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}

int main(){
    int vectorLength = 1024;
    explicitAdd(vectorLength);
    return 0;
}
