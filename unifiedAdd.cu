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

void unifiedMemAdd(int vectorLength){
    float* A {};
    float* B {};
    float* C {};
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    int threads = 256;
    int blocks = (vectorLength + threads - 1)/threads;
    cudaMemPrefetchAsync(A, vectorLength*sizeof(float), 0, 0);
    cudaMemPrefetchAsync(B, vectorLength*sizeof(float), 0, 0);
    cudaMemPrefetchAsync(C, vectorLength*sizeof(float), 0, 0);
    vecAdd<<<blocks, threads>>>(A,B,C,vectorLength);
    cudaDeviceSynchronize();
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
    free(comparisonResult);
}

int main(){
    int vectorLength = 1024;
    unifiedMemAdd(vectorLength);
    return 0;
}

