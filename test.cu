#include <iostream>
#include <math.h>

__global__ void add(int n, float *sum, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+=stride)
    sum[i] = x[i] + y[i];
}

int main(){
  int N {1<<20};
  float *x, *y, *sum;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  cudaMallocManaged(&sum, N*sizeof(float));

  
  for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }

 int blockSize = 256;
 int numBlocks = (N + blockSize - 1) / blockSize;
 cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
 cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);
 add<<<numBlocks, blockSize>>>(N, sum, x, y);
 cudaDeviceSynchronize(); 
 float maxError = 0.0f;
 for (int i = 0; i < N; i++)
   maxError = fmax(maxError, fabs(sum[i]-3.0f));
 std::cout << "Max error: " << maxError << std::endl;
 cudaFree(x);
 cudaFree(y);
 cudaFree(sum);

}
