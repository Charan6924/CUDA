#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
#define TILE 16

__global__ void matmul(float*a, float*b, float*c,int h, int w,int k){
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;
    for (int t = 0; t<cdiv(k,TILE);t++){
        int aCol = t*TILE + tx;
        tileA[ty][tx] = (row < h && aCol < k) ? a[row * k + aCol] : 0.0f;
        int bRow = t * TILE + ty;
        tileB[ty][tx] = (bRow < k && col < w) ? b[bRow * w + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; i++)
            sum += tileA[ty][i] * tileB[i][tx];

        __syncthreads();
    }
    if (row<h && col<w){
        c[row*w + col] = sum;
    }
}

torch::Tensor multiplyMatrices(torch::Tensor m, torch::Tensor n){
    CHECK_INPUT(m);
    CHECK_INPUT(n);
    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);
    TORCH_CHECK((k==n.size(0)), "size mismatch");
    torch::Tensor output {torch::zeros({h,w},m.options())};

    dim3 tpb(TILE,TILE);
    dim3 blocks(cdiv(w,TILE),cdiv(h,TILE));
    matmul<<<blocks,tpb>>>(m.data_ptr<float>(),n.data_ptr<float>(),output.data_ptr<float>(),h,w,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multiplyMatrices", &multiplyMatrices, "Matrix multiplication (CUDA)");
}
