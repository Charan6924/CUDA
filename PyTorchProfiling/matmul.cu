#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void matmul(float*a, float*b, float*c,int h, int w,int k){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r<h && col<w){
        float sum = 0.0f;
        for (int i=0;i<k;i++){
            sum += a[r*k+i] * b[i*w+col];
        }
        c[r * w + col] = sum;
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

    dim3 tpb(16,16);
    dim3 blocks(cdiv(w,tpb.x),cdiv(h,tpb.y));
    matmul<<<blocks,tpb>>>(m.data_ptr<float>(),n.data_ptr<float>(),output.data_ptr<float>(),h,w,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multiplyMatrices", &multiplyMatrices, "Matrix multiplication (CUDA)");
}
