from pathlib import Path
import pandas as pd
import torch
from torch.utils.cpp_extension import load_inline
pd.set_option('max_colwidth', None, 'display.max_rows', None, 'display.max_columns', None)
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"


cuda_source = '''
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);

    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);
    return result;
}

'''

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"
build_dir = Path('./load_inline_cuda')
build_dir.mkdir(exist_ok=True)

square_matrix_extension = load_inline(
    name='square_matrix_extension',   # Unique name for the extension
    cpp_sources=cpp_source,           # C++ source code containing the CPU implementation
    cuda_sources=cuda_source,         # CUDA source code for GPU implementation
    functions=['square_matrix'],      # List of functions to expose to Python
    with_cuda=True,                   # Enable CUDA support
    extra_cuda_cflags=["-O2"],        # Compiler flags for optimizing the CUDA code
    build_directory=str(build_dir),   # Directory to store the compiled extension
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix(a))
print(f"Module Path: {square_matrix_extension.__file__}")
pd.DataFrame([path.name for path in Path(square_matrix_extension.__file__).parent.iterdir()])
