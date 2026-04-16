from torch.utils.cpp_extension import load
import torch
import time

src = "matmul.cu"
module = load(name='matmul', sources=[src], verbose=True)

N = 4096
a = torch.randn(N, N, dtype=torch.float32).cuda()
b = torch.randn(N, N, dtype=torch.float32).cuda()

for _ in range(3):
    module.multiplyMatrices(a, b)
    torch.mm(a, b)
torch.cuda.synchronize()

runs = 10
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(runs):
    module.multiplyMatrices(a, b)
torch.cuda.synchronize()
custom_ms = (time.perf_counter() - start) * 1000 / runs

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(runs):
    torch.mm(a, b)
torch.cuda.synchronize()
cublas_ms = (time.perf_counter() - start) * 1000 / runs

custom_out = module.multiplyMatrices(a, b)
cublas_out = torch.mm(a, b)
max_err = (custom_out - cublas_out).abs().max().item()

print(f"\nMatrix size: {N}x{N}")
print(f"Number of runs : {runs}")
print(f"Custom kernel: {custom_ms:.2f} ms")
print(f"torch.mm (cuBLAS): {cublas_ms:.2f} ms")
print(f"Speedup (cuBLAS/custom): {custom_ms/cublas_ms:.1f}x slower")
print(f"Max absolute error vs torch.mm: {max_err:.6f}")
