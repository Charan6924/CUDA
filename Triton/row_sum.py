from sympy import N
import torch
import triton
import triton.language as tl

def builtin_row_sum(x:torch.Tensor):
    return x.sum(dim=1)

def triton_row_sum(x:torch.Tensor):
    M,N = x.shape
    y = torch.empty(M, device = 'cuda', dtype = x.dtype)
    BLOCK_SIZE=1024
    row_sum_kernel[(M,)](x,y,BLOCK_SIZE=BLOCK_SIZE) #type: ignore
    return y

@triton.jit
def row_sum_kernel(x_ptr,y_ptr,BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE],tl.float32)

    for start in range(0,N,BLOCK_SIZE):
        cols = start + tl.arange(0,BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask,other=0.0)
        acc += x
    
    result = tl.sum(acc,axis=0)
    tl.store(y_ptr+row,result)

def launch_kernel():
    raise NotImplementedError

launch_kernel()
