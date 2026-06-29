import torch
import triton
import triton.language as tl

def naive_matmul_relu(x:torch.Tensor,y:torch.Tensor):
    return torch.nn.functional.relu(x@y)




