import torch, os, math
import torchvision as tv
import torchvision.transforms.functional as tvf
from torchvision import io
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load_inline


#img = io.read_image('puppy.jpg')
#print(img.shape)
#print(img[:2,:3,:4])


def show_img(x, figsize=(4,3), **kwargs):
    plt.figure(figsize=figsize)
    plt.axis('off')
    if len(x.shape)==3: x = x.permute(1,2,0)  # CHW -> HWC
    plt.imshow(x.cpu(), **kwargs)
    plt.savefig('saved')


#img2 = tvf.resize(img, 150, antialias=True) #type: ignore
#ch,h,w = img.shape
#print(ch,h,w,h*w)

def rgb2grey_py(x):
    c,h,w = x.shape
    n = h*w
    x = x.flatten()
    res = torch.empty(n, dtype=x.dtype, device=x.device)
    for i in range(n): res[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n]
    return res.view(h,w)

#img_g = rgb2grey_py(img)
#show_img(img_g, cmap='gray')

def run_kernel(f, times, *args):
    for i in range(times): f(i, *args)

def rgb2grey_k(i, x, out, n):
    out[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n]


def rgb2grey_pyk(x):
    c,h,w = x.shape
    n = h*w
    x = x.flatten()
    res = torch.empty(n, dtype=x.dtype, device=x.device)
    run_kernel(rgb2grey_k, h*w, x, res, n)
    return res.view(h,w)


#img_g = rgb2grey_pyk(img2)
#show_img(img_g, cmap='gray')

os.environ['CUDA_LAUNCH_BLOCKING']='1'
def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")

cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_src = cuda_begin + r'''
__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (unsigned char)(0.2126f*x[i*3] + 0.7152f*x[i*3+1] + 0.0722f*x[i*3+2]);
}

torch::Tensor rgb_to_grayscale(torch::Tensor input) {
    CHECK_INPUT(input);
    int h = input.size(0);
    int w = input.size(1);
    auto output = torch::empty({h, w}, input.options());
    rgb_to_grayscale_kernel<<<cdiv(w*h, 256), 256>>>(input.data_ptr<unsigned char>(),output.data_ptr<unsigned char>(),w * h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
'''

cpp_src = "torch::Tensor rgb_to_grayscale(torch::Tensor input);"

module = load_cuda(cuda_src, cpp_src, ['rgb_to_grayscale'], verbose=True)
print([o for o in dir(module) if o[0] != '_'])

img = io.read_image('puppy.jpg')
img2 = tvf.resize(img, 150, antialias=True)  #type: ignore
img2_cuda = img2.permute(1, 2, 0).contiguous().cuda() 
img_g = module.rgb_to_grayscale(img2_cuda)
show_img(img_g, cmap='gray')
