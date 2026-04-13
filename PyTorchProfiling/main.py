import torch

def time_pytorch_function(func, input):
    for _ in range(5):
        func(input)  # Run the function 5 times to warm up the GPU

    start = torch.cuda.Event(enable_timing=True) 
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    func(input)  # Run the function to be timed
    end.record()  # Stop the timer
    torch.cuda.synchronize()  # Wait for the kernel to finish
    return start.elapsed_time(end)  # Return the elapsed time in milliseconds

b = torch.randn(10000, 10000).cuda()

def square_2(a):
    return a*a

def square_3(a):
    return a**2

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    torch.square(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    square_2(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with torch.autograd.profiler.profile(use_device='cuda') as prof:
    square_3(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
