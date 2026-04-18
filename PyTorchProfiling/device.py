import torch
props0 = torch.cuda.get_device_properties(0)
props1 = torch.cuda.get_device_properties(1)
props2 = torch.cuda.get_device_properties(2)
props3 = torch.cuda.get_device_properties(3)
print(props0)
print(props1)
print(props2)
print(props3)
