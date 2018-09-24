import torch

cuda_enabled = torch.cuda.is_available()
print("cuda_enabled: ", cuda_enabled)

if cuda_enabled:
    print("current_device: ", torch.cuda.current_device())
    print("device_count: ", torch.cuda.device_count())
  
    for i in range(torch.cuda.device_count()):
        print("device_name: ", torch.cuda.get_device_name(i))