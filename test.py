import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
else:
    print("running on CPU")
