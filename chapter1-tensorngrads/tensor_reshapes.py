import torch

x = torch.arange(1, 11)
print(x)

# Reshape into 5x2
x = x.view(5, 2)
print(x)

# Flatten back to 1D
x = x.view(-1)
print(x)
