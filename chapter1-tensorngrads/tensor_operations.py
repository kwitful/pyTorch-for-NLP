import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z = x + y
print("The resul of the addition is:", z)

# Multiplication
mult = x * y
print("The resul of the multiplication is:", mult)

# Matrix Multiplication
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
result = torch.mm(a, b)  # or a @ b
print(result)
