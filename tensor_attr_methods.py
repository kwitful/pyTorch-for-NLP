import torch

x = torch.tensor([[1, 2], [3, 4]])

# SHAPE - The dimension (size) of the tensor
print(x.shape)  # Output: torch.Size([2, 2])


# SIZE - The size of each dimension

print(x.size())  # Output: torch.Size([2, 2])
print(x.size(0)) # Output: 2 (rows)
print(x.size(1)) # Output: 2 (columns)


# NUMBER OF DIMENSIONS 
print(x.ndimension())  # Output: 2
print(x.dim())         # Output: 2 (same as .ndimension())



# DATA TYPE - The type of the data stored in the tensor
y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(y.dtype)  # Output: torch.float32

z = torch.tensor([1, 2, 3], dtype=torch.int64)
print(z.dtype)  # Output: torch.int64


# DEVICE - The storage place of the tensor: CPU | GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_tens = torch.tensor([1, 2, 3], device=device)
print(d_tens.device)  # Output: cuda:0 (if GPU is available)


# REQUIRES GRAD - Indicated gradient tracking for automatic differantiation
tra_tens = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(tra_tens.requires_grad)  # Output: True

# GRADIENTS - The gradients computed after backpropagation
x_tens = torch.tensor([2.0], requires_grad=True)
y_func = x_tens ** 2  # y = x^2
y_func.backward()  # Compute gradients
print(x_tens.grad)  # Output: tensor([4.])



# ITEM - Converts a single-element tensor into a Python number
i_tens = torch.tensor([3.5])
print(i_tens.item())  # Output: 3.5

