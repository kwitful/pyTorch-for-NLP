import torch



# Defining a tensor
x = torch.tensor([[1, 2], [3, 4], [4,7]])
print(x)

# Tensor shape and type
print(x.shape)
print(x.dtype)



# Scalar
a = torch.tensor(5)

# Vector
b = torch.tensor([1, 2, 3])

# Matrix
c = torch.tensor([[1, 2], [3, 4]])

# Higher Dimensions
d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(a.shape)  
print(b.shape)
print(c.shape)  
print(d.shape) 
