import torch

# Tensors with gradient tracking
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Perform computations
c = a ** 2 + b ** 2

# Compute gradients
c.backward()

print(a.grad)  # Output: 4 (dc/da)
print(b.grad)  # Output: 6 (dc/db)