import torch

# Step 1: Define with 'requires_grad=True'
x = torch.tensor(2.0, requires_grad=True)

# Step 2: Perform some operation
y = x ** 3

# Step 3: Calculate the gradient
y.backward()  # Computes dy/dx

# Step 4: Print the gradient
print(x.grad)  
