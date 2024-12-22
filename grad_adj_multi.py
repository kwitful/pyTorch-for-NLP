import torch

# Sample data
X = torch.tensor([[1.0], [2.0], [3.0]])  # Input features (3 samples, 1 feature)
Y = torch.tensor([[2.0], [4.0], [5.0]])  # Target values

# Model parameters
w = torch.randn(1, requires_grad=True)  # Initialize weight randomly
b = torch.randn(1, requires_grad=True) # Initialize bias randomly
lr = 0.01
epochs = 100

# Optimizer
optimizer = torch.optim.SGD([w, b], lr=lr)

for epoch in range(epochs):
    # Forward pass
    predictions = X @ w + b #Matric multiplication @
    loss = torch.mean((predictions - Y) ** 2)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
      print(f'epoch: {epoch+1}, loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}')

print(f'Final w:{w.item()}, Final b:{b.item()}')