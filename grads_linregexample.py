import torch

# Input (x) and target output (y)
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)


w = torch.tensor(0.0, requires_grad=True)  # Weight
b = torch.tensor(0.0, requires_grad=True)  # Bias


learning_rate = 0.01

# Forward pass
def model(x):
    return w * x + b

# Loss function (Mean Squared Error)
def loss_fn(y_pred, y):
    return ((y_pred - y) ** 2).mean()

for epoch in range(100):
    # Compute predictions
    y_pred = model(x)
    
    # Compute loss
    loss = loss_fn(y_pred, y)
    
    # Backpropagation
    loss.backward()

    # Update parameters
    with torch.no_grad():  # Temporarily disable gradient tracking
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # Clear gradients
        w.grad.zero_()
        b.grad.zero_()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}: Loss = {loss.item()}')

print(f'Final Weight: {w.item()}')
print(f'Final Bias: {b.item()}')