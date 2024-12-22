import torch

# Initial weight (guess)
w = torch.tensor(1.0, requires_grad=True)

# Target value (what we want)
target = 10.0

# Learning rate (how big the steps are)
lr = 0.1

# Example input (You would usually have multiple inputs)
x = torch.tensor(2.0) #Example input

for epoch in range(5):  # Repeat 5 times
    # Compute prediction and loss
    prediction = w * x # Correct prediction calculation
    loss = (prediction - target) ** 2

    # Compute gradient
    loss.backward()

    # Update weight using an optimizer (Recommended)
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()
    # Or Using an optimizer
    # optimizer = torch.optim.SGD([w], lr=lr)
    # optimizer.step()
    # optimizer.zero_grad()


    print(f"Epoch {epoch+1}: Weight = {w.item()}, Loss = {loss.item()}")