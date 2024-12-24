import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  

# Dataset
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# Model
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleANN()

# Loss and Optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Learning rate = 0.01

# Training
epochs = 100
losses = []  # To store losses for plotting

for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())  # Record loss
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Evaluation
x_test = torch.tensor([[5.0], [6.0]], dtype=torch.float32)
y_test_pred = model(x_test)
print("Predictions:", y_test_pred.detach().numpy())

# Plot the loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()