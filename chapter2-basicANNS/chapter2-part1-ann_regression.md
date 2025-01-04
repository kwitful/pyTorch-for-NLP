# Building Your First Artificial Neural Network for Regression

## 1. Introduction

In this tutorial, we will walk through how to build , train, and evaluate an **Artificial Neural Network (ANN)** using PyTorch to solve a **regression problem**. I don't feel very good for using an "ideally convenient" dataset. However, such datasets function as good utilities for comprehending how neural networks are built and maintained. So, we will predict California housing prices based on features such as average income and house age.

### What is Regression?
Regression is a type of machine learning task where the goal is to predict a **continuous value** based on input features. For example:
- Predicting house prices based on size and location.
- Forecasting stock prices based on historical data.
- Estimating temperature changes based on weather patterns.

### Why Use an ANN for Regression?
An ANN is a powerful machine learning model that mimics the structure of the human brain. It is made up of layers of interconnected nodes (neurons) that process data and learn patterns. ANNs are especially effective for tasks involving complex relationships, such as regression problems where multiple inputs influence the output.

---

## 2. Prerequisites

Before you proceed, ensure you have the following libraries installed:

- **Python**: The programming language we are using.
- **PyTorch**: A deep learning framework that makes it easier to create and train neural networks.
- **Scikit-learn**: A library for preprocessing and handling datasets.

Install them with:
```bash
pip install torch torchvision scikit-learn
```
---

## 3. The Code

### 3.1 Importing Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

- **`torch`**: Core PyTorch library for tensor computations.
- **`nn`**: Module for building neural networks.
- **`optim`**: Optimizers to adjust model weights for learning.
- **`fetch_california_housing`**: Preloaded dataset of California housing data.
- **`train_test_split`**: Splits data into training and testing sets.
- **`StandardScaler`**: Scales data for uniformity, improving model performance.

---

### 3.2 Loading and Preprocessing Data

```python
# Load the dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **California Housing Dataset**: Features include population, average income, and median house age. The target (output) is the median house price.
- **Standardization**: Ensures all features have a similar scale, preventing larger values from dominating the learning process.
- **Train-Test Split**: 80% of data is used for training and 20% for testing.

---

### 3.3 Converting Data to PyTorch Tensors

```python
# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)
```

- **PyTorch Tensors**: Similar to NumPy arrays but optimized for deep learning.
- **`view(-1, 1)`**: Reshapes target data into a 2D column vector, which matches the output format expected by the neural network.

---

### 3.4 Defining the Neural Network Model

```python
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

### Key Components Explained:
- **Input Layer**: Accepts input features (number of features = 8).
- **Hidden Layer**: Contains 64 neurons, applies the ReLU activation function to add non-linearity.
- **Output Layer**: Produces a single continuous value (predicted price).
- **Activation Function (ReLU)**: Introduces non-linearity by converting negative values to zero, allowing the network to model complex relationships.

---

### 3.5 Initializing Model, Loss, and Optimizer

```python
model = ANN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

- **Loss Function (MSE)**: Measures how far predictions are from actual values. Smaller values indicate better performance.
- **Optimizer (Adam)**: Adjusts model weights to minimize loss by following gradients (gradient descent).
- **Learning Rate**: Controls step size when updating weights.

---

### 3.6 Training the Model

```python
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

- **Epochs**: Number of complete passes through the training data.
- **Backward Propagation**: Computes gradients to update weights.
- **Progress Display**: Shows loss every 10 epochs.

---

### 3.7 Evaluating the Model

```python
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

    y_test_mean = y_test.mean()
    ss_total = ((y_test - y_test_mean) ** 2).sum()
    ss_residual = ((y_test - test_outputs) ** 2).sum()
    r2_score = 1 - (ss_residual / ss_total)
    print(f'R-squared Score: {r2_score.item():.4f}')
```

- **Test Loss**: Measures performance on unseen data.
- **R-squared Score**: Indicates how well the model explains variability in data (values close to 1 are better).

---

### 3.8 Saving the Model

```python
torch.save(model.state_dict(), "regression_model.pth")
print("Model saved successfully!")
```

- **Model Save**: Saves learned parameters for reuse.

---

## 4. Conclusion
You now have an understanding of how you can use PyTorch to build ANNs. Keep on nerding!
