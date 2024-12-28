# Building Your First Artificial Neural Network for Regression

## 1. Introduction

This tutorial demonstrates how to build and train an **Artificial Neural Network (ANN)** for regression tasks using PyTorch. The example focuses on predicting California housing prices based on numerical input features. Regression models are essential in scenarios where the target output is continuous, such as price prediction, temperature forecasting, and stock price analysis.

---

## 2. Prerequisites

Before starting, ensure you have the following installed:

- Python
- PyTorch
- Scikit-learn

Install them using:

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

- **torch**: PyTorch library for tensor computation and neural networks.
- **nn**: Submodule for building neural network layers.
- **optim**: Provides optimization algorithms like Adam and SGD.
- **fetch\_california\_housing**: Prebuilt dataset containing California housing data.
- **train\_test\_split**: Splits data into training and testing sets.
- **StandardScaler**: Scales features to ensure uniformity, improving training stability.

### 3.2 Loading and Preprocessing Data

```python
# Load and preprocess data
data = fetch_california_housing()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **California Housing Dataset**: Contains features like average income, house age, and population.
- **Splitting Data**: 80% of the data is used for training, and 20% for testing.
- **Standardization**: Ensures features have zero mean and unit variance, preventing features with large scales from dominating smaller ones.

### 3.3 Converting Data to PyTorch Tensors

```python
# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)
```

- **Tensor Conversion**: PyTorch requires data to be in tensor format for computation.
- **`view(-1, 1)`**: Reshapes target data into a 2D column vector to match the output shape of the model.

### 3.4 Defining the Model

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

- **Input Layer**: Matches the number of features in the dataset (8 features).
- **Hidden Layer**: Contains 64 neurons and uses ReLU activation for non-linearity.
- **Output Layer**: Produces a single value representing the predicted house price.

### 3.5 Initializing Model, Loss, and Optimizer

```python
model = ANN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

- **Loss Function**: Mean Squared Error (MSE) calculates the average squared difference between predicted and actual values.
- **Adam Optimizer**: Combines momentum and adaptive learning rates to speed up convergence.

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

- **Epochs**: Number of times the entire training dataset is passed through the model.
- **Forward Pass**: Computes predictions.
- **Loss Calculation**: Measures prediction errors.
- **Backward Pass**: Calculates gradients to minimize loss.
- **Weight Update**: Adjusts weights based on gradients.

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

- **Evaluation Mode**: Disables gradient computation.
- **Test Loss**: Evaluates model performance on unseen data.
- **R-squared Score**: Measures how well the predictions match the true values.

### 3.8 Saving the Model

```python
torch.save(model.state_dict(), "regression_model.pth")
print("Model saved successfully!")
```

- **Saving Model Weights**: Allows reuse of the trained model without retraining.

---

This concludes the tutorial. You now have a deeper understanding of building ANNs for regression tasks using PyTorch. Happy coding!
