# Building an Artificial Neural Network for Classification with PyTorch

## 1. Introduction
Before we begin, I would like to point out that I'm not all too happy about  using the iris dataset. However, as with the previous parts, using such datasets can help beginners understand how artificial neural networks are built and evaluated with PyTorch. In this tutorial, we will train, and evaluate an **Artificial Neural Network (ANN)** for a **classification task** using PyTorch. Specifically, we will classify the Iris flower dataset into three species.

### What is Classification?
Classification is a machine learning task where the goal is to categorize data into predefined labels or groups. For example:
- Spam email detection (Spam or Not Spam)
- Image recognition (Cats vs Dogs)
- Sentiment analysis (Positive, Negative, Neutral)

In this tutorial, we will classify flowers into one of three species (Setosa, Versicolor, Virginica) based on features like petal length, petal width, sepal length, and sepal width.

### Why Use an ANN for Classification?
Artificial Neural Networks (ANNs) are inspired by the way the human brain processes information. They consist of layers of interconnected nodes (neurons) that transform input data into meaningful predictions. ANNs are widely used because:
- They handle non-linear relationships well.
- They can capture complex patterns and dependencies in data.
- They are versatile and can adapt to various machine-learning tasks like regression and classification.

---

## 2. Prerequisites
### Libraries Required:
- **PyTorch**: Deep learning framework for tensor computation and neural networks.
- **Scikit-learn**: Tools for preprocessing data and dataset management.

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
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
```
- **`torch`**: The core PyTorch library for tensor computations and neural networks.
- **`nn`**: Module for defining and building neural networks.
- **`optim`**: Contains optimization algorithms to update model weights.
- **`load_iris`**: Loads the Iris dataset, a classic dataset for classification.
- **`train_test_split`**: Splits data into training and testing sets for evaluation.
- **`StandardScaler`**: Scales data to have a mean of 0 and a standard deviation of 1, improving model performance.

---

### 3.2 Loading and Preprocessing Data
```python
# Load dataset
data = load_iris()  
X = data.data  # Features
print("Feature Names:", data.feature_names)
y = data.target  # Labels
print("Classes:", data.target_names)

# Standardize features
scaler = StandardScaler()  
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)  
X_test = torch.FloatTensor(X_test)  
y_train = torch.LongTensor(y_train)  
y_test = torch.LongTensor(y_test) 
```
### 
1. **Dataset**: The Iris dataset has 150 samples of flowers, each with 4 measurements (sepal and petal dimensions) and a label indicating the species.
2. **Standardization**: Features are scaled to ensure equal importance and to speed up training.
3. **Splitting Data**: 80% of the data is used for training, and 20% is reserved for testing.
4. **Tensors**: PyTorch models work with tensors (similar to NumPy arrays), so we convert data into tensors.

---

### 3.3 Defining the Neural Network
```python
class ANN(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):  
        super(ANN, self).__init__()  
        self.fc1 = nn.Linear(input_size, hidden_size)  # Linear layer 1
        self.relu = nn.ReLU()  # Activation function (ReLU)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Linear layer 2

    def forward(self, x):  
        x = self.fc1(x)  # Pass input through layer 1
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Pass through output layer
        return x
```
### 
1. **Input Layer**: Receives the input features (4 features in the Iris dataset).
2. **Hidden Layer**: Adds complexity to the model, with 8 neurons and a ReLU activation function to handle non-linearity.
3. **Output Layer**: Outputs probabilities for the 3 classes (species).
4. **Forward Pass**: Defines how data flows through the network during prediction.

---

### 3.4 Model Initialization
```python
input_size = 4  # Number of features
hidden_size = 8  # Number of neurons in the hidden layer
output_size = 3  # Number of classes

model = ANN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Optimizer
```
### 
1. **Input and Output Sizes**: Match dataset dimensions.
2. **Loss Function**: CrossEntropyLoss measures the error between predicted and true classes.
3. **Optimizer**: Adam adjusts weights to minimize loss during training.

---

### 3.5 Training the Model
```python
num_epochs = 100  
for epoch in range(num_epochs):  
    outputs = model(X_train)  
    loss = criterion(outputs, y_train)  

    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  

    if (epoch + 1) % 10 == 0:  
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```
### 
1. **Epochs**: The number of times the model sees the entire dataset.
2. **Forward Pass**: Computes predictions.
3. **Loss Calculation**: Measures prediction error.
4. **Backward Pass**: Calculates gradients to improve weights.
5. **Weight Update**: Adjusts weights using the optimizer.

---

### 3.6 Model Evaluation
```python
with torch.no_grad():  
    model.eval()  
    test_outputs = model(X_test)  
    _, predicted = torch.max(test_outputs, 1)  
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)  
    print(f'Accuracy on test data: {accuracy * 100:.2f}%')
```
### 
1. **Evaluation Mode**: Disables gradient calculation for efficiency.
2. **Predictions**: Chooses the class with the highest probability.
3. **Accuracy**: Measures the percentage of correctly classified samples.

---

## 4. Conclusion
Typical accuracy ranges from **95-98%**, showing the network effectively classifies the Iris dataset.





