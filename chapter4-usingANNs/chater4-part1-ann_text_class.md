# Feedforward Neural Network for Text Classification with PyTorch

## 1. Introduction
By this point, we can safely assume that we have an understanding about the building blocks of neural networks. Now, let's start actually using them for NLP-related tasks. We shall start with text classification.

### What is Text Classification?
Text classification is the process of assigning predefined labels to textual data. For example:
- **Spam Detection**: Classify emails as spam or not spam.
- **Sentiment Analysis**: Determine if a review is positive, negative, or neutral.
- **Topic Categorization**: Group articles based on topics like sports, politics, etc.


---

## 2. Prerequisites
### Required Libraries:
```bash
pip install torch numpy scikit-learn nltk
```

### What Do These Libraries Do?
- **PyTorch**: Provides tools for building and training neural networks.
- **NumPy**: Handles numerical computations.
- **Scikit-learn**: Offers preprocessing tools and datasets.
- **nltk**: Provides text-processing utilities and datasets.

---

## 3. Loading and Preprocessing Data

### Step 1: Import Libraries and Dataset
```python
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Download movie_reviews dataset
nltk.download('movie_reviews')

# Load and label the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

texts, labels = zip(*documents)
labels = [1 if label == 'pos' else 0 for label in labels]  # Binary encoding
```
### Code Breakdown:
1. **Import Libraries**: Load required tools for data processing and neural network training.
2. **Download Dataset**: Fetch the **IMDb movie reviews dataset**, which consists of positive and negative reviews.
3. **Label Encoding**: Assign numerical values (1 = Positive, 0 = Negative) to labels for compatibility with PyTorch.

---

### Step 2: Feature Extraction (Bag-of-Words)
```python
# Convert text data into strings
texts = [" ".join(words) for words in texts]

# Apply CountVectorizer
vectorizer = CountVectorizer(max_features=5000, binary=True)
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Theory:
**Bag-of-Words (BoW)** is a method that represents text as a fixed-size vector based on word frequency:
- Each word corresponds to a column in the vector.
- Values indicate whether the word is present (binary encoding) or how often it appears.

### Code Explanation:
1. **Join Words**: Convert lists of words into sentences, as CountVectorizer expects strings.
2. **Vectorizer**: Extracts features based on word presence or frequency.
3. **Train-Test Split**: Splits the dataset into training (80%) and test (20%) sets for training and evaluation.

---

## 4. Building the Feedforward Neural Network

### Step 3: Define the Model
```python
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
```
### Neural Network Architecture:
1. **Input Layer**: Accepts the feature vector created by BoW.
2. **Hidden Layers**: Two layers with 128 and 64 neurons, activated by **ReLU** for non-linearity.
3. **Output Layer**: Uses **Sigmoid** to produce probabilities between 0 and 1 for binary classification.

### Why Use ReLU and Sigmoid?
- **ReLU**: Introduces non-linearity, enabling the network to learn complex patterns.
- **Sigmoid**: Squashes outputs to a range of [0,1], making it suitable for binary classification.

---

## 5. Model Training and Evaluation

### Step 4: Prepare Data for PyTorch
```python
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Model initialization
input_dim = X_train.shape[1]
model = FeedforwardNN(input_dim)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
### Code Explanation:
1. **Tensor Conversion**: Converts NumPy arrays into PyTorch tensors for compatibility.
2. **Loss Function**: **BCELoss** calculates errors for binary outputs.
3. **Optimizer**: **Adam** updates weights to minimize loss.

---

### Step 5: Train the Model
```python
epochs = 10
batch_size = 32

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
```
### How Training Works:
1. **Mini-Batch Processing**: Processes data in small batches for efficiency.
2. **Forward Pass**: Computes predictions.
3. **Loss Computation**: Measures error between predictions and actual labels.
4. **Backward Pass**: Updates weights using gradients.

---

### Step 6: Evaluate the Model
```python
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions = predictions.round()
    accuracy = (predictions.eq(y_test_tensor).sum().item()) / len(y_test_tensor)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
```
### Evaluation Process:
- **No Gradient Tracking**: Disables gradients to speed up computation.
- **Rounding**: Converts probabilities into class labels (0 or 1).
- **Accuracy**: Measures correct predictions.

---

## 6. Saving and Loading the Model
```python
# Save model
torch.save(model.state_dict(), "text_classifier.pth")

# Load model
model.load_state_dict(torch.load("text_classifier.pth"))
model.eval()
```

---

## 7. What's next?

Up next, we will learn about the different types of neural networks that can be utilized for NLP tasks. We will also take a look at some common problems that arise when we try to scale up the scope of neural networks and the solutions found to these problems. 

---

