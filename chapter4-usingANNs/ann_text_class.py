# Import required libraries
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Download and Load the Dataset
nltk.download('movie_reviews')

# Load movie reviews data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Separate data and labels
texts, labels = zip(*documents)
labels = [1 if label == 'pos' else 0 for label in labels]

# Step 2: Feature Extraction
# Convert list of words to a string for vectorization
texts = [" ".join(words) for words in texts]

# Initialize CountVectorizer (binary=True for bag-of-words)
vectorizer = CountVectorizer(max_features=5000, binary=True)

# Fit and transform the text data
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Define the Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)   # First hidden layer
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer
        self.fc3 = nn.Linear(64, 1)           # Output layer
        self.relu = nn.ReLU()                 # Activation function
        self.sigmoid = nn.Sigmoid()           # Sigmoid for binary classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Step 4: Prepare Data for PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define model, loss function, and optimizer
input_dim = X_train.shape[1]
model = FeedforwardNN(input_dim)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the Model
epochs = 10
batch_size = 32

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        # Get mini-batches
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss after each epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Step 6: Evaluate the Model
model.eval()
with torch.no_grad():
    # Make predictions
    predictions = model(X_test_tensor)
    predictions = predictions.round()  # Round predictions to 0 or 1
    accuracy = (predictions.eq(y_test_tensor).sum().item()) / len(y_test_tensor)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save the Model
# Save the model
torch.save(model.state_dict(), "text_classifier.pth")

