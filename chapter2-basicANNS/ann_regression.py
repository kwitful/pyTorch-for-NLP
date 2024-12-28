# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
# California housing data contains features like average income, house age, and population
# Target is the median house price
print("Loading dataset...")
data = fetch_california_housing()  # Fetch the dataset
X = data.data  # Feature variables
y = data.target  # Target labels (house prices)

# Standardize features (scale data to have mean 0 and standard deviation 1)
scaler = StandardScaler()  # Initialize a scaler object
X = scaler.fit_transform(X)  # Fit and transform data

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Use 20% data for testing, ensure reproducibility
)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)  # Convert to 32-bit float tensor
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).view(-1, 1)  # Reshape to 2D tensor

# Test labels should match target tensor size
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Define the neural network model
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.relu = nn.ReLU()  # ReLU activation function for non-linearity
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to output layer

    def forward(self, x):
        x = self.fc1(x)  # First layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)  # Output layer
        return x

# Model parameters
input_size = X_train.shape[1]  # Number of features (8 in the dataset)
hidden_size = 64  # Hidden layer with 64 neurons
output_size = 1  # Single output for regression task

# Initialize model, loss function, and optimizer
print("Initializing model...")
model = ANN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# Training loop
num_epochs = 100
print("Starting training...")
for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    # Forward pass: compute predictions
    outputs = model(X_train)

    # Compute loss
    loss = criterion(outputs, y_train)

    # Backward pass: compute gradients and update weights
    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Calculate gradients
    optimizer.step()  # Update weights

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate model performance on test data
print("Evaluating model...")
with torch.no_grad():
    model.eval()  # Set model to evaluation mode

    # Compute predictions for test data
    test_outputs = model(X_test)

    # Calculate test loss
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

    # Calculate R-squared (accuracy for regression)
    y_test_mean = y_test.mean()
    ss_total = ((y_test - y_test_mean) ** 2).sum()
    ss_residual = ((y_test - test_outputs) ** 2).sum()
    r2_score = 1 - (ss_residual / ss_total)
    print(f'R-squared Score: {r2_score.item():.4f}')

# Save the trained model
print("Saving model...")
torch.save(model.state_dict(), "regression_model.pth")
print("Model saved successfully!")
