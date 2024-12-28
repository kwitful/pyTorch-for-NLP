# Import required libraries
import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  

# Load the dataset
data = load_iris()  
X = data.data  # Feature variables (sepal/petal length and width)
y = data.target  # Target labels (flower species: 0, 1, 2)

# Standardize the features (scale data to have mean 0 and standard deviation 1)
scaler = StandardScaler()  
X = scaler.fit_transform(X)  # Fit and transform the data to standardize it

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 20% of data is for testing; random_state ensures reproducibility
)

# Convert the NumPy arrays into PyTorch tensors for processing
X_train = torch.FloatTensor(X_train)  
X_test = torch.FloatTensor(X_test)  
y_train = torch.LongTensor(y_train)  
y_test = torch.LongTensor(y_test) 

# Define the neural network class
class ANN(nn.Module):  # Inherits from PyTorch's base class nn.Module
    def __init__(self, input_size, hidden_size, output_size):  # Constructor initializes the network structure
        super(ANN, self).__init__()  # Calls the constructor of the parent class

        # Define the first fully connected layer (input layer to hidden layer)
        self.fc1 = nn.Linear(input_size, hidden_size)  # Linear transformation layer
        self.relu = nn.ReLU()  # ReLU activation function for non-linearity

        # Define the second fully connected layer (hidden layer to output layer)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Linear transformation layer

    # Define the forward pass (how data flows through the network)
    def forward(self, x):
        x = self.fc1(x)  # Pass data through the first layer
        x = self.relu(x)  # Apply ReLU activation function
        x = self.fc2(x)  # Pass data through the second layer (output layer)
        return x  # Return the output of the network

# Initialize the model parameters
input_size = 4  # Number of input features (4 features in the Iris dataset)
hidden_size = 8  # Number of neurons in the hidden layer
output_size = 3  # Number of output classes (3 flower species)

# Create an instance of the ANN model
model = ANN(input_size, hidden_size, output_size)

# Define the loss function (criterion) for classification tasks
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss is suitable for multi-class classification

# Define the optimizer for updating weights
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with a learning rate of 0.01

# Set the number of epochs for training
num_epochs = 100  # Number of complete passes through the dataset

# Start the training loop
for epoch in range(num_epochs):  # Repeat training for the specified number of epochs
    # Forward pass: compute predicted outputs
    outputs = model(X_train)  # Pass training data through the model

    # Calculate the loss between predicted and actual labels
    loss = criterion(outputs, y_train)  # Compute loss using cross-entropy function

    # Backward pass: reset gradients
    optimizer.zero_grad()  # Clear gradients to prevent accumulation from previous steps

    # Compute gradients for each parameter
    loss.backward()  # Perform backpropagation to calculate gradients

    # Update model weights based on gradients
    optimizer.step()  # Update weights using optimizer

    # Print loss every 10 epochs for monitoring progress
    if (epoch + 1) % 10 == 0:  # Display results every 10 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  # Show current loss

# Disable gradient computation for testing (saves memory and speeds up computation)
with torch.no_grad():  
    model.eval()  # Set the model to evaluation mode (disables dropout and batch normalization)

    # Forward pass: make predictions on the test set
    test_outputs = model(X_test)  # Compute outputs for test data

    # Get the predicted class for each input (highest probability)
    _, predicted = torch.max(test_outputs, 1)  # Returns the index of the max value along dimension 1

    # Calculate accuracy by comparing predictions with true labels
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)  # Correct predictions / total predictions
    print(f'Accuracy on test data: {accuracy * 100:.2f}%')  # Print accuracy percentage
