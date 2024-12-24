# PyTorch Gradients and Linear Regression Tutorial

This tutorial covers gradient computation and linear regression using PyTorch, based on the examples provided in `grads_intro.py` and `grads_linregexample.py`.

---

## 1. Gradient Computation Basics
Gradients are used in machine learning to optimize models by updating parameters to minimize loss functions.

```python
import torch

# Define tensors with gradient tracking enabled
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Perform computations
c = a ** 2 + b ** 2

# Compute gradients
c.backward()

print(a.grad)  # Output: 4 (dc/da)
print(b.grad)  # Output: 6 (dc/db)
```
- **`requires_grad=True`**: Tracks operations on the tensor for gradient computation.
- **`backward()`**: Computes gradients of the output with respect to input tensors.
- **`grad`**: Stores the computed gradients.

The gradients are partial derivatives of the function `c = a² + b²` with respect to `a` and `b`. These values (4 and 6) indicate how much the output changes when the inputs change.

---

## 2. Linear Regression with Gradients
Linear regression models a relationship between inputs (`x`) and outputs (`y`) using a line:

```
y = wx + b
```
Here, `w` is the weight, and `b` is the bias. Both are trainable parameters optimized using gradients.

```python
import torch

# Input (x) and target output (y)
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# Initialize weight and bias with gradient tracking
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01

# Define the model
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
```
### Step-by-Step Explanation of `.grad` Usage
1. **Why is `.grad` used?**
   The `.grad` attribute is used to store the computed gradients of the loss function with respect to the parameters (`w` and `b`). These gradients show the direction and magnitude of change needed to reduce the loss.

2. **How are gradients computed?**
   The line `loss.backward()` calculates gradients for `w` and `b` using the chain rule of differentiation. It computes how much the loss changes if we slightly increase or decrease `w` or `b`.

3. **How are gradients applied?**
   - `w.grad` and `b.grad` contain the gradients for `w` and `b`.
   - These gradients are used in the update step:
     ```python
     w -= learning_rate * w.grad
     b -= learning_rate * b.grad
     ```
     This step reduces the values of `w` and `b` in the direction of the negative gradient, minimizing the loss.

4. **Why do we zero the gradients?**
   After updating the weights, we clear the gradients with:
   ```python
   w.grad.zero_()
   b.grad.zero_()
   ```
   Gradients accumulate by default in PyTorch, so clearing them ensures that updates are calculated independently for each epoch, avoiding incorrect calculations.

---

This concludes the tutorial on PyTorch gradients and linear regression. This also concludes the firts chapter. In the next chapter, we will train a basic Artificial Neural Network with PyTorch. 

