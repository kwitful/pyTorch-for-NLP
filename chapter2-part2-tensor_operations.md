# PyTorch Tensor Operations Tutorial

This tutorial explains the basic operations, reshaping, saving, and slicing of tensors in PyTorch, using the examples provided in the repo Python files.

---

## 1. Basic Tensor Operations

### Arithmetic Operations

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Addition
z = x + y
print("Addition:", z)

# Multiplication
mult = x * y
print("Multiplication:", mult)

# Matrix Multiplication
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
result = torch.mm(a, b)  # or a @ b
print("Matrix Multiplication:", result)
```
- **Addition (`+`)**: Element-wise addition of tensors.
- **Multiplication (`*`)**: Element-wise multiplication.
- **Matrix Multiplication (`torch.mm` or `@`)**: Performs dot product between matrices.

---

## 2. Reshaping Tensors
**File:** `tensor_reshapes.py`
```python
import torch

x = torch.arange(1, 11)
print(x)

# Reshape into 5x2
x = x.view(5, 2)
print(x)

# Flatten back to 1D
x = x.view(-1)
print(x)
```
- **`view()`**: Changes the dimensions of the tensor without altering the data.
  - `5x2`: Reshapes tensor into 5 rows and 2 columns.
  - `-1`: Automatically infers size to flatten the tensor back to 1D.

---



---

## 3. Slicing Tensors
**File:** `tensor_slices.py`
```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Access row 0
print(x[0])

# Access element at row 1, column 2
print(x[1, 2])

# Select all rows but only first column
print(x[:, 0])
```
- **Row/Column Access**:
  - `x[0]`: Selects the first row.
  - `x[1, 2]`: Selects the value at row 1, column 2.
- **Column Selection**: `x[:, 0]` selects all rows but only the first column.

---
## 4. Saving and Loading Tensors
**File:** `tensor_save.py`
```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Save tensor
torch.save(x, 'tensor.pt')

# Load tensor
loaded_x = torch.load('tensor.pt')
print(loaded_x)
```
- **Saving (`torch.save`)**: Stores tensor in a file with extension `.pt`.
- **Loading (`torch.load`)**: Restores tensor from the file.
- Useful for saving model parameters and results.
---

---

This concludes the tutorial on PyTorch tensor operations, reshaping, saving, and slicing. Next, we will dive into tensor attributes and methods for advanced manipulations.

