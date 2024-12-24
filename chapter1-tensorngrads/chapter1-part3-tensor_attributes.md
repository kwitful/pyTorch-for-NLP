# PyTorch Tensor Attributes and Methods Tutorial

This tutorial explains the key attributes and methods for working with PyTorch tensors, based on the examples provided in `tensor_attr_methods.py`.

---

## 1. Tensor Shape and Size
Shape describes the number of rows and columns in a tensor, while size provides the same information and can also give the size of a specific dimension.

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])

# Shape of the tensor
print(x.shape)  # Output: torch.Size([2, 2])

# Size of each dimension
print(x.size())  # Output: torch.Size([2, 2])
print(x.size(0)) # Output: 2 (rows)
print(x.size(1)) # Output: 2 (columns)
```
---

## 2. Number of Dimensions
Check the number of dimensions in a tensor, which is useful when working with multi-dimensional data like images or sequences.

```python
# Number of dimensions
print(x.ndimension())  # Output: 2
print(x.dim())         # Output: 2
```
---

## 3. Data Types
Check and set the type of data stored in a tensor. Data type determines whether the tensor holds integers, floats, or other formats.

```python
# Data type of tensor elements
y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
z = torch.tensor([1, 2, 3], dtype=torch.int64)

print(y.dtype)  # Output: torch.float32
print(z.dtype)  # Output: torch.int64
```
---

## 4. Device Assignment
Check where a tensor is stored—either on a CPU or GPU—and move it between devices for faster processing.

```python
# Assign tensor to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_tens = torch.tensor([1, 2, 3], device=device)
print(d_tens.device)  # Output: cuda:0 (if GPU is available)
```
---

## 5. Gradient Tracking
Enable automatic differentiation, which tracks gradients during calculations. Gradients are essential for training machine learning models.

```python
# Enable gradient tracking
tra_tens = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(tra_tens.requires_grad)  # Output: True

# Compute gradients
x_tens = torch.tensor([2.0], requires_grad=True)
y_func = x_tens ** 2  # y = x^2
y_func.backward()  # Compute gradients
print(x_tens.grad)  # Output: tensor([4.])
```
---

## 6. Extracting Values
Convert a tensor with a single value into a regular Python number for easier use in calculations.

```python
# Extract scalar values
i_tens = torch.tensor([3.5])
print(i_tens.item())  # Output: 3.5
```
---


This concludes the tutorial on PyTorch tensor attributes and methods. 

