# PyTorch Tensor Definitions Tutorial

This tutorial covers the basics of defining tensors in PyTorch, as outlined in the repo Python files `tensor_definitions.py` and `tensor_definitions_continued.py`. We will explain each code snippet step-by-step.

---

## 1. Defining Tensors

### Scalar, Vector, Matrix, and Higher Dimensions
```python
import torch

# Defining a tensor
x = torch.tensor([[1, 2], [3, 4], [4,7]])
print(x)
```
This defines a **2D tensor** with shape `(3, 2)`. It is displayed as:
```
[[1, 2]
 [3, 4]
 [4, 7]]
```

### Shape and Data Type
```python
print(x.shape)
print(x.dtype)
```
- **`x.shape`**: Returns the dimensions of the tensor, e.g., `torch.Size([3, 2])`.
- **`x.dtype`**: Outputs the data type, e.g., `torch.int64` by default.

### Examples of Tensor Types
```python
# Scalar
a = torch.tensor(5)

# Vector
b = torch.tensor([1, 2, 3])

# Matrix
c = torch.tensor([[1, 2], [3, 4]])

# Higher Dimensions
d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```
- **Scalar**: Single value.
- **Vector**: 1D array.
- **Matrix**: 2D array.
- **Higher Dimensions**: Nested arrays with 3 or more dimensions.

```python
print(a.shape)  # torch.Size([]) - Scalar has no dimensions
print(b.shape)  # torch.Size([3]) - Vector has 1 dimension of size 3
print(c.shape)  # torch.Size([2, 2]) - Matrix with 2x2 shape
print(d.shape)  # torch.Size([2, 2, 2]) - 3D tensor
```

---

## 2. Creating Tensors from Different Sources

### From Python Lists
```python
import torch

# From Python lists
tens_list = torch.tensor([64,23,61,82,65])
```
Defines a tensor directly from a Python list.

### From Tuples
```python
# From Python tuples
tens_tuple = torch.tensor((42,36,75,84,73))
```
Defines a tensor from a Python tuple.

### From NumPy Arrays
```python
import numpy as np

# From Numpy arrays
tens_array = torch.tensor(np.array([53,78,92,77,72]))
```
Converts a NumPy array into a PyTorch tensor, allowing seamless integration with existing NumPy code.

---

## 3. Why Are Tensors Useful?
Tensors are the backbone of PyTorch because they:
- Represent data structures similar to NumPy arrays but optimized for **GPU computations**.
- Support **automatic differentiation** for building neural networks.
- Provide flexible APIs for creating and manipulating high-dimensional data, making them ideal for **deep learning tasks**.



---


