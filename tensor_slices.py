import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Access row 0
print(x[0])

# Access element at row 1, column 2
print(x[1, 2])

# Select all rows but only first column
print(x[:, 0])
