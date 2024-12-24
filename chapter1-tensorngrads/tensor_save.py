import torch 
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Save tensor
torch.save(x, 'tensor.pt')

# Load tensor
loaded_x = torch.load('tensor.pt')
print(loaded_x)
