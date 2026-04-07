import torch

torch.manual_seed(42)
tensor = torch.rand(5, 5, 5)

print("Original 5x5x5 tensor shape:", tensor.shape)

# 1. First slice
first_slice = tensor[0]
print("\n1. First slice (first_row):")
print(first_slice)

# 2. Last column
last_column = tensor[:, :, -1]
print("\n2. Last column:")
print(last_column)

# 3. Central 2x2
central_2x2 = tensor[2:4, 2:4, 2:4]
print("\n3. Central 2x2:")
print(central_2x2)

# 4. Even indexed elements
even_elements = tensor.flatten()[::2]
print("\n4. Even indexed elements (first 10):")
print(even_elements[:10])