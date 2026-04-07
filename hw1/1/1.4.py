import torch

torch.manual_seed(42)
tensor_24 = torch.arange(24)

print("Original tensor (24):", tensor_24.shape)

shape_2x12 = tensor_24.reshape(2, 12)
print("\n2x12:")
print(shape_2x12)

shape_3x8 = tensor_24.reshape(3, 8)
print("\n3x8:")
print(shape_3x8)

shape_4x6 = tensor_24.reshape(4, 6)
print("\n4x6:")
print(shape_4x6)

shape_2x3x4 = tensor_24.reshape(2, 3, 4)
print("\n2x3x4:")
print(shape_2x3x4)

shape_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)
print("\n2x2x2x3:")
print(shape_2x2x2x3)