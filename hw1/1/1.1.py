import torch

tensor1 = torch.rand(3, 4)
print("3x4 random:\n", tensor1)

tensor2 = torch.zeros(2, 3, 4)
print("\n2x3x4 zeros:", tensor2.shape)

tensor3 = torch.ones(5, 5)
print("\n5x5 ones:", tensor3.shape)

tensor4 = torch.arange(16).reshape(4, 4)
print("\n4x4 tensor 0-15:\n", tensor4)
