import torch

torch.manual_seed(42)

# 64 x 1024 x 1024
matrix1 = torch.rand(64, 1024, 1024)
print("64x1024x1024 shape:", matrix1.shape)

# 128 x 512 x 512
matrix2 = torch.rand(128, 512, 512)
print("\n128x512x512 shape:", matrix2.shape)

# 256 x 256 x 256
matrix3 = torch.rand(256, 256, 256)
print("\n256x256x256 shape:", matrix3.shape)