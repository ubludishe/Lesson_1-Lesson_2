import torch

# ??????? ??????? ? requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

# f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2*x*y*z

print("f(x,y,z) =", f.item())
print("x.grad =", x.grad)
print("y.grad =", y.grad)
print("z.grad =", z.grad)

f.backward()
print("\n????? backward():")
print("df/dx =", x.grad.item())
print("df/dy =", y.grad.item())
print("df/dz =", z.grad.item())