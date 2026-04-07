import torch

# ??????
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=False)
y_true = torch.tensor([2.0, 4.0, 6.0, 8.0], requires_grad=False)

# ????????? ?????? (y_pred = w * x + b)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# ????????????
y_pred = w * x + b

# MSE = (1/n) * ?(y_pred - y_true)^2
n = len(x)
mse = ((y_pred - y_true) ** 2).sum() / n

print("MSE =", mse.item())
print("w =", w.item(), "grad =", w.grad)
print("b =", b.item(), "grad =", b.grad)

# ?????????? ??????????
mse.backward()

print("\n????? backward():")
print("?MSE/?w =", w.grad.item())
print("?MSE/?b =", b.grad.item())