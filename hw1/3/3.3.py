import torch
import time
import pandas as pd


def measure_time(func, device='cpu', num_runs=20, warmup_runs=5):
    """???????? ????? ?????????? ???????"""
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # ???????
    for _ in range(warmup_runs):
        func()
        if device == 'cuda': torch.cuda.synchronize()

    # ??????
    times = []
    for _ in range(num_runs):
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            func()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            start = time.time()
            func()
            times.append((time.time() - start) * 1000)

    return sum(times) / len(times)


# ??????? ??????? ???????
size = 2048
a = torch.randn(size, size)
b = torch.randn(size, size)

# ?????????? ????????
operations = {
    'Matrix multiplication': lambda: torch.matmul(a, b),
    'Elementwise addition': lambda: a + b,
    'Elementwise multiplication': lambda: a * b,
    'Transpose': lambda: a.T,
    'Sum': lambda: a.sum()
}

# ??????? ???????????
results = []

for name, op in operations.items():
    # CPU
    cpu_time = measure_time(op, 'cpu')

    # GPU
    if torch.cuda.is_available():
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        op_gpu = operations[name].__code__.co_varnames[0] == 'a' and (lambda: torch.matmul(a_gpu, b_gpu)) or op
        gpu_time = measure_time(lambda: op_gpu(), 'cuda')
        speedup = cpu_time / gpu_time
    else:
        gpu_time = 0
        speedup = 0

    results.append([name, f"{cpu_time:.1f}", f"{gpu_time:.1f}", f"{speedup:.1f}x"])

# ????? ???????
print("????????          | CPU (??) | GPU (??) | ?????????")
print("-" * 45)
for row in results:
    print(f"{row[0]:18} | {row[1]:7} | {row[2]:7} | {row[3]:8}")