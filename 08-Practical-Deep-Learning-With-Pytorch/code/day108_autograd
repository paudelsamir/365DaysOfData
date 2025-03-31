import torch

def f(x):
    return x**3 + 2*x**2 + 3*x + 4

# derivative of f(x) manually computed
def df_manual(x):
    return 3*x**2 + 4*x + 3

# test at x = 2
x_val = 2.0
manual_grad = df_manual(x_val)
print(f"manual gradient at x={x_val}: {manual_grad}")

# pytorch autograd
x = torch.tensor(x_val, requires_grad=True)  # define tensor with gradient tracking
y = f(x)  # compute function value
y.backward()  # compute gradient

print(f"autograd gradient at x={x_val}: {x.grad.item()}")
