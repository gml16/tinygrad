
from tinygrad import Tensor
<<<<<<< Updated upstream

a = Tensor.randn(10, requires_grad=True)
loss = (a*(a ^ 0)).sum().backward()
print(a.grad.numpy())

import torch
a = torch.autograd.Variable(torch.Tensor(a.numpy()), requires_grad=True)
loss = (a*(a ^ 0)).sum().backward()
print(a.grad)
=======
import torch

# t1 = torch.ones(4, requires_grad=True)
# t2 = torch.ones(4, requires_grad=True)
# loss = (t1 == t2).sum().backward()
# print(t1.grad)

t = Tensor.ones(4, requires_grad=True)
# loss = (t*(t < 10)).sum().backward()
# print(t.grad.numpy())

a = torch.autograd.Variable(torch.Tensor(t.numpy()), requires_grad=True)
# loss = (a*(a < 0)).sum().backward()
# print(a.grad.numpy())

b = torch.autograd.Variable(torch.Tensor(t.numpy()), requires_grad=True)
# loss = (a*(a < 0)).sum().backward()
# print(a.grad.numpy())

# tt1 = Tensor.ones(4, requires_grad=True)
# tt2 = Tensor.ones(4, requires_grad=True)
# tt1 = torch.ones(4, requires_grad=True)
# tt2 = torch.ones(4, requires_grad=True)
# (tt1 == tt2).sum().backward()
>>>>>>> Stashed changes
