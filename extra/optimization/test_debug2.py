
from tinygrad import Tensor

a = Tensor.randn(10, requires_grad=True)
loss = (a*(a ^ 0)).sum().backward()
print(a.grad.numpy())

import torch
a = torch.autograd.Variable(torch.Tensor(a.numpy()), requires_grad=True)
loss = (a*(a ^ 0)).sum().backward()
print(a.grad)
