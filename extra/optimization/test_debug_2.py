from tinygrad.nn.optim import Adam
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.tensor import Tensor
  
def relu_2(t):
   return t * (t > 0)

class Net:
  def __init__(self):
    self.l1 = Linear(10,10)
  def __call__(self, x):
    return self.l1(x)
  
net = Net()
optim = Adam(get_parameters(net), 1e-3)
rd = Tensor.randn(10)

loss = (relu_2(net(rd) - rd)).abs().mean()
loss.backward()
print(loss.numpy())
print(net.l1.weight.grad.numpy())

loss = (net(rd) - rd).relu().abs().mean()
loss.backward()
print(loss.numpy())
print(net.l1.weight.grad.numpy())


# import torch

# rd = Tensor.randn(10)
# rand2 = Tensor.randn(10)

# loss = l1_loss_smooth(rand1, rand2)
# loss2 = l1_loss_smooth_2(rand1, rand2)

# print("l1", loss.numpy())
# print("l2", loss2.numpy())
# assert (loss.numpy() == loss2.numpy()).all()
