from tinygrad.nn.optim import Adam
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.tensor import Tensor

def l1_loss_smooth_2(predictions: Tensor, targets: Tensor, beta = 1.0) -> Tensor:
    diff = predictions-targets
    return (0.5*diff**2 / beta).clip(0, 0.5) + (diff.abs() - beta).relu()

def l1_loss_smooth(predictions: Tensor, targets: Tensor, beta = 1.0):
    diff = predictions-targets
    diff_abs = diff.abs()
    mask = diff_abs < beta
    return mask * (0.5*diff**2 / beta) + (1-mask) * (diff_abs - 0.5*beta)

class Net:
  def __init__(self):
    self.l1 = Linear(1021,10)
  def __call__(self, x):
    return self.l1(x)
  
# def relu_2(t):
#    return t * (t > 0)
  
net = Net()
optim = Adam(get_parameters(net), 1e-3)
rand1 = Tensor.randn(1021)
rand2 = Tensor.randn(10)

loss = l1_loss_smooth_2(net(rand1), rand2).mean()
print("loss type", type(loss))
loss.backward()
print(loss.numpy())

# loss = l1_loss_smooth_2(net(rand1), rand2).mean()
# loss.backward()
# print(loss.numpy())

# print("rand", rand1.numpy())
# loss = relu_2(rand1)
# print("relu", loss.numpy())
# loss = loss.mean()
# loss.backward()
# print(loss.numpy())

# print("lazy", rand2.LazyBuffer)
# from tinygrad.mlops import Less
# Less()

# import torch

# rand1 = Tensor.randn(10)
# rand2 = Tensor.randn(10)

# loss = l1_loss_smooth(rand1, rand2)
# loss2 = l1_loss_smooth_2(rand1, rand2)

# print("l1", loss.numpy())
# print("l2", loss2.numpy())
# assert (loss.numpy() == loss2.numpy()).all()


def l1_loss_smooth_2(predictions: Tensor, targets: Tensor, beta = 1.0):
    diff = predictions-targets
    return (0.5*diff**2 / beta).clip(0, 0.5) + (diff.abs() - beta).relu()

def l1_loss_smooth(predictions: Tensor, targets: Tensor, beta = 1.0):
    diff = predictions-targets
    diff_abs = diff.abs()
    mask = diff_abs < beta
    return mask * (0.5*diff**2 / beta) + (1-mask) * (diff_abs - 0.5*beta)

class Net:
  def __init__(self):
    self.l1 = Linear(1021,10)
  def __call__(self, x):
    return self.l1(x)
  
net = Net()
optim = Adam(get_parameters(net), 1e-3)

loss = l1_loss_smooth(net(Tensor.randn(1021)), Tensor.randn(10)).mean()
# optim.zero_grad()
loss.backward()
# optim.step()
print(loss.numpy())

