from tinygrad.nn.optim import Adam
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad import Tensor

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

loss = l1_loss_smooth_2(net(Tensor.randn(1021)), Tensor.randn(10)).mean()
# optim.zero_grad()
loss.backward()
# optim.step()
print(loss.numpy())

# import torch

# rand1 = Tensor.randn(10)
# rand2 = Tensor.randn(10)

# loss = l1_loss_smooth(rand1, rand2)
# loss2 = l1_loss_smooth_2(rand1, rand2)

# print("l1", loss.numpy())
# print("l2", loss2.numpy())
# assert (loss.numpy() == loss2.numpy()).all()
