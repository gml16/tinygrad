import os
import numpy as np
import math, traceback
from datetime import datetime
from typing import List, NamedTuple, Tuple
from tinygrad import Tensor, TinyJit
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.features.search import actions, bufs_from_lin, time_linearizer, get_linearizer_actions
from tinygrad.nn.optim import Adam
from tinygrad.tensor import dtypes
from tinygrad.codegen.linearizer import Linearizer
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats


USE_WANDB = False
SAVED_MODEL_PATH = "/tmp/dqn.safetensors" # if path does not exist the model is initialised from scratch
BS = 4
DELTA = 1e-4
MIN_EPSILON = 0.1
GAMMA = 0.9
EXP_REPLAY_SIZE = 10000
TRAIN_FREQ = 1
EVAL_FREQ = 100
SAVE_FREQ = 100
TARGET_UPDATE_FREQ = 200
LR = 1e-4
RNG_SEED = 0

INNER = 256
class DQN:
  def __init__(self):
    self.l1 = Linear(1021,INNER)
    self.l2 = Linear(INNER,INNER)
    self.l3 = Linear(INNER,1+len(actions))
  def __call__(self, x):
    x = self.l1(x).relu()
    x = self.l2(x).relu().dropout(0.9)
    return self.l3(x)

class Transition(NamedTuple):
  cur_feat: List[float]
  next_feat: List[float]
  act: int
  rew: float
  terminal: bool

class ExpReplay:
  def __init__(self):
    self.idx = 0
    self.cur_feats = None
    self.next_feats = None
    self.acts = None
    self.rews = None
    self.terminals = None
  def insert(self, transition):
    if self.cur_feats is None:
      self.cur_feats = np.array([transition.cur_feat])
      self.next_feats = np.array([transition.next_feat])
      self.acts = np.array([transition.act])
      self.rews = np.array([transition.rew])
      self.terminals = np.array([transition.terminal])
    elif self.idx == len(self.cur_feats):
      self.cur_feats = np.append(self.cur_feats, [transition.cur_feat], axis=0)
      self.next_feats = np.append(self.next_feats, [transition.next_feat], axis=0)
      self.acts = np.append(self.acts, [transition.act], axis=0)
      self.rews = np.append(self.rews, [transition.rew], axis=0)
      self.terminals = np.append(self.terminals, [transition.terminal], axis=0)
    else:
      self.cur_feats[self.idx] = transition.cur_feat
      self.next_feats[self.idx] = transition.next_feat
      self.acts[self.idx] = transition.act
      self.rews[self.idx] = transition.rew
      self.terminals[self.idx] = transition.terminal
    self.idx = (self.idx + 1) % EXP_REPLAY_SIZE
    assert len(self.cur_feats) == len(self.next_feats) == len(self.acts) == len(self.rews) == len(self.terminals), f"{len(self.cur_feats)}, {len(self.next_feats)}, {len(self.acts)}, {len(self.rews)}, {len(self.terminals)}"
  def sample(self):
    ids = np.random.choice(len(self.cur_feats), BS)
    return self.cur_feats[ids], self.next_feats[ids], self.acts[ids], self.rews[ids], self.terminals[ids]

def l1_loss_smooth(predictions: Tensor, targets: Tensor, beta = 1.0):
    diff = predictions-targets
    return (0.5*diff**2 / beta).clip(0, 0.5) + (diff.abs() - beta).relu()

def calculate_loss(q_net: DQN, target_net: DQN, transitions: Tuple) -> Tensor:
  # Transitions are tuple of shape (states, actions, rewards, next_states, dones)
  curr_state = Tensor(transitions[0])
  next_state = Tensor(transitions[1])
  act = Tensor(transitions[2]).unsqueeze(-1)
  rew = Tensor(transitions[3]).clip(-1, 1)
  terminal = Tensor(transitions[4])
  y = target_net(next_state)
  max_target_net = y.max(-1)
  net_pred = q_net(curr_state)
  is_not_over = (Tensor.ones(*terminal.shape) - terminal)
  # Bellman equation
  labels = rew + is_not_over * (GAMMA * max_target_net.detach())
  y_pred = net_pred.gather(idx=act, dim=-1).squeeze()
  loss = l1_loss_smooth(y_pred, labels)
  return loss.mean()

def get_next_action(feat: Tensor, q_net: DQN, target_net: DQN, lin: Linearizer, eps: float) -> Tuple[int, float]:
  # epsilon-greedy policy
  valid_action_mask = np.zeros((len(actions)+1), dtype=np.float32)
  for x in get_linearizer_actions(lin): valid_action_mask[x] = 1
  if np.random.random() < eps:
    q_val = np.zeros((len(actions),))
    idx = np.random.choice(len(valid_action_mask), p=valid_action_mask/sum(valid_action_mask))
  else:
    idx, q_val = get_greedy_action(feat, q_net, target_net, valid_action_mask)
  return idx, q_val

def get_greedy_action(feat, q_net, target_net, valid_action_mask, double_learning=True) -> Tuple[int, Tensor]:
  inputs = Tensor(feat)
  if double_learning:
    q_vals = q_net(inputs)
  else:
    q_vals = target_net(inputs)
  q_vals = q_vals.detach()
  q_vals = (q_vals + 1e6) * Tensor(valid_action_mask) # hack to select best valid action when all valid actions are negative 
  idx = q_vals.argmax().cast(dtypes.int64).numpy() # PR to remove the need to do this
  return idx, q_vals

def train(ast_strs, q_net, target_net, optim):
  expreplay = ExpReplay()
  eps = 1
  episode = 0
  while 1:
    Tensor.no_grad, Tensor.training = True, False
    idx = np.random.choice(ast_strs)
    lin = ast_str_to_lin(idx)
    try:
      next_feat = lin_to_feats(lin)
      rawbufs = bufs_from_lin(lin)
      tm = last_tm = base_tm = time_linearizer(lin, rawbufs)
    except:
      print("new lin failed with index", idx)
      print(traceback.format_exc())
      continue
    step = 0
    while 1:
      cur_feat = next_feat
      act, q_val = get_next_action(cur_feat, q_net, target_net, lin, eps)
      if act == 0:
        rew = 0
        break
      try:
        lin.apply_opt(actions[act-1])
        next_feat = lin_to_feats(lin)
        tm = time_linearizer(lin, rawbufs)
        if math.isinf(tm): raise Exception("failed")
        rew = (last_tm-tm)/base_tm
        last_tm = tm
      except:
        rew = -0.5
        break
      expreplay.insert(Transition(cur_feat, next_feat, act, rew, False))
      step+=1
    expreplay.insert(Transition(cur_feat, next_feat, act, rew, True))
    episode += 1
    eps = max(MIN_EPSILON, eps-DELTA)
    print(f"***** {episode=} {step=} {eps=} {base_tm*1e6:12.2f} -> {tm*1e6:12.2f} : {lin.colored_shape()}")
    wandb_log({"episode": episode, "epsilon": eps, "steps": step, "tm_diff": (base_tm-last_tm)/base_tm})
    if episode % TRAIN_FREQ == 0:
      Tensor.no_grad, Tensor.training = False, True
      batch = expreplay.sample()
      loss = calculate_loss(q_net, target_net, batch)
      optim.zero_grad()
      loss.backward()
      optim.step()
      wandb_log({"loss": loss.numpy().tolist(), "loss_grad": loss.grad.numpy().tolist()})
      if episode % (TRAIN_FREQ * SAVE_FREQ) == 0:
        model_path = f"qnets/qnet_{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}_ep{episode}.safetensors"
        safe_save(get_state_dict(q_net), model_path)
    if episode % TARGET_UPDATE_FREQ == 0:
      copy_dqn_to_target(q_net, target_net)
    if episode % EVAL_FREQ == 0:
      evaluate_net(episode, ast_strs, q_net, target_net)

def evaluate_net(episode, ast_strs, q_net, target_net):
  Tensor.no_grad, Tensor.training = True, False
  tries = 0
  while 1:
    idx = np.random.choice(ast_strs)
    lin = ast_str_to_lin(idx)
    try:
      next_feat = lin_to_feats(lin)
      rawbufs = bufs_from_lin(lin)
      last_tm = base_tm = time_linearizer(lin, rawbufs)
      break
    except:
      tries += 1
      if tries > 10:
        print("lin to time during evaluation failed 10 times")
        break
      continue
  while 1:
    cur_feat = next_feat
    act, _ = get_next_action(cur_feat, q_net, target_net, lin, eps=0)
    if act == 0:
      break
    try:
      lin.apply_opt(actions[act-1])
      next_feat = lin_to_feats(lin)
      tm = time_linearizer(lin, rawbufs)
      if math.isinf(tm): raise Exception("failed")
      last_tm = tm
    except:
      break
    wandb_log({"episode": episode, "eval_tm_diff": (base_tm-last_tm)/base_tm})

def wandb_log(*args, **kwargs):
  if USE_WANDB: wandb.log(*args, **kwargs)
  else: print(str(args) if len(args) > 0 else "", str(kwargs) if len(kwargs) > 0 else "")

def copy_dqn_to_target(q_net, target_net):
  state_dict = get_state_dict(q_net)
  for v in state_dict.values(): v.requires_grad = False
  load_state_dict(target_net, state_dict, verbose=False)
  for v in state_dict.values(): v.requires_grad = True

def main():
  if USE_WANDB:
    wandb.login()
    run = wandb.init(
      project="tinygrad-rl",
      config={
          "batch_size": BS,
          "delta": DELTA,
          "min_epsilon": MIN_EPSILON,
          "gamma": GAMMA,
          "exp_replay_size": EXP_REPLAY_SIZE,
          "train_freq": TRAIN_FREQ,
          "target_update_freq": TARGET_UPDATE_FREQ,
          "learning_rate": LR,
          "dqn_layers": " / ".join(list(get_state_dict(DQN()).keys())),
          "dqn_inner_size": INNER,
          "rng_seed": RNG_SEED
      },
    )
   
  np.random.seed(RNG_SEED)
  q_net = DQN()
  target_net = DQN()
  copy_dqn_to_target(q_net, target_net)
  if os.path.isfile(SAVED_MODEL_PATH): 
    print(f"loading model from {SAVED_MODEL_PATH}")
    load_state_dict(q_net, safe_load(SAVED_MODEL_PATH))
  else: print(f"no model to load from {SAVED_MODEL_PATH}")
  optim = Adam(get_parameters(q_net), LR)
  ast_strs = load_worlds()
  train(ast_strs, q_net, target_net, optim)

# TODO:
# Stack past 4 observations as the state
if __name__ == "__main__":
  try:
    import wandb
  except:
    USE_WANDB = False
  main()
