import os
import numpy as np
import math, random, traceback
from typing import List, NamedTuple, Tuple
from tinygrad import Tensor, TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from tinygrad.features.search import actions, bufs_from_lin, time_linearizer, get_linearizer_actions
from tinygrad.nn.optim import Adam
from extra.optimization.extract_policynet import PolicyNet
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats

USE_WANDB = False
BS = 32
DELTA = 0
MIN_EPSILON = 0.1
GAMMA = 0.9
EXP_REPLAY_SIZE = 10000
TRAIN_FREQ = 5

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

def calculate_loss(q_net: PolicyNet, target_net: PolicyNet, transitions: Tuple):
    # Transitions are tuple of shape (states, actions, rewards, next_states, dones)
    curr_state = Tensor(transitions[0])
    next_state = Tensor(transitions[1])
    act = Tensor(transitions[2]).unsqueeze(-1) # dtype=long
    rew = Tensor(transitions[3]) # TODO clamp between -1 and 1: .clamp(-1, 1) # dtype=float32
    terminal = Tensor(transitions[4]) #, dtype=DType.int)
    y = target_net(next_state)
    max_target_net = y.max(-1)[0]
    net_pred = q_net(curr_state)
    is_not_over = (Tensor.ones(*terminal.shape) - terminal)
    # Bellman equation
    labels = rew + is_not_over * (GAMMA * max_target_net.detach())
    y_pred = net_pred.gather(idx=act, dim=-1).squeeze()
    loss = ((labels - y_pred)**2)
    return loss.mean()

def get_next_action(feat, q_net, target_net, lin, eps):
  # mask valid actions
  valid_action_mask = np.zeros((len(actions)+1), dtype=np.float32)
  for x in get_linearizer_actions(lin): valid_action_mask[x] = 1
  # epsilon-greedy policy
  if np.random.random() < eps:
    q_val = np.zeros((len(actions),))
    idx = np.random.choice(len(valid_action_mask), p=valid_action_mask/sum(valid_action_mask))
  else:
    idx, q_val = get_greedy_action(feat, q_net, target_net, valid_action_mask)
  return idx, q_val

def get_greedy_action(feat, q_net, target_net, valid_action_mask, doubleLearning=True):
  # inputs = Tensor(feat).unsqueeze(0)
  q_vals = q_net(Tensor([feat])).exp()[0].numpy()

  q_vals *= valid_action_mask
  # probs /= sum(probs)
  # if doubleLearning:
  #   q_vals = self.dqn.q_network(inputs).detach().squeeze(0)
  # else:
  #   q_vals = self.dqn.target_network(
  #     inputs).detach().squeeze(0)
  idx = q_vals.max()
  # greedy_steps = np.array(idx, dtype=np.int32).flatten()
  return idx, q_vals

# @TinyJit
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
      wandb_log({"loss": loss.numpy()})

def wandb_log(*args, **kwargs):
  if USE_WANDB: wandb.log(*args, **kwargs)
  else: print(str(args) if len(args) > 0 else "", str(kwargs) if len(kwargs) > 0 else "")

if __name__ == "__main__":
  if USE_WANDB:
    try:
      import wandb
      wandb.login()
      run = wandb.init(
        project="tinygrad-rl",
        config={
            "batch_size": BS,
            "delta": DELTA,
            "min_epsilon": MIN_EPSILON,
            "gamma": GAMMA,
            "exp_replay_size": EXP_REPLAY_SIZE,
            "train_freq": TRAIN_FREQ
        },
      )
    except:
      USE_WANDB = False

  q_net = PolicyNet()
  target_net = PolicyNet()
  load_state_dict(target_net, get_state_dict(q_net))
  if os.path.isfile("/tmp/policynet.safetensors"): load_state_dict(q_net, safe_load("/tmp/policynet.safetensors"))
  optim = Adam(get_parameters(q_net))

  ast_strs = load_worlds()
  train(ast_strs, q_net, target_net, optim)

 
