"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data.dataloader import DataLoader
# from mingpt.utils import CfgNode as CN
# from scipy.spatial import distance
# import pandas as pd
# from tqdm import tqdm
# import gc
import random


KEY_PCT = 'prompt_chosen_tokens'
KEY_PRT = 'prompt_rejected_tokens'
KEY_CLM = 'chosen_loss_mask'
KEY_RLM = 'rejected_loss_mask'


def pad_tensor(seq, max_len, pad_value):
  """
  args:
    seq: A tensor of shape (seq_len,)
    max_len: The length to pad to
    pad_value: The value to pad with

  returns:
    A tensor of shape (max_len,)
  """

  pad_len = max_len - seq.shape[0]
  if pad_len <= 0:
    return seq[:max_len]
  return torch.cat([seq, torch.ones(pad_len, dtype=torch.long, device='cpu') * pad_value])

def get_max_len(examples):
  """
  args:
    examples:
      A list of examples, where each example is a dict with chosen and rejected
      input tensors (along with loss masks)
  
  returns:
    The length of the longest chosen or rejected input tensor
  """

  return max(
    max(len(example[KEY_PCT]), len(example[KEY_PRT])) for example in examples
  )

def get_padded_batch(examples, max_len, device='cpu'):
  """
  args:
    examples: A list of examples, each a dict with 4 key-value pairs
    max_len: The length each input tensor should be padded to
    device: The device to put the tensors on, default is 'cpu'
  
  returns:
    A tuple of 4 tensors, each of shape (batch_size, max_len)
  """
  try:
    chosen_tokens = torch.stack([
      pad_tensor(example[KEY_PCT].to(device), max_len, 1) for example in examples
    ]).to(device)

    rejected_tokens = torch.stack([
      pad_tensor(example[KEY_PRT].to(device), max_len, 1) for example in examples
    ]).to(device)

    chosen_loss_masks = torch.stack([
      pad_tensor(example[KEY_CLM].to(device), max_len, 0) for example in examples
    ]).to(device)

    rejected_loss_masks = torch.stack([
      pad_tensor(example[KEY_RLM].to(device), max_len, 0) for example in examples
    ]).to(device)
  except:
     print(examples)
     assert 0

  return chosen_tokens, rejected_tokens, chosen_loss_masks, rejected_loss_masks


def get_log_ps(logits, idxs, loss_mask):
  """
  args:
    logits: A tensor of shape (batch_size, seq_len, vocab_size)
    idxs: A torch.long tensor of shape (batch_size, seq_len)
    loss_mask: A torch.float tensor of shape (batch_size, seq_len)
  
  returns:
    A tensor of shape (batch_size,), the log probabilities of each sequence in the batch
  """

  idxs = idxs[:, 1:].unsqueeze(2)
  loss_mask = loss_mask[:, 1:]
  log_p_distributions = F.log_softmax(logits, dim=-1)[:, :-1]
  log_ps = torch.gather(log_p_distributions, dim=2, index=idxs).squeeze(2)
  return (log_ps * loss_mask).sum(dim=-1)


def loss_fn(
  chosen_policy_log_ps,
  rejected_policy_log_ps,
  chosen_ref_log_ps,
  rejected_ref_log_ps,
  beta=0.01
):
  """
  args:
    chosen_policy_log_ps: A tensor of shape (batch_size,)
    rejected_policy_log_ps: A tensor of shape (batch_size,)
    chosen_ref_log_ps: A tensor of shape (batch_size,)
    rejected_ref_log_ps: A tensor of shape (batch_size,)
    beta: The KL penalty parameter, default is 0.01 (from the paper)
  
  returns:
    A scalar tensor, the loss, and two scalar tensors, the chosen and rejected rewards
  """

  policy_log_ratio = chosen_policy_log_ps - rejected_policy_log_ps
  ref_log_ratio = chosen_ref_log_ps - rejected_ref_log_ps
  loss = -F.logsigmoid(beta * (policy_log_ratio - ref_log_ratio)).mean()
  
  # compute rewards too
  with torch.no_grad():
    chosen_reward = beta * (chosen_policy_log_ps - chosen_ref_log_ps).sum().cpu()
    rejected_reward = beta * (rejected_policy_log_ps - rejected_ref_log_ps).sum().cpu()

  return loss, chosen_reward, rejected_reward


def compute_loss(policy_model, ref_model, batch, beta):
  """
  args:
    policy_model: The policy model, $\pi_{\theta}$
    ref_model: The reference model, $\pi_{\text{ref}}$
    batch: A tuple of 4 tensors, each of shape (batch_size, max_len)

  returns:
    A scalar tensor, the loss, and two scalar tensors, the chosen and rejected rewards
  """

  chosen_tokens, rejected_tokens, chosen_loss_masks, rejected_loss_masks = batch
  chosen_tokens = chosen_tokens.to('cuda:0')
  rejected_tokens = rejected_tokens.to('cuda:0')
  chosen_loss_masks = chosen_loss_masks.to('cuda:0')
  rejected_loss_masks = rejected_loss_masks.to('cuda:0')

  chosen_policy_logits, _ = policy_model(chosen_tokens)
  chosen_policy_log_ps = get_log_ps(
    chosen_policy_logits, chosen_tokens, chosen_loss_masks
  )

  rejected_policy_logits, _ = policy_model(rejected_tokens)
  rejected_policy_log_ps = get_log_ps(
    rejected_policy_logits, rejected_tokens, rejected_loss_masks
  )

  with torch.no_grad():
    chosen_ref_logits, _ = ref_model(chosen_tokens)
    rejected_ref_logits, _ = ref_model(rejected_tokens)
    chosen_ref_log_ps = get_log_ps(
      chosen_ref_logits, chosen_tokens, chosen_loss_masks
    )
    rejected_ref_log_ps = get_log_ps(
      rejected_ref_logits, rejected_tokens, rejected_loss_masks
    )

  return loss_fn(
    chosen_policy_log_ps,
    rejected_policy_log_ps,
    chosen_ref_log_ps,
    rejected_ref_log_ps,
    beta=beta
  )

class DPOTrainer:


    def __init__(self, config, model, reference_model, train_dataset, val_sampler=None):
        self.config = config
        self.model = model
        self.reference_model = reference_model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.val_loss = None
        self.val_sampler = val_sampler
        self.max_iters = 3000

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.test_num_samples = 100

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        policy_model, reference_model, config = self.model, self.reference_model, self.config

        # setup the optimizer
        self.optimizer = policy_model.configure_optimizers(config)

        policy_model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            examples = random.sample(self.train_dataset, config.batch_size)
            max_len = min(278, get_max_len(examples))
            batch = get_padded_batch(examples, max_len)
            self.loss, self.chosen_reward, self.rejected_reward = compute_loss(
            policy_model, reference_model, batch, 0.1
            )
            self.loss.backward()
            self.optimizer.step() 
            self.optimizer.zero_grad()

            
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow
            self.trigger_callbacks('on_batch_end')
            #print('iter time:', self.iter_dt, 'iter: ', self.iter_num, 'loss: ', loss.item(), 'chosen_reward: ', chosen_reward.item(), 'rejected_reward: ', rejected_reward.item())
            

            # termination conditions
            if self.max_iters is not None and self.iter_num >= self.max_iters:
                break
            

