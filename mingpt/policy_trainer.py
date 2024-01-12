import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import numpy as np





def gae(config, values, rewards):
    advantages = torch.zeros_like(rewards, device=rewards.device)
    last_advantage = 0
    last_value = 0
    with torch.no_grad():
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + config.policy.gamma * last_value - values[ t]
            last_advantage = delta + config.policy.gamma * config.policy.lam * last_advantage
            advantages[ t] = last_advantage
            last_value = values[t]
        returns = advantages + values
    return advantages, returns

def ppo_loss(config, logprobs, values, old_logprobs, old_values, advantages,returns):
    values_clipped = torch.clamp(
        values,
        old_values - config.policy.cliprange_value,
        old_values + config.policy.cliprange_value,
    )
    n = len(values)
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2)) / n
    log_ratio = (logprobs - old_logprobs) 
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - config.policy.cliprange, 1.0 + config.policy.cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2)) / n
    # pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n
    loss = pg_loss + config.policy.vf_coef * vf_loss
    return loss

def logprobs_from_logits(logits, labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)

def loss_fn(config, mini_batch, model):
    query_tensors = mini_batch['query_tensors']
    response_tensors = mini_batch['response_tensors']
    old_logprobs = mini_batch['logprobs']
    old_values = mini_batch['values']
    old_rewards = mini_batch['rewards']
    
    response_length = old_rewards.shape[0]

    advantages, returns = gae(config, old_values, old_rewards)
    
    trajectories = torch.hstack([query_tensors, response_tensors])
    logits, values_pred = model(trajectories.reshape(1,-1))

    values_pred = values_pred[:, :-1]
    logprobs = logprobs_from_logits(logits[0, :-1, :], trajectories[ 1:])

    start = query_tensors.shape[0] - 1
    end = start + response_length
    logprobs, values_pred = (
        logprobs[start:end],
        values_pred[0, start:end],
    )

    loss = ppo_loss(
        config = config, 
        logprobs=logprobs,
        values=values_pred,
        old_logprobs=old_logprobs,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
    )

    return loss, old_rewards[-1].mean().item()

class Agent(nn.Module):
    def __init__(self, model, trainable=False):
        super().__init__()
        self.trainable = trainable
        self.model = model
        if not self.trainable:
            self.model = self.model.eval()
            self.model.requires_grad_(False)
        else:
            n_embd = self.model.lm_head.in_features
            num_labels = 1
            self.value_head = nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.GELU(),
                nn.Linear(n_embd, 4*n_embd),
                nn.GELU(),
                nn.Linear(4*n_embd, num_labels),
            ).to(self.model.config.device)
    
    def generate(self, input_ids, pad_id, **x):
        responses = []
        for idx in input_ids:
            response = self.model.generate_test(idx.reshape(1,-1), **x).tolist()[0]
            response = response + [pad_id.item()] * (x['max_token'] - len(response))
            responses.append(response)
        return responses

    def forward(self, input_ids, attention_mask=None):
        lm_logits, last_hidden_state = self.model.policy(input_ids)
        # last_hidden_state = outputs.hidden_states[-1]
        # lm_logits = self.logit_head(last_hidden_state)
        if self.trainable:
            value = self.value_head(last_hidden_state).squeeze(-1)
            return lm_logits, value
        else:
            return lm_logits


class CustomPromptDataGenerator():
    def __init__(self, prompt_dataset, prompt_batch_size):
        self.prompt_dataset = prompt_dataset
        self.prompt_batch_size = prompt_batch_size

    def __iter__(self):
        self.dataset_indices = np.arange(len(self.prompt_dataset))
        return self

    def __next__(self):
        if len(self.dataset_indices) >= self.prompt_batch_size:
            picked_indices = np.random.choice(np.arange(len(self.dataset_indices)),
                                              self.prompt_batch_size,
                                              replace=False)
            samples = self.prompt_dataset[self.dataset_indices[picked_indices]]
            self.dataset_indices = np.delete(self.dataset_indices, picked_indices)
            input_ids = torch.tensor(samples)
            # attention_mask = torch.ones_like(input_ids)
            # batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
            return input_ids
        else:
            raise StopIteration

class PolicyTrainer:

    @staticmethod
    def get_default_config():
        
        C = CN()
        
        C.seq_length = 278
        C.batch_size = 64
        C.lr =  0.00006
        C.prompt_size = 30
        C.prompt_batch_size = 128
        C.num_rollouts = 128
        C.epochs = 100
        C.ppo_epochs = 4
        C.kl_coef = 0.01
        C.gamma = 1
        C.lam = 0.95
        C.cliprange = 0.2
        C.cliprange_value = 0.2
        C.vf_coef = 1
        
        return C
    
    def __init__(self, config, reward_model,  prompt_dataset):
        super().__init__()
        self.config = config
        self.prompt_batch_size = config.prompt_batch_size
        self.prompt_dataset = prompt_dataset
        self.prompt_generator = CustomPromptDataGenerator(self.prompt_dataset, self.prompt_batch_size)
        self.prompt_iterator = iter(self.prompt_generator)
        self.reward_model = reward_model
        self.generate_kwargs = dict(
            max_token = self.prompt_dataset.config.max_length,
            itos=self.prompt_dataset.itos,
            beg_token = self.prompt_dataset.BOS_TOKEN,
            end_token = self.prompt_dataset.EOS_TOKEN,
            temperature=1.0,
            do_sample=False,
            top_k=None,
        )
        
    def logprobs_from_logits(self, logits, labels):
        logprobs = F.log_softmax(logits, dim=-1)
        logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
        return logprobs_labels.squeeze(-1)


    def reward_fn(self, samples):
        logits,_ = self.reward_model(samples)
        temperature = 0.3
        sentiments = torch.sigmoid(logits*temperature)[:,0].detach().cpu().tolist()
        return sentiments
    

    def make_experience(self, model, ref_model, num_rollouts=128):
        
        all_rollouts = []
        while len(all_rollouts) < num_rollouts:
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_generator = CustomPromptDataGenerator(self.prompt_dataset, self.prompt_batch_size)
                self.prompt_iterator = iter(self.prompt_generator)
                batch = next(self.prompt_iterator)
                
            device = model.model.config.device
            query = batch.to(device)

            eos_id = torch.tensor(self.prompt_dataset.stoi[self.prompt_dataset.EOS_TOKEN]).to(device)            
            query_response_list = model.generate(query, eos_id, **self.generate_kwargs)
            query_response_pad = torch.tensor(query_response_list).to(device)
            response = query_response_pad[:, query.shape[1]:]
            attention_mask = query_response_pad.not_equal(eos_id).long()

            with torch.no_grad():
                logits, values = model(
                    query_response_pad)
                ref_logits = ref_model(
                    query_response_pad)
            logprobs = self.logprobs_from_logits(logits[:, :-1, :], query_response_pad[:, 1:])
            ref_logprobs = self.logprobs_from_logits(ref_logits[:, :-1, :], query_response_pad[:, 1:])
            n_trajectories = query_response_pad.shape[0]
            values = values[:, :-1]

            start = batch.shape[1] 
            ends = start + attention_mask[:, start:].sum(1)
            truncated_values = [values[i, start : ends[i]] for i in range(n_trajectories)]
            truncated_logprobs = [logprobs[i, start : ends[i]] for i in range(n_trajectories)]

            # texts = tokenizer.batch_decode(trajectories, skip_special_tokens=True)
            scores = self.reward_fn(query_response_pad)
            rewards = -self.config.kl_coef * (logprobs - ref_logprobs)
            all_rewards = [None] * n_trajectories
            for i in range(n_trajectories):
                rs = rewards[i][start  : ends[i]]
                rs[-1] = scores[i]
                all_rewards[i] = rs

            new_rollout = [
                {
                    'query_tensors': query[i],
                    'response_tensors':response[i],
                    'logprobs':truncated_logprobs[i],
                    'values':truncated_values[i],
                    'rewards':all_rewards[i],
                }
                for i in range(n_trajectories)
            ]
            all_rollouts += new_rollout

        score = torch.tensor(scores).mean().detach().cpu().item()
        
        return all_rollouts, score
