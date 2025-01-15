"""
Trains a character-level language model.
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import normalize

from mobilitygpt.model import GPT
from mobilitygpt.trainer import Trainer
from mobilitygpt.ppo_policy_trainer import PolicyTrainer, Agent, loss_fn
from mobilitygpt.utils import set_seed, setup_logging, CfgNode as CN
import random
import pickle
import json
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm
import time
import pandas as pd
from .base_finetuner import BaseFinetuner

# -----------------------------------------------------------------------------




def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 128
    C.system.work_dir = './TS-TrajGen_Porto_synthetic/chargpt_adj_gravity_sample_0112'

    # data
    C.data = PromptDataset.get_default_config()

    #policy
    C.policy = PolicyTrainer.get_default_config()
    C.policy.learning_rate = 5e-3 # the model we're using is so small that we can go a bit faster

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mobility'

    # # trainer
    # C.trainer = Trainer.get_default_config()
    # C.trainer.learning_rate = 5e-3 # the model we're using is so small that we can go a bit faster

    return C

def get_reward_model_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 128
    C.system.work_dir = './TS-TrajGen_Porto_synthetic/chargpt_adj_gravity_sample_0112'

    # # data
    # C.data = PairwiseDataset.get_default_config()
    

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-3 # the model we're using is so small that we can go a bit faster

    return C


class PromptDataset(Dataset):
    """Dataset for PPO training with prompts"""
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 300
        C.max_length = 278
        return C
        
    def __init__(self, config, data, vocab, prompt_size):
        self.config = config
        self.EOS_TOKEN = '</S>'
        
        # Process raw data
        lines = data.strip().split('\n\n') 
        line_words = [[self.EOS_TOKEN]+l.strip().split(',')+[self.EOS_TOKEN] for l in lines]
        words = [item for sublist in line_words for item in sublist]
        data_size, vocab_size = len(words), len(vocab)
        
        # Create vocabulary mappings
        self.stoi = {ch:i for i,ch in enumerate(vocab)}
        self.itos = {i:ch for i,ch in enumerate(vocab)}
        self.stoi[self.EOS_TOKEN] = len(vocab)
        self.itos[len(vocab)] = self.EOS_TOKEN
        self.vocab_size = vocab_size + 1
        
        # Create prompt dataset
        all_input_ids = [[self.stoi[s] for s in traj] for traj in line_words]
        self.prompts_input_ids = np.array([item[:prompt_size] for item in all_input_ids if len(item)>prompt_size])
        
    def __len__(self):
        return len(self.prompts_input_ids)
        
    def __getitem__(self, ix):
        return self.prompts_input_ids[ix]

def od_pair_to_adjacency_matrix(od_pair_list):
    """
    Converts an od pair to an adjacency matrix.

    Args:
        od_pair (torch.Tensor): A tensor of od pairs, where each pair is a two-dimensional tensor of the form [source, destination].

    Returns:
        torch.Tensor: An adjacency matrix
    """

    od_pair= torch.tensor(od_pair_list)
    # Create a sparse adjacency matrix.
    adjacency_matrix = torch.sparse_coo_tensor(od_pair.t(), torch.ones(od_pair.size(0)))

    # Convert the sparse adjacency matrix to a dense adjacency matrix.
    adjacency_matrix = adjacency_matrix.to_dense()


    # Add two new rows of ones and columns at the end of adjacency matrix.
    adjacency_matrix = torch.cat((adjacency_matrix, torch.ones(1, adjacency_matrix.size(0))), 0)
    adjacency_matrix = torch.cat((adjacency_matrix, torch.ones(adjacency_matrix.size(0), 1)), 1)

    # Return the adjacency matrix.
    return adjacency_matrix


# -----------------------------------------------------------------------------

class PPOFinetuner(BaseFinetuner):
    def __init__(self, config, model, dataset, reward_model, gravity_sampling=False, prompt_size=4):
        super().__init__(config, model, dataset, gravity_sampling)
        self.prompt_size = prompt_size
        self.reward_model = reward_model
        
    def train(self, prompt_dataset):
        """Train the model using PPO"""
        # Create policy and reference models
        policy_model = Agent(self.model, trainable=True)
        ref_model = Agent(type(self.model)(self.model.config), trainable=False)
        ref_model.load_state_dict(self.model.state_dict())
        
        # Initialize trainer
        trainer = PolicyTrainer(
            config=self.config.policy,
            reward_model=self.reward_model,
            prompt_dataset=prompt_dataset
        )
        
        # Training loop
        best_score = 0
        for epoch in range(self.config.policy.epochs):
            # Generate experience
            rollouts, score = trainer.make_experience(policy_model, ref_model, 
                                                    self.config.policy.num_rollouts)
            
            # Update policy
            for batch in rollouts:
                for _ in range(self.config.policy.ppo_epochs):
                    loss, reward = trainer.compute_loss(self.config, batch, policy_model)
                    loss.backward()
                    trainer.optimizer.step()
                    trainer.optimizer.zero_grad()
                    
            print(f"Epoch {epoch+1}, Score: {score:.3f}")
            
            # Save best model
            if score > best_score:
                best_score = score
                torch.save(policy_model.model.state_dict(), 
                          f"{self.config.system.work_dir}/model_RL.pt")
                print("Best model saved")
    
    