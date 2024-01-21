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

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.reward_trainer import RewardTrainer
from mingpt.policy_trainer import PolicyTrainer, Agent, loss_fn
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import random
import pickle
import json
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm
import time
import pandas as pd
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
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 512
        C.max_length = 278
        return C

    def __init__(self, config, data, vocab, prompt_size):

        self.config = config
        self.EOS_TOKEN = '</S>'
        self.BOS_TOKEN = '<S>'
        
        lines = data.strip().split('\n\n') 
        line_words = [[self.BOS_TOKEN]+l.strip().split(',')+[self.EOS_TOKEN] for l in lines]
        words = [item for sublist in line_words for item in sublist]
        data_size, vocab_size = len(words), len(vocab)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        
        self.stoi[self.BOS_TOKEN] = len(vocab)
        self.itos[len(vocab)] = self.BOS_TOKEN
        
        self.stoi[self.EOS_TOKEN] = len(vocab)+1
        self.itos[len(vocab)+1] = self.EOS_TOKEN
        self.vocab_size = vocab_size + 2 

        
        all_input_ids = [[self.stoi[s] for s in traj] for traj in line_words]
        self.prompts_input_ids = np.array([item[:prompt_size] for item in all_input_ids if len(item)>prompt_size])

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

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
    adjacency_matrix = torch.cat((adjacency_matrix, torch.ones(2, adjacency_matrix.size(0))), 0)
    adjacency_matrix = torch.cat((adjacency_matrix, torch.ones(adjacency_matrix.size(0), 2)), 1)

    # Return the adjacency matrix.
    return adjacency_matrix


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    
    model_load=True
    num_samples = int(5e3)
    prompt_size = 4

    # get default config and overrides from the command line, if any
    reward_config = get_reward_model_config()
    config = get_config()
    # config.merge_from_args(sys.argv[1:])
    # print(config)
    setup_logging(config)
    setup_logging(reward_config, 'reward')
    set_seed(config.system.seed)


    # construct the training dataset
    text = open('TS-TrajGen_Porto.txt', 'r').read() # don't worry we won't run out of file handles
    porto_geo=pd.read_csv('Porto-Taxi/porto.geo')    
    geo_ids=porto_geo['geo_id'].apply(str).tolist()    

    prompt_dataset = PromptDataset(config.data, text, geo_ids, prompt_size)
    # pairs = create_comparison_dataset_ls(reward_config)
    # reward_dataset = PairwiseDataset(pairs, geo_ids)

    # construct the model
    reward_config.model.vocab_size = prompt_dataset.get_vocab_size()
    reward_config.model.block_size = prompt_dataset.get_block_size()
    reward_model = GPT(reward_config.model, reward_model=True)
    ckpt_path = os.path.join(reward_config.system.work_dir, "reward_model.pt")
    reward_model.load_state_dict(torch.load(ckpt_path))
    reward_model = reward_model.to('cuda')
    
    #Freeze gradients of reward model as it is not gonna be updated during policy update
    reward_model = reward_model.eval()
    
    porto_rel=pd.read_csv('Porto-Taxi/porto.rel')    
    porto_rel['combined'] = porto_rel.apply(lambda x: list([x['origin_id'], x['destination_id']]),axis=1)
    od_list=porto_rel['combined'].tolist()
    adj_matrix=od_pair_to_adjacency_matrix(od_list)
    adj_matrix = adj_matrix.to('cuda')    

    config.model.vocab_size = prompt_dataset.get_vocab_size()
    config.model.block_size = prompt_dataset.get_block_size()
    gravity = pd.read_csv('Porto-Taxi/Porto_Taxi_trajectory_train_w_gravity.csv').gravity.values
    gravity = torch.Tensor(gravity)
    
    minimum, maximum = torch.min(gravity), torch.max(gravity)    
    gravity  = (gravity-minimum)/(maximum-minimum) 
    model_input = GPT(config.model, adj_matrix=adj_matrix, gravity=gravity)
    ckpt_path = os.path.join(config.system.work_dir, "model.pt")
    model_input.load_state_dict(torch.load(ckpt_path))
    model_input = model_input.to(config.model.device)
    
    ref_model = Agent(model_input, trainable=False)
    model = Agent(model_input, trainable=True)
    
    rollout_creator = PolicyTrainer(config.policy, reward_model, prompt_dataset)
    opt = torch.optim.AdamW(model.parameters(), config.policy.learning_rate)
    
    # calculate statistic for test data
    df_porto=pd.read_csv('Porto-Taxi/Porto_Taxi_trajectory_test.csv')
    samples=df_porto.sample(n=100)
    rid_list_list=samples.rid_list.values.tolist()
    porto_geo=pd.read_csv('Porto-Taxi/porto.geo')
    OD_test=[]
    length_test=[]
    for traj in rid_list_list:
        link_ids=list(map(int, traj.split(',')))
        OD_test.append(link_ids[0])
        OD_test.append(link_ids[-1])
        length = porto_geo[porto_geo['geo_id'].isin(link_ids)].length.sum()
        length_test.append(length)

    total_steps = (config.policy.num_rollouts//config.policy.batch_size)*config.policy.ppo_epochs*config.policy.epochs
    tbar = tqdm(initial=0, total=config.policy.epochs)
    all_scores = []
    best_score = 0
    for i in range(config.policy.epochs):
        
        # filling in the storage (phase 1)
        # store.clear_history()
        rollouts, score = rollout_creator.make_experience(model, ref_model, config.policy.num_rollouts)
        # store.push(rollouts)
        # train_dataloader = store.create_loader(args.batch_size, shuffle=True)
        
        # loss calculation and graident optimization (Phase 2)
        for batch in rollouts:
            for _ in range(config.policy.ppo_epochs):
                loss, reward = loss_fn(config, batch, model)
                loss.backward()
                opt.step()
                opt.zero_grad()
        tbar.update()
        all_scores.append(score)
        tbar.set_description(f"| score: {score:.3f} |")
        if score > best_score:
            best_score = score
            ckpt_path = os.path.join(config.system.work_dir, "model_RL.pt")
            torch.save(model.model.state_dict(), ckpt_path)
            print("Best model saved")
    
    