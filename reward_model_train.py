"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import normalize

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.reward_trainer import RewardTrainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import random
import pickle
import json
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm
# -----------------------------------------------------------------------------




def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 128
    C.system.work_dir = './TS-TrajGen_Porto_synthetic/chargpt_adj_gravity_sample_0112'

    # data
    C.data = PairwiseDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-3 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class PairwiseDataset(Dataset):
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 300
        C.max_length = 278
        return C

    def __init__(self, config, pairs, vocab, data):
        
        self.config = config
        self.EOS_TOKEN = '</S>'
        # self.BOS_TOKEN = '<S>'
        
        lines = data.strip().split('\n\n') 
        line_words = [[self.EOS_TOKEN]+l.strip().split(',')+[self.EOS_TOKEN] for l in lines]
        words = [item for sublist in line_words for item in sublist]
        origins = [s[1] for s in line_words]
        # vocab=list(set(words))
        # chars = sorted(list(set(data)))
        data_size, vocab_size = len(words), len(vocab)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        
        self.stoi[self.EOS_TOKEN] = len(vocab)
        self.itos[len(vocab)] = self.EOS_TOKEN
        self.vocab_size = vocab_size + 1 

        # self.stoi[self.EOS_TOKEN] = len(vocab)+1
        # self.itos[len(vocab)+1] = self.EOS_TOKEN      
        
        self.chosen_input_ids = []
        self.rejected_input_ids = []
        
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_ids = [self.stoi[c] for c in chosen]
            rejected_ids = [self.stoi[c] for c in rejected]
            
            self.chosen_input_ids.append(chosen_ids)
            self.rejected_input_ids.append(rejected_ids)

        self.num_trajs = len(lines)
        self.data = words
        self.trajs = line_words
        self.origins = origins
        
    def __len__(self):
        return len(self.chosen_input_ids)

    def get_block_size(self):
        return self.config.block_size

    def get_vocab_size(self):
        return self.vocab_size
    
    def __getitem__(self, idx):
        x = torch.tensor([self.chosen_input_ids[idx],self.rejected_input_ids[idx]])        
        y = torch.tensor([0, 1])
        # y = torch.tensor([0] * len(idx) + [1] * len(idx))
        return x, y


def create_comparison_dataset_ls(config):
    file = open(config.system.work_dir+'/preference_dataset','rb')
    sequence_length = config.data.max_length
    data = pickle.load(file)
    EOS_TOKEN = '</S>'
    
    pairs = []
    for sample in data:
        chosen = None
        rejected = None
        if sample['choice'] == 0:
            chosen =  sample['candidate_0']
            rejected = sample['candidate_1']
        else:
            chosen = sample['candidate_1']
            rejected = sample['candidate_0']

        if len(chosen) < sequence_length:
            chosen = chosen + [EOS_TOKEN] * (sequence_length - len(chosen))
        if len(rejected) < sequence_length:
            rejected = rejected + [EOS_TOKEN] * (sequence_length - len(rejected))
        pair = {
            'chosen': chosen[:sequence_length],
            'rejected': rejected[:sequence_length]
        }
        pairs.append(pair)
    return pairs

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    
    model_load=True
    num_samples = int(5e3)

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)


    # construct the training dataset
    text = open('TS-TrajGen_Porto.txt', 'r').read() # don't worry we won't run out of file handles
    porto_geo=pd.read_csv('Porto-Taxi/porto.geo')    
    geo_ids=porto_geo['geo_id'].apply(str).tolist()    

    # train_dataset = CharDataset(config.data, text, geo_ids)

    pairs = create_comparison_dataset_ls(config)
    reward_dataset = PairwiseDataset(config.data, pairs, geo_ids, text)


    # construct the model
    config.model.vocab_size = reward_dataset.get_vocab_size()
    config.model.block_size = reward_dataset.get_block_size()
    reward_model = GPT(config.model, reward_model=True)
    reward_model = reward_model.to('cuda')


    # split the dataset into a training and validation set
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:

# ##############
    dataset_size = len(reward_dataset)
    indices = list(range(dataset_size))
    
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # construct the trainer object    
    trainer = RewardTrainer(config.trainer, reward_model, reward_dataset, train_sampler=train_sampler, val_sampler=valid_sampler)


    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 250 == 0:
            ckpt_path = os.path.join(config.system.work_dir, "reward_model.pt")
            torch.save(reward_model.state_dict(), ckpt_path)
            reward_model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    