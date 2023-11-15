"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import random
import pickle
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
# -----------------------------------------------------------------------------




def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 128
    C.system.work_dir = './TS-TrajGen_Porto_synthetic/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data, vocab):
        self.config = config
        self.EOS_TOKEN = '</S>'
        self.BOS_TOKEN = '<S>'
        
        lines = data.strip().split('\n\n') 
        line_words = [[self.BOS_TOKEN]+l.strip().split(',')+[self.EOS_TOKEN] for l in lines]
        words = [item for sublist in line_words for item in sublist]
        # vocab=list(set(words))
        # chars = sorted(list(set(data)))
        data_size, vocab_size = len(words), len(vocab)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        
        self.stoi[self.BOS_TOKEN] = len(vocab)
        self.itos[len(vocab)] = self.BOS_TOKEN
        
        self.stoi[self.EOS_TOKEN] = len(vocab)+1
        self.itos[len(vocab)+1] = self.EOS_TOKEN
        
        self.vocab_size = vocab_size + 2 
        self.data = words

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y



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
    
    
    model_load=False
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
    
    
    train_dataset = CharDataset(config.data, text, geo_ids)

    
    porto_rel=pd.read_csv('Porto-Taxi/porto.rel')    
    porto_rel['combined'] = porto_rel.apply(lambda x: list([x['origin_id'], x['destination_id']]),axis=1)
    od_list=porto_rel['combined'].tolist()
    


    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # split the dataset into a training and validation set
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    adj_matrix=od_pair_to_adjacency_matrix(od_list)
    adj_matrix = adj_matrix.to('cuda')

    # construct the trainer object    
    trainer = Trainer(config.trainer, model, train_dataset, train_sampler=train_sampler, val_sampler=valid_sampler, adj_matrix=adj_matrix)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                # context = random.sample(train_dataset.data,1)
                context = [train_dataset.BOS_TOKEN]

                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=None, adj_matrix=adj_matrix)[0]
                completion = ','.join([train_dataset.itos[int(i)] for i in y])
                # print(completion)
            # save the latest model
            # print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    def validation_end_callback(trainer):
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f} val loss {trainer.val_loss:.5f}")

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.add_callback('validation_end', validation_end_callback)
    
    if model_load:
        ckpt_path = os.path.join(config.system.work_dir, "model.pt")
        model.load_state_dict(torch.load(ckpt_path))
    else:    
        # run the optimization
        trainer.run()
    
    syntehtic_links=[]
    for i in tqdm(range(num_samples)):
        try:
            # context = random.sample(train_dataset.data,1)
            context = [train_dataset.BOS_TOKEN]

            x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
            y = model.generate_test(x, train_dataset.itos, train_dataset.EOS_TOKEN, temperature=1.0, do_sample=True, top_k=40, adj_matrix=adj_matrix)[0]
            d = []
            for i in y[1:]:
                d.append(int(train_dataset.itos[int(i)]))

            
            # d = [int(train_dataset.itos[int(i)]) for i in y]
            syntehtic_links.append(d)
        except:
            pass
    
    file = open(config.system.work_dir+'/test_trajectories.txt','wb')
    pickle.dump(syntehtic_links,file)

        # with open(config.system.work_dir+'/test_trajectories.txt', 'a') as f:
        #     f.write(str(txt)+'\n')
    
    

    