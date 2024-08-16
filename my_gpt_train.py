"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.functional import normalize

from mobilitygpt.model import GPT, get_lora_model
from mobilitygpt.trainer import Trainer
from mobilitygpt.utils import set_seed, setup_logging, CfgNode as CN
import random
import pickle
import pandas as pd
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm
# -----------------------------------------------------------------------------


def get_config(dataset):

    C = CN()
    
    C.device = 'cuda'

    # system
    C.system = CN()
    C.system.seed = 160
    C.system.work_dir = './Trajs_'+dataset+'_synthetic/chargpt_DP'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mobility'
    C.model.device = C.device 
    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-3 # the model we're using is so small that we can go a bit faster
    C.trainer.device = C.device 
    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 81
        C.max_length = 81
        return C

    def __init__(self, config, data, vocab):
        self.config = config
        self.EOS_TOKEN = '</S>'
        # self.BOS_TOKEN = '<S>'
        
        lines = data.strip().split('\n\n') 
        line_words = [[self.EOS_TOKEN]+l.strip().split(',')+[self.EOS_TOKEN] for l in lines]
        words = [item for sublist in line_words for item in sublist]
        origins = [s[1] for s in line_words]
        data_size, vocab_size = len(words), len(vocab)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        
        self.stoi[self.EOS_TOKEN] = len(vocab)
        self.itos[len(vocab)] = self.EOS_TOKEN
        self.vocab_size = vocab_size + 1 
    
        
        self.num_trajs = len(lines)
        self.data = words
        self.trajs = line_words
        self.origins = origins
        # self.max_length = max(len(t) for t in self.trajs)

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
    
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

    # Find the maximum index in the OD pair
    max_index = np.max(od_pair_list)
    
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = torch.zeros((max_index + 1, max_index + 1), dtype=int)
    
    # Set the elements in the adjacency matrix based on the OD pair
    for origin, destination in od_pair_list:
        adjacency_matrix[origin, destination] = 1

    # Add two new rows of ones and columns at the end of adjacency matrix.
    adjacency_matrix = torch.cat((adjacency_matrix, torch.ones(1, adjacency_matrix.size(0))), 0)
    adjacency_matrix = torch.cat((adjacency_matrix, torch.ones(adjacency_matrix.size(0), 1)), 1)

    # Return the adjacency matrix.
    return adjacency_matrix

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    
    dataset = "SF"
    
    model_load = False
    gravity_sampling = False
    adj = True
    lora = True

    lora_training = True
    random_trajs = False
    eps=10

    create_RL_dataset =False
    create_DPO_dataset = False
    num_samples = int(5e3)
    prompt_size = 4
    
    # split the dataset into a training and validation set
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    
    # assert create_DPO_dataset == True and create_RL_dataset == False, "Only one of the two datasets can be created at a time"
    
    # get default config and overrides from the command line, if any
    config = get_config(dataset)
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    if random_trajs:
        text = open('Trajs_'+dataset+'_random.txt', 'r').read() # don't worry we won't run out of file handles
    else:
        text = open('Trajs_'+dataset+'.txt', 'r').read() # don't worry we won't run out of file handles
        
    geo=pd.read_csv(dataset+'-Taxi/roadmap.geo')    
    geo_ids=geo['geo_id'].apply(str).tolist()    

    train_dataset = CharDataset(config.data, text, geo_ids)
    
    rel=pd.read_csv(dataset+'-Taxi/roadmap.rel')    
    rel['combined'] = rel.apply(lambda x: list([x['origin_id'], x['destination_id']]),axis=1)
    od_list=rel['combined'].tolist()
    if adj:
        adj_matrix=od_pair_to_adjacency_matrix(od_list)
        adj_matrix = adj_matrix.to(config.device)    
    else:
        adj_matrix = None    

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    DP=False

    if lora:
        config.model.lora_rank = 8
        config.model.lora_alpha = 16
        config.model.lora_dropout = 0.00

        model = GPT(config.model, adj_matrix = adj_matrix)
        
        if lora_training:

            ckpt_path = os.path.join(config.system.work_dir, "model_base.pt")
            
            state_dict = torch.load(ckpt_path)
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            model.load_state_dict(state_dict)
            print("Marking model as LoRA fine-tunable...")
            model = get_lora_model(model)
            print("Done.")
            DP=True
    else:
        model = GPT(config.model, adj_matrix = adj_matrix)

    # Creating data indices for training and validation splits:
##############
    if gravity_sampling:
        gravity = pd.read_csv(dataset+'-Taxi/'+dataset+'_Taxi_trajectory_train_w_gravity.csv').gravity.values
        #load gravity and normalize
        gravity = torch.Tensor(gravity)
        minimum, maximum = torch.min(gravity), torch.max(gravity)    
        gravity  = (gravity-minimum)/(maximum-minimum) 
        num_trajs = len(train_dataset.trajs)
        indices = list(range(num_trajs))
        split = int(np.floor(validation_split * num_trajs))
    
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
    
        train_gravity, val_gravity= gravity[train_indices], gravity[val_indices]
    
        train_sampler = WeightedRandomSampler(train_gravity, len(train_gravity))
        valid_sampler = WeightedRandomSampler(val_gravity, len(val_gravity))
    else:
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

    # construct the trainer object    
    trainer = Trainer(config.trainer, model, train_dataset, train_sampler=train_sampler, val_sampler=valid_sampler, DP = DP, eps = eps)
##############

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # print("saving model")
            model.eval()
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
        
        
    if (not create_RL_dataset) and (not create_DPO_dataset):    
        print('Creating synthetic trajectories')
        syntehtic_links=[]
        for i in tqdm(range(num_samples)):
            # context = random.sample(train_dataset.data,1)
            origin = random.sample(train_dataset.origins,1)[0]
            context = [train_dataset.EOS_TOKEN, origin]
            x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
            y = model.generate_test(x, train_dataset.itos, train_dataset.EOS_TOKEN, max_token = config.data.max_length, temperature=1.0, do_sample=True, top_k=None)[0]
            d = []
            for i in y[1:]:
                if train_dataset.itos[int(i)]==train_dataset.EOS_TOKEN:
                    break
                else:
                    d.append(int(train_dataset.itos[int(i)]))
            
            syntehtic_links.append(d)
        
        if lora_training:
            file = open(config.system.work_dir+'/test_trajectories_lora_eps'+str(eps)+'.txt','wb')
            pickle.dump(syntehtic_links,file)
        else:
            ckpt_path = os.path.join(config.system.work_dir, "model_base.pt")
            torch.save(model.state_dict(), ckpt_path)
            file = open(config.system.work_dir+'/test_trajectories.txt','wb')
            pickle.dump(syntehtic_links,file)

    elif create_RL_dataset:
        print('Creating RL dataset')
        preference_data=[]
        for i in tqdm(range(num_samples)):
    
            traj = random.sample(train_dataset.trajs,1)[0]
            traj_first_n = traj[:prompt_size]
            
            x = torch.tensor([train_dataset.stoi[s] for s in traj_first_n], dtype=torch.long)[None,...].to('cuda')        
            traj_length = geo[geo['geo_id'].isin([int(t) for t in traj[1:-1]])].length.sum()
    
            y = model.generate_test(x, train_dataset.itos, train_dataset.EOS_TOKEN, temperature=1.0, do_sample=True, top_k=None)[0]
            candidate_0 = [train_dataset.itos[int(i)] for i in y]                
            length_0 = geo[geo['geo_id'].isin([int(t) for t in candidate_0[1:]])].length.sum()
    
            y = model.generate_test(x, train_dataset.itos, train_dataset.EOS_TOKEN, temperature=1.0, do_sample=True, top_k=None)[0]
            candidate_1 = [train_dataset.itos[int(i)] for i in y]   
            length_1 = geo[geo['geo_id'].isin([int(t) for t in candidate_1[1:]])].length.sum()
            
            choice = np.argmin([abs(traj_length - length_0), abs(traj_length - length_1)])
            d = {'input': traj, 'candidate_0':candidate_0, 'candidate_1':candidate_1, 'choice':choice}
            preference_data.append(d)
        
        file = open(config.system.work_dir+'/preference_dataset_adj','wb')
        pickle.dump(preference_data,file)
    elif create_DPO_dataset:
        print("Creating DPO dataset")
        preference_data=[]
        for i in tqdm(range(num_samples)):
    
            traj = random.sample(train_dataset.trajs,1)[0]
            traj_first_n = traj[:prompt_size]
            
            x = torch.tensor([train_dataset.stoi[s] for s in traj_first_n], dtype=torch.long)[None,...].to('cuda')        
            traj_length = geo[geo['geo_id'].isin([int(t) for t in traj[1:-1]])].length.sum()
    
            y = model.generate_test(x, train_dataset.itos, train_dataset.EOS_TOKEN, temperature=1.0, do_sample=True, top_k=None)[0]
            candidate_0 = [train_dataset.itos[int(i)] for i in y]                
            # length_0 = geo[geo['geo_id'].isin([int(t) for t in candidate_0[1:]])].length.sum()
    
            # y = model.generate_test(x, train_dataset.itos, train_dataset.EOS_TOKEN, temperature=1.0, do_sample=True, top_k=None)[0]
            # candidate_1 = [train_dataset.itos[int(i)] for i in y]   
            # length_1 = geo[geo['geo_id'].isin([int(t) for t in candidate_1[1:]])].length.sum()
            
            # choice = np.argmin([abs(traj_length - length_0), abs(traj_length - length_1)])
            # chosen_response = candidate_0 if choice==0 else candidate_1
            # rejected_response = candidate_1 if choice==0 else candidate_0

            prompt_tokens = torch.tensor([train_dataset.stoi[s] for s in traj_first_n], dtype=torch.long)[None,...].to('cuda')
            chosen_response_tokens = torch.tensor([train_dataset.stoi[s] for s in traj], dtype=torch.long)[None,...].to('cuda')
            rejected_response_tokens = torch.tensor([train_dataset.stoi[s] for s in candidate_0], dtype=torch.long)[None,...].to('cuda')

            prompt_chosen_tokens = torch.cat([prompt_tokens, chosen_response_tokens], dim=1)
            prompt_rejected_tokens = torch.cat([prompt_tokens, rejected_response_tokens], dim=1)

            chosen_loss_mask = torch.cat(
                [torch.zeros(prompt_tokens.shape), torch.ones(chosen_response_tokens.shape)], dim=1
            )
            rejected_loss_mask = torch.cat(
                [torch.zeros(prompt_tokens.shape), torch.ones(rejected_response_tokens.shape)], dim=1
            )

            dataset_example = {
                'prompt_chosen_tokens': prompt_chosen_tokens.squeeze(),
                'prompt_rejected_tokens': prompt_rejected_tokens.squeeze(),
                'chosen_loss_mask': chosen_loss_mask.squeeze(),
                'rejected_loss_mask': rejected_loss_mask.squeeze()
            }

            preference_data.append(dataset_example)
        
        torch.save(preference_data, config.system.work_dir+'/dpo_preference_dataset.pt')




    