"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mobilitygpt.utils import CfgNode as CN
from scipy.spatial import distance
import pandas as pd
from tqdm import tqdm
import gc
import random

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'cuda'
        # dataloder parameters
        C.num_workers = 8
        # optimizer parameters
        C.max_iters = 3000
        C.batch_size = 64
        C.learning_rate = 1e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, train_sampler, val_sampler):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.val_loss = None
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

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

    def test_statistic(self, links, porto_geo, OD_test, length_test):
        OD_synth=[]
        length_synth=[]
        for link_ids in links:
            OD_synth.append(link_ids[0])
            OD_synth.append(link_ids[-1])
            length = porto_geo[porto_geo['geo_id'].isin(link_ids)].length.sum()
            length_synth.append(length)

            
        js_OD=distance.jensenshannon(OD_synth, OD_test)
        js_length=distance.jensenshannon(length_synth, length_test)
        return js_OD, js_length

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        val_loader = DataLoader(
            self.train_dataset,
            sampler=self.val_sampler,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        
        # calculate statistic for test data
        df_porto=pd.read_csv('Porto-Taxi/Porto_Taxi_trajectory_test.csv')
        samples=df_porto.sample(n=self.test_num_samples)
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

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # # add validation
            # if self.iter_num % 100 == 0:
            #     model.eval()
            #     running_vloss = 0.0
            #     # Disable gradient computation and reduce memory consumption.
            #     with torch.no_grad():
            #         for i, vdata in enumerate(val_loader):
            #             vinputs, vlabels = vdata
            #             vinputs, vlabels = vinputs.to(self.device), vlabels.to(self.device)
            #             voutputs, vloss = model(vinputs, vlabels)
            #             running_vloss += vloss.item()

            #     self.val_loss = running_vloss / (i + 1)
            #     print('LOSS train {} valid {}'.format(self.loss, self.val_loss ))
            #     model.train()
            #     self.trigger_callbacks('validation_end')

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break


            # # add statistical testing
            # if self.iter_num % 1000 == 0:
            #     try:
            #         start = time.time()
            #         syntehtic_links=[]
            #         for i in tqdm(range(self.test_num_samples)):
            #             # context = random.sample(train_dataset.data,1)
            #             origin = random.sample(self.train_dataset.origins,1)[0]
            #             context = [self.train_dataset.BOS_TOKEN, origin]
            #             x = torch.tensor([self.train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(self.device)
            #             y = model.generate_test(x, self.train_dataset.itos, self.train_dataset.EOS_TOKEN, temperature=0.2, do_sample=True, top_k=None, max_token=500)[0]
            #             d = []
            #             for i in y[1:]:
            #                 d.append(int(self.train_dataset.itos[int(i)]))

            #             # d = [int(train_dataset.itos[int(i)]) for i in y]
            #             syntehtic_links.append(d)
            #         js_OD, js_length = self.test_statistic(syntehtic_links, porto_geo, OD_test, length_test)
            #         print('test time: {}, JS OD {}, JS length {}'.format(time.time()-start, js_OD, js_length))
            #     except Exception as e:
            #         print(e)
            # model.train()
            # _ = gc.collect()