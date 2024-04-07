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
from opacus import PrivacyEngine 

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

    def __init__(self, config, model, train_dataset, train_sampler, val_sampler, DP=False):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.val_loss = None
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.DP = DP

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

        if self.DP:
            privacy_engine = PrivacyEngine()
            model, self.optimizer, train_loader = privacy_engine.make_private(
                module = model,
                optimizer = self.optimizer,
                data_loader = train_loader,
                max_grad_norm = 1.0,
                noise_multiplier = 1.0,
                )
            print("Training model with DP...")

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

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
