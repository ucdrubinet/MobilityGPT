import torch
import os
import numpy as np
import pandas as pd
from mobilitygpt.model import GPT
from mobilitygpt.trainer import Trainer
from finetuners.supervised_finetuning import SupervisedFinetuner
from finetuners.dpo_finetuning import DPOFinetuner
from finetuners.ppo_reward_finetuning import PPORewardFinetuner
from finetuners.ppo_finetuning import PPOFinetuner
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

def train_base_model(model, dataset, config, adj_matrix=None):
    """Train the base model"""
    print("Starting base model training")
    
    # Create samplers
    train_sampler, val_sampler = create_samplers(dataset, config)
    
    # Initialize trainer
    trainer = Trainer(
        config=config.training,
        model=model,
        train_dataset=dataset,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        DP=config.training.dp_training,
        eps=config.training.dp_epsilon
    )
    
    # Add callbacks
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: "
                  f"train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 500 == 0:
            save_checkpoint(trainer.model, config, "model_pretrain.pt")
            
    def validation_end_callback(trainer):
        print(f"Validation loss: {trainer.val_loss:.5f}")
    
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.add_callback('validation_end', validation_end_callback)
    
    # Run training
    trainer.run()
    
    # Save final model
    save_checkpoint(model, config, "model_pretrain.pt")

def train_supervised(model, dataset, config):
    """Train with supervised finetuning"""
    print("Starting supervised finetuning")
    supervised_finetuner = SupervisedFinetuner(
        config=config,
        model=model,
        dataset=dataset,
        gravity_sampling=config.training.gravity_sampling,
        dp_epsilon=config.training.dp_epsilon
    )
    supervised_finetuner.train()
    save_checkpoint(model, config, "model_supervised.pt")

def train_ppo(model, dataset, config):
    """Train with PPO finetuning"""
    print("Starting PPO training")
    # First train reward model
    reward_finetuner = PPORewardFinetuner(
        config=config,
        model=model,
        dataset=dataset,
        gravity_sampling=config.training.gravity_sampling
    )
    pairs = reward_finetuner.create_comparison_dataset_ls(config)
    reward_finetuner.train(pairs)
    
    # Load trained reward model
    reward_model = GPT(config.model, reward_model=True)
    reward_model.load_state_dict(torch.load(f"{config.system.work_dir}/reward_model.pt"))
    reward_model.eval()
    
    # PPO training
    ppo_finetuner = PPOFinetuner(
        config=config,
        model=model,
        dataset=dataset,
        reward_model=reward_model,
        gravity_sampling=config.training.gravity_sampling,
        prompt_size=config.training.prompt_size
    )
    ppo_finetuner.train()
    save_checkpoint(model, config, "model_ppo.pt")

def train_dpo(model, dataset, config):
    """Train with DPO finetuning"""
    print("Starting DPO training")
    dpo_finetuner = DPOFinetuner(
        config=config,
        model=model,
        dataset=dataset,
        prompt_size=config.training.prompt_size
    )
    
    # Create and split dataset
    preference_data = dpo_finetuner.create_dpo_dataset(config.training.num_samples)
    train_size = int(0.8 * len(preference_data))
    train_data = preference_data[:train_size]
    val_data = preference_data[train_size:]
    
    # Create reference model
    reference_model = type(model)(config.model)
    reference_model.load_state_dict(model.state_dict())
    reference_model.eval()
    
    # Train
    dpo_finetuner.train(train_data, val_data, reference_model)
    save_checkpoint(model, config, "model_dpo.pt")
    
    # Save dataset
    torch.save(preference_data, os.path.join(config.system.work_dir, 'dpo_preference_dataset.pt'))

def create_samplers(dataset, config):
    """Create train and validation samplers"""
    if config.training.gravity_sampling:
        return create_gravity_samplers(dataset, config)
    return create_random_samplers(dataset, config)

def create_gravity_samplers(dataset, config):
    """Create samplers with gravity-based weighting"""
    gravity = pd.read_csv(f"{config.data.dataset}-Taxi/{config.data.dataset}_Taxi_trajectory_train_w_gravity.csv").gravity.values
    gravity = torch.Tensor(gravity)
    minimum, maximum = torch.min(gravity), torch.max(gravity)    
    gravity = (gravity-minimum)/(maximum-minimum) 
    
    num_trajs = len(dataset.trajs)
    indices = list(range(num_trajs))
    split = int(np.floor(config.training.validation_split * num_trajs))
    
    if config.training.shuffle_dataset:
        np.random.seed(config.training.random_seed)
        np.random.shuffle(indices)
        
    train_indices, val_indices = indices[split:], indices[:split]
    train_gravity, val_gravity = gravity[train_indices], gravity[val_indices]
    
    return (WeightedRandomSampler(train_gravity, len(train_gravity)),
            WeightedRandomSampler(val_gravity, len(val_gravity)))

def create_random_samplers(dataset, config):
    """Create samplers with random sampling"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.training.validation_split * dataset_size))
    
    if config.training.shuffle_dataset:
        np.random.seed(config.training.random_seed)
        np.random.shuffle(indices)
        
    train_indices, val_indices = indices[split:], indices[:split]
    
    return (SubsetRandomSampler(train_indices),
            SubsetRandomSampler(val_indices))

def save_checkpoint(model, config, filename):
    """Save model checkpoint"""
    ckpt_path = os.path.join(config.system.work_dir, filename)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}") 