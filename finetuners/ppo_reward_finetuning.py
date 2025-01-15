import os
import torch
from .base_finetuner import BaseFinetuner
from mobilitygpt.reward_trainer import RewardTrainer
from mobilitygpt.utils import CfgNode as CN


class PPORewardFinetuner(BaseFinetuner):
    """Finetuner for training the reward model used in PPO"""
    
    def __init__(self, config, model, dataset, gravity_sampling=False):
        super().__init__(config, model, dataset, gravity_sampling)

    def train(self, pairs_data, raw_data, vocab):
        """
        Train the reward model using pairwise preference data
        
        Args:
            pairs_data: List of dictionaries containing chosen/rejected trajectory pairs
            raw_data: Raw trajectory data
            vocab: Vocabulary for tokenization
        """        
        # Create train/val samplers
        train_sampler, val_sampler = self.create_data_samplers()
        
        # Initialize trainer
        trainer = RewardTrainer(
            config=self.config.trainer,
            model=self.model,
            train_dataset=self.dataset,
            train_sampler=train_sampler,
            val_sampler=val_sampler
        )
        
        # Add callbacks for logging and checkpointing
        def batch_end_callback(trainer):
            if trainer.iter_num % 10 == 0:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: "
                      f"train loss {trainer.loss.item():.5f}")
            if trainer.iter_num % 250 == 0:
                self._save_checkpoint(trainer)
                
        def validation_end_callback(trainer):
            print(f"Validation loss: {trainer.val_loss:.5f}")
                
        trainer.set_callback('on_batch_end', batch_end_callback)
        trainer.add_callback('validation_end', validation_end_callback)
        
        # Run training
        trainer.run()
        
    def _save_checkpoint(self, trainer):
        """Save model checkpoint"""
        ckpt_path = os.path.join(self.config.system.work_dir, "reward_model.pt")
        torch.save(trainer.model.state_dict(), ckpt_path)
        trainer.model.train()
        print(f"Saved checkpoint to {ckpt_path}")