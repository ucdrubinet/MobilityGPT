import os
import torch
from mobilitygpt.ppo_policy_trainer import PolicyTrainer, Agent
from .base_finetuner import BaseFinetuner
from datasets.prompt_dataset import PromptDataset


class PPOFinetuner(BaseFinetuner):
    """Finetuner for PPO training"""
    
    def __init__(self, config, model, dataset, reward_model, gravity_sampling=False, prompt_size=4):
        super().__init__(config, model, dataset, gravity_sampling)
        self.prompt_size = prompt_size
        self.reward_model = reward_model
        
    def train(self):
        """Train the model using PPO"""
        print("Starting PPO training")
        
        # Create prompt dataset
        prompt_dataset = PromptDataset(
            config=self.config.data,
            data=self.dataset.data,
            vocab=self.dataset.vocab,
            prompt_size=self.prompt_size
        )
        
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
        best_score = float('-inf')
        for epoch in range(self.config.policy.epochs):
            # Generate experience
            rollouts, score = trainer.make_experience(
                policy_model, 
                ref_model,
                self.config.policy.num_rollouts
            )
            
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
                self._save_checkpoint(policy_model.model)
                print("Best model saved")
                
    def _save_checkpoint(self, model):
        """Save model checkpoint"""
        ckpt_path = os.path.join(self.config.system.work_dir, "model_ppo.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
    
    