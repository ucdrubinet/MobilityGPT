from .base_finetuner import BaseFinetuner
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset
from mobilitygpt.dpo_trainer import DPOTrainer

class DPOFinetuner(BaseFinetuner):
    def __init__(self, config, model, dataset, gravity_sampling=False, prompt_size=4):
        super().__init__(config, model, dataset, gravity_sampling)
        self.prompt_size = prompt_size
        
    def train(self, train_data, val_data=None, reference_model=None):
        """
        Train the model using DPO (Direct Preference Optimization)
        
        Args:
            train_data: List of preference data examples
            val_data: Optional validation data
            reference_model: Reference model for DPO training
        """
        if reference_model is None:
            # Clone the current model state as reference model
            reference_model = type(self.model)(self.model.config)
            reference_model.load_state_dict(self.model.state_dict())
            reference_model.eval()  # Reference model should be in eval mode
            
        trainer = DPOTrainer(
            config=self.config.trainer,
            model=self.model,
            reference_model=reference_model,
            train_dataset=train_data,
            val_sampler=None if val_data is None else self.create_data_samplers()[1]
        )
        
        def batch_end_callback(trainer):
            if trainer.iter_num % 10 == 0:
                print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: "
                      f"train loss {trainer.loss.item():.5f} "
                      f"chosen reward {trainer.chosen_reward.item():.5f} "
                      f"rejected reward {trainer.rejected_reward.item():.5f}")

        trainer.set_callback('on_batch_end', batch_end_callback)
        trainer.run()
        
    def create_dpo_dataset(self, num_samples):
        print("Creating DPO dataset")
        preference_data = []
        
        for _ in tqdm(range(num_samples)):
            traj = random.sample(self.dataset.trajs, 1)[0]
            traj_first_n = traj[:self.prompt_size]
            
            x = torch.tensor([self.dataset.stoi[s] for s in traj_first_n], dtype=torch.long)[None,...].to('cuda')
            
            y = self.model.generate_test(x, self.dataset.itos, self.dataset.EOS_TOKEN, 
                                       temperature=1.0, do_sample=True, top_k=None)[0]
            candidate_0 = [self.dataset.itos[int(i)] for i in y]

            prompt_tokens = torch.tensor([self.dataset.stoi[s] for s in traj_first_n], 
                                       dtype=torch.long)[None,...].to('cuda')
            chosen_response_tokens = torch.tensor([self.dataset.stoi[s] for s in traj], 
                                                dtype=torch.long)[None,...].to('cuda')
            rejected_response_tokens = torch.tensor([self.dataset.stoi[s] for s in candidate_0], 
                                                  dtype=torch.long)[None,...].to('cuda')

            dataset_example = self._create_dpo_example(
                prompt_tokens, chosen_response_tokens, rejected_response_tokens
            )
            preference_data.append(dataset_example)
        
        return preference_data
    
    def _create_dpo_example(self, prompt_tokens, chosen_response_tokens, rejected_response_tokens):
        prompt_chosen_tokens = torch.cat([prompt_tokens, chosen_response_tokens], dim=1)
        prompt_rejected_tokens = torch.cat([prompt_tokens, rejected_response_tokens], dim=1)

        chosen_loss_mask = torch.cat(
            [torch.zeros(prompt_tokens.shape), torch.ones(chosen_response_tokens.shape)], dim=1
        )
        rejected_loss_mask = torch.cat(
            [torch.zeros(prompt_tokens.shape), torch.ones(rejected_response_tokens.shape)], dim=1
        )

        return {
            'prompt_chosen_tokens': prompt_chosen_tokens.squeeze(),
            'prompt_rejected_tokens': prompt_rejected_tokens.squeeze(),
            'chosen_loss_mask': chosen_loss_mask.squeeze(),
            'rejected_loss_mask': rejected_loss_mask.squeeze()
        }