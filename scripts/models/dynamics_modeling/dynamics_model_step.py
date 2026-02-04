import torch
from typing import Tuple
from .xlstm_dm import XLSTM_DM

def dm_step(dynamics_model:XLSTM_DM, 
            tokens_batch:torch.Tensor, 
            rewards_batch:torch.Tensor, 
            terminations_batch:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    dynamics_model.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        next_latents_pred, rewards_pred, terminations_pred = dynamics_model.forward(tokens_batch=tokens_batch)

        # To do: Loss functions

        # rewards_loss()
        # dynamics_loss(next_latents)
        # representations_loss(next_latents)
        
    # return reward_loss, termination_loss, dynamic_loss, representation_loss
    pass

