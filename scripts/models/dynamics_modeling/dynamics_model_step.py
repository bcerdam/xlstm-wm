import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, cross_entropy
from typing import Tuple
from .xlstm_dm import XLSTM_DM
from scripts.models.categorical_vae.sampler import latent_unimix


def dm_fwd_step(dynamics_model:XLSTM_DM, 
                latents_batch:torch.Tensor,
                tokens_batch:torch.Tensor, 
                rewards_batch:torch.Tensor, 
                terminations_batch:torch.Tensor, 
                batch_size:int, 
                sequence_length:int,
                latent_dim:int, 
                codes_per_latent:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    dynamics_model.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        next_latents_pred, rewards_pred, terminations_pred = dynamics_model.forward(tokens_batch=tokens_batch)

        next_latents_pred = next_latents_pred.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))
        latents_batch = latents_batch.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))

        print(next_latents_pred.min(), next_latents_pred.max())

        rewards_loss = mse_loss(input=rewards_pred.squeeze(dim=-1), target=rewards_batch.float())

        terminations_loss = binary_cross_entropy_with_logits(input=terminations_pred.squeeze(dim=-1), target=terminations_batch.float())

        dynamics_loss = cross_entropy(input=next_latents_pred[:, :-1], target=latents_batch[:, 1:])
        
    return rewards_loss, terminations_loss, dynamics_loss