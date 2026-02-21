import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, kl_div
from typing import Tuple
from .xlstm_dm import XLSTM_DM


def dm_fwd_step(dynamics_model:XLSTM_DM, 
                latents_batch:torch.Tensor,
                tokens_batch:torch.Tensor, 
                rewards_batch:torch.Tensor, 
                terminations_batch:torch.Tensor, 
                batch_size:int, 
                sequence_length:int,
                latent_dim:int, 
                codes_per_latent:int, 
                latents_sampled_batch:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    dynamics_model.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        next_latents_pred, rewards_pred, terminations_pred, hidden_state = dynamics_model.forward(tokens_batch=tokens_batch)

        next_latents_pred = next_latents_pred.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))
        latents_batch = latents_batch.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))

        rewards_loss = mse_loss(input=rewards_pred.squeeze(dim=-1), target=rewards_batch.float())

        terminations_loss = binary_cross_entropy_with_logits(input=terminations_pred.squeeze(dim=-1), target=terminations_batch.float())

        latents_sampled_batch = latents_sampled_batch.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))

        dynamics_kl = kl_div(input=latents_sampled_batch.detach(), target=next_latents_pred)
        dynamics_loss = torch.max(1, dynamics_kl)

        representation_kl = kl_div(input=latents_sampled_batch, target=next_latents_pred.detach())
        representation_loss = torch.max(1, representation_kl)
        
    return rewards_loss, terminations_loss, dynamics_loss, representation_loss