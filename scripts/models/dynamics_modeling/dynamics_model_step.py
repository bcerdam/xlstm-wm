import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from typing import Tuple
from .xlstm_dm import XLSTM_DM
from scripts.loss_functions.kl_free_bits import dynamics_representation_loss
from scripts.models.categorical_vae.sampler import latent_unimix


def dm_step(dynamics_model:XLSTM_DM, 
            latents_batch:torch.Tensor,
            tokens_batch:torch.Tensor, 
            rewards_batch:torch.Tensor, 
            terminations_batch:torch.Tensor, 
            free_bits:int, 
            batch_size:int, 
            sequence_length:int,
            latent_dim:int, 
            codes_per_latent:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    dynamics_model.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        next_latents_pred, rewards_pred, terminations_pred = dynamics_model.forward(tokens_batch=tokens_batch)
        next_latents_pred = next_latents_pred.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))
        next_latents_pred_logits = latent_unimix(latents_batch=next_latents_pred, uniform_mixture_percentage=0.01)


        rewards_loss = mse_loss(input=rewards_pred.squeeze(dim=-1), target=rewards_batch.float())

        terminations_loss = binary_cross_entropy_with_logits(input=terminations_pred.squeeze(dim=-1), target=terminations_batch.float())

        dynamics_loss, dynamics_kl_div = dynamics_representation_loss(free_bits=free_bits, 
                                                                      next_latent_gt=latents_batch[:, 1:].detach(), 
                                                                      next_latent_pred=next_latents_pred_logits[:, :-1])
        
        representations_loss, representations_kl_div = dynamics_representation_loss(free_bits=free_bits, 
                                                                                    next_latent_gt=latents_batch[:, 1:], 
                                                                                    next_latent_pred=next_latents_pred_logits[:, :-1].detach())
        
    return rewards_loss, terminations_loss, dynamics_loss, representations_loss, dynamics_kl_div, representations_kl_div