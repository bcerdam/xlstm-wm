import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from torch.distributions import OneHotCategorical
from torch.distributions.kl import kl_divergence
from typing import Tuple
from .xlstm_dm import XLSTM_DM
from ..categorical_vae.sampler import latent_unimix


def dm_fwd_step(dynamics_model:XLSTM_DM, 
                posterior:torch.Tensor,
                tokens_batch:torch.Tensor, 
                rewards_batch:torch.Tensor, 
                terminations_batch:torch.Tensor, 
                batch_size:int, 
                sequence_length:int,
                latent_dim:int, 
                codes_per_latent:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    dynamics_model.train()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        next_latents_pred, rewards_pred, terminations_pred, hidden_state = dynamics_model.forward(tokens_batch=tokens_batch)

        next_latents_pred = next_latents_pred.view(size=(batch_size, sequence_length, latent_dim, codes_per_latent))
        prior = latent_unimix(latents_batch=next_latents_pred, uniform_mixture_percentage=0.01)
        # latents_batch = latents_batch.view(batch_size, sequence_length, latent_dim, codes_per_latent)

        rewards_loss = mse_loss(input=rewards_pred[:, :-1].squeeze(dim=-1), target=rewards_batch[:, 1:].float())
        terminations_loss = binary_cross_entropy_with_logits(input=terminations_pred[:, :-1].squeeze(dim=-1), target=terminations_batch[:, 1:].float())

        post_logits = posterior[:, 1:]
        prior_logits = prior[:, :-1]

        dynamics_kl = kl_divergence(OneHotCategorical(logits=post_logits.detach()), OneHotCategorical(logits=prior_logits))
        dynamics_loss = torch.clamp(dynamics_kl.sum(dim=-1).mean(), min=1.0)

        representation_kl = kl_divergence(OneHotCategorical(logits=post_logits), OneHotCategorical(logits=prior_logits.detach()))
        representation_loss = torch.clamp(representation_kl.sum(dim=-1).mean(), min=1.0)

    return rewards_loss, terminations_loss, dynamics_loss, representation_loss