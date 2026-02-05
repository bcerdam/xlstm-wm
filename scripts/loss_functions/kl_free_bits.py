import torch
import torch.nn as nn
from typing import Tuple
from torch.distributions import  OneHotCategorical, kl_divergence


def dynamics_representation_loss(free_bits:int, 
                                 next_latent_gt:torch.Tensor, 
                                 next_latent_pred:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
    next_latent_gt_distribution = OneHotCategorical(logits=next_latent_gt)
    next_latent_pred_distribution = OneHotCategorical(logits=next_latent_pred)
    mean_kl_div = kl_divergence(p=next_latent_gt_distribution, q=next_latent_pred_distribution).sum(dim=-1).mean()
    loss_value = torch.max(torch.ones_like(mean_kl_div)*free_bits, mean_kl_div)
    return loss_value, mean_kl_div