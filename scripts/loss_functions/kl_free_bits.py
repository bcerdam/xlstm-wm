import torch
import torch.nn as nn
from typing import Tuple
from torch.distributions import  OneHotCategorical, kl_divergence


def dynamics_representation_loss(free_bits:int, 
                                 next_latent_gt:torch.Tensor, 
                                 next_latent_pred:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    
    next_latent_gt_distribution = OneHotCategorical(logits=next_latent_gt)
    next_latent_pred_distribution = OneHotCategorical(logits=next_latent_pred)
    kl_div = kl_divergence(p=next_latent_gt_distribution, q=next_latent_pred_distribution).sum(dim=-1)
    kl_div = kl_div.mean()
    real_kl_div = kl_div
    kl_div = torch.max(torch.ones_like(kl_div)*free_bits, kl_div)
    return kl_div, real_kl_div
    # loss_value = torch.max(torch.ones_like(kl_div)*free_bits, kl_div)
    # return loss_value.mean(), kl_div.mean()



# class CategoricalKLDivLossWithFreeBits(nn.Module):
#     def __init__(self, free_bits) -> None:
#         super().__init__()
#         self.free_bits = free_bits

#     def forward(self, p_logits, q_logits):
#         p_dist = OneHotCategorical(logits=p_logits)
#         q_dist = OneHotCategorical(logits=q_logits)
#         kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
#         kl_div = reduce(kl_div, "B L D -> B L", "sum")
#         kl_div = kl_div.mean()
        # real_kl_div = kl_div
        # kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        # return kl_div, real_kl_div
