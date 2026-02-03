import torch
from .tokenizer import Tokenizer


def tokenize(tokenizer:Tokenizer, latents_sampled_batch:torch.Tensor, actions_batch:torch.Tensor) -> torch.Tensor:
    tokens = tokenizer.forward(latents_sampled_batch=latents_sampled_batch, actions_batch=actions_batch)
    return tokens