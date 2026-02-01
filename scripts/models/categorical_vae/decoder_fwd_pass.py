import torch
from .decoder import CategoricalDecoder

def forward_pass_decoder(categorical_decoder:CategoricalDecoder, 
                         latents_batch: torch.Tensor, 
                         batch_size:int, 
                         sequence_length:int, 
                         latent_dim:int, 
                         codes_per_latent:int) -> torch.Tensor:

    latents_batch = latents_batch.view(batch_size*sequence_length, latent_dim, codes_per_latent)
    upscaled_features = categorical_decoder.forward(latent=latents_batch)
    return upscaled_features.view(batch_size, sequence_length, 3, 64, 64)