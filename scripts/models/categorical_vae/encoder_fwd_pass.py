import torch
from .encoder import CategoricalEncoder

def forward_pass_encoder(categorical_encoder:CategoricalEncoder, 
                         observations_batch: torch.tensor, 
                         batch_size:int, 
                         sequence_length:int, 
                         latent_dim:int, 
                         codes_per_latent:int) -> torch.tensor:

    observations_batch = observations_batch.view(batch_size*sequence_length, 3, 64, 64)
    latent_features = categorical_encoder.forward(observation=observations_batch)
    return latent_features.view(batch_size, sequence_length, latent_dim, codes_per_latent)