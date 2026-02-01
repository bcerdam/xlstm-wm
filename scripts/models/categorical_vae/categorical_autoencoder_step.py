import torch
import torch.nn.functional as F
from .encoder import CategoricalEncoder
from .decoder import CategoricalDecoder
from .encoder_fwd_pass import forward_pass_encoder
from .decoder_fwd_pass import forward_pass_decoder
from .sampler import sample


def autoencoder_step(categorical_encoder:CategoricalEncoder, 
                     categorical_decoder:CategoricalDecoder, 
                     observations_batch:torch.Tensor, 
                     batch_size:int, 
                     sequence_length:int, 
                     latent_dim:int, 
                     codes_per_latent:int,
                     optimizer:torch.optim.Optimizer, 
                     scaler:torch.amp.GradScaler) -> float:
    
    categorical_encoder.train()
    categorical_decoder.train()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        latents_batch = forward_pass_encoder(categorical_encoder=categorical_encoder, 
                                            observations_batch=observations_batch, 
                                            batch_size=batch_size, 
                                            sequence_length=sequence_length, 
                                            latent_dim=latent_dim, 
                                            codes_per_latent=codes_per_latent)        

        latents_sampled_batch = sample(latents_batch=latents_batch)

        reconstructed_observations_batch = forward_pass_decoder(categorical_decoder=categorical_decoder, 
                                                                latents_batch=latents_sampled_batch, 
                                                                batch_size=batch_size, 
                                                                sequence_length=sequence_length, 
                                                                latent_dim=latent_dim, 
                                                                codes_per_latent=codes_per_latent)
        
        reconstruction_loss = F.mse_loss(reconstructed_observations_batch, observations_batch)
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(reconstruction_loss).backward()
    scaler.unscale_(optimizer)
    
    torch.nn.utils.clip_grad_norm_(categorical_encoder.parameters(), 1000.0)
    torch.nn.utils.clip_grad_norm_(categorical_decoder.parameters(), 1000.0)
    
    scaler.step(optimizer)
    scaler.update()

    return reconstruction_loss.item(), latents_sampled_batch