# 2. Encode observations, sample from them
# 3. Tokenize sampled latent and actions
# 4. Reuse inference -> dream function to gather bath of imagined data (frame, reward, termination)
# 5. Encode imagined frames, sample from them
# 6. Concat imagined latent and hidden state
# 7. Using data from 6., train the agent. (Lambda returns, Loss functions, then EMA)

import torch
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from scripts.models.categorical_vae.sampler import sample
from inference import dream


def train_agent(replay_buffer_path:str, 
                context_length:int, 
                imagination_horizon:int, 
                imagination_batch_size:int, 
                env_actions:int, 
                latent_dim:int, 
                codes_per_latent:int,  
                encoder:CategoricalEncoder, 
                decoder:CategoricalDecoder, 
                tokenizer:Tokenizer, 
                xlstm_dm:XLSTM_DM, 
                device:str, 
                decode:bool) -> Tuple:
    
    dataset = AtariDataset(replay_buffer_path=replay_buffer_path, sequence_length=context_length)
    dataloader = DataLoader(dataset=dataset, batch_size=imagination_batch_size, shuffle=True)

    observation_batch, action_batch, reward_batch, termination_batch = next(iter(dataloader))
    observation_batch = observation_batch.to(device)
    action_batch = action_batch.to(device)
    reward_batch = reward_batch.to(device)
    termination_batch = termination_batch.to(device)
    current_batch_size = observation_batch.shape[0]

    with torch.no_grad():
        latent_batch = encoder(observations_batch=observation_batch, 
                               batch_size=current_batch_size, 
                               sequence_length=context_length, 
                               latent_dim=latent_dim, 
                               codes_per_latent=codes_per_latent)
        
        latent_sampled_batch = sample(latents_batch=latent_batch, 
                                      batch_size=current_batch_size, 
                                      sequence_length=context_length)
        # [1024, 8, 512] -> [1024, 16, 32, 32], [1024, 16, 1], [1024, 16], [1024, 16, 512]
        tokens_batch = tokenizer.forward(latents_sampled_batch=latent_sampled_batch, actions_batch=action_batch)

        _, imagined_latent, imagined_reward, imagined_termination, hidden_state = dream(xlstm_dm=xlstm_dm, 
                                                                                        decoder=decoder, 
                                                                                        tokenizer=tokenizer, 
                                                                                        tokens=tokens_batch, 
                                                                                        imagination_horizon=imagination_horizon, 
                                                                                        latent_dim=latent_dim, 
                                                                                        codes_per_latent=codes_per_latent, 
                                                                                        batch_size=current_batch_size, 
                                                                                        env_actions=env_actions, 
                                                                                        device=device, 
                                                                                        decode=decode)
        
        imagined_latent = torch.cat(imagined_latent, dim=1)
        imagined_reward = torch.stack(imagined_reward, dim=1)
        imagined_termination = torch.stack(imagined_termination, dim=1)
        hidden_state = torch.stack(hidden_state, dim=1) 
        
        print(imagined_latent.shape, imagined_reward.shape, imagined_termination.shape, hidden_state.shape)
        

        

        


