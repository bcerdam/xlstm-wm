# 6. Concat imagined latent and hidden state
# 7. Using data from 6., train the agent. (Lambda returns, Loss functions, then EMA)

import torch
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from scripts.models.categorical_vae.sampler import sample


def dream(xlstm_dm:XLSTM_DM, 
          tokenizer:Tokenizer,
          tokens:torch.Tensor, 
          imagination_horizon:int, 
          latent_dim:int, 
          codes_per_latent:int, 
          batch_size:int, 
          env_actions:int,
          device:str) -> Tuple:
    
    imagined_latents = []
    imagined_rewards = []
    imagined_terminations = []
    hidden_states = []

    for step in range(imagination_horizon):
        latent, reward, termination, hidden_state = xlstm_dm.forward(tokens_batch=tokens)
        next_latent = latent[:, -1:, :].view(batch_size, 1, latent_dim, codes_per_latent)
        next_latent_sample = sample(latents_batch=next_latent, batch_size=batch_size, sequence_length=1)

        imagined_latents.append(next_latent_sample)
        imagined_rewards.append(reward[:, -1, :])
        imagined_terminations.append(termination[:, -1, :])
        hidden_states.append(hidden_state[:, -1, :])

        next_action = torch.zeros((batch_size, 1, env_actions), device=device)
        random_indices = torch.randint(0, env_actions, (batch_size,), device=device)
        next_action[torch.arange(batch_size), 0, random_indices] = 1.0

        next_token = tokenizer.forward(latents_sampled_batch=next_latent_sample, actions_batch=next_action)
        tokens = torch.cat([tokens[:, 1:], next_token], dim=1)

    imagined_latents = torch.cat(imagined_latents, dim=1)
    imagined_rewards = torch.stack(imagined_rewards, dim=1)
    imagined_terminations = torch.stack(imagined_terminations, dim=1)
    hidden_states = torch.stack(hidden_states, dim=1)    

    return imagined_latents, imagined_rewards, imagined_terminations, hidden_states


def train_agent(replay_buffer_path:str, 
                context_length:int, 
                imagination_horizon:int, 
                imagination_batch_size:int, 
                env_actions:int, 
                latent_dim:int, 
                codes_per_latent:int,  
                encoder:CategoricalEncoder, 
                tokenizer:Tokenizer, 
                xlstm_dm:XLSTM_DM, 
                device:str) -> Tuple:
    
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

        tokens_batch = tokenizer.forward(latents_sampled_batch=latent_sampled_batch, actions_batch=action_batch)

        imagined_latent, imagined_reward, imagined_termination, hidden_state = dream(xlstm_dm=xlstm_dm, 
                                                                                     tokenizer=tokenizer, 
                                                                                     tokens=tokens_batch, 
                                                                                     imagination_horizon=imagination_horizon, 
                                                                                     latent_dim=latent_dim, 
                                                                                     codes_per_latent=codes_per_latent, 
                                                                                     batch_size=current_batch_size, 
                                                                                     env_actions=env_actions, 
                                                                                     device=device)
        # [1024, 16, 3584]
        env_state = torch.concat([imagined_latent, hidden_state], dim=-1)

            

        

        


