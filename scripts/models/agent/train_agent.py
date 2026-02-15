# 1. Gather random context
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


def train_agent(replay_buffer_path:str, 
                context_length:int, 
                imagination_batch_size:int) -> Tuple:
    
    dataset = AtariDataset(replay_buffer_path=replay_buffer_path, sequence_length=context_length)
    dataloader = DataLoader(dataset=dataset, batch_size=imagination_batch_size, shuffle=True)

    observation_batch, action_batch, reward_batch, termination_batch = next(iter(dataloader))
    
    print(observation_batch.shape)
    print(action_batch.shape)
    print(reward_batch.shape)
    print(termination_batch.shape)
    
    # idx = np.random.randint(0, len(dataset), size=imagination_batch_size)
    # observation_batch, action_batch, reward_batch, termination_batch = dataset[idx]
    # print(observation_batch.shape, action_batch.shape, reward_batch.shape, termination_batch.shape)
    # observations, actions, _, _ = dataset[idx] 
    # observations = torch.from_numpy(observations).unsqueeze(0).to(DEVICE)
    # actions = torch.from_numpy(actions).unsqueeze(0).to(DEVICE)