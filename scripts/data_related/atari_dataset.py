import os
import h5py
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset


class AtariDataset(Dataset):
    def __init__(self, sequence_length:int) -> None:

        self.sequence_length = sequence_length
        self.valid_indices = np.array([], dtype=np.int64)

        self.observations = None
        self.actions = None
        self.rewards = None
        self.terminations = None
        self.episode_starts = None


    def update(self, observations: np.ndarray, 
                     actions: np.ndarray, 
                     rewards: np.ndarray, 
                     terminations: np.ndarray, 
                     episode_starts: np.ndarray) -> None:
        
        if self.observations is None:
            self.observations = observations
            self.actions = actions
            self.rewards = rewards
            self.terminations = terminations
            self.episode_starts = episode_starts
        else:
            self.observations = np.concatenate([self.observations, observations], axis=0)
            self.actions = np.concatenate([self.actions, actions], axis=0)
            self.rewards = np.concatenate([self.rewards, rewards], axis=0)
            self.terminations = np.concatenate([self.terminations, terminations], axis=0)
            self.episode_starts = np.concatenate([self.episode_starts, episode_starts], axis=0)

        total_samples = self.observations.shape[0]
        invalid_mask = np.zeros(total_samples, dtype=bool)
        start_indices = np.where(self.episode_starts)[0]

        for t in start_indices:
            start_range = max(0, t - self.sequence_length + 1)
            end_range = t
            invalid_mask[start_range : end_range] = True

        invalid_mask[-self.sequence_length + 1:] = True
        all_indices = np.arange(total_samples)
        self.valid_indices = all_indices[~invalid_mask]


    def __len__(self):
        return len(self.valid_indices)
    

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        real_idx = self.valid_indices[index]
        end_index = real_idx+self.sequence_length
        return (self.observations[real_idx:end_index], 
                self.actions[real_idx:end_index], 
                self.rewards[real_idx:end_index], 
                self.terminations[real_idx:end_index])