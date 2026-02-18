import os
import h5py
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset


class AtariDataset(Dataset):
    def __init__(self, replay_buffer_path:str, sequence_length:int) -> None:

        self.sequence_length= sequence_length
        self.valid_indices = np.array([], dtype=np.int64)

        self.observations = None
        self.actions = None
        self.rewards = None
        self.terminations = None
        self.episode_starts = None

        if replay_buffer_path and os.path.exists(replay_buffer_path):
             with h5py.File(replay_buffer_path, 'r') as f:
                self.update(
                    f['observations'][:], 
                    f['actions'][:], 
                    f['rewards'][:], 
                    f['terminations'][:], 
                    f['episode_starts'][:]
                )


    def update(self, observations: np.ndarray, 
                     actions: np.ndarray, 
                     rewards: np.ndarray, 
                     terminations: np.ndarray, 
                     episode_starts: np.ndarray) -> None:
        
        obs = np.array(observations)
        acts = np.array(actions)
        rews = np.array(rewards)
        terms = np.array(terminations)
        starts = np.array(episode_starts)

        if self.observations is None:
            self.observations = obs
            self.actions = acts
            self.rewards = rews
            self.terminations = terms
            self.episode_starts = starts
        else:
            self.observations = np.concatenate([self.observations, obs], axis=0)
            self.actions = np.concatenate([self.actions, acts], axis=0)
            self.rewards = np.concatenate([self.rewards, rews], axis=0)
            self.terminations = np.concatenate([self.terminations, terms], axis=0)
            self.episode_starts = np.concatenate([self.episode_starts, starts], axis=0)

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