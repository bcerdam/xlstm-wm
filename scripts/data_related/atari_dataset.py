import h5py
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset


class AtariDataset(Dataset):
    def __init__(self, replay_buffer_path:str, sequence_length:int) -> None:
        self.sequence_length= sequence_length
        with h5py.File(replay_buffer_path, 'r') as replay_buffer_file:
            self.observations = replay_buffer_file['observations'][:]
            self.actions = replay_buffer_file['actions'][:]
            self.rewards = replay_buffer_file['rewards'][:]
            self.terminations = replay_buffer_file['terminations'][:]
            self.episode_starts = replay_buffer_file['episode_starts'][:]

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