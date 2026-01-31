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


    def __len__(self):
        return self.observations.shape[0]
    

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        end_index = index+self.sequence_length
        return (self.observations[index:end_index], 
                self.actions[index:end_index], 
                self.rewards[index:end_index], 
                self.terminations[index:end_index])