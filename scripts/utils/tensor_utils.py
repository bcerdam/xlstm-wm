import torch
import numpy as np
from ..data_related.atari_dataset import AtariDataset
from torch.utils.data import default_collate


def normalize_observation(observation:np.ndarray) -> np.ndarray:
    normalized_observation = (observation.astype(np.float32)/127.5)-1.0
    return normalized_observation


def reshape_observation(observation:np.ndarray) -> np.ndarray:
    reshaped_observation = np.moveaxis(observation, -1, 0)
    return reshaped_observation


def random_replay_batch(atari_dataset:AtariDataset, batch_size:int, sequence_length:int, device:str) -> torch.Tensor:
    indices = torch.randint(high=len(atari_dataset)-sequence_length, size=(batch_size,))
    batch = default_collate([atari_dataset[i] for i in indices])
    return [item.to(device) for item in batch]