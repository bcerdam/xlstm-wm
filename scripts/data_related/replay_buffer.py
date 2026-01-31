import os
import h5py
import numpy as np
from typing import List

def update_replay_buffer(replay_buffer_path:str, 
                         observations:List[np.ndarray], 
                         actions:List[np.int64], 
                         rewards:List[np.float64], 
                         terminations:List[np.bool]) -> None:
    
    if os.path.dirname(replay_buffer_path):
        os.makedirs(os.path.dirname(replay_buffer_path), exist_ok=True)

    data_dict = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'terminations': np.array(terminations)
    }

    with h5py.File(replay_buffer_path, 'a') as f:
        for key, val in data_dict.items():
            if key not in f:
                f.create_dataset(key, data=val, maxshape=(None, *val.shape[1:]))
            else:
                dset = f[key]
                dset.resize(dset.shape[0] + val.shape[0], axis=0)
                dset[-val.shape[0]:] = val