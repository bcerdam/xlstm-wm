import numpy as np


def normalize_observation(observation):
    normalized_observation = (observation.astype(np.float32)/127.5)-1.0
    return normalized_observation


def reshape_observation(observation):
    reshaped_observation = np.moveaxis(observation, -1, 0)
    return reshaped_observation