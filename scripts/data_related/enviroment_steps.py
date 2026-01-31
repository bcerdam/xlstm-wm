import numpy as np
import gymnasium as gym
import ale_py
from ..utils.tensor_utils import normalize_observation, reshape_observation
from gymnasium.wrappers import AtariPreprocessing, ClipReward
from typing import Tuple, List


def gather_steps(env_name: str, 
                 env_steps_per_epoch: int,
                 frameskip: int,
                 noop_max: int,
                 episodic_life: bool,
                 min_reward: float,
                 max_reward: float,
                 observation_height_width: int) -> Tuple[List[np.ndarray], List[np.int64], List[np.float64], List[np.bool]]:
    
    gym.register_envs(ale_py)
    env = gym.make(id=env_name, frameskip=1)
    env = AtariPreprocessing(env=env, 
                             noop_max=noop_max, 
                             frame_skip=frameskip, 
                             screen_size=observation_height_width, 
                             terminal_on_life_loss=episodic_life, 
                             grayscale_obs=False)
    env = ClipReward(env=env, min_reward=min_reward, max_reward=max_reward)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminations = [] 
    observation, info = env.reset()
    for step in range(env_steps_per_epoch):
        random_action = env.action_space.sample()
        observation, reward, termination, truncated, info = env.step(random_action)
        observation = reshape_observation(normalize_observation(observation=observation))

        all_observations.append(observation)
        all_actions.append(random_action)
        all_rewards.append(reward)
        all_terminations.append(termination)

        if termination or truncated:
            observation, info = env.reset()

    return all_observations, all_actions, all_rewards, all_terminations