import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, ClipReward


def gather_steps(env_name: str, 
                 env_steps_per_epoch: int,
                 epsilon_greedy: float,
                 frameskip: int,
                 noop_max: int,
                 episodic_life: bool,
                 min_reward: float,
                 max_reward: float,
                 observation_height_width: int) -> None:
    
    gym.register_envs(ale_py)
    env = gym.make(id=env_name, max_episode_steps=env_steps_per_epoch, frameskip=1)
    env = AtariPreprocessing(env=env, 
                             noop_max=noop_max, 
                             frame_skip=frameskip, 
                             screen_size=observation_height_width, 
                             terminal_on_life_loss=episodic_life, 
                             grayscale_obs=False)
    env = ClipReward(env=env, min_reward=min_reward, max_reward=max_reward)

    observation, info = env.reset()
    for step in range(env_steps_per_epoch):
        random_action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(random_action)
