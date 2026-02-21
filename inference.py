import torch
import argparse
import yaml
import numpy as np
import gymnasium as gym
import ale_py
import cv2
from scripts.utils.tensor_utils import normalize_observation, reshape_observation
from gymnasium.wrappers import AtariPreprocessing, ClipReward
from typing import List, Tuple
from scripts.utils.debug_utils import save_dream_video
from scripts.utils.tensor_utils import env_n_actions
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from scripts.models.categorical_vae.sampler import sample
from scripts.models.agent.actor import Actor
from torch.distributions import OneHotCategorical


def collect_steps(env_name:str, 
                  frameskip:int, 
                  noop_max:int, 
                  observation_height_width:int, 
                  episodic_life:bool, 
                  min_reward:float, 
                  max_reward:float,  
                  context_length:int, 
                  env_actions:int, 
                  device:str, 
                  batch_size:int) -> Tuple[torch.Tensor, torch.Tensor]:
    
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
    all_episode_starts = []

    observation, info = env.reset()
    observation = reshape_observation(normalize_observation(observation=observation))
    episode_start = True
    for _ in range(context_length):
        all_observations.append(observation)
        
        act = env.action_space.sample()
        act_one_hot = np.zeros(env_actions, dtype=np.float32)
        act_one_hot[act] = 1.0
        
        all_actions.append(act_one_hot)
        
        obs, _, terminated, truncated, _ = env.step(act)
        if terminated or truncated:
            obs, _ = env.reset()

        observation = reshape_observation(normalize_observation(observation=obs))
            
    env.close()

    observations = torch.from_numpy(np.stack(all_observations)).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
    actions = torch.from_numpy(np.stack(all_actions)).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    
    return observations, actions


def dream(xlstm_dm:XLSTM_DM, 
          decoder:CategoricalDecoder, 
          tokenizer:Tokenizer,
          tokens:torch.Tensor, 
          imagination_horizon:int, 
          latent_dim:int, 
          codes_per_latent:int, 
          batch_size:int, 
          env_actions:int,
          device:str, 
          actor:Actor) -> Tuple:
    
    imagined_frames = []
    imagined_actions = []
    imagined_latents = []
    imagined_rewards = []
    imagined_terminations = []
    hidden_states = []

    for step in range(imagination_horizon):
        latent, reward, termination, hidden_state = xlstm_dm.forward(tokens_batch=tokens)
        imagined_rewards.append(reward[:, -1, :])
        imagined_terminations.append(termination[:, -1, :])
        hidden_states.append(hidden_state[:, -1, :])

        next_latent = latent[:, -1:, :].view(batch_size, 1, latent_dim, codes_per_latent)
        next_latent_sample = sample(latents_batch=next_latent, batch_size=batch_size, sequence_length=1)

        decoded_latent = decoder.forward(latents_batch=next_latent_sample, 
                                        batch_size=batch_size, 
                                        sequence_length=1, 
                                        latent_dim=latent_dim, 
                                        codes_per_latent=codes_per_latent).squeeze(1).cpu().numpy()
        
        imagined_frames.append(decoded_latent)

        imagined_latents.append(next_latent_sample)

        #
        flattened_latent = next_latent_sample.view(batch_size, -1)
        current_hidden = hidden_state[:, -1, :]
        env_state = torch.cat([flattened_latent, current_hidden], dim=-1)

        action_logits = actor.forward(state=env_state)
        policy = OneHotCategorical(logits=action_logits)
        next_action = policy.sample()
        
        imagined_actions.append(next_action)

        next_token = tokenizer.forward(latents_sampled_batch=next_latent_sample, actions_batch=next_action.unsqueeze(dim=1))
        tokens = torch.cat([tokens[:, 1:], next_token], dim=1)

    return imagined_frames, imagined_latents, imagined_rewards, imagined_terminations, hidden_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', default='config/train_wm.yaml', type=str, help='Path to env parameters .yaml file')
    parser.add_argument('--env_cfg', default='config/env.yaml', type=str, help='Path to env parameters .yaml file')
    parser.add_argument('--inference_cfg', default='config/inference.yaml', type=str, help='Path to inference hyperparameters .yaml file')
    args = parser.parse_args()

    with open(args.train_cfg, 'r') as file_train, open(args.env_cfg, 'r') as file_env, open(args.inference_cfg, 'r') as file_inference:
        train_cfg = yaml.safe_load(file_train)['train_wm']
        inference_cfg = yaml.safe_load(file_inference)['inference']
        env_cfg = yaml.safe_load(file_env)['env']
    
    ENV_NAME = env_cfg['env_name']
    WEIGHTS_PATH = inference_cfg['weights_path']
    VIDEO_PATH = f'output/videos/dream/dream_{ENV_NAME}.mp4'
    FPS = inference_cfg['fps']
    BATCH_SIZE = inference_cfg['batch_size']
    CONTEXT_LENGTH = inference_cfg['context_length']
    IMAGINATION_HORIZON = inference_cfg['imagination_horizon']
    LATENT_DIM = train_cfg['latent_dim']
    CODES_PER_LATENT = train_cfg['codes_per_latent']
    DEVICE = 'cuda'
    ENV_ACTIONS = env_n_actions(env_name=ENV_NAME)
    EMBEDDING_DIM = train_cfg['embedding_dim']
    SEQUENCE_LENGTH = train_cfg['sequence_length']
    NUM_BLOCKS = train_cfg['num_blocks']
    SLSTM_AT = train_cfg['slstm_at']
    DROPOUT = train_cfg['dropout']
    ADD_POST_BLOCKS_NORM = train_cfg['add_post_blocks_norm']
    CONV1D_KERNEL_SIZE = train_cfg['conv1d_kernel_size']
    NUM_HEADS = train_cfg['num_heads']
    QKV_PROJ_BLOCKSIZE = train_cfg['qkv_proj_blocksize']
    BIAS_INIT = train_cfg['bias_init']
    PROJ_FACTOR = train_cfg['proj_factor']
    ACT_FN = train_cfg['act_fn']
    FRAMESKIP = env_cfg['frameskip']
    NOOP_MAX = env_cfg['noop_max']
    OBSERVATION_HEIGHT_WIDTH = env_cfg['observation_height_width']
    EPISODIC_LIFE = env_cfg['episodic_life']
    MIN_REWARD = env_cfg['min_reward']
    MAX_REWARD = env_cfg['max_reward']


    dataset = AtariDataset(sequence_length=CONTEXT_LENGTH)
    encoder = CategoricalEncoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    decoder = CategoricalDecoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    tokenizer = Tokenizer(latent_dim=LATENT_DIM, 
                          codes_per_latent=CODES_PER_LATENT, 
                          env_actions=ENV_ACTIONS, 
                          embedding_dim=EMBEDDING_DIM, 
                          sequence_length=SEQUENCE_LENGTH).to(DEVICE)
    xlstm_dm = XLSTM_DM(sequence_length=SEQUENCE_LENGTH, 
                        num_blocks=NUM_BLOCKS, 
                        embedding_dim=EMBEDDING_DIM, 
                        slstm_at=SLSTM_AT, 
                        dropout=DROPOUT, 
                        add_post_blocks_norm=ADD_POST_BLOCKS_NORM, 
                        conv1d_kernel_size=CONV1D_KERNEL_SIZE, 
                        qkv_proj_blocksize=QKV_PROJ_BLOCKSIZE, 
                        num_heads=NUM_HEADS, 
                        latent_dim=LATENT_DIM, 
                        codes_per_latent=CODES_PER_LATENT, 
                        bias_init=BIAS_INIT, 
                        proj_factor=PROJ_FACTOR, 
                        act_fn=ACT_FN).to(DEVICE)
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    def clean_state_dict(sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

    encoder.load_state_dict(clean_state_dict(checkpoint['encoder']))
    decoder.load_state_dict(clean_state_dict(checkpoint['decoder']))
    tokenizer.load_state_dict(clean_state_dict(checkpoint['tokenizer']))
    xlstm_dm.load_state_dict(clean_state_dict(checkpoint['dynamics']))

    encoder.eval()
    decoder.eval()
    tokenizer.eval()
    xlstm_dm.eval()
    
    observations, actions = collect_steps(env_name=ENV_NAME, 
                                          frameskip=FRAMESKIP, 
                                          noop_max=NOOP_MAX, 
                                          observation_height_width=OBSERVATION_HEIGHT_WIDTH, 
                                          episodic_life=EPISODIC_LIFE, 
                                          min_reward=MIN_REWARD, 
                                          max_reward=MAX_REWARD, 
                                          context_length=CONTEXT_LENGTH, 
                                          env_actions=ENV_ACTIONS, 
                                          device=DEVICE, 
                                          batch_size=BATCH_SIZE)

    with torch.no_grad():
        latents = encoder.forward(observations_batch=observations, 
                                  batch_size=BATCH_SIZE, 
                                  sequence_length=CONTEXT_LENGTH, 
                                  latent_dim=LATENT_DIM, 
                                  codes_per_latent=CODES_PER_LATENT)
        
        latents_sampled_batch = sample(latents, batch_size=BATCH_SIZE, sequence_length=CONTEXT_LENGTH)

        tokens = tokenizer.forward(latents_sampled_batch=latents_sampled_batch, actions_batch=actions)

        imagined_frames, _, imagined_rewards, imagined_terminations, _ = dream(xlstm_dm=xlstm_dm, 
                                                                               decoder=decoder,
                                                                               tokenizer=tokenizer,
                                                                               tokens=tokens, 
                                                                               imagination_horizon=IMAGINATION_HORIZON, 
                                                                               latent_dim=LATENT_DIM, 
                                                                               codes_per_latent=CODES_PER_LATENT, 
                                                                               batch_size=BATCH_SIZE, 
                                                                               env_actions=ENV_ACTIONS, 
                                                                               device=DEVICE)
        
        # save_dream_video(imagined_frames=imagined_frames, video_path=VIDEO_PATH, fps=FPS)
        save_dream_video(imagined_frames=imagined_frames, 
                         imagined_rewards=imagined_rewards, 
                         imagined_terminations=imagined_terminations, 
                         video_path=VIDEO_PATH, 
                         fps=FPS)