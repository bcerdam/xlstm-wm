import torch
import argparse
import yaml
import os
import shutil
import lpips
import copy
from torch.utils.data import DataLoader
from scripts.data_related.enviroment_steps import gather_steps
from scripts.data_related.replay_buffer import update_replay_buffer
from scripts.data_related.atari_dataset import AtariDataset
from scripts.utils.tensor_utils import random_replay_batch, env_n_actions
from scripts.utils.debug_utils import plot_current_loss, save_checkpoint
from scripts.models.categorical_vae.categorical_autoencoder_step import autoencoder_fwd_step
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from scripts.models.dynamics_modeling.dynamics_model_step import dm_fwd_step
from scripts.models.dynamics_modeling.total_loss import total_loss_step
from scripts.models.agent.train_agent import train_agent
from scripts.models.agent.critic import Critic
from scripts.models.agent.actor import Actor

import warnings
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_wm_cfg', default='config/train_wm.yaml', type=str, help='Path to train wm parameters .yaml file')
    parser.add_argument('--train_agent_cfg', default='config/train_agent.yaml', type=str, help='Path to train agent parameters .yaml file')
    parser.add_argument('--env_cfg', default='config/env.yaml', type=str, help='Path to env parameters .yaml file')
    args = parser.parse_args()

    with open(args.train_wm_cfg, 'r') as file_train_wm, open(args.env_cfg, 'r') as file_env, open(args.train_agent_cfg, 'r') as file_train_agent:
        train_wm_cfg = yaml.safe_load(file_train_wm)['train_wm']

        train_agent_cfg = yaml.safe_load(file_train_agent)['train_agent']

        env_file_content = yaml.safe_load(file_env)
        env_cfg = env_file_content['env']
        dataset_cfg = env_file_content['dataset']

    if os.path.exists('data'):
            shutil.rmtree('data')

    if os.path.exists('output/logs'):
        shutil.rmtree('output/logs')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    REPLAY_BUFFER_PATH = dataset_cfg['replay_buffer_path']
    ENV_NAME = env_cfg['env_name']
    ENV_ACTIONS = env_n_actions(ENV_NAME)

    EPOCHS = train_wm_cfg['epochs']
    TRAINING_STEPS_PER_EPOCH = train_wm_cfg['training_steps_per_epoch']
    BATCH_SIZE = train_wm_cfg['batch_size']
    SEQUENCE_LENGTH = train_wm_cfg['sequence_length']
    WORLD_MODEL_LEARNING_RATE = train_wm_cfg['world_model_learning_rate']

    # autoencoder
    LATENT_DIM = train_wm_cfg['latent_dim']
    CODES_PER_LATENT = train_wm_cfg['codes_per_latent']

    # xlstm
    EMBEDDING_DIM = train_wm_cfg['embedding_dim']
    NUM_BLOCKS = train_wm_cfg['num_blocks']
    SLSTM_AT = train_wm_cfg['slstm_at']
    DROPOUT = train_wm_cfg['dropout']
    ADD_POST_BLOCKS_NORM = train_wm_cfg['add_post_blocks_norm']
    CONV1D_KERNEL_SIZE = train_wm_cfg['conv1d_kernel_size']
    QKV_PROJ_BLOCKSIZE = train_wm_cfg['qkv_proj_blocksize']
    NUM_HEADS = train_wm_cfg['num_heads']
    BIAS_INIT = train_wm_cfg['bias_init']
    PROJ_FACTOR = train_wm_cfg['proj_factor']
    ACT_FN = train_wm_cfg['act_fn']

    # actor critic agent
    IMAGINATION_BATCH_SIZE = train_agent_cfg['imagination_batch_size']
    CONTEXT_LENGTH = train_agent_cfg['context_length']
    IMAGINATION_HORIZON = train_agent_cfg['imagination_horizon']
    GAMMA = train_agent_cfg['gamma']
    LAMBDA = train_agent_cfg['lambda']
    NABLA = train_agent_cfg['entropy_coeff']
    EMA_SIGMA = train_agent_cfg['ema_sigma']
    AGENT_LEARNING_RATE = train_agent_cfg['learning_rate']

    categorical_encoder = CategoricalEncoder(latent_dim=LATENT_DIM, 
                                             codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    categorical_decoder = CategoricalDecoder(latent_dim=LATENT_DIM, 
                                             codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    tokenizer = Tokenizer(latent_dim=LATENT_DIM, 
                          codes_per_latent=CODES_PER_LATENT, 
                          env_actions=ENV_ACTIONS, 
                          embedding_dim=EMBEDDING_DIM, 
                          sequence_length=SEQUENCE_LENGTH).to(DEVICE)
    dynamics_model = XLSTM_DM(sequence_length=SEQUENCE_LENGTH, 
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
    lpips_model = lpips.LPIPS(net='alex').to(DEVICE).requires_grad_(False).eval()

    critic = Critic(latent_dim=LATENT_DIM, 
                    codes_per_latent=CODES_PER_LATENT, 
                    embedding_dim=EMBEDDING_DIM).to(DEVICE)
    
    ema_critic = copy.deepcopy(critic).requires_grad_(False).to(DEVICE)

    actor = Actor(latent_dim=LATENT_DIM, 
                  codes_per_latent=CODES_PER_LATENT, 
                  embedding_dim=EMBEDDING_DIM, 
                  env_actions=ENV_ACTIONS).to(DEVICE)

    OPTIMIZER = torch.optim.Adam(list(categorical_encoder.parameters()) + 
                                 list(categorical_decoder.parameters()) +
                                 list(tokenizer.parameters()) + 
                                 list(dynamics_model.parameters()),
                                 lr=WORLD_MODEL_LEARNING_RATE)
    
    AGENT_OPTIMIZER = torch.optim.Adam(list(critic.parameters()) +
                                       list(actor.parameters()),  
                                       lr=AGENT_LEARNING_RATE)

    SCALER = torch.amp.GradScaler(enabled=True)

    training_steps_finished = 0
    for epoch in range(EPOCHS):
        observations, actions, rewards, terminations, episode_starts = gather_steps(**env_cfg)
        update_replay_buffer(replay_buffer_path=REPLAY_BUFFER_PATH, 
                            observations=observations, 
                            actions=actions,
                            rewards=rewards, 
                            terminations=terminations, 
                            episode_starts=episode_starts)
        atari_dataset = AtariDataset(replay_buffer_path=REPLAY_BUFFER_PATH, sequence_length=SEQUENCE_LENGTH)
        agent_dataset = AtariDataset(replay_buffer_path=REPLAY_BUFFER_PATH, sequence_length=CONTEXT_LENGTH)
        agent_dataloader = DataLoader(dataset=agent_dataset, batch_size=IMAGINATION_BATCH_SIZE, shuffle=True)

        epoch_loss_history = []
        for step in range(TRAINING_STEPS_PER_EPOCH):
            observations_batch, actions_batch, rewards_batch, terminations_batch = random_replay_batch(atari_dataset=atari_dataset, 
                                                                                                       batch_size=BATCH_SIZE, 
                                                                                                       sequence_length=SEQUENCE_LENGTH,
                                                                                                       device=DEVICE)
            
            reconstruction_loss, latents_sampled_batch = autoencoder_fwd_step(categorical_encoder=categorical_encoder, 
                                                                              categorical_decoder=categorical_decoder, 
                                                                              observations_batch=observations_batch, 
                                                                              batch_size=BATCH_SIZE, 
                                                                              sequence_length=SEQUENCE_LENGTH, 
                                                                              latent_dim=LATENT_DIM, 
                                                                              codes_per_latent=CODES_PER_LATENT,
                                                                              lpips_loss_fn=lpips_model)
            
            tokens_batch = tokenizer.forward(latents_sampled_batch=latents_sampled_batch.detach(), actions_batch=actions_batch)
            
            rewards_loss, terminations_loss, dynamics_loss = dm_fwd_step(dynamics_model=dynamics_model,
                                                                         latents_batch=latents_sampled_batch, 
                                                                         tokens_batch=tokens_batch, 
                                                                         rewards_batch=rewards_batch, 
                                                                         terminations_batch=terminations_batch, 
                                                                         batch_size=BATCH_SIZE, 
                                                                         sequence_length=SEQUENCE_LENGTH, 
                                                                         latent_dim=LATENT_DIM, 
                                                                         codes_per_latent=CODES_PER_LATENT)
            
            mean_total_loss = total_loss_step(reconstruction_loss=reconstruction_loss, 
                                              reward_loss=rewards_loss, 
                                              termination_loss=terminations_loss, 
                                              dynamics_loss=dynamics_loss,
                                              categorical_encoder=categorical_encoder, 
                                              categorical_decoder=categorical_decoder, 
                                              tokenizer=tokenizer, 
                                              dynamics_model=dynamics_model, 
                                              optimizer=OPTIMIZER, 
                                              scaler=SCALER)
            
            observation_batch, action_batch, reward_batch, termination_batch = next(iter(agent_dataloader))
            mean_actor_loss, mean_critic_loss, mean_imagined_reward = train_agent(observation_batch=observation_batch, 
                                                                                  action_batch=action_batch, 
                                                                                  reward_batch=reward_batch, 
                                                                                  termination_batch=termination_batch, 
                                                                                  context_length=CONTEXT_LENGTH, 
                                                                                  imagination_horizon=IMAGINATION_HORIZON, 
                                                                                  env_actions=ENV_ACTIONS, 
                                                                                  latent_dim=LATENT_DIM, 
                                                                                  codes_per_latent=CODES_PER_LATENT, 
                                                                                  encoder=categorical_encoder, 
                                                                                  tokenizer=tokenizer, 
                                                                                  xlstm_dm=dynamics_model, 
                                                                                  actor=actor, 
                                                                                  critic=critic,
                                                                                  ema_critic=ema_critic,
                                                                                  device=DEVICE, 
                                                                                  gamma=GAMMA, 
                                                                                  lambda_p=LAMBDA, 
                                                                                  ema_sigma=EMA_SIGMA, 
                                                                                  nabla=NABLA, 
                                                                                  optimizer=AGENT_OPTIMIZER, 
                                                                                  scaler=SCALER)
            
            training_steps_finished += 1

            if training_steps_finished % 10**4 == 0:
                save_checkpoint(encoder=categorical_encoder,
                                decoder=categorical_decoder,
                                tokenizer=tokenizer,
                                dynamics=dynamics_model,
                                optimizer=OPTIMIZER,
                                scaler=SCALER,
                                step=training_steps_finished)
            
            step_metrics = {
                'total': mean_total_loss.item(),
                'reconstruction': reconstruction_loss.item(),
                'reward': rewards_loss.item(),
                'termination': terminations_loss.item(),
                'dynamics': dynamics_loss.item()
            }
            
            epoch_loss_history.append(step_metrics)

        plot_current_loss(new_losses=epoch_loss_history, 
                          training_steps_per_epoch=TRAINING_STEPS_PER_EPOCH, 
                          epochs=EPOCHS)