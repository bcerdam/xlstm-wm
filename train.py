import argparse
import yaml
import os
import shutil
from scripts.data_related.enviroment_steps import gather_steps
from scripts.data_related.replay_buffer import update_replay_buffer
from scripts.data_related.atari_dataset import AtariDataset
from scripts.utils.tensor_utils import random_replay_batch
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.encoder_fwd_pass import forward_pass_encoder
from scripts.models.categorical_vae.sampler import sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', default='config/train.yaml', type=str, help='Path to env parameters .yaml file')
    parser.add_argument('--env_cfg', default='config/env.yaml', type=str, help='Path to env parameters .yaml file')
    args = parser.parse_args()

    with open(args.train_cfg, 'r') as file_train, open(args.env_cfg, 'r') as file_env:
        train_cfg = yaml.safe_load(file_train)['train']

        env_file_content = yaml.safe_load(file_env)
        env_cfg = env_file_content['env']
        dataset_cfg = env_file_content['dataset']

    if os.path.exists('data'):
            shutil.rmtree('data')

    EPOCHS = train_cfg['epochs']
    REPLAY_BUFFER_PATH = dataset_cfg['replay_buffer_path']
    BATCH_SIZE = train_cfg['batch_size']
    SEQUENCE_LENGTH = train_cfg['sequence_length']
    LATENT_DIM = train_cfg['latent_dim']
    CODES_PER_LATENT = train_cfg['codes_per_latent']

    categorical_encoder = CategoricalEncoder(latent_dim=LATENT_DIM, codes_per_latent=CODES_PER_LATENT)
    for epoch in range(EPOCHS):
        observations, actions, rewards, terminations = gather_steps(**env_cfg)
        update_replay_buffer(replay_buffer_path=REPLAY_BUFFER_PATH, 
                            observations=observations, 
                            actions=actions,
                            rewards=rewards, 
                            terminations=terminations)
        atari_dataset = AtariDataset(replay_buffer_path=REPLAY_BUFFER_PATH, sequence_length=SEQUENCE_LENGTH)

        observations_batch, actions_batch, rewards_batch, terminations_batch = random_replay_batch(atari_dataset=atari_dataset, 
                                                                                                   batch_size=BATCH_SIZE, 
                                                                                                   sequence_length=SEQUENCE_LENGTH)

        latents_batch = forward_pass_encoder(categorical_encoder=categorical_encoder, 
                                             observations_batch=observations_batch, 
                                             batch_size=BATCH_SIZE, 
                                             sequence_length=SEQUENCE_LENGTH, 
                                             latent_dim=LATENT_DIM, 
                                             codes_per_latent=CODES_PER_LATENT)        

        latents_sampled_batch = sample(latents_batch=latents_batch)