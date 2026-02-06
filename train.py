import torch
import argparse
import yaml
import os
import shutil
import lpips
from scripts.data_related.enviroment_steps import gather_steps
from scripts.data_related.replay_buffer import update_replay_buffer
from scripts.data_related.atari_dataset import AtariDataset
from scripts.utils.tensor_utils import random_replay_batch, env_n_actions
from scripts.utils.debug_utils import plot_current_loss, save_checkpoint
from scripts.models.categorical_vae.categorical_autoencoder_step import autoencoder_step
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.sequence_preprocessing.tokenizer import Tokenizer
from scripts.models.sequence_preprocessing.tokenizer_fwd_pass import tokenize
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from scripts.models.dynamics_modeling.dynamics_model_step import dm_step
from scripts.loss_functions.total_loss import total_loss_step


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

    if os.path.exists('output/logs'):
        shutil.rmtree('output/logs')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = train_cfg['epochs']
    TRAINING_STEPS_PER_EPOCH = train_cfg['training_steps_per_epoch']
    REPLAY_BUFFER_PATH = dataset_cfg['replay_buffer_path']
    BATCH_SIZE = train_cfg['batch_size']
    SEQUENCE_LENGTH = train_cfg['sequence_length']
    LATENT_DIM = train_cfg['latent_dim']
    CODES_PER_LATENT = train_cfg['codes_per_latent']
    WORLD_MODEL_LEARNING_RATE = train_cfg['world_model_learning_rate']
    EMBEDDING_DIM = train_cfg['embedding_dim']
    ENV_NAME = env_cfg['env_name']
    ENV_ACTIONS = env_n_actions(ENV_NAME)
    NUM_BLOCKS = train_cfg['num_blocks']
    SLSTM_AT = train_cfg['slstm_at']
    DROPOUT = train_cfg['dropout']
    ADD_POST_BLOCKS_NORM = train_cfg['add_post_blocks_norm']
    CONV1D_KERNEL_SIZE = train_cfg['conv1d_kernel_size']
    QKV_PROJ_BLOCKSIZE = train_cfg['qkv_proj_blocksize']
    NUM_HEADS = train_cfg['num_heads']
    FREE_BITS = train_cfg['free_bits']
    DYNAMICS_BETA = train_cfg['dynamics_beta']
    REPRESENTATIONS_BETA = train_cfg['representations_beta']

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
                              codes_per_latent=CODES_PER_LATENT).to(DEVICE)
    lpips_model = lpips.LPIPS(net='alex').to(DEVICE).requires_grad_(False).eval()

    OPTIMIZER = torch.optim.Adam(list(categorical_encoder.parameters()) + 
                                 list(categorical_decoder.parameters()) +
                                 list(tokenizer.parameters()) + 
                                 list(dynamics_model.parameters()),
                                 lr=WORLD_MODEL_LEARNING_RATE)
    SCALER = torch.amp.GradScaler(enabled=True)
    for epoch in range(EPOCHS):
        if epoch % 100 == 0:
             print(f'Training Epoch: {epoch}...')

        # Gather data
        observations, actions, rewards, terminations, episode_starts = gather_steps(**env_cfg)
        update_replay_buffer(replay_buffer_path=REPLAY_BUFFER_PATH, 
                            observations=observations, 
                            actions=actions,
                            rewards=rewards, 
                            terminations=terminations, 
                            episode_starts=episode_starts)
        atari_dataset = AtariDataset(replay_buffer_path=REPLAY_BUFFER_PATH, sequence_length=SEQUENCE_LENGTH)

        epoch_loss_history = []
        for step in range(TRAINING_STEPS_PER_EPOCH):
            observations_batch, actions_batch, rewards_batch, terminations_batch = random_replay_batch(atari_dataset=atari_dataset, 
                                                                                                       batch_size=BATCH_SIZE, 
                                                                                                       sequence_length=SEQUENCE_LENGTH,
                                                                                                       device=DEVICE)
            
            # Train World Model
            categorical_autoencoder_reconstruction_loss, latents_sampled_batch, latents_batch_logits = autoencoder_step(categorical_encoder=categorical_encoder, 
                                                                                                                        categorical_decoder=categorical_decoder, 
                                                                                                                        observations_batch=observations_batch, 
                                                                                                                        batch_size=BATCH_SIZE, 
                                                                                                                        sequence_length=SEQUENCE_LENGTH, 
                                                                                                                        latent_dim=LATENT_DIM, 
                                                                                                                        codes_per_latent=CODES_PER_LATENT,
                                                                                                                        optimizer=OPTIMIZER,
                                                                                                                        scaler=SCALER, 
                                                                                                                        lpips_loss_fn=lpips_model)
            
            tokens_batch = tokenize(tokenizer=tokenizer, 
                                    latents_sampled_batch=latents_sampled_batch, 
                                    actions_batch=actions_batch)
            

            rewards_loss, terminations_loss, dynamics_loss, representations_loss, dynamics_kl_div, representations_kl_div = dm_step(dynamics_model=dynamics_model,
                                                                                                                                    latents_batch=latents_batch_logits, 
                                                                                                                                    tokens_batch=tokens_batch, 
                                                                                                                                    rewards_batch=rewards_batch, 
                                                                                                                                    terminations_batch=terminations_batch, 
                                                                                                                                    free_bits=FREE_BITS, 
                                                                                                                                    batch_size=BATCH_SIZE, 
                                                                                                                                    sequence_length=SEQUENCE_LENGTH, 
                                                                                                                                    latent_dim=LATENT_DIM, 
                                                                                                                                    codes_per_latent=CODES_PER_LATENT)

            mean_total_loss = total_loss_step(reconstruction_loss=categorical_autoencoder_reconstruction_loss, 
                                              reward_loss=rewards_loss, 
                                              termination_loss=terminations_loss, 
                                              dynamics_loss=dynamics_loss, 
                                              representation_loss=representations_loss, 
                                              batch_size=BATCH_SIZE, 
                                              sequence_length=SEQUENCE_LENGTH, 
                                              dynamics_beta=DYNAMICS_BETA, 
                                              representations_beta=REPRESENTATIONS_BETA, 
                                              categorical_encoder=categorical_encoder, 
                                              categorical_decoder=categorical_decoder, 
                                              tokenizer=tokenizer, 
                                              dynamics_model=dynamics_model, 
                                              optimizer=OPTIMIZER, 
                                              scaler=SCALER)
            
            step_metrics = {
                'total': mean_total_loss.item(),
                'reconstruction': categorical_autoencoder_reconstruction_loss.item(),
                'reward': rewards_loss.item(),
                'termination': terminations_loss.item(),
                'dynamics': dynamics_loss.item()*DYNAMICS_BETA,
                'representation': representations_loss.item()*REPRESENTATIONS_BETA,
                'dynamics_kl': dynamics_kl_div.item(),
                'representation_kl': representations_kl_div.item()
            }
            
            epoch_loss_history.append(step_metrics)

        # Metrics
        plot_current_loss(new_losses=epoch_loss_history, 
                          training_steps_per_epoch=TRAINING_STEPS_PER_EPOCH, 
                          epochs=EPOCHS)