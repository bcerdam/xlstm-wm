import torch
import numpy as np
from typing import List
from scripts.utils.tensor_utils import env_n_actions
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder
from scripts.models.dynamics_modeling.tokenizer import Tokenizer
from scripts.models.dynamics_modeling.xlstm_dm import XLSTM_DM
from scripts.models.categorical_vae.sampler import sample


def dream(xlstm_dm:XLSTM_DM, tokens:torch.Tensor, imagination_horizon:int) -> List:
    imagined_frames = []

    print(tokens.shape)
    latent, reward, termination = xlstm_dm.forward(tokens_batch=tokens)
    print(latent.shape, reward.shape, termination.shape)


    # for step in range(imagination_horizon):
    pass


if __name__ == '__main__':
    env_name = 'ALE/Breakout-v5'
    dataset_path = 'data/replay_buffer.h5'
    weights_path = 'output/checkpoints/checkpoint_step_10000.pth'
    batch_size = 1
    context_length = 8
    imagination_horizon = 16
    latent_dim = 32
    codes_per_latent = 32
    device = 'cuda'
    env_actions = env_n_actions(env_name=env_name)
    embedding_dim = 256
    sequence_length = 64
    num_blocks = 2
    slstm_at = []
    dropout = 0.1
    add_post_blocks_norm = True
    conv1d_kernel_size = 4
    num_heads = 4
    qkv_proj_blocksize = 4

    dataset = AtariDataset(replay_buffer_path=dataset_path, sequence_length=context_length)
    encoder = CategoricalEncoder(latent_dim=latent_dim, codes_per_latent=codes_per_latent).to(device)
    decoder = CategoricalDecoder(latent_dim=latent_dim, codes_per_latent=codes_per_latent).to(device)
    tokenizer = Tokenizer(latent_dim=latent_dim, 
                          codes_per_latent=codes_per_latent, 
                          env_actions=env_actions, 
                          embedding_dim=embedding_dim, 
                          sequence_length=sequence_length).to(device)
    xlstm_dm = XLSTM_DM(sequence_length=sequence_length, 
                        num_blocks=num_blocks, 
                        embedding_dim=embedding_dim, 
                        slstm_at=slstm_at, 
                        dropout=dropout, 
                        add_post_blocks_norm=add_post_blocks_norm, 
                        conv1d_kernel_size=conv1d_kernel_size, 
                        qkv_proj_blocksize=qkv_proj_blocksize, 
                        num_heads=num_heads, 
                        latent_dim=latent_dim, 
                        codes_per_latent=codes_per_latent).to(device)
    

    checkpoint = torch.load(weights_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    tokenizer.load_state_dict(checkpoint['tokenizer'])
    xlstm_dm.load_state_dict(checkpoint['dynamics'])

    encoder.eval()
    decoder.eval()
    tokenizer.eval()
    xlstm_dm.eval()
    
    idx = np.random.randint(0, len(dataset))
    observations, actions, _, _ = dataset[idx] 
    observations = torch.from_numpy(observations).unsqueeze(0).to(device)
    actions = torch.from_numpy(actions).unsqueeze(0).to(device)

    with torch.no_grad():
        latents = encoder.forward(observations_batch=observations, 
                                  batch_size=batch_size, 
                                  sequence_length=context_length, 
                                  latent_dim=latent_dim, 
                                  codes_per_latent=codes_per_latent)
        
        latents_sampled_batch = sample(latents, batch_size=batch_size, sequence_length=context_length)

        tokens = tokenizer.forward(latents_sampled_batch=latents_sampled_batch, actions_batch=actions)

        dream(xlstm_dm=xlstm_dm, tokens=tokens, imagination_horizon=imagination_horizon)


    # 2. Feed it to xLSTM dynamics model, autoregressively generate for L = 16 frames, gather next latent, reward and termination

    # 3. Random policy for now, generate next action

    # 4. Repeat 1. with generated latent and action

    # At the end, save imagined rollout for debugging