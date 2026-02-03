import h5py
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import sys
from typing import List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.models.categorical_vae.encoder_fwd_pass import forward_pass_encoder
from scripts.models.categorical_vae.sampler import sample
from scripts.models.categorical_vae.decoder_fwd_pass import forward_pass_decoder
from scripts.data_related.atari_dataset import AtariDataset
from scripts.models.categorical_vae.encoder import CategoricalEncoder
from scripts.models.categorical_vae.decoder import CategoricalDecoder


def inspect_dataset(h5_path:str) -> None:
    with h5py.File(h5_path, 'r') as f:
        print(f"Inspecting: {h5_path}")
        for key in f.keys():
            dset = f[key]
            print(f"\nColumn: {key}")
            print(f"  Rows: {dset.shape[0]}")
            print(f"  Dtype: {dset.dtype}")
            print(f"  Min: {np.min(dset)}")
            print(f"  Max: {np.max(dset)}")


def rollout_video(h5_path:str, start_idx:int, steps:int, video_fps:int, output_path:str) -> None:
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(h5_path, 'r') as f:
        observations = f['observations'][start_idx : start_idx + steps]
        rollout_obs = np.moveaxis(observations, 1, -1)
        rollout_obs = ((rollout_obs + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        height, width = rollout_obs.shape[1], rollout_obs.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        for frame in rollout_obs:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()


def plot_current_loss(new_losses: List[float], training_steps_per_epoch: int, epochs: int) -> None:
    epoch_mean_loss = np.array(new_losses).mean()
    
    output_dir = 'output/logs'
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, 'loss_history.npy')

    if os.path.exists(history_path):
        try:
            loss_history = np.load(history_path).tolist()
        except:
            loss_history = []
    else:
        loss_history = []

    loss_history.append(epoch_mean_loss)
    np.save(history_path, np.array(loss_history))


    max_x = epochs * training_steps_per_epoch
    x_values = np.arange(1, len(loss_history) + 1) * training_steps_per_epoch

    plt.figure(figsize=(6, 2))
    plt.style.use('default') 
    plt.plot(x_values, loss_history, color='black', linewidth=0.75, linestyle='-')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max_x)
    
    ax = plt.gca()
    def k_formatter(x, pos):
        return f'{int(x/1000)}K'
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))
    
    plt.xlabel("Total Training Steps", fontsize=6)
    plt.ylabel("Epoch Mean Loss", fontsize=6)
    plt.title(f"Training Progress (Epoch {len(loss_history)})", fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=4)
    
    output_path = os.path.join(output_dir, 'loss_plot.jpeg')
    plt.tight_layout()
    plt.savefig(output_path, format='jpeg', dpi=150, bbox_inches='tight')
    plt.close()


def save_checkpoint(encoder, decoder, epoch, path="output/checkpoints"):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, os.path.join(path, f"checkpoint_autoencoder_epoch_{epoch}.pth"))


def visualize_reconstruction(dataset_path:str, 
                             weights_path:str, 
                             device:torch.device, 
                             sequence_length:int, 
                             latent_dim:int, 
                             codes_per_latent:int, 
                             epoch:int,
                             video_path="output/videos") -> None:

    os.makedirs(video_path, exist_ok=True)

    dataset = AtariDataset(replay_buffer_path=dataset_path, sequence_length=sequence_length)
    encoder = CategoricalEncoder(latent_dim=latent_dim, codes_per_latent=codes_per_latent).to(device)
    decoder = CategoricalDecoder(latent_dim=latent_dim, codes_per_latent=codes_per_latent).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()
    
    idx = np.random.randint(0, len(dataset) - sequence_length)
    obs_seq, _, _, _ = dataset[idx] 
    obs_seq = torch.from_numpy(obs_seq)
        
    model_input = obs_seq.unsqueeze(0).to(device)
    with torch.no_grad():
        latents = forward_pass_encoder(encoder, model_input, 1, sequence_length, latent_dim, codes_per_latent)
        sampled_latents = sample(latents)
        reconstructions = forward_pass_decoder(decoder, sampled_latents, 1, sequence_length, latent_dim, codes_per_latent)

    model_input = model_input.cpu()
    reconstructions = reconstructions.cpu()

    orig_np = (model_input[0].permute(0, 2, 3, 1).numpy() + 1) * 127.5
    recon_np = (reconstructions[0].permute(0, 2, 3, 1).numpy() + 1) * 127.5

    orig_np = orig_np.astype(np.uint8)
    recon_np = recon_np.astype(np.uint8)

    save_file = os.path.join(video_path, f"epoch_{epoch}_reconstruction.mp4")
    L, height, width, _ = orig_np.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, 15.0, (width * 2, height))

    for t in range(L):
        frame_orig = cv2.cvtColor(orig_np[t], cv2.COLOR_RGB2BGR)
        frame_recon = cv2.cvtColor(recon_np[t], cv2.COLOR_RGB2BGR)
        
        combined_frame = np.concatenate((frame_orig, frame_recon), axis=1)
        out.write(combined_frame)

    out.release()


if __name__ == '__main__':
    h5_path = 'data/replay_buffer.h5'
    start_idx = 500
    steps = 100
    video_fps = 15
    output_path = 'output/video/rollout_video.mp4'
    sequence_length = 64
    latent_dim = 32
    codes_per_latent = 32
    epoch = 100

    # inspect_dataset(h5_path=h5_path)

    # rollout_video(h5_path=h5_path, 
    #               start_idx=start_idx, 
    #               steps=steps,
    #               video_fps=video_fps, 
    #               output_path=output_path)

    visualize_reconstruction(dataset_path='data/replay_buffer.h5', 
                             weights_path='output/checkpoints/checkpoint_autoencoder_epoch_100.pth', 
                             device='cuda',
                             sequence_length=sequence_length, 
                             latent_dim=latent_dim, 
                             codes_per_latent=codes_per_latent, 
                             epoch=epoch)