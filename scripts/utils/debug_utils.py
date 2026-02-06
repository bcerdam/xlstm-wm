import h5py
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import sys
from typing import List, Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.models.categorical_vae.sampler import sample
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


def plot_current_loss(new_losses: List[Dict[str, float]], training_steps_per_epoch: int, epochs: int) -> None:
    # 1. Calculate Mean for the current Epoch
    # Assumes new_losses is a list of dicts: [{'total': 0.5, 'recon': 0.1, ...}, ...]
    keys = new_losses[0].keys()
    epoch_means = {k: np.mean([d[k] for d in new_losses]) for k in keys}
    
    output_dir = 'output/logs'
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, 'loss_history.npy')

    # 2. Load and Update History
    if os.path.exists(history_path):
        try:
            # Load existing history
            loaded_history = np.load(history_path, allow_pickle=True).item()
            
            # Migration check: If old history was just a list (from previous code version), convert it
            if isinstance(loaded_history, list):
                loss_history = {'total': loaded_history} 
                # Fill missing keys for old epochs with nan so they don't plot
                for k in keys:
                    if k != 'total':
                        loss_history[k] = [np.nan] * len(loaded_history)
            else:
                loss_history = loaded_history
        except:
            loss_history = {k: [] for k in keys}
    else:
        loss_history = {k: [] for k in keys}

    # Append new means
    for k in keys:
        if k not in loss_history: loss_history[k] = [] # Handle new keys if config changes
        loss_history[k].append(epoch_means[k])

    np.save(history_path, loss_history)

    # 3. Plotting
    current_epoch = len(loss_history['total'])
    max_x = epochs * training_steps_per_epoch
    x_values = np.arange(1, current_epoch + 1) * training_steps_per_epoch

    # Create subplot: 1 row, 2 columns (Left: Losses, Right: KL Divergences)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plt.style.use('default') 

    # --- Plot 1: Losses ---
    # Define styles for individual losses
    loss_styles = {
        'total':          {'color': '#D32F2F', 'alpha': 1.0,  'label': 'Total', 'linewidth': 1.5}, # Red
        'reconstruction': {'color': '#1976D2', 'alpha': 0.6,  'label': 'Recon', 'linewidth': 1.0}, # Blue
        'reward':         {'color': '#388E3C', 'alpha': 0.6,  'label': 'Reward', 'linewidth': 1.0}, # Green
        'termination':    {'color': '#FBC02D', 'alpha': 0.6,  'label': 'Term',   'linewidth': 1.0}, # Yellow/Orange
        'dynamics':       {'color': '#7B1FA2', 'alpha': 0.5,  'label': 'Dyn',    'linewidth': 1.0}, # Purple
        'representation': {'color': '#E64A19', 'alpha': 0.5,  'label': 'Rep',    'linewidth': 1.0}, # Orange
    }

    for key, style in loss_styles.items():
        if key in loss_history:
            ax1.plot(x_values, loss_history[key], **style)

    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_xlim(0, max_x)
    ax1.set_xlabel("Total Training Steps", fontsize=8)
    ax1.set_ylabel("Loss", fontsize=8)
    ax1.set_title("Loss Components", fontsize=10)
    ax1.legend(fontsize=6, loc='upper right')
    
    # Format X-axis (Kilo-steps)
    def k_formatter(x, pos): return f'{int(x/1000)}K'
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))
    ax1.tick_params(axis='both', which='major', labelsize=6)


    # --- Plot 2: KL Divergences ---
    kl_styles = {
        'dynamics_kl':       {'color': '#7B1FA2', 'alpha': 0.9, 'label': 'Dyn KL'},
        'representation_kl': {'color': '#E64A19', 'alpha': 0.9, 'label': 'Rep KL'},
    }

    for key, style in kl_styles.items():
        if key in loss_history:
            ax2.plot(x_values, loss_history[key], **style)

    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_xlim(0, max_x)
    ax2.set_xlabel("Total Training Steps", fontsize=8)
    ax2.set_ylabel("Nats / Bits", fontsize=8)
    ax2.set_title("KL Divergences", fontsize=10)
    ax2.legend(fontsize=6, loc='upper right')
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))
    ax2.tick_params(axis='both', which='major', labelsize=6)

    # Save
    plt.suptitle(f"Training Progress (Epoch {current_epoch})", fontsize=10)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_plot.jpeg')
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
        latents = encoder.forward(observations_batch=model_input, batch_size=1, sequence_length=sequence_length, latent_dim=latent_dim, codes_per_latent=codes_per_latent)
        sampled_latents = sample(latents)
        reconstructions = decoder.forward(latents_batch=sampled_latents, batch_size=1, sequence_length=sequence_length, latent_dim=latent_dim, codes_per_latent=codes_per_latent)

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
    start_idx = 0
    steps = 200
    video_fps = 15
    output_path = 'output/videos/rollout/rollout_video_1.mp4'
    sequence_length = 64
    latent_dim = 32
    codes_per_latent = 32
    epoch = 100

    # inspect_dataset(h5_path=h5_path)

    rollout_video(h5_path=h5_path, 
                  start_idx=start_idx, 
                  steps=steps,
                  video_fps=video_fps, 
                  output_path=output_path)

    # visualize_reconstruction(dataset_path='data/replay_buffer.h5', 
    #                          weights_path='output/checkpoints/checkpoint_autoencoder_epoch_100.pth', 
    #                          device='cuda',
    #                          sequence_length=sequence_length, 
    #                          latent_dim=latent_dim, 
    #                          codes_per_latent=codes_per_latent, 
    #                          epoch=epoch)