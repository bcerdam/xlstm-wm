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
    # 1. Calculate Averages
    keys = new_losses[0].keys()
    epoch_means = {k: np.mean([d[k] for d in new_losses]) for k in keys}
    
    # 2. Load/Update History
    output_dir = 'output/logs'
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, 'loss_history.npy')

    if os.path.exists(history_path):
        loss_history = np.load(history_path, allow_pickle=True).item()
    else:
        loss_history = {}

    for k, v in epoch_means.items():
        loss_history.setdefault(k, []).append(v)

    np.save(history_path, loss_history)

    # 3. Setup Plotting
    current_epoch = len(loss_history['total'])
    max_x = epochs * training_steps_per_epoch
    x_values = np.arange(1, current_epoch + 1) * training_steps_per_epoch

    # Create 4 Subplots vertically (Height increased to 8)
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 8), dpi=200, sharex=True)
    
    # --- SUBPLOT 1: World Model Loss ---
    ax_wm = axes[0]
    wm_styles = {
        'total':          {'color': '#D32F2F', 'label': 'Total'},
        'reconstruction': {'color': '#1976D2', 'label': 'Recon'},
        'reward':         {'color': '#388E3C', 'label': 'Reward'},
        'termination':    {'color': '#FBC02D', 'label': 'Term'},
        'dynamics':       {'color': '#00BCD4', 'label': 'Dyn'},
    }
    
    for key, style in wm_styles.items():
        if key in loss_history:
            ax_wm.plot(x_values, loss_history[key], linewidth=1.0, alpha=0.9, **style)
    
    ax_wm.set_title("World Model Loss", fontsize=7, fontweight='bold')
    ax_wm.set_ylabel("Loss", fontsize=6)
    ax_wm.legend(fontsize=5, loc='upper right', framealpha=0.8)
    ax_wm.grid(True, linestyle='--', alpha=0.3)

    # --- SUBPLOT 2: Actor Critic Loss ---
    ax_ac = axes[1]
    ac_styles = {
        'actor':  {'color': '#E91E63', 'label': 'Actor'},
        'critic': {'color': '#673AB7', 'label': 'Critic'},
    }

    for key, style in ac_styles.items():
        if key in loss_history:
            ax_ac.plot(x_values, loss_history[key], linewidth=1.0, alpha=0.9, **style)

    ax_ac.set_title("Actor Critic Loss", fontsize=7, fontweight='bold')
    ax_ac.set_ylabel("Loss", fontsize=6)
    ax_ac.legend(fontsize=5, loc='upper right', framealpha=0.8)
    ax_ac.grid(True, linestyle='--', alpha=0.3)

    # --- SUBPLOT 3: Mean Imagined Reward ---
    ax_im = axes[2]
    if 'imagined_reward' in loss_history:
        ax_im.plot(x_values, loss_history['imagined_reward'], color='#FF9800', linewidth=1.0, label='Imagined')

    ax_im.set_title("Mean Imagined Reward", fontsize=7, fontweight='bold')
    ax_im.set_ylabel("Reward", fontsize=6)
    ax_im.legend(fontsize=5, loc='upper left', framealpha=0.8)
    ax_im.grid(True, linestyle='--', alpha=0.3)

    # --- SUBPLOT 4: Real Mean Reward ---
    ax_real = axes[3]
    if 'real_reward' in loss_history:
        ax_real.plot(x_values, loss_history['real_reward'], color='#4CAF50', linewidth=1.0, label='Real')

    # CHANGED TITLE HERE
    ax_real.set_title("Mean Real Episode Return", fontsize=7, fontweight='bold')
    ax_real.set_ylabel("Score", fontsize=6)
    ax_real.set_xlabel("Total Training Steps", fontsize=6)
    ax_real.legend(fontsize=5, loc='upper left', framealpha=0.8)
    ax_real.grid(True, linestyle='--', alpha=0.3)

    # --- Formatting ---
    ax_real.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}K'))
    ax_real.set_xlim(0, max_x)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot.jpeg'), format='jpeg', dpi=200, bbox_inches='tight')
    plt.close()


def save_checkpoint(encoder, decoder, tokenizer, dynamics, optimizer, scaler, step, path="output/checkpoints"):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'step': step,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'tokenizer': tokenizer.state_dict(),
        'dynamics': dynamics.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }, os.path.join(path, f"checkpoint_step_{step}.pth"))


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


def save_dream_video(imagined_frames:List[np.ndarray], video_path:str, fps:int) -> None:
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    _, _, height, width = imagined_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in imagined_frames:
        frame = frame[0]
        frame = np.transpose(frame, (1, 2, 0))
        frame = (frame + 1) * 127.5
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

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

    # rollout_video(h5_path=h5_path, 
    #               start_idx=start_idx, 
    #               steps=steps,
    #               video_fps=video_fps, 
    #               output_path=output_path)

    # visualize_reconstruction(dataset_path='data/replay_buffer.h5', 
    #                          weights_path='output/checkpoints/checkpoint_autoencoder_epoch_100.pth', 
    #                          device='cuda',
    #                          sequence_length=sequence_length, 
    #                          latent_dim=latent_dim, 
    #                          codes_per_latent=codes_per_latent, 
    #                          epoch=epoch)