import h5py
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from typing import List
from scripts.models.categorical_vae.encoder_fwd_pass import forward_pass_encoder
from scripts.models.categorical_vae.sampler import sample
from scripts.models.categorical_vae.decoder_fwd_pass import forward_pass_decoder


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

    plt.figure(figsize=(10, 6))
    plt.style.use('default') 
    plt.plot(x_values, loss_history, color='black', linewidth=1.5, linestyle='-')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max_x)
    
    ax = plt.gca()
    def k_formatter(x, pos):
        return f'{int(x/1000)}K'
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))
    
    plt.xlabel("Total Training Steps")
    plt.ylabel("Epoch Mean Loss")
    plt.title(f"Training Progress (Epoch {len(loss_history)})")
    
    output_path = os.path.join(output_dir, 'loss_plot.jpeg')
    plt.savefig(output_path, format='jpeg', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    h5_path = 'data/replay_buffer.h5'
    start_idx = 200
    steps = 100
    video_fps = 15
    output_path = 'output/video/rollout_video.mp4'

    # inspect_dataset(h5_path=h5_path)

    # rollout_video(h5_path=h5_path, 
    #               start_idx=start_idx, 
    #               steps=steps,
    #               video_fps=video_fps, 
    #               output_path=output_path)