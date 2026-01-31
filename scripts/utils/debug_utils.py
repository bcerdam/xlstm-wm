import h5py
import cv2
import os
import numpy as np


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