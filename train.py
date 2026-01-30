import argparse
import yaml
import os
import shutil
from scripts.data_related.enviroment_steps import gather_steps
from scripts.data_related.replay_buffer import update_replay_buffer


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

    for epoch in range(EPOCHS):
        observations, actions, rewards, terminations = gather_steps(**env_cfg)
        update_replay_buffer(replay_buffer_path=dataset_cfg['replay_buffer_path'], 
                            observations=observations, 
                            actions=actions, 
                            rewards=rewards, 
                            terminations=terminations)