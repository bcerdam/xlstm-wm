import argparse
import yaml
from env_data import gather_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_cfg', default='config/train.yaml', type=str, help='Path to env parameters .yaml file')
    parser.add_argument('--env_cfg', default='config/env.yaml', type=str, help='Path to env parameters .yaml file')
    args = parser.parse_args()

    with open(args.train_cfg, 'r') as file_train, open(args.env_cfg, 'r') as file_env:
        train_cfg = yaml.safe_load(file_train)['train']
        env_cfg = yaml.safe_load(file_env)['env']

    gather_steps(**env_cfg)

