## To do
### 1. Clean sweep_config
### 2. Run on all models
### 3. Try new sweep values if results are not satisfactory
### 4. Find a way to store transformed datasets

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import wandb
import copy
import argparse
from pathlib import Path

from views_forecasts.extensions import *
from utils import fetch_data, transform_data, get_config_path, get_config_from_path, retrain_transformed_sweep, evaluate


def train():
    wandb.init(config=common_config)
    wandb.config.update(model_config, allow_val_change=True)
    retrain_transformed_sweep(Datasets_transformed)
    evaluate('calib', para_transformed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method for sweeping on W&B')
    parser.add_argument('-l', metavar='level', type=str, required=True, choices=['cm', 'pgm'])
    parser.add_argument('-c', metavar='config', type=str, required=True, help='Path to the configuration directory')
    args = parser.parse_args()

    level = args.l
    config_path = Path(args.c)

    transforms = ['raw', 'log', 'normalize', 'standardize']
    Datasets_transformed = {}
    para_transformed = {}
    qslist, Datasets = fetch_data(level)
    for t in transforms:
        Datasets_transformed[t], para_transformed[t] = transform_data(Datasets, t)

    common_config_path, wandb_config_path, model_config_path, sweep_config_path = get_config_path(config_path)
    common_config = get_config_from_path(common_config_path, 'common')
    wandb_config = get_config_from_path(wandb_config_path, 'wandb')

    for sweep_file in sweep_config_path.iterdir():
        if sweep_file.is_file():
            model_file = model_config_path / sweep_file.name
            if not model_file.is_file():
                raise FileNotFoundError(f'The corresponding model configuration file {model_file} does not exist.')

            sweep_config = get_config_from_path(sweep_file, 'sweep')
            model_config = get_config_from_path(model_file, 'model')

            sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'],
                                   entity=wandb_config['entity'])
            wandb.agent(sweep_id, function=train)
