# To do
# 1. Clean sweep_config
# 2. Run on all models
# 3. Try new sweep values if results are not satisfactory
# 4. Find a way to store transformed datasets

from utils import fetch_data, transform_data, get_config_path, get_config_from_path, retrain_transformed_sweep, evaluate
from views_forecasts.extensions import *
from pathlib import Path
import argparse
import copy
import wandb
import numpy as np
import warnings
warnings.filterwarnings("ignore")


PARA_DICT_h = {
    'rf': ['transform', 'clf_name', 'reg_name'],
    'xgb': ['transform', 'clf_name', 'reg_name'],
    'gbm': ['transform', 'clf_name', 'reg_name'],
    'lgb': ['transform', 'clf_name', 'reg_name']

}

PARA_DICT = {
    'rf': ['transform', 'n_estimators', 'n_jobs', 'learning_rate'],
    'xgb': ['transform', 'n_estimators', 'n_jobs', 'learning_rate'],
    'gbm': ['transform', 'n_estimators', 'n_jobs', 'learning_rate'],
    'lgb': ['transform', 'n_estimators', 'n_jobs', 'learning_rate']
}


def train():
    run = wandb.init(config=common_config,
                     project=wandb_config['project'], entity=wandb_config['entity'])
    wandb.config.update(model_config, allow_val_change=True)

    run_name = ''
    for para in sweep_paras:
        run_name += f'{para}_{run.config[para]}_'
    run_name = run_name.rstrip('_')
    wandb.run.name = run_name

    retrain_transformed_sweep(Datasets_transformed, sweep_paras)
    evaluate('calib', para_transformed, by_group=True)
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method for sweeping on W&B')
    parser.add_argument('-l', metavar='level', type=str,
                        required=True, choices=['cm', 'pgm'])
    parser.add_argument('-c', metavar='config', type=str,
                        required=True, help='Path to the configuration directory')
    args = parser.parse_args()

    level = args.l
    config_path = Path(args.c)

    transforms = ['raw', 'log', 'normalize', 'standardize']
    Datasets_transformed = {}
    para_transformed = {}
    qslist, Datasets = fetch_data(level)
    for t in transforms:
        Datasets_transformed[t], para_transformed[t] = transform_data(
            Datasets, t, by_group=True)

    common_config_path, wandb_config_path, model_config_path, sweep_config_path = get_config_path(
        config_path)
    common_config = get_config_from_path(common_config_path, 'common')
    wandb_config = get_config_from_path(wandb_config_path, 'wandb')

    for sweep_file in sweep_config_path.iterdir():
        if sweep_file.is_file():
            model_file = model_config_path / sweep_file.name
            if not model_file.is_file():
                raise FileNotFoundError(
                    f'The corresponding model configuration file {model_file} does not exist.')

            sweep_config = get_config_from_path(sweep_file, 'sweep')
            model_config = get_config_from_path(model_file, 'model')

            if sweep_file.stem.split('_')[-2] == 'hurdle':
                model = sweep_file.stem.split('_')[-1]
                sweep_paras = PARA_DICT_h[model]
                sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'],
                                       entity=wandb_config['entity'])
                wandb.agent(sweep_id, function=train)
            else:
                model = sweep_file.stem.split('_')[-1]
                sweep_paras = PARA_DICT[model]
                sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'],
                                       entity=wandb_config['entity'])
                wandb.agent(sweep_id, function=train)
