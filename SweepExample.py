import numpy as np
import warnings
warnings.filterwarnings("ignore")
import wandb
import copy

from views_forecasts.extensions import *
from utils import fetch_data, transform_data, retrain_transformed, evaluate
from ModelConfig import model_config
from CommonConfig import common_config, wandb_config


sweep_configuration = {
    "name": model_config["modelname"],
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        "n_estimators": {"values": [100, 200, 300]},
        "n_jobs": {"values": [12]},
        "learning_rate": {"values": [0.05, 0.1, 0.2]}
    }
}

# class Trainer:
#     def __init__(self):
#         self.Datasets_transformed = {}
#         self.para_transformed = {}
#         self.transforms = []
#
#     def setup(self):
#         level = 'cm'
#         qslist, Datasets = fetch_data(level)
#         for t in self.transforms:
#             self.Datasets_transformed[t], self.para_transformed[t] = transform_data(Datasets, t)
#
#     def train(self):
#         retrain_transformed(self.Datasets_transformed, self.transforms)
#         evaluate('calib', self.transforms, self.para_transformed, evaluate_raw=False)

def train():
    wandb.init(config=common_config, project=wandb_config['project'])
    wandb.config.update(model_config, allow_val_change=True)
    retrain_transformed(Datasets_transformed, transforms)
    evaluate('calib', transforms, para_transformed, evaluate_raw=False)


if __name__ == '__main__':
    level = 'cm'
    transforms = ['log']
    Datasets_transformed = {}
    para_transformed = {}
    qslist, Datasets = fetch_data(level)
    for t in transforms:
        Datasets_transformed[t], para_transformed[t] = transform_data(Datasets, t)

    # transforms = ['log', 'normalize', 'standardize']

    sweep_id = wandb.sweep(sweep_configuration, project=wandb_config['project'])
    wandb.agent(sweep_id, function=train)
