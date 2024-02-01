from util.utils import *
from pathlib import Path
import argparse
import wandb


def train() -> None:
    """
    This function initializes a run with Weights & Biases (wandb), updates the configuration with the model configuration, 
    constructs a run name based on the sweep parameters, retrains the model with transformed datasets, evaluates the model, 
    and finally finishes the run.

    The function uses the following global variables:
    - common_config: A dictionary containing common configuration parameters for the wandb run.
    - wandb_config: A dictionary containing the project and entity for the wandb run.
    - model_config: A dictionary containing the model configuration parameters.
    - sweep_paras: A list of parameters to be included in the sweep.
    - Datasets_transformed: Transformed datasets to be used for retraining the model.
    - para_transformed: Transformed parameters to be used for model evaluation.

    Note: This function does not return anything. The results of the run are logged and managed by wandb.
    """
    run = wandb.init()
    model_paras = [para for para in run.config.keys() if para != 'transform']
    run_name = f'transform_{run.config["transform"]}'
    for para in model_paras:
        run_name += f'_{para}_{run.config[para]}'
    wandb.run.name = run_name

    wandb.config.update(common_config, allow_val_change=True)
    wandb.config.update(model_config, allow_val_change=True)

    retrain_transformed_sweep(Datasets_transformed, model_paras)
    evaluate('calib', para_transformed, by_group=True, plot_map=True)
    run.finish()


if __name__ == '__main__':
    # this is the main block of code that will be called when running the script from the command line with arguments

    parser = argparse.ArgumentParser(description='Method for sweeping on W&B')
    parser.add_argument('-l', metavar='level', type=str,
                        required=True, choices=['cm', 'pgm'])
    parser.add_argument('-c', metavar='config', type=str,
                        required=True, help='Path to the configuration directory')
    parser.add_argument('-m', metavar='modelname',
                        help='Name of the model to implement')
    args = parser.parse_args()

    level = args.l
    config_path = Path(args.c)
    model_name = args.m

    transforms = ['raw', 'log', 'normalize', 'standardize']
    Datasets_transformed = {}
    para_transformed = {}
    qslist, Datasets = i_fetch_data(level)
    Datasets_transformed, para_transformed = transform_data(
        Datasets, transforms, level=level, by_group=True)

    common_config_path, wandb_config_path, model_config_path, sweep_config_path = get_config_path(
        config_path)
    common_config = get_config_from_path(common_config_path, 'common')
    wandb_config = get_config_from_path(wandb_config_path, 'wandb')

    for sweep_file in sweep_config_path.iterdir():
        if sweep_file.is_file():
            # Skip if a specific model name is provided and it doesn't match the file
            model_name_from_file = sweep_file.stem
            if model_name and model_name != model_name_from_file:
                continue

            model_file = model_config_path / sweep_file.name
            if not model_file.is_file():
                raise FileNotFoundError(
                    f'The corresponding model configuration file {model_file} does not exist.')

            sweep_config = get_config_from_path(sweep_file, 'sweep')
            model_config = get_config_from_path(model_file, 'model')

            model = sweep_file.stem.split('_')[-1]
            sweep_id = wandb.sweep(
                sweep_config, project=wandb_config['project'], entity=wandb_config['entity'])
            wandb.agent(sweep_id, function=train)

            print(f'Finish sweeping over model {sweep_file.stem}')
            print('**************************************************************')
