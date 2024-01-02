import warnings
warnings.filterwarnings("ignore")
import wandb
import argparse
from pathlib import Path
from utils import i_fetch_data, transform_data, get_config_path, get_config_from_path, retrain_transformed_sweep, evaluate


def train():
    run = wandb.init()
    model_paras = [para for para in run.config.keys() if para != 'transform']
    run_name = f'transform_{run.config["transform"]}'
    for para in model_paras:
        run_name += f'_{para}_{run.config[para]}'
    wandb.run.name = run_name
    # print(run_name)

    wandb.config.update(common_config, allow_val_change=True)
    wandb.config.update(model_config, allow_val_change=True)

    retrain_transformed_sweep(Datasets_transformed, model_paras)
    evaluate('calib', para_transformed, by_group=True, plot_map=True)
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method for sweeping on W&B')
    parser.add_argument('-l', metavar='level', required=True, choices=['cm', 'pgm'])
    parser.add_argument('-c', metavar='config', required=True, help='Path to the configuration directory')
    parser.add_argument('-m', metavar='modelname', help='Name of the model to implement')
    args = parser.parse_args()

    level = args.l
    config_path = Path(args.c)
    model_name = args.m

    # Fetch and transform data
    transforms = ['raw', 'log', 'normalize', 'standardize']
    qslist, Datasets = i_fetch_data(level)
    Datasets_transformed, para_transformed = transform_data(Datasets, transforms, level, by_group=True)

    # Get config path and config file
    common_config_path, wandb_config_path, model_config_path, sweep_config_path = get_config_path(config_path)
    common_config = get_config_from_path(common_config_path, 'common')
    wandb_config = get_config_from_path(wandb_config_path, 'wandb')
    #
    # for sweep_file in sweep_config_path.iterdir():
    #     if sweep_file.is_file():
    #
    #         # Skip if a specific model name is provided and it doesn't match the file
    #         model_name_from_file = sweep_file.stem
    #         if model_name and model_name != model_name_from_file:
    #             continue
    #
    #         model_file = model_config_path / sweep_file.name
    #         if not model_file.is_file():
    #             raise FileNotFoundError(f'The corresponding model configuration file {model_file} does not exist.')
    #
    #         sweep_config = get_config_from_path(sweep_file, 'sweep')
    #         model_config = get_config_from_path(model_file, 'model')
    #
    #         if sweep_file.stem.split('_')[-2] == 'hurdle':
    #             continue  # Currently Hurdle models are not supported
    #         model = sweep_file.stem.split('_')[-1]
    #         sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'],
    #                                entity=wandb_config['entity'])
    #         wandb.agent(sweep_id, function=train)
    #
    #         print(f'Finish sweeping over model {sweep_file.stem}')
    #         print('**************************************************************')
