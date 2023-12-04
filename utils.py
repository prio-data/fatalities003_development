import wandb
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['WANDB_SILENT'] = 'true'

from sklearn.metrics import mean_squared_error
from xgboost import XGBRFRegressor

from viewser import Queryset
from FetchData import ReturnQsList, fetch_cm_data_from_model_def, RetrieveFromList
from views_runs import storage, StepshiftedModels
from views_partitioning.data_partitioner import DataPartitioner
from views_runs.run_result import RunResult
from views_forecasts.extensions import *



def fetch_data(level: str) -> (Tuple[List[Queryset], List[Dict[str, pd.DataFrame]]]):
    print('Fetching and transforming data')
    qslist = ReturnQsList(level)
    Datasets = fetch_cm_data_from_model_def(qslist)
    return qslist, Datasets


def transform_data(Datasets: List[Dict[str, pd.DataFrame]], transform: str, b = 1, a = 0) -> (List[Dict[str, pd.DataFrame]], pd.DataFrame):
    Datasets_transformed = copy.deepcopy(Datasets)
    if transform == 'raw':
        return Datasets_transformed, None
    elif transform == 'log':
        for dataset in Datasets_transformed:
            dataset['df']['ged_sb_dep'] = np.log(dataset['df']['ged_sb_dep']+1) # same as transform.ops.ln
        return Datasets_transformed, None
    elif transform == 'normalize':
        dict_max_min = {'Name': [], 'max_val': [], 'min_val': []}
        for dataset in Datasets_transformed:
            min_val = dataset['df']['ged_sb_dep'].min()
            max_val = dataset['df']['ged_sb_dep'].max()
            dataset['df']['ged_sb_dep'] = (b - a) * (dataset['df']['ged_sb_dep'] - min_val) / (max_val - min_val) + a
            dict_max_min['Name'].append(dataset['Name'])
            dict_max_min['max_val'].append(max_val)
            dict_max_min['min_val'].append(min_val)
        df = pd.DataFrame(dict_max_min)
    elif transform == 'standardize':
        dict_mean_std = {'Name': [], 'mean': [], 'std': []}
        for dataset in Datasets_transformed:
            mean = dataset['df']['ged_sb_dep'].mean()
            std =  dataset['df']['ged_sb_dep'].std()
            dataset['df']['ged_sb_dep'] = (dataset['df']['ged_sb_dep'] - mean) / std
            dict_mean_std['Name'].append(dataset['Name'])
            dict_mean_std['mean'].append(mean)
            dict_mean_std['std'].append(std)
        df = pd.DataFrame(dict_mean_std)
    else:
        raise ValueError("Wrong transformation, only support 'log', 'normalize', 'standardize'.")
    return Datasets_transformed, df


def get_config_path(config_path: Path) -> Path:
    common_config_path = config_path / 'common_config.py'
    wandb_config_path = config_path / 'wandb_config.py'
    model_config_path = config_path / 'model_config'
    sweep_config_path = config_path / 'sweep_config'

    if not common_config_path.is_file():
        raise FileNotFoundError(f'The common configuration file {common_config_path} does not exist.')
    if not wandb_config_path.is_file():
        raise FileNotFoundError(f'The common configuration file {wandb_config_path} does not exist.')
    if not model_config_path.exists() or not model_config_path.is_dir():
        raise FileNotFoundError(f'The directory {model_config_path} does not exist or is not a directory.')
    if not sweep_config_path.exists() or not sweep_config_path.is_dir():
        raise FileNotFoundError(f'The directory {sweep_config_path} does not exist or is not a directory.')

    return common_config_path, wandb_config_path, model_config_path, sweep_config_path


def get_config_from_path(path: Path, config_name: str) -> Dict:
    if config_name not in ['common', 'wandb', 'sweep', 'model']:
        raise ValueError("Wrong configuration name, only support 'common', 'wandb', 'sweep', 'model'.")
    config = {}
    exec(path.read_text(), config)
    config_name = config_name + '_config'
    return config[config_name]


def retrain_transformed_sweep(Datasets_transformed):
    modelstore = storage.Storage()
    level = wandb.config['level']
    run_id = wandb.config['run_id']
    steps = wandb.config['steps']
    transform = wandb.config['transform']
    calib_partitioner_dict = wandb.config['calib_partitioner_dict']
    test_partitioner_dict = wandb.config['test_partitioner_dict']
    future_partitioner_dict = wandb.config['future_partitioner_dict']
    force_retrain = wandb.config['force_retrain']

    wandb.config.update({'predstore_calib': level + '_' + wandb.config['modelname'] + '_calib'})
    wandb.config.update({'predstore_test': level + '_' + wandb.config['modelname'] + '_test'})

    model = globals()[wandb.config['algorithm']](n_estimators=wandb.config['n_estimators'],
                                                 learning_rate=wandb.config['learning_rate'],
                                                 n_jobs=wandb.config['n_jobs'])

    ## Training
    print(f'Training model {wandb.config["modelname"]}')

    print(f'Calibration partition ({transform})')
    RunResult_calib = RunResult.retrain_or_retrieve(
        retrain=force_retrain,
        store=modelstore,
        partitioner=DataPartitioner({"calib": calib_partitioner_dict}),
        stepshifted_models=StepshiftedModels(model, steps, wandb.config['depvar']),
        dataset=RetrieveFromList(Datasets_transformed[transform], wandb.config['data_train']),
        queryset_name=wandb.config['queryset'],
        partition_name="calib",
        timespan_name="train",
        storage_name=wandb.config['modelname'] + f'_calib_{transform}',
        author_name="test0",
    )
    wandb.config.update(
        {f'predstore_calib_{transform}': level + '_' +
                                         wandb.config['modelname'] +
                                         f'_calib_{transform}_estimators_{wandb.config["n_estimators"]}_lr_{wandb.config["learning_rate"]}'})
    print('Getting predictions')
    try:
        predictions_calib = pd.DataFrame.forecasts.read_store(run=run_id,
                                                              name=wandb.config[f'predstore_calib_{transform}'])
    except KeyError:
        print(wandb.config[f'predstore_calib_{transform}'], ', run', run_id, 'does not exist, predicting')
        predictions_calib = RunResult_calib.run.predict("calib", "predict", RunResult_calib.data)
        predictions_calib.forecasts.set_run(run_id)
        predictions_calib.forecasts.to_store(name=wandb.config[f'predstore_calib_{transform}'])

    print('Test partition ({transform})')
    RunResult_test = RunResult.retrain_or_retrieve(
        retrain=force_retrain,
        store=modelstore,
        partitioner=DataPartitioner({"test": test_partitioner_dict}),
        stepshifted_models=StepshiftedModels(model, steps, wandb.config['depvar']),
        dataset=RetrieveFromList(Datasets_transformed[transform], wandb.config['data_train']),
        queryset_name=wandb.config['queryset'],
        partition_name="test",
        timespan_name="train",
        storage_name=wandb.config['modelname'] + f'_test_{transform}',
        author_name="test0",
    )
    wandb.config.update(
        {f'predstore_test_{transform}': level + '_' + wandb.config['modelname'] + f'_test_{transform}_estimators_{wandb.config["n_estimators"]}_lr_{wandb.config["learning_rate"]}'})
    print('Getting predictions')
    try:
        predictions_test = pd.DataFrame.forecasts.read_store(run=run_id,
                                                             name=wandb.config[f'predstore_test_{transform}'])
    except KeyError:
        print(wandb.config[f'predstore_test_{transform}'], ', run', run_id, 'does not exist, predicting')
        predictions_test = RunResult_test.run.predict("test", "predict", RunResult_test.data)
        predictions_test.forecasts.set_run(run_id)
        predictions_test.forecasts.to_store(name=wandb.config[f'predstore_test_{transform}'])
    print('**************************************************************')


def evaluate(target, para_transformed, retransform=True, b = 1, a = 0):
    print(f'Evaluating model {wandb.config["modelname"]}')
    if target not in ['calib', 'test']:
        raise ValueError("Wrong target name, only support 'calib' and 'test'.")

    transform = wandb.config['transform']
    steps = wandb.config['steps']
    run_id = wandb.config['run_id']
    stepcols = [wandb.config['depvar']]
    for step in steps:
        stepcols.append('step_pred_' + str(step))
    pred_cols = [f'step_pred_{str(i)}' for i in steps]

    name = wandb.config[f'predstore_{target}_{transform}']
    df = pd.DataFrame.forecasts.read_store(run=run_id, name=name).replace([np.inf, -np.inf], 0)[stepcols]

    if retransform:
        if transform == 'log':
            df = np.exp(df) - 1
        elif transform == 'normalize':
            df_para = para_transformed[transform]
            df_para_model = df_para[df_para['Name'] == wandb.config['data_train']]
            max_val = df_para_model['max_val'].iloc[0]
            min_val = df_para_model['min_val'].iloc[0]
            df = (df - a) / (b - a) * (max_val - min_val) + min_val
        elif transform == 'standardize':
            df_para = para_transformed[transform]
            df_para_model = df_para[df_para['Name'] == wandb.config['data_train']]
            mean = df_para_model['mean'].iloc[0]
            std = df_para_model['std'].iloc[0]
            df = df * std + mean

    df['mse'] = df.apply(lambda row: mean_squared_error([row['ged_sb_dep']] * 36,
                                                        [row[col] for col in pred_cols]), axis=1)

    print(f'mse_{transform}', df['mse'].mean())
    wandb.log({'mse': df['mse'].mean()})
