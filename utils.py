from new_metrics import *
from views_forecasts.extensions import *
from views_runs.run_result import RunResult
from views_partitioning.data_partitioner import DataPartitioner
from views_runs import storage, StepshiftedModels
from FetchData import ReturnQsList, fetch_cm_data_from_model_def, RetrieveFromList
from viewser import Queryset
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.metrics import mean_squared_error
import os
import wandb
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import traceback

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression

from xgboost import XGBRegressor
from xgboost import XGBClassifier
from xgboost import XGBRFRegressor, XGBRFClassifier

from lightgbm import LGBMClassifier, LGBMRegressor

from ViewsEstimators import *

warnings.filterwarnings("ignore")
os.environ['WANDB_SILENT'] = 'true'


def log_transform_raw_data(df):
    transformed_df = np.log(df + 1)
    return transformed_df


def standardize_raw_data(df):
    mean_val = df.mean()
    std_val = df.std()
    standardized_df = (df - mean_val) / std_val
    return standardized_df


def normalize_raw_data(df, b, a):
    x_min = df.min()
    x_max = df.max()
    x_norm = (b - a) * (df - x_min) / (x_max - x_min) + a
    return x_norm


def fetch_data(level: str) -> (Tuple[List[Queryset], List[Dict[str, pd.DataFrame]]]):
    print('Fetching query sets')
    qslist = ReturnQsList(level)
    print('Fetching datasets')
    Datasets = fetch_cm_data_from_model_def(qslist)
    return qslist, Datasets


def normalize_retransform(x, min_val, max_val, b=1, a=0):
    return (x - a) / (b - a) * (max_val - min_val) + min_val


def standardize_retransform(x, mean_val, std_val):
    return x * std_val + mean_val


def transform_data(Datasets, transform, b=1, a=0, by_group=False):
    Datasets_transformed = copy.deepcopy(Datasets)
    if transform == 'raw':
        return Datasets_transformed, None

    elif transform == 'log':
        for dataset in Datasets_transformed:
            dataset['df']['ged_sb_dep'] = np.log(
                dataset['df']['ged_sb_dep'] + 1)
        return Datasets_transformed, None

    elif transform == 'normalize':
        dict_max_min = {}
        for dataset in Datasets_transformed:
            if by_group:
                min_values = dataset['df'].groupby(level='country_id')[
                    'ged_sb_dep'].min()
                max_values = dataset['df'].groupby(level='country_id')[
                    'ged_sb_dep'].max()

                dict_max_min[dataset['Name']] = pd.DataFrame(
                    {'min_val': min_values, 'max_val': max_values})

            else:
                min_values = dataset['df']['ged_sb_dep'].min()
                max_values = dataset['df']['ged_sb_dep'].max()
                dict_max_min[dataset['Name']] = pd.DataFrame(
                    {'min_val': [min_values], 'max_val': [max_values]})

            dataset['df']['ged_sb_dep'] = (b - a) * (dataset['df']['ged_sb_dep'] - min_values) / (
                max_values - min_values) + a
            dataset['df']['ged_sb_dep'].fillna(0, inplace=True)
        return Datasets_transformed, dict_max_min

    elif transform == 'standardize':
        dict_mean_std = {}
        for dataset in Datasets_transformed:
            if by_group:
                mean_values = dataset['df'].groupby(level='country_id')[
                    'ged_sb_dep'].mean()
                std_values = dataset['df'].groupby(level='country_id')[
                    'ged_sb_dep'].std()
                dict_mean_std[dataset['Name']] = pd.DataFrame(
                    {'mean_val': mean_values, 'std_val': std_values})

            else:
                mean_values = dataset['df']['ged_sb_dep'].mean()
                std_values = dataset['df']['ged_sb_dep'].std()
                dict_mean_std[dataset['Name']] = pd.DataFrame(
                    {'mean_val': [mean_values], 'std_val': [std_values]})

            dataset['df']['ged_sb_dep'] = (
                dataset['df']['ged_sb_dep'] - mean_values) / std_values

            dataset['df']['ged_sb_dep'].fillna(0, inplace=True)
        return Datasets_transformed, dict_mean_std

    else:
        raise ValueError(
            "Wrong transformation, only support 'log', 'normalize', 'standardize'.")


def get_config_path(config_path: Path) -> Path:
    common_config_path = config_path / 'common_config.py'
    wandb_config_path = config_path / 'wandb_config.py'
    model_config_path = config_path / 'model_config'
    sweep_config_path = config_path / 'sweep_config'

    if not common_config_path.is_file():
        raise FileNotFoundError(
            f'The common configuration file {common_config_path} does not exist.')
    if not wandb_config_path.is_file():
        raise FileNotFoundError(
            f'The common configuration file {wandb_config_path} does not exist.')
    if not model_config_path.exists() or not model_config_path.is_dir():
        raise FileNotFoundError(
            f'The directory {model_config_path} does not exist or is not a directory.')
    if not sweep_config_path.exists() or not sweep_config_path.is_dir():
        raise FileNotFoundError(
            f'The directory {sweep_config_path} does not exist or is not a directory.')

    return common_config_path, wandb_config_path, model_config_path, sweep_config_path


def get_config_from_path(path: Path, config_name: str) -> Dict:
    if config_name not in ['common', 'wandb', 'sweep', 'model']:
        raise ValueError(
            "Wrong configuration name, only support 'common', 'wandb', 'sweep', 'model'.")
    config = {}
    exec(path.read_text(), config)
    config_name = config_name + '_config'
    return config[config_name]


def retrain_transformed_sweep(Datasets_transformed, sweep_paras):
    modelstore = storage.Storage()
    level = wandb.config['level']
    run_id = wandb.config['run_id']
    steps = wandb.config['steps']
    transform = wandb.config['transform']
    calib_partitioner_dict = wandb.config['calib_partitioner_dict']
    test_partitioner_dict = wandb.config['test_partitioner_dict']
    future_partitioner_dict = wandb.config['future_partitioner_dict']
    force_retrain = wandb.config['force_retrain']

    # Get a suffix for identification
    suffix = ''
    for para in sweep_paras:
        suffix += f'_{para}_{wandb.config[para]}'

    # Get all the model parameters by soft coding (transform shouldn't be passed to model)
    model_paras = [para for para in sweep_paras if para != 'transform']
    parameters = {para: wandb.config[para] for para in model_paras}
    print(parameters)
    model = globals()[wandb.config['algorithm']](**parameters)

    # Training
    print(f'Training model {wandb.config["modelname"]}')

    print(f'Calibration partition ({transform})')
    RunResult_calib = RunResult.retrain_or_retrieve(
        retrain=force_retrain,
        store=modelstore,
        partitioner=DataPartitioner({"calib": calib_partitioner_dict}),
        stepshifted_models=StepshiftedModels(
            model, steps, wandb.config['depvar']),
        dataset=RetrieveFromList(
            Datasets_transformed[transform], wandb.config['data_train']),
        queryset_name=wandb.config['queryset'],
        partition_name="calib",
        timespan_name="train",
        storage_name=wandb.config['modelname'] + '_calib' + suffix,
        author_name="test0",
    )
    wandb.config.update(
        {f'predstore_calib_{transform}': level + '_' + wandb.config['modelname'] + '_calib' + suffix})
    print('Getting predictions')
    try:
        predictions_calib = pd.DataFrame.forecasts.read_store(run=run_id,
                                                              name=wandb.config[f'predstore_calib_{transform}'])
        print(predictions_calib)
    except KeyError:
        print(wandb.config[f'predstore_calib_{transform}'],
              ', run', run_id, 'does not exist, predicting')
        predictions_calib = RunResult_calib.run.predict(
            "calib", "predict", RunResult_calib.data)
        predictions_calib.forecasts.set_run(run_id)
        predictions_calib.forecasts.to_store(
            name=wandb.config[f'predstore_calib_{transform}'])

    print(f'Test partition ({transform})')
    RunResult_test = RunResult.retrain_or_retrieve(
        retrain=force_retrain,
        store=modelstore,
        partitioner=DataPartitioner({"test": test_partitioner_dict}),
        stepshifted_models=StepshiftedModels(
            model, steps, wandb.config['depvar']),
        dataset=RetrieveFromList(
            Datasets_transformed[transform], wandb.config['data_train']),
        queryset_name=wandb.config['queryset'],
        partition_name="test",
        timespan_name="train",
        storage_name=wandb.config['modelname'] + '_test' + suffix,
        author_name="test0",
    )
    wandb.config.update(
        {f'predstore_test_{transform}': level + '_' + wandb.config['modelname'] + '_test' + suffix})
    print('Getting predictions')
    try:
        predictions_test = pd.DataFrame.forecasts.read_store(run=run_id,
                                                             name=wandb.config[f'predstore_test_{transform}'])
    except KeyError:
        print(wandb.config[f'predstore_test_{transform}'],
              ', run', run_id, 'does not exist, predicting')
        predictions_test = RunResult_test.run.predict(
            "test", "predict", RunResult_test.data)
        predictions_test.forecasts.set_run(run_id)
        predictions_test.forecasts.to_store(
            name=wandb.config[f'predstore_test_{transform}'])
    print('**************************************************************')


def evaluate(target, para_transformed, retransform=True, by_group=False, b=1, a=0):
    '''
    :param target: 'calib' or 'test
    :param para_transformed: the dict that is generated by transform_data
    :param retransform: transform the data back if True
    :param retransform_by_group: transform the data back by country_id if True. Make sure it is the same value in transform_data
    '''

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
    df = pd.DataFrame.forecasts.read_store(run=run_id, name=name).replace([
        np.inf, -np.inf], 0)[stepcols]
    print(df)
# raw Outcomes evaluation - converting predictions to raw
    if retransform:
        if transform == 'log':
            df = np.exp(df) - 1
            print(df)
        elif transform == 'normalize':
            df_para_model = para_transformed[transform][wandb.config['data_train']]
            if by_group:
                df = df.apply(lambda row: normalize_retransform(row,
                                                                df_para_model['min_val'].loc[row.name[1]],
                                                                df_para_model['max_val'].loc[row.name[1]]), axis=1)
            else:
                max_val = df_para_model['max_val'].iloc[0]
                min_val = df_para_model['min_val'].iloc[0]
                df = (df - a) / (b - a) * (max_val - min_val) + min_val
        elif transform == 'standardize':
            df_para_model = para_transformed[transform][wandb.config['data_train']]

            if by_group:
                df = df.apply(lambda row: standardize_retransform(row,
                                                                  df_para_model['mean_val'].loc[row.name[1]],
                                                                  df_para_model['std_val'].loc[row.name[1]]), axis=1)
            else:
                mean = df_para_model['mean'].iloc[0]
                std = df_para_model['std'].iloc[0]
                df = df * std + mean

    df['mse'] = df.apply(lambda row: mean_squared_error([row['ged_sb_dep']] * 36,
                                                        [row[col] for col in pred_cols]), axis=1)
    print(df['mse'])
    print(df)
    print(df['mse'].mean())

    df['tloss'] = df.apply(lambda row: tweedie_loss([row['ged_sb_dep']] * 36,
                                                    [row[col] for col in pred_cols], pow=1.5, eps=np.exp(-100)), axis=1)
    df['kld'] = df.apply(lambda row: kl_divergence([row['ged_sb_dep']] * 36,
                                                   [row[col] for col in pred_cols], eps=np.exp(-100)), axis=1)
    df['jefd'] = df.apply(lambda row: jeffreys_divergence([row['ged_sb_dep']] * 36,
                                                          [row[col] for col in pred_cols], eps=np.exp(-100)), axis=1)
    df['jend'] = df.apply(lambda row: jenson_shannon_divergence([row['ged_sb_dep']] * 36,
                                                                [row[col] for col in pred_cols], eps=np.exp(-100)), axis=1)
    print(f'mse_{transform}', df['mse'].mean())
    print(f'kld_{transform}', df['kld'].mean())

    wandb.log({'mse_raw': df['mse'].mean()})
    wandb.log({'tloss': df['tloss'].mean()})
    wandb.log({'kld': df['kld'].mean()})
    wandb.log({'jefd': df['jefd'].mean()})
    wandb.log({'jend': df['jend'].mean()})
    print('**************************************************************')

    # try:
    #     df['mse_log'] = df.apply(lambda row: mean_squared_error(
    #         [log_transform_raw_data(row['ged_sb_dep'])] * 36,
    #         [log_transform_raw_data(row[col]) for col in pred_cols]), axis=1)
    # except Exception as e:
    #     traceback.print_exc()
    print('the transformed mse', df)
    df.to_csv('raw.csv')

    #####
    for step_number in [1, 3, 6, 9, 12, 36]:
        wandb.log({f'mse_raw_step_{step_number}': mean_squared_error(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        # add other metrics here too
        wandb.log({f'tloss_step_{step_number}': tweedie_loss(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'kld_step_{step_number}': kl_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jefd_step_{step_number}': jeffreys_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jend_step_{step_number}': jenson_shannon_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})

    ####

    # +++++++++++++++++++LOG
    df = pd.DataFrame.forecasts.read_store(run=run_id, name=name).replace([
        np.inf, -np.inf], 0)[stepcols]
# raw Outcomes evaluation - converting predictions to raw
    if retransform:
        if transform == 'log':
            df = np.exp(df) - 1
        elif transform == 'normalize':
            df_para_model = para_transformed[transform][wandb.config['data_train']]
            if by_group:
                df = df.apply(lambda row: normalize_retransform(row,
                                                                df_para_model['min_val'].loc[row.name[1]],
                                                                df_para_model['max_val'].loc[row.name[1]]), axis=1)
            else:
                max_val = df_para_model['max_val'].iloc[0]
                min_val = df_para_model['min_val'].iloc[0]
                df = (df - a) / (b - a) * (max_val - min_val) + min_val
        elif transform == 'standardize':
            df_para_model = para_transformed[transform][wandb.config['data_train']]

            if by_group:
                df = df.apply(lambda row: standardize_retransform(row,
                                                                  df_para_model['mean_val'].loc[row.name[1]],
                                                                  df_para_model['std_val'].loc[row.name[1]]), axis=1)
            else:
                mean = df_para_model['mean'].iloc[0]
                std = df_para_model['std'].iloc[0]
                df = df * std + mean

    print(np.log(df+1).isna().any().any())
    df = np.log(df+1)
    df.fillna(0, inplace=True)
    df['mse_log'] = df.apply(lambda row: mean_squared_error(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)

    print(f'mse_log_{transform}', df['mse_log'].mean())
    # add other metrics here too
    df['tloss_log'] = df.apply(lambda row: tweedie_loss(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)
    df['kld_log'] = df.apply(lambda row: kl_divergence(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)
    df['jefd_log'] = df.apply(lambda row: jeffreys_divergence(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)
    df['jend_log'] = df.apply(lambda row: jenson_shannon_divergence(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)

    wandb.log({'mse_log': df['mse_log'].mean()})
    wandb.log({'tloss_log': df['tloss_log'].mean()})
    wandb.log({'kld_log': df['kld_log'].mean()})
    wandb.log({'jefd_log': df['jefd_log'].mean()})
    wandb.log({'jend_log': df['jend_log'].mean()})

    print('the transformed mse', df)
    df.to_csv('log.csv')
    #####
    for step_number in [1, 3, 6, 9, 12, 36]:
        wandb.log({f'mse_log_step_{step_number}': mean_squared_error(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
    # add other metrics here too
        wandb.log({f'tloss_log_step_{step_number}': tweedie_loss(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'kld_log_step_{step_number}': kl_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jefd_log_step_{step_number}': jeffreys_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jend_log_step_{step_number}': jenson_shannon_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})

    ####

# ++++++++++++++++++++++++++++ STANDARDIZE

    df = pd.DataFrame.forecasts.read_store(run=run_id, name=name).replace([
        np.inf, -np.inf], 0)[stepcols]
    # raw Outcomes evaluation - converting predictions to raw
    if retransform:
        if transform == 'log':
            df = np.exp(df) - 1
        elif transform == 'normalize':
            df_para_model = para_transformed[transform][wandb.config['data_train']]
            if by_group:
                df = df.apply(lambda row: normalize_retransform(row,
                                                                df_para_model['min_val'].loc[row.name[1]],
                                                                df_para_model['max_val'].loc[row.name[1]]), axis=1)
            else:
                max_val = df_para_model['max_val'].iloc[0]
                min_val = df_para_model['min_val'].iloc[0]
                df = (df - a) / (b - a) * (max_val - min_val) + min_val
        elif transform == 'standardize':
            df_para_model = para_transformed[transform][wandb.config['data_train']]

            if by_group:
                df = df.apply(lambda row: standardize_retransform(row,
                                                                  df_para_model['mean_val'].loc[row.name[1]],
                                                                  df_para_model['std_val'].loc[row.name[1]]), axis=1)
            else:
                mean = df_para_model['mean'].iloc[0]
                std = df_para_model['std'].iloc[0]
                df = df * std + mean
    df = standardize_raw_data(df)
    df['mse_standardize'] = df.apply(lambda row: mean_squared_error(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)

    print(f'mse_standardize_{transform}', df['mse_standardize'].mean())
    # add other metrics here too
    df['tloss_standardize'] = df.apply(lambda row: tweedie_loss(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)
    df['kld_standardize'] = df.apply(lambda row: kl_divergence(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)
    df['jefd_standardize'] = df.apply(lambda row: jeffreys_divergence(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)
    df['jend_standardize'] = df.apply(lambda row: jenson_shannon_divergence(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)

    wandb.log({'mse_standardize': df['mse_standardize'].mean()})
    wandb.log({'tloss_standardize': df['tloss_standardize'].mean()})
    wandb.log({'kld_standardize': df['kld_standardize'].mean()})
    wandb.log({'jefd_standardize': df['jefd_standardize'].mean()})
    wandb.log({'jend_standardize': df['jend_standardize'].mean()})

    print('the transformed mse', df)
    df.to_csv('stan.csv')

    #####
    for step_number in [1, 3, 6, 9, 12, 36]:
        wandb.log({f'mse_standardize_step_{step_number}': mean_squared_error(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        # add other metrics here too
        wandb.log({f'tloss_standardize_step_{step_number}': tweedie_loss(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'kld_standardize_step_{step_number}': kl_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jefd_standardize_step_{step_number}': jeffreys_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jend_standardize_step_{step_number}': jenson_shannon_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})

    ####

    # +++++++++++++++++Normalize
    df = pd.DataFrame.forecasts.read_store(run=run_id, name=name).replace([
        np.inf, -np.inf], 0)[stepcols]
    # raw Outcomes evaluation - converting predictions to raw
    if retransform:
        if transform == 'log':
            df = np.exp(df) - 1
        elif transform == 'normalize':
            df_para_model = para_transformed[transform][wandb.config['data_train']]
            if by_group:
                df = df.apply(lambda row: normalize_retransform(row,
                                                                df_para_model['min_val'].loc[row.name[1]],
                                                                df_para_model['max_val'].loc[row.name[1]]), axis=1)
            else:
                max_val = df_para_model['max_val'].iloc[0]
                min_val = df_para_model['min_val'].iloc[0]
                df = (df - a) / (b - a) * (max_val - min_val) + min_val
        elif transform == 'standardize':
            df_para_model = para_transformed[transform][wandb.config['data_train']]

            if by_group:
                df = df.apply(lambda row: standardize_retransform(row,
                                                                  df_para_model['mean_val'].loc[row.name[1]],
                                                                  df_para_model['std_val'].loc[row.name[1]]), axis=1)
            else:
                mean = df_para_model['mean'].iloc[0]
                std = df_para_model['std'].iloc[0]
                df = df * std + mean
    df = normalize_raw_data(df, 1, 0)
    df['mse_normalize'] = df.apply(lambda row: mean_squared_error(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)

    print(f'mse_normalize_{transform}', df['mse_normalize'].mean())
    # add other metrics here too
    df['tloss_normalize'] = df.apply(lambda row: tweedie_loss(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)
    df['kld_normalize'] = df.apply(lambda row: kl_divergence(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)
    df['jefd_normalize'] = df.apply(lambda row: jeffreys_divergence(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)
    df['jend_normalize'] = df.apply(lambda row: jenson_shannon_divergence(
        [row['ged_sb_dep']] * 36, [row[col] for col in pred_cols]), axis=1)

    wandb.log({'mse_normalize': df['mse_normalize'].mean()})
    wandb.log({'tloss_normalize': df['tloss_normalize'].mean()})
    wandb.log({'kld_normalize': df['kld_normalize'].mean()})
    wandb.log({'jefd_normalize': df['jefd_normalize'].mean()})
    wandb.log({'jend_normalize': df['jend_normalize'].mean()})
    print('the transformed mse', df)
    df.to_csv('norm.csv')

    #####
    for step_number in [1, 3, 6, 9, 12, 36]:
        wandb.log({f'mse_normalize_step_{step_number}': mean_squared_error(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        # add other metrics here too
        wandb.log({f'tloss_normalize_step_{step_number}': tweedie_loss(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'kld_normalize_step_{step_number}': kl_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jefd_normalize_step_{step_number}': jeffreys_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jend_normalize_step_{step_number}': jenson_shannon_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
    ####
