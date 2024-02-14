import os
import wandb
import copy
from pathlib import Path
import warnings

from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
os.environ['WANDB_SILENT'] = 'true'
from diskcache import Cache
cache = Cache('./cache', size_limit=100000000000)

from views_runs import storage, StepshiftedModels
from views_partitioning.data_partitioner import DataPartitioner
from views_runs.run_result import RunResult
from views_forecasts.extensions import *
from dataloader.FetchData import *
from ViewsEstimators import *
from util.utils_map import GeoPlotter
from util.new_metrics import *


def fetch_data(level: str) -> (Tuple[List[Queryset], List[Dict[str, pd.DataFrame]]]):
    print('Fetching query sets')
    qslist = ReturnQsList(level)
    print('Fetching datasets')
    if level == 'cm':
        Datasets = fetch_cm_data_from_model_def(qslist)
    elif level == 'pgm':
        Datasets = fetch_pgm_data_from_model_def(qslist)
    return qslist, Datasets

@cache.memoize(typed=True, expire=None, tag='data')
def i_fetch_data(level):
    return fetch_data(level)


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


def normalize_retransform(x, min_val, max_val, b=1, a=0):
    return (x - a) / (b - a) * (max_val - min_val) + min_val


def standardize_retransform(x, mean_val, std_val):
    return x * std_val + mean_val


def transform_data(Datasets, transforms, level, b=1, a=0, by_group=True):
    Datasets_transformed = {}
    para_transformed = {}

    if level == 'cm':
        target = 'country_id'
    elif level == 'pgm':
        target = 'priogrid_gid'

    for transform in transforms:
        Datasets_copy = copy.deepcopy(Datasets)
        if transform == 'raw':
            Datasets_transformed[transform] = Datasets_copy
            para_transformed[transform] = None

        elif transform == 'log':
            for dataset in Datasets_copy:
                dataset['df']['ged_sb_dep'] = np.log(dataset['df']['ged_sb_dep'] + 1)
            Datasets_transformed[transform] = Datasets_copy
            para_transformed[transform] = None

        elif transform == 'normalize':
            dict_max_min = {}
            for dataset in Datasets_copy:
                if by_group:
                    min_values = dataset['df'].groupby(level=target)['ged_sb_dep'].min()
                    max_values = dataset['df'].groupby(level=target)['ged_sb_dep'].max()

                    dict_max_min[dataset['Name']] = pd.DataFrame({'min_val': min_values, 'max_val': max_values})

                else:
                    min_values = dataset['df']['ged_sb_dep'].min()
                    max_values = dataset['df']['ged_sb_dep'].max()
                    dict_max_min[dataset['Name']] = pd.DataFrame({'min_val': [min_values], 'max_val': [max_values]})

                dataset['df']['ged_sb_dep'] = (b - a) * (dataset['df']['ged_sb_dep'] - min_values) / (
                            max_values - min_values) + a
                dataset['df']['ged_sb_dep'].fillna(0, inplace=True)
            Datasets_transformed[transform] = Datasets_copy
            para_transformed[transform] = dict_max_min

        elif transform == 'standardize':
            dict_mean_std = {}
            for dataset in Datasets_copy:
                if by_group:
                    mean_values = dataset['df'].groupby(level=target)['ged_sb_dep'].mean()
                    std_values = dataset['df'].groupby(level=target)['ged_sb_dep'].std()
                    dict_mean_std[dataset['Name']] = pd.DataFrame({'mean_val': mean_values, 'std_val': std_values})

                else:
                    mean_values = dataset['df']['ged_sb_dep'].mean()
                    std_values = dataset['df']['ged_sb_dep'].std()
                    dict_mean_std[dataset['Name']] = pd.DataFrame({'mean_val': [mean_values], 'std_val': [std_values]})

                dataset['df']['ged_sb_dep'] = (dataset['df']['ged_sb_dep'] - mean_values) / std_values
                dataset['df']['ged_sb_dep'].fillna(0, inplace=True)
            Datasets_transformed[transform] = Datasets_copy
            para_transformed[transform] = dict_mean_std

        else:
            raise ValueError("Wrong transformation, only support 'raw', 'log', 'normalize', 'standardize'.")
    return Datasets_transformed, para_transformed


def get_config_path(config_path: Path) -> Path:
    common_config_path = config_path / 'common_config.py'
    wandb_config_path = config_path / 'wandb_config.py'
    model_config_path = config_path / 'model_config'
    sweep_config_path = config_path / 'sweep_config'

    if not common_config_path.is_file():
        raise FileNotFoundError(f'The common configuration file {common_config_path} does not exist.')
    if not wandb_config_path.is_file():
        raise FileNotFoundError(f'The wandb configuration file {wandb_config_path} does not exist.')
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


def steps_evaluation(df, steps, transform='raw'):
    for step_number in steps:
        wandb.log({f'mse_{transform}_step_{step_number}': mean_squared_error(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        # add other metrics here too
        wandb.log({f'tloss_{transform}_step_{step_number}': tweedie_loss(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'kld_{transform}_step_{step_number}': kl_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jefd_{transform}_step_{step_number}': jeffreys_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})
        wandb.log({f'jend_{transform}_step_{step_number}': jenson_shannon_divergence(
            df['ged_sb_dep'], df[f'step_pred_{step_number}'])})


def metrics_evaluation(df, pred_cols, transform='raw'):
    df['mse'] = df.apply(lambda row: mean_squared_error([row['ged_sb_dep']] * 36,
                                                        [row[col] for col in pred_cols]), axis=1)


    df['tloss'] = df.apply(lambda row: tweedie_loss([row['ged_sb_dep']] * 36,
                                                    [row[col] for col in pred_cols], pow=1.5, eps=np.exp(-100)), axis=1)
    df['kld'] = df.apply(lambda row: kl_divergence([row['ged_sb_dep']] * 36,
                                                   [row[col] for col in pred_cols], eps=np.exp(-100)), axis=1)
    df['jefd'] = df.apply(lambda row: jeffreys_divergence([row['ged_sb_dep']] * 36,
                                                          [row[col] for col in pred_cols], eps=np.exp(-100)), axis=1)
    df['jend'] = df.apply(lambda row: jenson_shannon_divergence([row['ged_sb_dep']] * 36,
                                                                [row[col] for col in pred_cols], eps=np.exp(-100)), axis=1)

    wandb.log({f'mse_{transform}': df['mse'].mean()})
    wandb.log({f'tloss_{transform}': df['tloss'].mean()})
    wandb.log({f'kld_{transform}': df['kld'].mean()})
    wandb.log({f'jefd_{transform}': df['jefd'].mean()})
    wandb.log({f'jend_{transform}': df['jend'].mean()})

    print(f'''mse_{transform}: {df["mse"].mean()}, 
          tloss_{transform}: {df["tloss"].mean()}, 
          kld_{transform}: {df["kld"].mean()}, 
          jefd_{transform}: {df["jefd"].mean()}, 
          jend_{transform}: {df["jend"].mean()}''')


def retransform_data(df, transform, para_transformed, by_group=True, b=1, a=0):
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
    return df


def retrain_transformed_sweep(Datasets_transformed, model_paras):
    modelstore = storage.Storage()
    run_id = wandb.config['run_id']
    steps = wandb.config['steps']
    transform = wandb.config['transform']
    calib_partitioner_dict = wandb.config['calib_partitioner_dict']
    test_partitioner_dict = wandb.config['test_partitioner_dict']
    future_partitioner_dict = wandb.config['future_partitioner_dict']
    force_retrain = wandb.config['force_retrain']

    # Get a suffix for identification
    suffix = f'_transform_{wandb.config["transform"]}'
    for para in model_paras:
        suffix += f'_{para}_{wandb.config[para]}'

    # Get all the model parameters (transform shouldn't be passed to model)
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
        stepshifted_models=StepshiftedModels(model, steps, wandb.config['depvar']),
        dataset=RetrieveFromList(Datasets_transformed[transform], wandb.config['data_train']),
        queryset_name=wandb.config['queryset'],
        partition_name="calib",
        timespan_name="train",
        storage_name=wandb.config['modelname'] + '_calib' + suffix,
        author_name="test0",
    )
    wandb.config.update(
        {f'predstore_calib_{transform}': wandb.config['modelname'] + '_calib' + suffix})
    print('Getting predictions')
    try:
        predictions_calib = pd.DataFrame.forecasts.read_store(run=run_id,
                                                              name=wandb.config[f'predstore_calib_{transform}'])
    except KeyError:
        print(wandb.config[f'predstore_calib_{transform}'], ', run', run_id, 'does not exist, predicting')
        predictions_calib = RunResult_calib.run.predict("calib", "predict", RunResult_calib.data)
        predictions_calib.forecasts.set_run(run_id)
        predictions_calib.forecasts.to_store(name=wandb.config[f'predstore_calib_{transform}'])

    print(f'Test partition ({transform})')
    RunResult_test = RunResult.retrain_or_retrieve(
        retrain=force_retrain,
        store=modelstore,
        partitioner=DataPartitioner({"test": test_partitioner_dict}),
        stepshifted_models=StepshiftedModels(model, steps, wandb.config['depvar']),
        dataset=RetrieveFromList(Datasets_transformed[transform], wandb.config['data_train']),
        queryset_name=wandb.config['queryset'],
        partition_name="test",
        timespan_name="train",
        storage_name=wandb.config['modelname'] + '_test' + suffix,
        author_name="test0",
    )
    wandb.config.update(
        {f'predstore_test_{transform}': wandb.config['modelname'] + '_test' + suffix})
    print('Getting predictions')
    try:
        predictions_test = pd.DataFrame.forecasts.read_store(run=run_id,
                                                             name=wandb.config[f'predstore_test_{transform}'])
    except KeyError:
        print(wandb.config[f'predstore_test_{transform}'], ', run', run_id, 'does not exist, predicting')
        predictions_test = RunResult_test.run.predict("test", "predict", RunResult_test.data)
        predictions_test.forecasts.set_run(run_id)
        predictions_test.forecasts.to_store(name=wandb.config[f'predstore_test_{transform}'])


def evaluate(target, para_transformed, retransform=True, by_group=False, b=1, a=0, plot_map=False):
    '''
    :param target: 'calib' or 'test'
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
    level = wandb.config['level']
    stepcols = [wandb.config['depvar']]

    for step in steps:
        stepcols.append('step_pred_' + str(step))
    pred_cols = [f'step_pred_{str(i)}' for i in steps]

    step_eval = [1, 3, 6, 9, 12, 36] # target steps for evaluation

    name = wandb.config[f'predstore_{target}_{transform}']
    df = pd.DataFrame.forecasts.read_store(run=run_id, name=name).replace([np.inf, -np.inf], 0)[stepcols]

    # Retransform the data (save for HH's idea)
    df_retransform = retransform_data(df, transform, para_transformed, by_group, b, a)
    
    # raw outcomes evaluation 
    if retransform:
        df = df_retransform.copy(deep=True)

    if plot_map:
        plotter = GeoPlotter()
        months = df.index.levels[0].tolist()
        step_preds = [f'step_pred_{i}' for i in steps]
        # Temporarily only predict the first month
        for month in [months[0]]:
            for step in step_preds:
                if level == 'cm':
                    fig = plotter.plot_cm_map(df, month, step, transform)
                elif level == 'pgm':
                    fig = plotter.plot_pgm_map(df, month, step, transform)
                wandb.log({f'month_{month}_{step}': wandb.Image(fig)})
                

    metrics_evaluation(df, pred_cols)
    steps_evaluation(df, step_eval)

    print('mse', df['mse'].mean())
    wandb.log({'mse': df['mse'].mean()})

    # df.to_csv('raw.csv')

    """ 
    As requested by HH, after retransforming the data back, we need to transform the predictions again using all the transformations, 
    i.e., raw, log, normalize, standardize. 
    """
    # ++++++++++++++++++++ Log
    df = df_retransform.copy(deep=True)
    df = np.log(df+1)
    df.fillna(0, inplace=True)

    metrics_evaluation(df, pred_cols, transform='log')
    steps_evaluation(df, step_eval, transform='log')

    # df.to_csv('log.csv')

    # ++++++++++++++++++++ Standardize
    df = df_retransform.copy(deep=True) 
    df = standardize_raw_data(df)

    metrics_evaluation(df, pred_cols, transform='standardize')
    steps_evaluation(df, step_eval, transform='standardize')

    # df.to_csv('stan.csv')

    #++++++++++++++++++++ Normalize
    df = df_retransform.copy(deep=True)
    df = normalize_raw_data(df, b, a)

    metrics_evaluation(df, pred_cols, transform='normalize')
    steps_evaluation(df, step_eval, transform='normalize')

    # df.to_csv('norm.csv')  
