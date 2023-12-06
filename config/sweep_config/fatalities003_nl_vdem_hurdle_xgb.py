sweep_configuration = {
    "name": 'fatalities003_nl_vdem_hurdle_xgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_vdem_hurdle_xgb'},
        'algorithm': {'value': 'HurdleRegression'},
        "transform": {"values": ['log', 'standardize', 'raw', 'normalize']},
        'clf_name': {'value': 'XGBClassifier'},
        'reg_name': {'value': 'XGBRegressor'},
        'depvar': {'value': 'ged_sb_dep'},
        'data_train': {'value': 'vdem_short'},
        'queryset': {'value': 'fatalities003_vdem_short'},
        'preprocessing': {'value': 'float_it'},
        'level': {'value': 'cm'},
        'description': {'value': ''},
        'long_description': {'value': ''}
    }
}
