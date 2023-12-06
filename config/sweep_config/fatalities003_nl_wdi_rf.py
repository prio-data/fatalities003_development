sweep_configuration = {
    "name": 'fatalities003_nl_wdi_rf',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_wdi_rf'},
        'algorithm': {'value': 'XGBRFRegressor'},
        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'n_estimators': {'values': [300, 100]},
        'n_jobs': {'values': [12, 15]},
        "learning_rate": {"values": [0.05, 0.1, 0.2]},
        'max_depth': {'values': [12, 13]},
        'min_child_weight': {'values': [12, 13]},
        'subsample': {'values': [0.5, 0.7]},
        'colsample_bytree': {'values': [0.5, 0.7]},

        'depvar': {'value': 'ged_sb_dep'},
        'data_train': {'value': 'wdi_short'},
        'queryset': {'value': 'fatalities003_wdi_short'},
        'preprocessing': {'value': 'float_it'},
        'level': {'value': 'cm'},
        'description': {'value': ''},
        'long_description': {'value': ''}
    }
}
