sweep_configuration = {
    "name": 'fatalities003_nl_topics_xgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_topics_xgb'},
        'algorithm': {'value': 'XGBRegressor'},
        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'n_estimators': {'values': [80, 90]},
        'learning_rate': {'values': [0.05, 1]},
        'n_jobs': {'values': [12, 13]},
        'max_depth': {'values': [12, 13]},
        'min_child_weight': {'values': [12, 13]},
        'subsample': {'values': [0.5, 0.7]},
        'colsample_bytree': {'values': [0.5, 0.7]},
        'depvar': {'value': 'ged_sb_dep'},
        'data_train': {'value': 'topics_003'},
        'queryset': {'value': 'fatalities003_topics'},
        'preprocessing': {'value': 'float_it'},
        'level': {'value': 'cm'},
        'description': {'value': ''},
        'long_description': {'value': ''}
    }
}
