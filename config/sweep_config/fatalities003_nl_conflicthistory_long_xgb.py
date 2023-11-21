sweep_configuration = {
    "name": 'fatalities003_nl_conflicthistory_long_xgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value':'fatalities003_nl_conflicthistory_long_xgb'},
        'algorithm': {'value':'XGBRegressor'},
        "transform": {"values": ['log', 'standardize', 'raw', 'normalize']},
        "n_estimators": {"values": [100, 200, 300]},
        "n_jobs": {"values": [12,13]},
        "learning_rate": {"values": [0.05, 0.1, 0.2]},
        'depvar': {'value':'ged_sb_dep'},
        'data_train': {'value':'conflictlong_ln'},
        'queryset': {'value':'fatalities003_conflict_history_long'},
        'preprocessing': {'value':'float_it'},
        'level': {'value':'cm'},
        'description': {'value':''},
        'long_description': {'value':''}
    }
}
