sweep_configuration = {
    "name": 'fatalities003_nl_baseline_rf',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_baseline_rf'},
        'algorithm': {'value': 'XGBRFRegressor'},
        "transform": {"values": ['log', 'standardize', 'raw', 'normalize']},
        "n_estimators": {"values": [100, 200, 300]},
        "n_jobs": {"values": [12, 13]},
        "learning_rate": {"values": [0.05, 0.1, 0.2]},
        'max_depth': {'values': [12, 13]},
        'min_child_weight': {'values': [12, 13]},
        'subsample': {'values': [0.5, 0.7]},
        'colsample_bytree': {'values': [0.5, 0.7]},
        'depvar': {'value': 'ged_sb_dep'},
        'data_train': {'value': 'baseline003'},
        'queryset': {'value': 'fatalities003_baseline'},
        'preprocessing': {'value': 'float_it'},
        'level': {'value': 'cm'},
        'description': {'value': 'Baseline model with a few conflict history features as well as log population, random forests regression model.'},
        'long_description': {'value': 'A very simple model with only five data columns (each column representing one feature): The number of fatalities in the same country at $t-1$, three decay functions of time since there was at least five fatalities in a single month, for each of the UCDP conflict types -- state-based, one-sided, or non-state conflict -- and log population size (Hegre2020RP, Pettersson2021JPR).'}
    }
}
