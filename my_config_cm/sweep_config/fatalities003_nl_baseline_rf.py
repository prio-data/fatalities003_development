sweep_config = {
    "name": 'fatalities003_nl_baseline_rf',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {

        "transform": {"values": ['log', 'standardize', 'raw', 'normalize']},
        "n_estimators": {"values": [100]},
        "n_jobs": {"values": [12]},
        "learning_rate": {"values": [0.05]},
        'max_depth': {'values': [12]},
        'min_child_weight': {'values': [12]},
        'subsample': {'values': [0.5]},
        'colsample_bytree': {'values': [0.5]},
    }
}
