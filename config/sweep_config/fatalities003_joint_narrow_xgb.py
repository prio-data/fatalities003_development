sweep_configuration = {
    "name": 'fatalities003_joint_narrow_xgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {

        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'n_estimators': {'values': [250, 122]},
        "learning_rate": {"values": [0.05, 0.1, 0.2]},
        'n_jobs': {'values': [12, 13]},
        'max_depth': {'values': [12, 13]},
        'min_child_weight': {'values': [12, 13]},
        'subsample': {'values': [0.5, 0.7]},
        'colsample_bytree': {'values': [0.5, 0.7]},

    }
}
