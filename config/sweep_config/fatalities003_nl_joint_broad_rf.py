sweep_configuration = {
    "name": 'fatalities003_nl_joint_broad_rf',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_joint_broad_rf'},
        'algorithm': {'value': 'XGBRFRegressor'},
        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'n_estimators': {'values': [250, 100]},
        'n_jobs': {'values': [12, 32]},  # Assuming nj is defined earlier
        "learning_rate": {"values": [0.05, 0.1, 0.2]},

        'depvar': {'value': ['ged_sb_dep']},
        'data_train': {'value': 'joint_broad'},
        'queryset': {'value': 'fatalities003_joint_broad'},
        'preprocessing': {'value': 'float_it'},
        'level': {'value': 'cm'},
        'description': {'value': ''},
        'long_description': {'value': ''}
    }
}
