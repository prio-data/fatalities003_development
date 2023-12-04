sweep_configuration = {
    "name": 'fatalities003_joint_narrow_xgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_joint_narrow_xgb'},
        'algorithm': {'value': 'XGBRFRegressor'},
        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'n_estimators': {'values': [250, 122]},
        "learning_rate": {"values": [0.05, 0.1, 0.2]},

        'n_jobs': {'values': [12, 13]},  # Assuming nj is defined earlier
        'depvar': {'value': 'ged_sb_dep'},
        'data_train': {'value': 'joint_narrow'},
        'queryset': {'value': 'fatalities003_joint_narrow'},
        'preprocessing': {'value': 'float_it'},
        'level': {'value': 'cm'},
        'description': {'value': ''},
        'long_description': {'value': ''}
    }
}
