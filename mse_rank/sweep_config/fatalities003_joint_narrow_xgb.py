sweep_config = {
    'name': 'fatalities003_joint_narrow_xgb',
    'method': 'grid',
    'metric': {
        'goal': 'minimize',
        'name': 'mse',
    },
    'parameters': {

        'transform': {'values': ['log', 'standardize', 'raw', 'normalize']},
        'n_estimators': {'values': [250]},
        'learning_rate': {'values': [0.1]},
        'n_jobs': {'values': [12]},


    }
}
