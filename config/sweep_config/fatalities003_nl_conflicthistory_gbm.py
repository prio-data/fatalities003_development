sweep_config = {
    'name': 'fatalities003_nl_conflicthistory_gbm',
    'method': 'grid',
    'metric': {
        'goal': 'minimize',
        'name': 'mse',
    },
    'parameters': {

        'transform': {'values': ['log', 'standardize', 'raw', 'normalize']},
        'n_estimators': {'values': [100, 200, 300]},
        'n_jobs': {'values': [12]},
        'learning_rate': {'values': [0.05, 0.1, 0.2]},
        'max_depth': {'values': [1, 3]},
        'min_samples_split': {'values': [1, 2]},
        'min_samples_leaf': {'values': [1, 3]},

    }
}
