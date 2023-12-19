sweep_config = {
    "name": 'fatalities003_nl_all_pca3_xgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {

        "transform": {"values": ['log', 'standardize', 'raw', 'normalize']},
        'n_estimators': {'values': [100]},
        'learning_rate': {'values': [0.05]},
        'n_jobs': {'values': [12]},


    }
}
