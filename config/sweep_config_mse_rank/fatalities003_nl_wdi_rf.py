sweep_config = {
    "name": 'fatalities003_nl_wdi_rf',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {

        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'n_estimators': {'values': [300]},
        'n_jobs': {'values': [12]},
        "learning_rate": {"values": [0.05]},



    }
}
