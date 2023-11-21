sweep_configuration = {
    "name": 'fatalities003_nl_topics_rf',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_topics_rf'},
        'algorithm': {'value': 'XGBRFRegressor'},
        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'n_estimators': {'values': [250,230]},
        'n_jobs': {'values': [10,12]},  # Assuming nj is defined earlier
        'depvar': {'value': 'ged_sb_dep'},
        'data_train': {'value': 'topics_003'},
        'queryset': {'value': 'fatalities003_topics'},
        'preprocessing': {'value': 'float_it'},
        'level': {'value': 'cm'},
        'description': {'value': ''},
        'long_description': {'value': ''}
    }
}
