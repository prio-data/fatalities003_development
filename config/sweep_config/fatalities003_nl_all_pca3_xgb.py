sweep_configuration = {
    "name": 'fatalities003_nl_all_pca3_xgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_all_pca3_xgb'},
        'algorithm': {'value': 'XGBRegressor'},
        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']},
        'n_estimators': {'values': [100,122]},
        'learning_rate': {'values': [0.05,1]},
        'n_jobs': {'values': [12,13]},  # Assuming nj is defined earlier
        'depvar': {'value': 'ged_sb_dep'},
        'data_train': {'value': 'all_features'},
        'queryset': {'value': 'fatalities003_all_features'},
        'preprocessing': {'value': 'pca_it'},
        'level': {'value': 'cm'},
        'description': {'value': ''},
        'long_description': {'value': ''}
    }
}
