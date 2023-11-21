sweep_configuration = {
    "name": 'fatalities003_nl_wdi_rf',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': ['fatalities003_nl_wdi_rf']},
        'algorithm': {'value': ['XGBRFRegressor']},
        "transform": {
            "values": ['mse_calib_log', 'mse_calib_standardize', 'mse_calib', 'mse_calib_normalize']
        },
        'n_estimators': {'values': [300,100]},
        'n_jobs': {'values': [12,15]},  # Assuming nj is defined earlier
        'depvar': {'value': ['ged_sb_dep']},
        'data_train': {'value': ['wdi_short']},
        'queryset': {'value': ['fatalities003_wdi_short']},
        'preprocessing': {'value': ['float_it']},
        'level': {'value': ['cm']},
        'description': {'value': ['']},
        'long_description': {'value': ['']}
    }
}
