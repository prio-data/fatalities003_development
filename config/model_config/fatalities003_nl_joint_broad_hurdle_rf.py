model_config = {
    'modelname': 'fatalities003_nl_wdi_rf',
    'algorithm': 'XGBRFRegressor',
    'n_estimators': 300,
    'n_jobs': 12,  # Assuming nj is defined somewhere earlier in your code
    'depvar': "ged_sb_dep",
    'data_train': 'wdi_short',
    'queryset': "fatalities003_wdi_short",
    'preprocessing': 'float_it',
    'level': 'cm',
    'description': '',
    'long_description': ''
}
