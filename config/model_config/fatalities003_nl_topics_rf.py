model_config = {
    'modelname': 'fatalities003_nl_topics_rf',
    'algorithm': 'XGBRFRegressor',
    'n_estimators': 250,
    'n_jobs': 12,  # Assuming nj is defined somewhere earlier in your code
    'depvar': 'ged_sb_dep',
    'data_train': 'topics_003',
    'queryset': 'fatalities003_topics',
    'preprocessing': 'float_it',
    'level': 'cm',
    'description': '',
    'long_description': ''
}