model_config = {
    'modelname': 'fatalities003_nl_all_pca3_xgb',
    'algorithm': 'XGBRegressor',
    'n_estimators': 100,
    'learning_rate': 0.05,
    'n_jobs': 12,  # Assuming nj is defined somewhere earlier in your code
    'depvar': 'ged_sb_dep',
    'data_train': 'all_features',
    'queryset': 'fatalities003_all_features',
    'preprocessing': 'pca_it',
    'level': 'cm',
    'description': '',
    'long_description': ''
}