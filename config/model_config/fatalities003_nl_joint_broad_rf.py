model_config = {
    'modelname': 'fatalities003_nl_joint_broad_rf',
    'algorithm': 'XGBRFRegressor',
    'n_estimators': 250,
    'n_jobs': 12,  # Assuming nj is defined somewhere earlier in your code
    'depvar': 'ged_sb_dep',
    'data_train': 'joint_broad',
    'queryset': 'fatalities003_joint_broad',
    'preprocessing': 'float_it',
    'level': 'cm',
    'description': '',
    'long_description': ''
}
