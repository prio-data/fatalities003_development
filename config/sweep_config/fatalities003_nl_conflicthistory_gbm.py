sweep_configuration = {
    "name": 'fatalities003_nl_conflicthistory_gbm',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value':['fatalities003_nl_conflicthistory_gbm']},
        'algorithm': {'value':['GradientBoostingRegressor']},
        "transform": {"values": ['mse_calib_log', 'mse_calib_standardize', 'mse_calib','mse_calib_normalize']},
        "n_estimators": {"values": [100, 200, 300]},
        #"n_jobs": {"values": [12]},
        #"learning_rate": {"values": [0.05, 0.1, 0.2]}
        'depvar': {'value':['ged_sb_dep']},
        'data_train': {'value':['conflict_ln']},
        'queryset': {'value':['fatalities003_conflict_history']},
        'preprocessing': {'value':['float_it']},
        'level': {'value':['cm']},
        'description': {'value': ['A collection of variables that together map the conflict history of a country, scikit gradient boosting regression model.']},
        'long_description': {'value':['']}
    }
}
