sweep_configuration = {
    "name": 'fatalities003_nl_conflicthistory_hurdle_lgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': ['fatalities003_nl_conflicthistory_hurdle_lgb']},
        'algorithm': {'value': ['HurdleRegression']},
        "transform": {"values": ['mse_calib_log', 'mse_calib_standardize', 'mse_calib','mse_calib_normalize']},
        #"n_estimators": {"values": [100, 200, 300]},
        #"n_jobs": {"values": [12]},
        #"learning_rate": {"values": [0.05, 0.1, 0.2]}
        'clf_name': {'value':['LGBMClassifier']},
        'reg_name': {'value':['LGBMRegressor']},
        'depvar': {'value':['ged_sb_dep']},
        'data_train': {'value':['conflict_ln']},
        'queryset': {'value':['fatalities003_conflict_history']},
        'preprocessing': {'value':['float_it']},
        'level': {'value':['cm']},
        'description': {'value': ['']},
        'long_description': {'value':['']}
    }
}
