sweep_configuration = {
    "name": 'fatalities003_nl_joint_broad_hurdle_rf',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': ['fatalities003_nl_joint_broad_hurdle_rf']},
        'algorithm': {'value': ['HurdleRegression']},
        "transform": {
            "values": ['mse_calib_log', 'mse_calib_standardize', 'mse_calib', 'mse_calib_normalize']
        },
        'clf_name': {'value': ['RFClassifier']},
        'reg_name': {'value': ['RFRegressor']},
        'depvar': {'value': ['ged_sb_dep']},
        'data_train': {'value': ['joint_broad']},
        'queryset': {'value': ['fatalities003_joint_broad']},
        'preprocessing': {'value': ['float_it']},
        'level': {'value': ['cm']},
        'description': {'value': ['']},
        'long_description': {'value': ['']}
    }
}
