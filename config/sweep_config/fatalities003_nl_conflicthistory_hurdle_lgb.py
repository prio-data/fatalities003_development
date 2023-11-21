sweep_configuration = {
    "name": 'fatalities003_nl_conflicthistory_hurdle_lgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_conflicthistory_hurdle_lgb'},
        'algorithm': {'value': 'HurdleRegression'},
        "transform": {"values": ['log', 'standardize', 'raw', 'normalize']},
        'clf_name': {'value':'LGBMClassifier'},
        'reg_name': {'value':'LGBMRegressor'},
        'depvar': {'value':'ged_sb_dep'},
        'data_train': {'value':'conflict_ln'},
        'queryset': {'value':'fatalities003_conflict_history'},
        'preprocessing': {'value':'float_it'},
        'level': {'value':'cm'},
        'description': {'value': ''},
        'long_description': {'value':''}
    }
}
