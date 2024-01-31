sweep_config = {
    "name": 'fatalities003_nl_conflicthistory_nonlog_hurdle_lgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {

        "transform": {"values": ['log', 'standardize', 'raw', 'normalize']},
        'clf_name': {'value': 'LGBMClassifier'},
        'reg_name': {'value': 'LGBMRegressor'},
    }
}