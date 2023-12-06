sweep_configuration = {
    "name": 'fatalities003_nl_conflicthistory_hurdle_lgb',
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
