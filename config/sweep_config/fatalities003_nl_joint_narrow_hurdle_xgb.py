sweep_configuration = {
    "name": 'fatalities003_nl_joint_narrow_hurdle_xgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {

        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'clf_name': {'value': 'XGBClassifier'},
        'reg_name': {'value': 'XGBRegressor'},

    }
}
