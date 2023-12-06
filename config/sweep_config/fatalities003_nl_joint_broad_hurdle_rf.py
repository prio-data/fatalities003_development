sweep_configuration = {
    "name": 'fatalities003_nl_joint_broad_hurdle_rf',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {

        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'clf_name': {'value': 'RFClassifier'},
        'reg_name': {'value': 'RFRegressor'},

    }
}
