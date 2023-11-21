sweep_configuration = {
    "name": 'fatalities003_nl_joint_narrow_hurdle_xgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_joint_narrow_hurdle_xgb'},
        'algorithm': {'value': 'HurdleRegression'},
        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'clf_name': {'value': 'XGBClassifier'},
        'reg_name': {'value': 'XGBRegressor'},
        'depvar': {'value': 'ged_sb_dep'},
        'data_train': {'value': 'joint_narrow'},
        'queryset': {'value': 'fatalities003_joint_narrow'},
        'preprocessing': {'value': 'float_it'},
        'level': {'value': 'cm'},
        'description': {'value': ''},
        'long_description': {'value': ''}
    }
}
