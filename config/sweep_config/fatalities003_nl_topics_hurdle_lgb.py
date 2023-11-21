sweep_configuration = {
    "name": 'fatalities003_nl_topics_hurdle_lgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_topics_hurdle_lgb'},
        'algorithm': {'value': 'HurdleRegression'},
        "transform": {
            "values": ['log', 'standardize', 'raw', 'normalize']
        },
        'clf_name': {'value': 'LGBMClassifier'},
        'reg_name': {'value': 'LGBMRegressor'},
        'depvar': {'value': 'ged_sb_dep'},
        'data_train': {'value': 'topics_003'},
        'queryset': {'value': 'fatalities003_topics'},
        'preprocessing': {'value': 'float_it'},
        'level': {'value': 'cm'},
        'description': {'value': ''},
        'long_description': {'value': ''}
    }
}
