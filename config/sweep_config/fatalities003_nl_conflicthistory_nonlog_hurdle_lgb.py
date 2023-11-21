sweep_configuration = {
    "name": 'fatalities003_nl_conflicthistory_nonlog_hurdle_lgb',
    "method": "grid",
    "metric": {
        "goal": "minimize",
        'name': "mse",
    },
    "parameters": {
        'modelname': {'value': 'fatalities003_nl_conflicthistory_nonlog_hurdle_lgb'},
        'algorithm': {'value': 'HurdleRegression'},
        "transform": {"values": ['log', 'standardize', 'raw', 'normalize']},
        'clf_name': {'value':'LGBMClassifier'},
        'reg_name': {'value':'LGBMRegressor'},
        'depvar': {'value':'ged_sb_dep'},
        'data_train': {'value': 'conflict_nonlog'},
        'queryset': {'value': 'fatalities003_conflict_history_nonlog'},
        'preprocessing': {'value':'float_it'},
        'level': {'value':'cm'},
        'description': {'value': 'A collection of variables that together map the conflict history of a country, random forests regression model.'},
        'long_description': {'value': 'A collection of variables that together map the conflict history of a country. The features include lagged dependent variables for each conflict type as coded by the UCDP (state-based, one-sided, or non-state) for up to each of the preceding six months, decay functions of time since conflict caused 5, 100, and 500 deaths in a month, for each type of violence, whether ACLED (https://doi.org/10.1177/0022343310378914 recorded similar violence, and whether there was recent violence in any neighboring countries.'}
    }
}
