sweep_config = {
    "name": "fatalities003_pgm_conflict_treelag_hurdle",
    "method": "grid",
    "metric": {
        "goal": "minimize",
        "name": "mse",
    },
    "parameters": {
        "transform": {"values": ["log", "standardize", "raw", "normalize"]},
        "clf_name": {"value": "XGBClassifier"},
        "reg_name": {"value": "XGBRegressor"}
    }
}