sweep_config = {
    "name": "fatalities003_pgm_broad_hurdle_lgbm",
    "method": "grid",
    "metric": {
        "goal": "minimize",
        "name": "mse",
    },
    "parameters": {
        "transform": {"values": ["log", "standardize", "raw", "normalize"]},
        "clf_name": {"value": "LGBMClassifier"},
        "reg_name": {"value": "LGBMRegressor"}
    }
}