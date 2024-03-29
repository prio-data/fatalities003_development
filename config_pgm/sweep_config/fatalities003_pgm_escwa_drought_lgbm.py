sweep_config = {
    "name": "fatalities003_pgm_escwa_drought_lgbm",
    "method": "grid",
    "metric": {
        "goal": "minimize",
        "name": "mse",
    },
    "parameters": {
        "transform": {"values": ["log", "standardize", "raw", "normalize"]},
        "n_estimators": {"values": [100, 200]},
        "learning_rate": {"values": [0.05, 0.1]},
        "n_jobs": {"values": [12]}
    }
}