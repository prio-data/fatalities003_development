sweep_config = {
    "name": "fatalities003_pgm_baseline_lgbm",
    "method": "grid",
    "metric": {
        "goal": "minimize",
        "name": "mse",
    },
    "parameters": {
        "transform": {"values": ["log"]},
        "n_estimators": {"values": [100]},
        "learning_rate": {"values": [0.05]},
        "n_jobs": {"values": [1]}
    }
}