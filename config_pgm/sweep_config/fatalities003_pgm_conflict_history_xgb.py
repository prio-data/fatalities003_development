sweep_config = {
    "name": "fatalities003_pgm_conflict_history_xgb",
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