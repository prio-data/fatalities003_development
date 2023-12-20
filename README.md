# Weights-and-Biases
Hyperparameter tuning for Fatalities003 models

## 1. The configuration structure
```bash config folder structure
├── cm_config
│   ├── model_config
│   │   ├── modelname1.py
│   │   ├── modelname2.py
│   │   └── ...
│   ├── sweep_config
│   │   ├── modelname1.py
│   │   ├── modelname2.py
│   │   └── ...
│   ├── common_config.py
│   └── wandb_config.py
│ 
└── pgm_config
    ├── model_config
    │   ├── modelname1.py
    │   ├── modelname2.py
    │   └── ...
    ├── sweep_config
    │   ├── modelname1.py
    │   ├── modelname2.py
    │   └── ...
    ├── common_config.py
    └── wandb_config.py
```

## 2. Run the code
To sweep over all the models in the config fold: 
```console
python Sweep.py -l cm -c my_config_cm
```
To sweep over the specific model: 
```console
python Sweep.py -l cm -c my_config_cm -m fatalities003_nl_baseline_rf
```
