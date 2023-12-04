# Weights-and-Biases
Hyperparameter tuning for Fatalities003 models

## 1. The configuration structure
```bash config folder structure
config
├── model_config
│   ├── modelname1.py
│   ├── modelname2.py
│   ├── ...
├── sweep_config
│   ├── modelname1.py
│   ├── modelname2.py
│   ├── ...
├── common_config.py
└── wandb_config.py
```

## 2. Run the code
```console
python SweepExample.py -l cm -c my_config
```
