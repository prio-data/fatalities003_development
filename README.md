# Weights-and-Biases
Logging Fatalities003 models on Weights-and-Biases

## 1. The configuration structure
```bash config folder structure
<<<<<<< HEAD
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
=======
my_config
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
>>>>>>> 6967f39e33a3a00d96dd55c272bdfe1f3ced890f
```
## 2. Add model configuration files to model_config & sweep_config

<<<<<<< HEAD
## 2. Run the code
To sweep over all the models in the config fold: 
```console
python Sweep.py -l cm -c my_config_cm
```
To sweep over the specific model: 
```console
python Sweep.py -l cm -c my_config_cm -m fatalities003_nl_baseline_rf
=======
## 3. Run the code
```console
python sweep.py -l cm -c my_config
>>>>>>> 6967f39e33a3a00d96dd55c272bdfe1f3ced890f
```
