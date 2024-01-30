# Weights-and-Biases
Logging Fatalities003 models on Weights-and-Biases

## 1. The configuration structure
```bash config folder structure
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
```
## 2. Add model configuration files to model_config & sweep_config

## 3. Run the code
```console
python sweep.py -l cm -c my_config
```
