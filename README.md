# Weights-and-Biases
Logging Fatalities003 models on Weights-and-Biases

## 1. The configuration structure
```bash config folder structure
├── config_cm
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
└── config_pgm
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
Before running the code, change 'project' and 'entity' that match your w&b configuration in wandb_config.py in the config folder.
```console
wandb_config = {
    'project': 'your_project_name',
    'entity': 'your_entity_name'
}
```
To sweep over all the models in the config fold: 
```console
python sweep.py -l cm -c config_cm
```
To sweep over the specific model: 
```console
python sweep.py -l cm -c config_cm -m fatalities003_nl_baseline_rf
```

Note: if there is an error similar to 'Booster' object has no attribute 'handle', this is because the boost model version is different from the one used in your enviroment. Set 'force_retrain' in common_config.py to True and rerun the codes.


## 3. Evaluation
The model is trained on the dataset that is transformed using raw, log, standardize, and normalize (transform_1). Then we transform the outputs back to make them comparable to the actual ones (transform_2). Finally we do the evaluation of the four sets of prediction models (one for each transform) against all the four transforms of the outcome (transform_3). So, on W&B, you can expect to see:
1. Metrics such as mse, tloss, kld, jefd, jend after transform_3.
2. Same metrics for steps 1, 3, 6, 9, 12, 36 after transfrom_3.
3. Plots of predicted fatalities, absolute error, squared error, and squared logarithmic error on cm/pgm level after transform_2. Now we only plot fatalities in the first predicted month with all steps.
4. Hyperparameter tuning results (mse) if it is your goal.