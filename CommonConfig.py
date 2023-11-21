
common_config = {
    'level': 'cm',
    'run_id': 'Fatalities003',
    'steps': [*range(1, 36 + 1, 1)],
    'calib_partitioner_dict': {"train": (121, 396), "predict": (397, 444)},
    'test_partitioner_dict': {"train": (121, 444), "predict": (445, 492)},
    'future_partitioner_dict': {"train": (121, 492), "predict": (493, 504)},
    'FutureStart': 508,
    'get_future': False,
    'force_retrain': True
}

wandb_config = {
    'project': 'training_example_2',
    'entity': 'model-development-and-deployment'
}
