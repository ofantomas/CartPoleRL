import os
import uuid
import json
import numpy as np
import torch

try:
    import wandb
except ImportError:
    pass


def detorchify(data_dict):
    res = dict()

    mappers = {
        np.int64: int,
        np.float32: float,
        np.float64: float,
        torch.Tensor: lambda x: x.item(),
    }

    for k, v in data_dict.items():
        if type(v) in mappers:
            v = mappers[type(v)](v)
        res[k] = v
    return res


class LoggingManager:
    def __init__(self):
        self.loggers = []

    def add_logger(self, logger):
        self.loggers.append(logger)

    def log_config(self, config_dict):
        if len(self.loggers) == 0:
            print("Current experiment setting:")
            print(config_dict)
        else:
            for logger in self.loggers:
                logger.log_config(config_dict)

    def __call__(self, metrics_dict):
        if len(self.loggers) == 0:
            print(metrics_dict)
        else:
            for logger in self.loggers:
                logger.log_metrics(metrics_dict)


class LocalLogger:
    def __init__(self, log_folder, exp_name):
        self.current_run_folder = os.path.join(log_folder, f'{exp_name}-{uuid.uuid4()}')
        os.makedirs(self.current_run_folder, exist_ok=False)

    def log_metrics(self, metrics_dict):
        with open(os.path.join(self.current_run_folder, 'metrics.json'), 'a') as outF:
            json.dump(detorchify(metrics_dict), outF)
            outF.write('\n')

    def log_config(self, config_dict):
        with open(os.path.join(self.current_run_folder, 'config.json'), 'w') as outF:
            json.dump(config_dict, outF)


class WandbLogger:
    def __init__(self, project_name, exp_name):
        wandb.init(project=project_name, reinit=True, name=exp_name)

    def log_config(self, config_dict):
        for k, v in config_dict.items():
            setattr(wandb.config, k, v)

    def log_metrics(self, metrics_dict):
        wandb.log(metrics_dict)


class ConsoleLogger:
    def __init__(self, metric_names=None, log_each_n=1):
        self.metric_names = metric_names
        self.log_each_n = log_each_n
        self.counter = 0

    def log_config(self, config_dict):
        print(config_dict)

    def log_metrics(self, metrics_dict):
        self.counter = (self.counter + 1) % self.log_each_n
        if self.counter != 0:
            return
        if self.metric_names is None:
            print(metrics_dict)
        else:
            print({k: v for k, v in metrics_dict.items() if k in self.metric_names})
