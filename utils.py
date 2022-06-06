import random
import numpy as np
import torch
import importlib

def l1(tensor):
    return tensor.flatten().abs().sum()

def l2(tensor):
    return torch.pow(tensor.flatten(), 2).sum()

def regularization(param_dict, alpha):
    l = []
    for param in param_dict:
        tensor_list = [t for t in param['params']]
        for t in tensor_list:
            l.append((alpha * t.abs() + (1.0 - alpha) * torch.pow(t, 2)).sum())
    return sum(l)

def load_template(module: str, model_name: str, model_args: dict):
    module = importlib.import_module(module)
    template = getattr(module, model_name)
    model = template(**model_args)
    return model

def seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark= False
