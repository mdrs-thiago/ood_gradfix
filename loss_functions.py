import torch.nn as nn

def get_traditional_criterion(name, **kwargs):
    name = name.lower()
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'mse':
        return nn.MSELoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {name}")