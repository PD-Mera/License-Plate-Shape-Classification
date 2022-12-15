import torch.nn as nn
import os
import torch


def calculate_weights(training_config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights = []
    for classname in training_config['class']['name']:
        weights.append(len(os.listdir(os.path.join(training_config['path'], classname))))
    
    total_datasize = sum(weights)
    weights = [(1.0 - (x / total_datasize)) for x in weights]
    return torch.tensor(weights, dtype=torch.float32).to(device)


class ClsfLoss():
    def __init__(self):
        super(ClsfLoss, self).__init__()
        pass
    def forward(self, x):
        return x


def init_loss(training_config):
    if training_config['loss'] == 'custom':
        loss = ClsfLoss()
    elif training_config['loss'] == 'CE':
        weights = calculate_weights(training_config)
        loss = nn.CrossEntropyLoss(weight=weights)
    return loss