import torch.nn as nn
from wsd.model.simple_clf import *


def get_model(config, voc_size, n_classes):
    if config['model_type'].lower() == 'simple_emb':
        ptr_mdl = ModelSimpleEmb(config, voc_size)
        clf = ModelSimpleClf(config, n_classes)
        model = nn.Sequential(ptr_mdl, clf)

    return model
