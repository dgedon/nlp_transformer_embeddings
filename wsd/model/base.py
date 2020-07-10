import torch.nn as nn
from wsd.model.simple_clf import *


def get_model(config, voc_size, n_classes, pretrain_stage_config=None, pretrain_stage_ckpt=None):
    if config['model_type'].lower() == 'simple_emb':
        """"
        simple model consisting of:
        embedding layer + simple FF classifier
        """
        ptr_mdl = ModelSimpleEmb(config, voc_size)
        clf = ModelSimpleClf(config, n_classes)
        # combine model
        model = nn.Sequential(ptr_mdl, clf)

    if config['model_type'].lower() == 'transf_word':
        from wsd.model.transformer_pretrain import MyTransformer
        from wsd.model.selection import WordSelModel
        """
        Model consisting of:
        word embedding transformer + select features for word + simple FF classifier
        """
        # pretrained word model
        pretrained = MyTransformer(pretrain_stage_config, voc_size)
        if pretrain_stage_ckpt is not None:
            pretrained.load_state_dict(pretrain_stage_ckpt['model'])
        ptr_mdl = pretrained.get_pretrained()
        # word selector
        word_sel_mdl = WordSelModel()
        # simple classifier
        clf = ModelSimpleClf(config, n_classes)
        # combine model
        model = nn.Sequential(ptr_mdl, word_sel_mdl, clf)

    return model
