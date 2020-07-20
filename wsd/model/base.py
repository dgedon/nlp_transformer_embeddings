import torch.nn as nn
from wsd.model.simple_clf import *


def get_model(config, clf_input, pretrain_stage_config=None, pretrain_stage_ckpt=None):
    voc_size = clf_input.voc_size
    char_voc_size = clf_input.char_voc_size
    n_classes = clf_input.n_classes

    if config['model_type'].lower() == 'simple_word':
        """"
        simple model consisting of:
        word embedding layer + simple FF classifier
        """
        input_dim = config['emb_dim']
        # define model
        word_emb_mdl = ModelSimpleWordEmb(config, voc_size)
        clf = ModelSimpleClf(config, input_dim, n_classes)
        # combine model
        model = nn.Sequential(word_emb_mdl, clf)

    elif config['model_type'].lower() == 'simple_char':
        """"
        simple model consisting of:
        character embedding layer + simple FF classifier
        """
        input_dim = config['emb_dim']
        # define model
        char_emb_mdl = ModelSimpleCharEmb(config, char_voc_size)
        clf = ModelSimpleClf(config, input_dim, n_classes)
        # combine model
        model = nn.Sequential(char_emb_mdl, clf)

    elif config['model_type'].lower() == 'simple_word_char':
        """
        simple model consisiting of:
        word embedding layer, char embedding layer + simple FF classifier
        """
        input_dim = 2*config['emb_dim']
        # define model
        word_char_emb_mdl = ModelSimpleWordCharEmb(config, voc_size, char_voc_size)
        clf = ModelSimpleClf(config, input_dim, n_classes)
        # combine model
        model = nn.Sequential(word_char_emb_mdl, clf)

    elif config['model_type'].lower() == 'transformer_word':
        from wsd.model.transformer_pretrain import MyTransformer
        """
        Model consisting of:
        word embedding transformer + simple FF classifier
        """
        # pretrained word model
        pretrained = MyTransformer(pretrain_stage_config, clf_input, train_words=True)
        if pretrain_stage_ckpt is not None:
            pretrained.load_state_dict(pretrain_stage_ckpt['model'])
        ptr_mdl = pretrained.get_pretrained(config['finetuning'])
        # simple classifier
        inp_dim = config['emb_dim']
        clf = ModelSimpleClf(config, inp_dim, n_classes)
        # combine model
        model = nn.Sequential(ptr_mdl, clf)

    return model
