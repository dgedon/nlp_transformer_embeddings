from tqdm import tqdm
from wsd.model.simple_clf import *


class MyBiSequential(nn.Module):
    def __init__(self, par1, par2, seq, con_dim):
        super(MyBiSequential, self).__init__()
        # parallel models
        self.par1 = par1
        self.par2 = par2
        # sequential model
        self.seq = seq
        # concatenation dimension
        self.con_dim = con_dim

    def forward(self, src):
        # parallel models
        src1 = self.par1(src)
        src2 = self.par2(src)
        # concatenation
        src3 = torch.cat([src1, src2], self.con_dim)
        # sequential model
        out = self.seq(src3)

        return out


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
        pretrained = MyTransformer(pretrain_stage_config, clf_input)
        if pretrain_stage_ckpt is not None:
            pretrained.load_state_dict(pretrain_stage_ckpt['model'])
        ptr_mdl = pretrained.get_pretrained(config['finetuning'], train_words=True)
        # simple classifier
        inp_dim = config['emb_dim']
        clf = ModelSimpleClf(config, inp_dim, n_classes)
        # combine model
        model = nn.Sequential(ptr_mdl, clf)

    elif config['model_type'].lower() == 'transformer_char':
        from wsd.model.transformer_pretrain import MyTransformer
        """
        Model consisting of:
        char embedding transformer + simple FF classifier
        """
        # pretrained char model
        pretrained = MyTransformer(pretrain_stage_config, clf_input)
        if pretrain_stage_ckpt is not None:
            pretrained.load_state_dict(pretrain_stage_ckpt['model'])
        ptr_mdl = pretrained.get_pretrained(config['finetuning'])
        # simple classifier
        inp_dim = config['emb_dim']
        clf = ModelSimpleClf(config, inp_dim, n_classes)
        # combine model
        model = nn.Sequential(ptr_mdl, clf)

    elif config['model_type'].lower() == 'transformer_word_char':
        from wsd.model.transformer_pretrain import MyTransformer
        """
        Model consisting of:
        word embedding transformer + char embedding transformer + simple FF classifier
        """
        # extract pretrained word and char information
        pretrain_stage_config_word, pretrain_stage_config_char = pretrain_stage_config
        pretrain_stage_ckpt_word, pretrain_stage_ckpt_char = pretrain_stage_ckpt

        # pretrained word model
        pretrained_word = MyTransformer(pretrain_stage_config_word, clf_input)
        if pretrain_stage_ckpt_word is not None:
            pretrained_word.load_state_dict(pretrain_stage_ckpt_word['model'])
        ptr_mdl_word = pretrained_word.get_pretrained(config['finetuning'])

        # pretrained word model
        pretrained_char = MyTransformer(pretrain_stage_config_char, clf_input)
        if pretrain_stage_ckpt_char is not None:
            pretrained_char.load_state_dict(pretrain_stage_ckpt_char['model'])
        ptr_mdl_char = pretrained_char.get_pretrained(config['finetuning'])

        # simple classifier
        inp_dim = 2*config['emb_dim']
        clf = ModelSimpleClf(config, inp_dim, n_classes)

        # combine model
        model = MyBiSequential(ptr_mdl_char, ptr_mdl_word, clf, con_dim=1)

    tqdm.write("...choosing {}...".format(config['model_type']))
    return model
