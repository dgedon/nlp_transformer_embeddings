from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
import os
import json
import datetime

def get_linear_schedule(optimizer, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0.
    """

    def lr_lambda(current_step: int):
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def define_folder(settings, args):
    # Generate output folder if needed and save config file
    if settings.folder[-1] == '/':
        folder = os.path.join(settings.folder, 'output_' +
                              str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))
    else:
        folder = settings.folder
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')

    return folder

def check_pretrained_model(args, settings, folder):
    try:
        if args.model_type.lower() in ['simple_word', 'simple_word_char', 'simple_char']:
            raise Exception('no pretrained simple embedding model')
        ckpt_pretrain_stage = torch.load(os.path.join(folder, 'pretrain_model.pth'),
                                         map_location=lambda storage, loc: storage)
        config_pretrain_stage = os.path.join(folder, 'pretrain_config.json')
        with open(config_pretrain_stage, 'r') as f:
            config_dict_pretrain_stage = json.load(f)
        tqdm.write("...Found pretrained model...")
    except:
        ckpt_pretrain_stage = None
        config_dict_pretrain_stage = None
        tqdm.write("...Did not found pretrained model...")
    # check for possible second model (char model)
    if settings.folder_model2[-1] != '/' and args.model_type.lower() == 'transformer_word_char':
        ckpt_pretrain2_stage = torch.load(os.path.join(settings.folder_model2, 'pretrain_model.pth'),
                                          map_location=lambda storage, loc: storage)
        config_pretrain2_stage = os.path.join(folder, 'pretrain_config.json')
        with open(config_pretrain2_stage, 'r') as f:
            config_dict_pretrain2_stage = json.load(f)
        # combine both pretrain configs in a tuple
        ckpt_pretrain_stage = (ckpt_pretrain_stage, ckpt_pretrain2_stage)
        config_dict_pretrain_stage = (config_dict_pretrain_stage, config_dict_pretrain2_stage)

    return ckpt_pretrain_stage, config_dict_pretrain_stage