import json
import torch
import argparse
import datetime
from warnings import warn
from tqdm import tqdm
import torch.nn as nn
import os
import numpy as np

from wsd.classifier_pretrain import TextClassifierPretrain
from wsd.model.transformer_pretrain import MyTransformer
from wsd.utils_pretrain import get_linear_schedule_with_warmup


if __name__ == '__main__':
    ###################################
    # Options and folders
    ###################################
    tqdm.write("Define pretrain_wsd.py settings...")

    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--seed', type=int, default=2,
                               help='random seed for number generator (default: 2)')
    config_parser.add_argument('--epochs', type=int, default=40,
                               help='maximum number of epochs (default: 100)')
    config_parser.add_argument('--batch_size', type=int, default=16,
                               help='batch size (default: 16).')
    config_parser.add_argument('--valid_split', type=float, default=0.30,
                               help='fraction of the data used for validation (default: 0.1).')
    config_parser.add_argument('--lr', type=float, default=5e-5,
                               help='learning rate (default: 5e-5)')
    config_parser.add_argument('--clip_value', type=float, default=1.0,
                               help='maximum value for the gradient norm (default: 1.0)')
    config_parser.add_argument("--max_char_voc_size", type=int, default=None,
                               help='maximal size of the character vocabulary (default: None)')
    config_parser.add_argument("--bag_of_chars", type=bool, default=True,
                               help='Use bag of chars for transformers instead of [words in doc, chars in word] '
                                    '(default: True)')
    config_parser.add_argument("--max_token", type=int, default=1_000_000,
                               help='maximal number of used tokens for pretraining (default: tbd). '
                                    'Max 103 mio for words.')
    config_parser.add_argument("--max_valid_token", type=int, default=10_000,
                               help='maximal number of used tokens for validation in pretraining (default: tbd). '
                                    'Max 217 k for words.')
    # parameters for transformer
    config_parser.add_argument('--transformer_type', choices=['words', 'chars'], default='chars',
                               help="Type of transformer to learn. Options: words, chars.")
    config_parser.add_argument('--seq_length', type=int, default=128,
                               help="Transformer training fixed sequence length. Default is 128.")
    config_parser.add_argument('--num_heads', type=int, default=8,
                               help="Number of attention heads. Default is 4.")
    config_parser.add_argument('--num_trans_layers', type=int, default=4,
                               help="Number of transformer blocks. Default is 3.")
    config_parser.add_argument('--emb_dim', type=int, default=128,
                               help="Internal dimension of transformer. Default is 128.")
    config_parser.add_argument('--dim_inner', type=int, default=256,
                               help="Size of the FF network in the transformer. Default is 256.")
    config_parser.add_argument('--dropout_trans', type=float, default=0.1,
                               help='dropout rate of transformer (default: 0.1).')
    config_parser.add_argument('--activation_trans', choices=['relu', 'gelu'], default='gelu',
                               help='activation function of transformer (default: relu).')
    config_parser.add_argument('--perc_masked_token', type=float, default=0.15,
                               help="Percentage of total masked token. Default is 0.15.")
    args, rem_args = config_parser.parse_known_args()

    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--data_folder', type=str, default=os.getcwd() + '/wsd/data/',
                            help='data folder.')
    sys_parser.add_argument('--data_file_train', type=str, default='wsd_train.txt',
                            help='data file for training.')
    sys_parser.add_argument('--data_file_test', type=str, default='wsd_test.txt',
                            help='data file for validation.')
    sys_parser.add_argument('--cuda', action='store_true',
                            help='use cuda for computations. (default: False)')
    sys_parser.add_argument('--folder', default=os.getcwd() + '/wsd/',
                            help='output folder. If we pass /PATH/TO/FOLDER/ ending with `/`,'
                                 'it creates a folder `output_YYYY-MM-DD_HH_MM_SS_MMMMMM` inside it'
                                 'and save the content inside it. If it does not ends with `/`, the content is saved'
                                 'in the folder provided.')
    settings, unk = sys_parser.parse_known_args(rem_args)
    #  Final parser is needed for generating help documentation
    parser = argparse.ArgumentParser(parents=[sys_parser, config_parser])
    _, unk = parser.parse_known_args(unk)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # Set device
    device = torch.device('cuda:0' if settings.cuda else 'cpu')

    # Set output folder and save config
    if settings.folder[-1] == '/':
        folder = os.path.join(settings.folder, 'output_' +
                              str(datetime.datetime.now()).replace(":", "_").replace(" ", "_").replace(".", "_"))
    else:
        folder = settings.folder
    # Create output folder if needed
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass
    with open(os.path.join(folder, 'pretrain_config.json'), 'w') as f:
        json.dump(vars(args), f, indent='\t')

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tqdm.write("Done!")
    ###################################
    # Define classifier
    ###################################
    tqdm.write("Define classifier...")

    # only need this for the preprocessing and to split the data
    clf = TextClassifierPretrain(vars(args), device)

    tqdm.write("Done!")
    ###################################
    # Data loading & pre-processing
    ###################################
    tqdm.write("Load data and pre-process...")

    path_finetuning = os.path.join(settings.data_folder, settings.data_file_train)
    path_pretrain = os.path.join(settings.data_folder, 'pretrain')
    train_loader, valid_loader = clf.preprocess(path_finetuning, path_pretrain, folder)

    tqdm.write("Done!")
    ###################################
    # Define the Model
    ###################################
    tqdm.write("Define model...")

    model = MyTransformer(vars(args), clf)
    model.to(device=device)
    clf.set_model(model)

    tqdm.write("Done!")
    ###################################
    # Define the optimizer
    ###################################
    tqdm.write("Define optimizer...")

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in clf.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in clf.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    #adam_beta1 = 0.9
    #adam_beta2 = 0.999
    #learning_rate = 5e-5
    #adam_epsilon = 1e-8
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        #betas=(adam_beta1, adam_beta2),
        #eps=adam_epsilon,
    )
    # optimizer = torch.optim.Adam(clf.model.parameters(), args.lr)
    clf.optimizer = optimizer

    tqdm.write("Done!")
    ###################################
    # Define the learning scheduler
    ###################################
    tqdm.write("Define scheduler...")

    temp = 1 if len(train_loader.dataset.x) % args.batch_size != 0 else 0
    num_training_steps = len(train_loader.dataset.x)//args.batch_size + temp
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_training_steps)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_factor)
    clf.scheduler = scheduler

    tqdm.write("Done!")
    ###################################
    # Define the loss
    ###################################
    tqdm.write("Define loss...")

    loss = nn.CrossEntropyLoss(reduction='sum')
    clf.loss_fun = loss

    tqdm.write("Done!")
    ###################################
    # Train the model
    ###################################
    tqdm.write("Train the model:")

    clf.train_model(train_loader, valid_loader, folder)
