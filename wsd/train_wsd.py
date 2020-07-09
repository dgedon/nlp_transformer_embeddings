# Generic imports
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
from warnings import warn
import datetime
import json
import pandas as pd

# user defined inputs
from wsd.utils import read_data
from wsd.classifier import TextClassifier, make_predictions
from wsd.model.base import get_model

if __name__ == '__main__':
    ###################################
    # Options and folders
    ###################################
    tqdm.write("Define train_wsd.py settings...")

    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    # Learning parameters
    config_parser.add_argument('--seed', type=int, default=2,
                               help='random seed for number generator (default: 2)')
    config_parser.add_argument('--epochs', type=int, default=100,
                               help='maximum number of epochs (default: 200)')
    config_parser.add_argument('--batch_size', type=int, default=256,
                               help='batch size (default: 128).')
    config_parser.add_argument('--valid_split', type=float, default=0.30,
                               help='fraction of the data used for validation (default: 0.3).')
    config_parser.add_argument('--lr', type=float, default=1e-3,
                               help='learning rate (default: 0.001)')
    config_parser.add_argument('--milestones', nargs='+', type=int, default=[75, 125, 175],
                               help='milestones for lr scheduler (default: [75, 125, 175])')
    config_parser.add_argument("--lr_factor", type=float, default=0.1,
                               help='reducing factor for the lr in a plateau (default: 0.1)')
    # Model parameters
    config_parser.add_argument("--model_type", type=str, default='simple_emb',
                               help='model type. Options: simple_emb')
    config_parser.add_argument("--max_voc_size", type=int, default=None,
                               help='maximal size of the vocabulary (default: None)')
    config_parser.add_argument("--emb_dim", type=int, default=512,
                               help='dimension of embeddings (default: 512)')
    config_parser.add_argument("--dropout", type=float, default=0.3,
                               help='dropout value (default: 0.3)')
    config_parser.add_argument("--hidden_size_simpleclf", type=int, default=500,
                               help='hidden dimension of simple classifier (default: 500)')

    args, rem_args = config_parser.parse_known_args()

    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--data_folder', type=str, default=os.getcwd() + '/data/',
                            help='data folder.')
    sys_parser.add_argument('--data_file_train', type=str, default='wsd_train.txt',
                            help='data file for training.')
    sys_parser.add_argument('--data_file_test', type=str, default='wsd_test.txt',
                            help='data file for validation.')
    sys_parser.add_argument('--data_file_test_blind', type=str, default='wsd_test_blind.txt',
                            help='data file for testing.')
    sys_parser.add_argument('--cuda', action='store_true',
                            help='use cuda for computations. (default: False)')
    sys_parser.add_argument('--folder', default=os.getcwd() + '/',
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

    # Set seed
    torch.manual_seed(args.seed)

    tqdm.write("Done!")
    ###################################
    # Define classifier
    ###################################
    tqdm.write("Define classifier...")

    clf = TextClassifier(vars(args), device)

    tqdm.write("Done!")
    ###################################
    # Data loading & pre-processing
    ###################################
    tqdm.write("Load data and pre-process...")

    # preprocess data
    train_loader, valid_loader = clf.preprocess(data_path=os.path.join(settings.data_folder, settings.data_file_train))

    tqdm.write("Done!")
    ###################################
    # Define the Model
    ###################################
    tqdm.write("Define model...")

    model = get_model(vars(args), clf.voc_size, clf.n_classes)
    model.to(device=device)
    clf.set_model(model)

    tqdm.write("Done by choosing {}!".format(args.model_type))
    ###################################
    # Define the optimizer
    ###################################
    tqdm.write("Define optimizer...")

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    clf.optimizer = optimizer

    tqdm.write("Done!")
    ###################################
    # Define the learning scheduler
    ###################################
    tqdm.write("Define scheduler...")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_factor)
    clf.scheduler = scheduler

    tqdm.write("Done!")
    ###################################
    # Define the loss
    ###################################
    tqdm.write("Define loss...")

    loss = torch.nn.CrossEntropyLoss()
    clf.loss_fun = loss

    tqdm.write("Done!")
    ###################################
    # Train the model
    ###################################
    tqdm.write("Train the model:")

    clf.train(train_loader, valid_loader, folder)

    tqdm.write("Model training done!")
    ###################################
    # Test the model
    ###################################
    tqdm.write("Test the model:")

    data_test_path = os.path.join(settings.data_folder, settings.data_file_test)
    make_predictions(clf, data_test_path)

    tqdm.write("Finished script!")
