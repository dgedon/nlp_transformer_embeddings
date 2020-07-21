# Generic imports
import torch
import os
import argparse
from tqdm import tqdm
from warnings import warn
import datetime
import json
import numpy as np

# user defined inputs
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
    config_parser.add_argument('--batch_size', type=int, default=32,
                               help='batch size (default: 32).')
    config_parser.add_argument('--valid_split', type=float, default=0.30,
                               help='fraction of the data used for validation (default: 0.3).')
    config_parser.add_argument('--lr', type=float, default=1e-3,
                               help='learning rate (default: 0.001)')
    config_parser.add_argument('--milestones', nargs='+', type=int, default=[40, 60, 80],
                               help='milestones for lr scheduler (default: [75, 125, 175])')
    config_parser.add_argument("--lr_factor", type=float, default=0.1,
                               help='reducing factor for the lr in a plateau (default: 0.1)')
    # Model parameters
    config_parser.add_argument("--model_type", choices=['simple_word', 'simple_char', 'simple_word_char',
                                                        'transformer_word', 'transformer_char', 'transformer_word_char'],
                               default='transformer_word_char',
                               help='model type.')
    config_parser.add_argument("--max_voc_size", type=int, default=None,
                               help='maximal size of the vocabulary (default: None)')
    config_parser.add_argument("--max_char_voc_size", type=int, default=None,
                               help='maximal size of the character vocabulary (default: None)')
    config_parser.add_argument("--emb_dim", type=int, default=128,
                               help='dimension of embeddings (default: 128)')
    config_parser.add_argument("--dropout", type=float, default=0.5,
                               help='dropout rate (default: 0.3)')
    config_parser.add_argument("--hidden_size_simpleclf", type=int, default=500,
                               help='hidden dimension of simple classifier (default: 500)')
    config_parser.add_argument('--finetuning', type=bool, default=False,
                               help='when there is a pre-trained model, by default it '
                                    'freezes the weights of the pre-trained model, but with this option'
                                    'these weight will be fine-tuned during training. Default is False')
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
    sys_parser.add_argument('--folder', default=os.getcwd() + '/wsd/server/pretrain_word_char',  # '/wsd/'
                            help='output folder. If we pass /PATH/TO/FOLDER/ ending with `/`,'
                                 'it creates a folder `output_YYYY-MM-DD_HH_MM_SS_MMMMMM` inside it'
                                 'and save the content inside it. If it does not ends with `/`, the content is saved'
                                 'in the folder provided.')
    sys_parser.add_argument('--folder_model2', default=os.getcwd() + '/wsd/server/pretrain_word_char/pretrain_char',  # '/'
                            help='Folder where second model is stored. Specify as `/` if no second model is used.')
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
    # Check if there is pretrained model in the given folder
    try:
        if args.model_type.lower() in ['simple_word', 'simple_word_char', 'simple_char']:
            raise Exception('no pretrained simple embedding model')
        ckpt_pretrain_stage = torch.load(os.path.join(folder, 'pretrain_model.pth'),
                                         map_location=lambda storage, loc: storage)
        config_pretrain_stage = os.path.join(folder, 'pretrain_config.json')
        with open(config_pretrain_stage, 'r') as f:
            config_dict_pretrain_stage = json.load(f)
        tqdm.write("Found pretrained model!")
        # adapt voc size, bag_of_chars
        args.max_voc_size = config_dict_pretrain_stage['max_voc_size']
        args.bag_of_chars = True
    except:
        ckpt_pretrain_stage = None
        config_dict_pretrain_stage = None
        pretrain_ids = []
        tqdm.write("Did not found pretrained model!")
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


    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    train_loader, valid_loader = clf.preprocess(data_path=os.path.join(settings.data_folder, settings.data_file_train))

    tqdm.write("Done!")
    ###################################
    # Define the Model
    ###################################
    tqdm.write("Define model...")

    model = get_model(vars(args), clf, config_dict_pretrain_stage, ckpt_pretrain_stage)
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

    try:
        clf.train(train_loader, valid_loader, folder)

        tqdm.write("Model training done!")
        ###################################
        # Test the model
        ###################################
        tqdm.write("Test the model:")

        data_test_path = os.path.join(settings.data_folder, settings.data_file_test)
        make_predictions(clf, data_test_path, folder)

        tqdm.write("Finished script!")
    except:
        tqdm.write("Model training interrupted. Test the best model:")

        data_test_path = os.path.join(settings.data_folder, settings.data_file_test)
        make_predictions(clf, data_test_path, folder)

        tqdm.write("Finished script!")
