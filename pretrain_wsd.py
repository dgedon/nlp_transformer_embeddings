import json
import torch
import argparse
import datetime
import pandas as pd
from warnings import warn
from tqdm import tqdm
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import os
import numpy as np
import math

from wsd.classifier import TextClassifier
from wsd.model.transformer_pretrain import MyTransformer


def train_model(model, loss, optimizer, scheduler, train_loader, valid_loader, folder, device, train_words):
    history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "lr", ])
    best_loss = np.Inf
    for ep in range(args.epochs):
        train_loss = selfsupervised(ep, model, optimizer, train_loader, loss, device, args, train_words, train=True)
        valid_loss = selfsupervised(ep, model, optimizer, valid_loader, loss, device, args, train_words, train=False)
        # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        # Print message
        message = 'Epoch {:2d}: \tTrain Loss {:2.3e} ' \
                  '\tValid Loss {:2.3e} \tLearning Rate {:1.2e}\t' \
            .format(ep, train_loss, valid_loss, learning_rate)
        tqdm.write(message)

        # Save history
        history = history.append({"epoch": ep, "train_loss": train_loss, "valid_loss": valid_loss, "lr": learning_rate},
                                 ignore_index=True)
        history.to_csv(os.path.join(folder, 'pretrain_history.csv'), index=False)

        # Save best model
        if best_loss > valid_loss:
            # Save model
            torch.save({'epoch': ep,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'pretrain_model.pth'))
            # Update best validation loss
            best_loss = valid_loss
            tqdm.write("Save model (best)!")
        # Save last model
        if ep == args.epochs - 1:
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(folder, 'pretrain_final_model.pth'))
            tqdm.write("Save model (last)!")
        # Call optimizer step
        scheduler.step()


def selfsupervised(ep, model, optimizer, loader, loss, device, args, train_words, train):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    n_entries = 0
    str_name = 'train' if train else 'val'
    desc = "Epoch {:2d}: {} - Loss: {:2.3e}"
    bar = tqdm(initial=0, leave=True, total=len(loader.dataset.x), desc=desc.format(ep, str_name, 0), position=0)

    # loop over all batches
    for i, batch in enumerate(loader):
        # Send to device
        x_batch, y_batch, word_pos_batch, x_char_batch, src_key_padding_mask = batch
        x_batch = x_batch.to(device=device)
        x_char_batch = x_char_batch.to(device=device)
        src_key_padding_mask = src_key_padding_mask.to(device=device)
        # create model input and targets
        if train_words:
            inp, target = model.get_input_and_targets(x_batch, src_key_padding_mask)
        else:
            inp, target = model.get_input_and_targets(x_char_batch)

        if train:
            # Reinitialize grad
            model.zero_grad()
            # Forward pass
            output = model(inp, src_key_padding_mask)
            ll = loss(output, target)
            # Backward pass
            ll.backward()
            clip_grad_norm_(model.parameters(), args.clip_value)
            # Optimize
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(inp, src_key_padding_mask)
                ll = loss(output, target)
        # Update
        total_loss += ll.detach().cpu().numpy()
        bs = x_batch.size(0)
        n_entries += bs
        # Update train bar
        bar.desc = desc.format(ep, str_name, total_loss / n_entries)
        bar.update(bs)
    bar.close()
    return total_loss / n_entries


if __name__ == '__main__':
    ###################################
    # Options and folders
    ###################################
    tqdm.write("Define pretrain_wsd.py settings...")

    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--seed', type=int, default=2,
                               help='random seed for number generator (default: 2)')
    config_parser.add_argument('--epochs', type=int, default=100,
                               help='maximum number of epochs (default: 100)')
    config_parser.add_argument('--batch_size', type=int, default=32,
                               help='batch size (default: 32).')
    config_parser.add_argument('--valid_split', type=float, default=0.30,
                               help='fraction of the data used for validation (default: 0.1).')
    config_parser.add_argument('--lr', type=float, default=1e-3,
                               help='learning rate (default: 0.001)')
    config_parser.add_argument('--milestones', nargs='+', type=int,
                               default=[40, 60, 80],
                               help='milestones for lr scheduler (default: [40, 60, 80])')
    config_parser.add_argument("--lr_factor", type=float, default=0.1,
                               help='reducing factor for the lr in a plateau (default: 0.1)')
    config_parser.add_argument('--clip_value', type=float, default=1.0,
                               help='maximum value for the gradient norm (default: 1.0)')
    config_parser.add_argument("--max_voc_size", type=int, default=5000,
                               help='maximal size of the vocabulary (default: None)')
    config_parser.add_argument("--max_char_voc_size", type=int, default=None,
                               help='maximal size of the character vocabulary (default: None)')
    # parameters for transformer
    config_parser.add_argument('--transformer_type', type=str, default='words',
                               help="Type of transformer to learn. Options: words, chars.")
    config_parser.add_argument('--num_heads', type=int, default=4,
                               help="Number of attention heads. Default is 4.")
    config_parser.add_argument('--num_trans_layers', type=int, default=3,
                               help="Number of transformer blocks. Default is 4.")
    config_parser.add_argument('--emb_dim', type=int, default=128,
                               help="Internal dimension of transformer. Default is 128.")
    config_parser.add_argument('--dim_inner', type=int, default=256,
                               help="Size of the FF network in the transformer. Default is 256.")
    config_parser.add_argument('--dropout_trans', type=float, default=0.2,
                               help='dropout rate of transformer (default: 0.2).')
    config_parser.add_argument('--perc_masked_token', type=int, default=0.15,
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

    train_words = True if args.transformer_type.lower() == 'words' else False
    model = MyTransformer(vars(args), clf, train_words)
    model.to(device=device)

    tqdm.write("Done!")
    ###################################
    # Define the optimizer
    ###################################
    tqdm.write("Define optimizer...")

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    tqdm.write("Done!")
    ###################################
    # Define the learning scheduler
    ###################################
    tqdm.write("Define scheduler...")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_factor)

    tqdm.write("Done!")
    ###################################
    # Define the loss
    ###################################
    tqdm.write("Define loss...")

    loss = nn.CrossEntropyLoss()  # nn.MSELoss(reduction='sum')

    tqdm.write("Done!")
    ###################################
    # Train the model
    ###################################
    tqdm.write("Train the model:")

    train_model(model, loss, optimizer, scheduler, train_loader, valid_loader, folder, device, train_words)
