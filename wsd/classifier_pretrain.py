from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# user defined function
from wsd.utils_train import read_data_dataset_finetuning
from wsd.utils_pretrain import read_data_dataset_pretrain, DocumentBatcherPretrain, DocumentDatasetPretrain
from wsd.vocabulary import Vocabulary


# %% Text classifier
class TextClassifierPretrain:
    """A text classifier based on a neural network."""

    def __init__(self, args, device):
        self.do_chars = True if args["transformer_type"] == 'chars' else False
        self.tokenizer_choice = args['tokenizer']

        self.seed = args['seed']
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.milestones = args['milestones']
        self.lr_factor = args['lr_factor']
        self.clip_value = args['clip_value']

        self.seq_length = args['seq_length']
        if self.do_chars:
            self.max_voc_size = args['max_char_voc_size']
        else:
            self.max_voc_size = args['max_voc_size']
        self.max_token = args['max_token']
        self.max_valid_token = args['max_valid_token']

        self.device = device

        # define vocabulary
        self.voc = Vocabulary(max_voc_size=self.max_voc_size, character=self.do_chars,
                              tokenizer_choice=self.tokenizer_choice)

    def preprocess(self, data_path_finetuning, data_path_pretrain, folder):
        """Carry out the document preprocessing, then build `DataLoader`s for the training and validation sets."""

        # build the vocabulary according to finetuning data
        x_finetune, _, _, _ = read_data_dataset_finetuning(data_path_finetuning)
        self.voc.build(x_finetune)
        tqdm.write("...Vocabulary built (size {:})...".format(len(self.voc)))
        # save vocabulary
        torch.save({'character': self.voc.character,
                    'bag_of_chars': self.voc.bag_of_chars,
                    'max_voc_size': self.voc.max_voc_size,
                    'stoi': self.voc.stoi,
                    'itos': self.voc.itos},
                   os.path.join(folder, 'voc.pth'))

        # get pretraining data (just one long string)
        path = os.path.join(data_path_pretrain, 'wiki.train.tokens')
        x_train = read_data_dataset_pretrain(path, get_characters=self.do_chars, max_tokens=self.max_token,
                                             tokenizer_choice=self.tokenizer_choice)
        path = os.path.join(data_path_pretrain, 'wiki.valid.tokens')
        x_valid = read_data_dataset_pretrain(path, get_characters=self.do_chars, max_tokens=self.max_valid_token,
                                             tokenizer_choice=self.tokenizer_choice)

        batcher = DocumentBatcherPretrain(self.seq_length)
        # encode pretraining data with vocabulary from finetuning data
        encoded_train = self.voc.encode_pretrain(x_train)
        train_dataset = DocumentDatasetPretrain(encoded_train, self.seq_length)
        train_loader = DataLoader(train_dataset, self.batch_size, collate_fn=batcher)

        encoded_valid = self.voc.encode_pretrain(x_valid)
        valid_dataset = DocumentDatasetPretrain(encoded_valid, self.seq_length)
        valid_loader = DataLoader(valid_dataset, self.batch_size, collate_fn=batcher)

        return train_loader, valid_loader

    def set_model(self, model):
        """Provide a neural network model for this document classifier."""
        self.model = model
        self.model_best = model

    def train_model(self, train_loader, valid_loader, folder):
        history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "lr", ])
        best_loss = np.Inf
        for ep in range(self.epochs):
            train_loss = self.selfsupervised(ep, train_loader, train=True)
            valid_loss, pred_acc, pred_naive = self.selfsupervised(ep, valid_loader, train=False)
            # Get learning rate
            for param_group in self.optimizer.param_groups:
                learning_rate = param_group["lr"]
            # Print message
            message = 'Epoch {:2d}: \tTrain Loss {:2.3e} ' \
                      '\tValid Loss {:2.3e} \tLearning Rate {:1.2e}\tPred Acc {:2.4f}%\tPred Naive {:2.4f}%' \
                .format(ep, train_loss, valid_loss, learning_rate, pred_acc * 100, pred_naive * 100)
            tqdm.write(message)

            # Save history
            history = history.append(
                {"epoch": ep, "train_loss": train_loss, "valid_loss": valid_loss, "lr": learning_rate,
                 "pred_acc": pred_acc}, ignore_index=True)
            history.to_csv(os.path.join(folder, 'pretrain_history.csv'), index=False)

            # Save best model
            if best_loss > valid_loss:
                # Save model
                torch.save({'epoch': ep,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                           os.path.join(folder, 'pretrain_model.pth'))
                # Update best validation loss
                best_loss = valid_loss
                tqdm.write("Save model (best)!")
            # Save last model
            if ep == self.epochs - 1:
                torch.save({'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                           os.path.join(folder, 'pretrain_final_model.pth'))
                tqdm.write("Save model (last)!")
            # Call optimizer step
            self.scheduler.step()

    def selfsupervised(self, ep, loader, train):
        if train:
            self.model.train()
        else:
            self.model.eval()
        # for accuracy
        logsm = nn.LogSoftmax(dim=1)
        pred_values = 0
        pred_naive = 0
        n_values = 0
        # for loss
        total_loss = 0
        n_entries = 0
        str_name = 'train' if train else 'val'
        desc = "Epoch {:2d}: {} - Loss: {:2.3e}"
        bar = tqdm(initial=0, leave=True, total=len(loader.dataset.x) // self.seq_length,
                   desc=desc.format(ep, str_name, 0), position=0)

        # loop over all batches
        for i, batch in enumerate(loader):
            # Send to device
            batch = batch.to(device=self.device)
            # create model input and targets
            inp, target, indices = self.model.get_input_and_targets(batch)

            if train:
                # Reinitialize grad
                self.model.zero_grad()
                # Forward pass
                output = self.model(inp, indices)
                ll = self.loss_fun(output, target)
                # Backward pass
                ll.backward()
                clip_grad_norm_(self.model.parameters(), self.clip_value)
                # Optimizer
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output = self.model(inp, indices)
                    ll = self.loss_fun(output, target)
                    # get correct prediction rate
                    out = logsm(output).argmax(dim=1)
                    pred_values += (out == target).cpu().sum().numpy()
                    pred_naive += (3 * torch.ones_like(out) == target).cpu().sum().numpy()
                    n_values += target.cpu().numel()
            # Update
            total_loss += ll.detach().cpu().numpy()
            bs = batch.size(0)
            n_entries += bs
            # Update train bar
            bar.desc = desc.format(ep, str_name, total_loss / n_entries)
            bar.update(bs)
        bar.close()
        if train:
            return total_loss / n_entries
        else:
            return total_loss / n_entries, pred_values / n_values, pred_naive / n_values
