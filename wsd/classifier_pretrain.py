from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# user defined function
from wsd.utils_train import read_data_dataset_finetuning, DocumentBatcher, DocumentDataset
from wsd.utils_pretrain import read_data_dataset_pretrain, DocumentBatcherPretrain, DocumentDatasetPretrain, \
    read_data_dataset_pretrain_new, DocumentBatcherPretrain_new, DocumentDatasetPretrain_new
from wsd.vocabulary import Vocabulary, VocabularyUpdated


# %% Text classifier
class TextClassifierPretrain:
    """A text classifier based on a neural network."""

    def __init__(self, args, device):
        self.character = True if args["transformer_type"] == 'chars' else False

        self.seed = args['seed']
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.clip_value = args['clip_value']

        self.seq_length = args['seq_length']
        if self.character:
            self.max_voc_size = args['max_char_voc_size']
        else:
            # not used
            self.max_voc_size = None
        self.max_token = args['max_token']
        self.max_valid_token = args['max_valid_token']

        self.device = device

        # define vocabulary
        self.voc = VocabularyUpdated(max_voc_size=self.max_voc_size, character=self.character)

    def preprocess(self, data_path_finetuning, data_path_pretrain, folder):
        """Carry out the document preprocessing, then build `DataLoader`s for the training and validation sets."""

        # get pretraining data
        path = os.path.join(data_path_pretrain, 'wiki.train.tokens')
        x_train = read_data_dataset_pretrain_new(path, get_characters=self.character, max_tokens=self.max_token)
        path = os.path.join(data_path_pretrain, 'wiki.valid.tokens')
        x_valid = read_data_dataset_pretrain_new(path, get_characters=self.character, max_tokens=self.max_valid_token)

        if self.character:
            self.voc.build(x_train)
            tqdm.write("...Vocabulary built (size {:})...".format(len(self.voc)))
            # save vocabulary
            torch.save({'character': self.voc.character,
                        'max_voc_size': self.voc.max_voc_size,
                        'stoi': self.voc.stoi,
                        'itos': self.voc.itos},
                       os.path.join(folder, 'voc.pth'))
        else:
            tqdm.write("...Vocabulary built (size {:})...".format(len(self.voc)))

        """# build the vocabulary
        self.voc.build(x_train)
        tqdm.write("...Vocabulary built (size {:})...".format(len(self.voc)))
        # save vocabulary
        torch.save({'character': self.voc.character,
                    'max_voc_size': self.voc.max_voc_size,
                    'stoi': self.voc.stoi,
                    'itos': self.voc.itos},
                   os.path.join(folder, 'voc.pth'))"""

        batcher = DocumentBatcherPretrain_new(self.voc, self.seq_length)
        # encode pretraining data with vocabulary from finetuning data
        encoded_train = self.voc.encode(x_train)
        train_dataset = DocumentDatasetPretrain_new(encoded_train)  # , self.seq_length)
        train_loader = DataLoader(train_dataset, self.batch_size, collate_fn=batcher)

        encoded_valid = self.voc.encode(x_valid)
        valid_dataset = DocumentDatasetPretrain_new(encoded_valid)  # , self.seq_length)
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
            """# Call optimizer step
            self.scheduler.step()"""

    def selfsupervised(self, ep, loader, train):
        if train:
            self.model.train()
        else:
            self.model.eval()
        # for accuracy
        logsm = nn.LogSoftmax(dim=-1)
        pred_values = 0
        pred_naive = 0
        n_values = 0
        # for loss
        total_loss = 0
        n_entries = 0
        str_name = 'train' if train else 'val'
        desc = "Epoch {:2d}: {}, Loss: {:2.3e}, Pred Acc {:2.3f}%"
        temp = 1 if len(loader.dataset.x) % self.batch_size != 0 else 0
        bar_len = len(loader.dataset.x) // self.batch_size + temp
        bar = tqdm(initial=0, leave=True, total=bar_len,
                   desc=desc.format(ep, str_name, 0, 0), position=0)

        # loop over all batches
        for i, batch in enumerate(loader):
            # create model input and targets
            inp, target = self.model.get_input_and_targets(batch)
            # Send to device
            inp = inp.to(device=self.device)
            target = target.to(device=self.device)

            if train:
                # Reinitialize grad
                self.model.zero_grad()
                # Forward pass
                output = self.model(inp)
                ll = self.loss_fun(output.view(-1, len(self.voc)), target.view(-1))
                # Backward pass
                ll.backward()
                clip_grad_norm_(self.model.parameters(), self.clip_value)
                # Optimizer
                self.optimizer.step()
                self.scheduler.step()

            else:
                with torch.no_grad():
                    output = self.model(inp)
                    ll = self.loss_fun(output.view(-1, len(self.voc)), target.view(-1))
            # get correct prediction rate
            batch = batch.to(device=self.device)
            mask = (inp != 1).int()
            out = logsm(output).argmax(dim=-1)
            pred_values += (out == batch * mask).cpu().sum().numpy()
            pred_naive += (5 * torch.ones_like(out) == target*mask).cpu().sum().numpy()
            n_values += mask.sum().item()

            # Update
            total_loss += ll.detach().cpu().numpy()
            bs = batch.size(0)
            n_entries += bs
            # Update train bar
            bar.desc = desc.format(ep, str_name, total_loss / n_entries, pred_values / n_values * 100)
            bar.update()  # bs
        bar.close()
        if train:
            return total_loss / n_entries
        else:
            return total_loss / n_entries, pred_values / n_values, pred_naive / n_values

    def get_input_and_targets(self, x):
        inp = x
        target = inp.clone()

        # sample tokens in each sequence with probability self.perc_masked_token
        probability_matrix = torch.full(target.shape, self.perc_masked_token)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        target[~masked_indices] = -100  # only compute loss on masked tokens

        # 80% we replace the masked input with as mask
        indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
        inp[indices_replaced] = self.voc_mask_id

        # 10% we replace masked input token with a random token
        indices_random = torch.bernoulli(torch.full(target.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.voc_size, target.shape, dtype=torch.long).to(device=inp.device)
        inp[indices_random] = random_words[indices_random]

        # remaining 10% we leave the correct token as input
        return inp, target
