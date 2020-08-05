from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

# user defined function
from wsd.utils_train import read_data_dataset_finetuning, DocumentBatcher, DocumentDataset
from wsd.vocabulary import Vocabulary


# %% Text classifier
class TextClassifier:
    """A text classifier based on a neural network."""

    def __init__(self, args, settings, device):
        self.model_type = args['model_type']
        self.seed = args['seed']
        self.epochs = args['epochs']
        self.valid_split = args['valid_split']
        self.batch_size = args['batch_size']

        self.word_seq_length = args["seq_length_words"]
        self.char_seq_length = args["seq_length_chars"]

        self.max_voc_size = None
        self.max_char_voc_size = args['max_char_voc_size']
        self.shuffle = True

        self.device = device

        # define vocabulary
        if self.model_type in ['simple_word', 'simple_word_char', 'simple_char']:
            self.voc = Vocabulary(max_voc_size=self.max_voc_size)
            self.char_voc = Vocabulary(max_voc_size=self.max_char_voc_size, character=True)
        if self.model_type in ['transformer_word', 'transformer_char', 'transformer_word_char']:
            folder = settings["folder"]
            if self.model_type in ['transformer_word_char']:
                folder = settings["folder_model2"]
            try:
                temp = torch.load(os.path.join(folder, 'voc.pth'))
                self.max_char_voc_size = temp['max_char_voc_size']
                self.stoi = temp['stoi']
                self.itos = temp['itos']
            except:
                self.max_char_voc_size = args['max_char_voc_size']
                self.stoi = None
                self.itos = None

            self.voc = Vocabulary(max_voc_size=self.max_voc_size)
            self.char_voc = Vocabulary(max_voc_size=self.max_char_voc_size, character=True,
                                       stoi=self.stoi, itos=self.itos)

        self.lbl_enc = LabelEncoder()

    def preprocess(self, data_path):
        """Carry out the document preprocessing, then build `DataLoader`s for the training and validation sets."""

        # read the training data
        x, y, word_pos, _ = read_data_dataset_finetuning(data_path)

        # split training data
        x_train, x_valid, y_train, y_valid, word_pos_train, word_pos_valid = train_test_split(x, y, word_pos,
                                                                                              test_size=self.valid_split,
                                                                                              random_state=self.seed,
                                                                                              shuffle=self.shuffle)

        self.voc_size = len(self.voc)
        tqdm.write("...word vocabulary built (size {:})...".format(self.voc_size))
        # also build a vocabulary for characters (only if not pre-loaded)
        if self.stoi is None:
            self.char_voc.build(x_train)
        self.char_voc_size = len(self.char_voc)
        tqdm.write("...char vocabulary built (size {:})...".format(self.char_voc_size))

        self.lbl_enc.fit(y_train)
        self.n_classes = len(self.lbl_enc.classes_)

        # define data batcher (same padding for self.voc and self.char_voc)
        self.batcher = DocumentBatcher(self.voc, self.word_seq_length, self.char_seq_length)

        # batch the training data
        train_dataset = DocumentDataset(self.voc.encode(x_train), self.lbl_enc.transform(y_train),
                                        word_pos_train, self.char_voc.encode(x_train))
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=self.shuffle, collate_fn=self.batcher)
        # batch the validation data
        valid_dataset = DocumentDataset(self.voc.encode(x_valid), self.lbl_enc.transform(y_valid),
                                        word_pos_valid, self.char_voc.encode(x_valid))
        valid_loader = DataLoader(valid_dataset, self.batch_size, shuffle=self.shuffle, collate_fn=self.batcher)

        return train_loader, valid_loader

    def set_model(self, model):
        """Provide a neural network model for this document classifier."""
        self.model = model
        self.model_best = model

    def train(self, train_loader, valid_loader, folder):
        """Train the model. We assume that a dataset and a model have already been provided."""

        # We'll log the loss and accuracy scores encountered during training.
        history = pd.DataFrame(columns=["epoch", "train_loss", "valid_loss", "lr", "train_acc", "valid_acc"])

        best_valid_acc = -np.Inf
        for ep in range(self.epochs):
            # Train and evaluate
            train_loss, train_acc = self.epoch(ep, train_loader, do_train=True)
            valid_loss, valid_acc = self.epoch(ep, valid_loader, do_train=False)

            # Get learning rate
            for param_group in self.optimizer.param_groups:
                learning_rate = param_group["lr"]
            # Print message
            message = 'Epoch {:2d}: \tTrain Loss {:2.3e} ' \
                      '\tValid Loss {:2.3e} \tLearning Rate {:1.2e}\t' \
                      'Train Acc: {:2.4f} \tValid Acc: {:2.4f} ' \
                .format(ep, train_loss, valid_loss, learning_rate, train_acc, valid_acc)
            tqdm.write(message)
            # Save history
            history = history.append({"epoch": ep, "train_loss": train_loss, "valid_loss": valid_loss,
                                      "lr": learning_rate, "train_acc": train_acc,
                                      "valid_acc": valid_acc, }, ignore_index=True)
            history.to_csv(os.path.join(folder, 'history.csv'), index=False)

            # Save best model
            if valid_acc > best_valid_acc:
                # Save model
                torch.save({'epoch': ep,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                           os.path.join(folder, 'model.pth'))
                self.model_best.load_state_dict(self.model.state_dict())
                # Update best validation accuracy
                best_valid_acc = valid_acc
                tqdm.write("Save model (best)!")
            # Save last model
            if ep == self.epochs - 1:
                torch.save({'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                           os.path.join(folder, 'final_model.pth'))
                tqdm.write("Save model (last)!")

    def epoch(self, ep, batches, do_train):
        """Runs the neural network for one epoch, using the given batches.
        If an optimizer is provided, this is training data and we will update the model
        after each batch. Otherwise, this is assumed to be validation data.

        Returns the loss and accuracy over the epoch."""
        if do_train:
            self.model.train()
        else:
            self.model.eval()
        str_name = 'train' if do_train else 'valid'

        n_correct = 0
        n_instances = 0
        total_loss = 0
        # training progress bar
        train_desc = "Epoch {:2d}: {} - Loss: {:.6f}"
        train_bar = tqdm(initial=0, leave=True, total=len(batches.dataset.x),
                         desc=train_desc.format(ep, str_name, 0), position=0)

        for i, batch in enumerate(batches):
            y_batch, x_batch, x_char_batch = batch
            y_batch = y_batch.to(self.device)
            x_batch = x_batch.to(self.device)
            x_char_batch = x_char_batch.to(self.device)

            # Reinitialize grad
            self.model.zero_grad()
            self.optimizer.zero_grad()
            # Forward pass
            model_inp = (x_batch, x_char_batch)
            scores = self.model(model_inp)
            # Compute the loss for this batch.
            loss = self.loss_fun(scores, y_batch)

            if do_train:
                # Backward pass
                loss.backward()
                # Optimize
                self.optimizer.step()

            # Update
            total_loss += loss.detach().cpu().numpy()
            bs = y_batch.size(0)
            n_instances += bs

            # Compute the number of correct predictions, for the accuracy.
            guesses = scores.argmax(dim=1)
            n_correct += (guesses == y_batch).sum().item()

            # Update train bar
            train_bar.desc = train_desc.format(ep, str_name, total_loss / n_instances)
            train_bar.update(bs)

        train_bar.close()
        # Call optimizer step
        if do_train:
            self.scheduler.step()

        return total_loss / n_instances, n_correct / n_instances

    def predict(self, x, word_pos):
        """Run a trained document classifier on a set of documents and return the predictions."""
        batcher = DocumentBatcher(self.voc, self.word_seq_length, self.char_seq_length)

        # Build a DataLoader to generate the batches, as above.
        dummy_labels = [self.lbl_enc.classes_[0] for _ in x]

        dataset = DocumentDataset(self.voc.encode(x), self.lbl_enc.transform(dummy_labels),
                                  word_pos, self.char_voc.encode(x))
        loader = DataLoader(dataset, self.batch_size, shuffle=False, collate_fn=batcher)

        # Apply the model to all the batches and aggregate the predictions.
        self.model_best.eval()
        output = []
        for y_batch, x_batch, x_char_batch in loader:
            # to device
            y_batch = y_batch.to(self.device)
            x_batch = x_batch.to(self.device)
            x_char_batch = x_char_batch.to(self.device)
            # run
            model_inp = (x_batch, x_char_batch)
            # evaluate
            scores = self.model_best(model_inp)
            guesses = scores.argmax(dim=1)
            output.extend(self.lbl_enc.inverse_transform(guesses.cpu().numpy()))
        return output


# %% Predictor
def make_predictions(clf, data_test_path, folder):
    # read the test data
    x_test, y_test_true, word_pos_test, _ = read_data_dataset_finetuning(data_test_path)
    # make predictions
    y_test_pred = clf.predict(x_test, word_pos_test)

    true_pred = 0
    false_pred = 0
    for l_true, l_pred in zip(y_test_true, y_test_pred):
        if l_true == l_pred:
            true_pred += 1
        else:
            false_pred += 1

    accuracy = true_pred / (true_pred + false_pred)
    tqdm.write('Accuracy of model on test set is {:.3f}%'.format(accuracy * 100))

    final_acc = pd.DataFrame(columns=["test_acc"])
    # Save history
    final_acc = final_acc.append({"test_acc": accuracy, }, ignore_index=True)
    final_acc.to_csv(os.path.join(folder, 'history_testing.csv'), index=False)
