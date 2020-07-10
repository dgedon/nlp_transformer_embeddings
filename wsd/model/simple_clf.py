import torch.nn as nn
import torch

class ModelSimpleEmb(nn.Module):
    def __init__(self, config, voc_size):
        super().__init__()
        self.voc_size = voc_size
        self.emb_dim = config['emb_dim']
        self.dropout = config['dropout']

        self.embedding = nn.Embedding(self.voc_size, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, src):
        x, word_pos, _ = src
        embedded = self.embedding(x)
        cBoW = embedded.mean(dim=1)
        out = self.dropout(cBoW)

        return out


class ModelSimpleClf(nn.Module):
    def __init__(self, config, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim = config['emb_dim']
        self.hidden_size = config['hidden_size_simpleclf']
        self.dropout = config['dropout']

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=self.emb_dim, out_features=self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.n_classes),
        )

    def forward(self, src):
        out = self.classifier(src)

        return out
