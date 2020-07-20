import torch.nn as nn
import torch


class ModelSimpleWordEmb(nn.Module):
    def __init__(self, config, voc_size):
        super().__init__()
        self.voc_size = voc_size
        self.emb_dim = config['emb_dim']
        self.dropout = config['dropout']

        self.embedding = nn.Embedding(self.voc_size, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, src):
        word_pos, x, _, x_char, _ = src
        word_emb = self.embedding(x)
        cbow = word_emb.mean(dim=1)
        out = self.dropout(cbow)

        return out


class ModelSimpleCharEmb(nn.Module):
    def __init__(self, config, char_voc_size):
        super().__init__()
        self.char_voc_size = char_voc_size
        self.emb_dim = config['emb_dim']
        self.dropout = config['dropout']

        self.embedding = nn.Embedding(self.char_voc_size, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, src):
        word_pos, x, _, x_char, _ = src
        char_emb = self.embedding(x_char)
        # reshape to [batch_size, doc_len*char_len, emb_dim]
        char_emb = char_emb.view(char_emb.size(0), -1, self.emb_dim)
        # get continuous bag of characters
        cboc = char_emb.mean(dim=1)
        out = self.dropout(cboc)

        return out


class ModelSimpleWordCharEmb(nn.Module):
    def __init__(self, config, voc_size, char_voc_size):
        super().__init__()
        self.char_voc_size = char_voc_size
        self.voc_size = voc_size
        self.emb_dim = config['emb_dim']
        self.dropout = config['dropout']

        self.word_embedding = nn.Embedding(self.voc_size, self.emb_dim)
        self.char_embedding = nn.Embedding(self.char_voc_size, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, src):
        word_pos, x, _, x_char, _ = src

        # word embedding
        word_emb = self.word_embedding(x)
        cbow = word_emb.mean(dim=1)
        cbow = self.dropout(cbow)

        # character embedding
        char_emb = self.char_embedding(x_char)
        # reshape to [batch_size, doc_len*word_len, emb_dim]
        char_emb = char_emb.view(char_emb.size(0), -1, self.emb_dim)
        # get continuous bag of characters
        cboc = char_emb.mean(dim=1)
        cboc = self.dropout(cboc)

        # concatenate cbow and cboc
        out = torch.cat([cbow, cboc], 1)

        return out


class ModelSimpleClf(nn.Module):
    def __init__(self, config, inp_dim, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.inp_dim = inp_dim
        self.hidden_size = config['hidden_size_simpleclf']
        self.dropout = config['dropout']

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=self.inp_dim, out_features=self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.n_classes),
        )

    def forward(self, src):
        out = self.classifier(src)

        return out
