import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


# %% Extract Pre-trained block
class PretrainedTransformerBlock(nn.Module):
    """Get reusable part from MyTransformer and return new model. Include Linear block with the given output_size."""

    def __init__(self, pretrained, freeze=True):
        super(PretrainedTransformerBlock, self).__init__()
        self.freeze = freeze
        self.emb_dim = pretrained._modules['decoder'].in_features

        self.embedding = pretrained._modules['embedding']
        self.pos_encoder = pretrained._modules['pos_encoder']
        self.transformer_encoder = pretrained._modules['transformer_encoder']

        if self.freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

    def forward(self, src):
        word_pos, x, _, x_char, src_key_padding_mask = src

        # process data (no mask in transformer used)
        src1 = self.embedding(x) * math.sqrt(self.emb_dim)
        src2 = src1.transpose(0, 1)
        src3 = self.pos_encoder(src2)
        out1 = self.transformer_encoder(src3, src_key_padding_mask=src_key_padding_mask)

        out2 = out1.transpose(0, 1)
        # generate continuous bag of features
        cbof = out2.mean(dim=1)

        return cbof


# %% Positional Encoder
class PositionalEncoding(nn.Module):
    # This is the positional encoding according to paper "Attention is all you need".
    # Could be changed to learnt encoding
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=10000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# %% Transformer model
class MyTransformer(nn.Module):
    """My Transformer:
    inspired by https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, args, clf, train_words):
        super(MyTransformer, self).__init__()

        self.train_words = train_words

        if train_words:
            self.voc_size = clf.voc_size
            self.voc_mask_id = clf.voc.get_mask_idx()
        else:
            # this is actually for characters but for compatibility we use the same variable name
            self.voc_size = clf.char_voc_size
            self.voc_mask_id = clf.char_voc.get_mask_idx()
        self.perc_masked_token = args['perc_masked_token']

        self.emb_dim = args["emb_dim"]

        self.embedding = nn.Embedding(self.voc_size, self.emb_dim)
        self.pos_encoder = PositionalEncoding(self.emb_dim, args['dropout_trans'])
        encoder_layers = TransformerEncoderLayer(self.emb_dim, args['num_heads'], args['dim_inner'],
                                                 args['dropout_trans'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, args['num_trans_layers'])
        self.decoder = nn.Linear(self.emb_dim, self.voc_size)

    def forward(self, src, src_key_padding_mask):
        # process data
        src1 = self.embedding(src) * math.sqrt(self.emb_dim)
        src2 = src1.transpose(0, 1)
        src3 = self.pos_encoder(src2)
        src4 = self.transformer_encoder(src3, src_key_padding_mask=src_key_padding_mask)

        out1 = self.decoder(src4)
        output = out1.permute(1, 2, 0)

        return output

    def get_pretrained(self, finetuning=False):
        freeze = not finetuning
        return PretrainedTransformerBlock(self, freeze)

    def get_input_and_targets(self, x, src_key_padding_mask):
        # inputs are masked docs
        inp = copy.deepcopy(x)
        # for each document in the batch
        for i, _ in enumerate(x):
            sz = (~src_key_padding_mask[i, :]).cpu().int().sum()
            # number of masked tokens
            # add a uniform random number U(-0.5,+0.5) for probabilistic rounding
            temp = np.random.rand() - 0.5
            num_mask = int(self.perc_masked_token * sz + temp)
            # get masking indices
            masked_idx = np.random.choice(sz, num_mask, replace=False)
            # mask indices
            inp[i, masked_idx] = self.voc_mask_id

        # targets are the original docs
        target = copy.deepcopy(x)
        inp_padding_mask = src_key_padding_mask

        """"
        #This could be used if the input format is not (batch_size, word/char_len) but 
        #if it is (batch_size, doc_len, char_len) 
        
        # input are masked chars
        inp = copy.deepcopy(x)
        # for each document in the batch (i=1:batch_size)
        for i, doc in enumerate(x):
            # for each word in that document (j=1:max_doc_length)
            for j, _ in enumerate(doc):
                sz = (~src_key_padding_mask[i, j, :]).cpu().int().sum()
                # number of masked tokens
                # add a uniform random number for probabilistic rounding
                temp = np.random.rand()
                num_mask = int(self.perc_masked_token * sz + temp)
                # get masking indices
                masked_idx = np.random.choice(sz, num_mask, replace=False)
                # mask indices
                inp[i, j, masked_idx] = self.voc_mask_id

        # targets are the original chars
        target = copy.deepcopy(x)
        #inp = inp.view(x.size(0), -1)
        inp_padding_mask = src_key_padding_mask # .view(x.size(0), -1)
        """
        # return input, target
        return inp, inp_padding_mask, target
