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
        self.train_words = pretrained.train_words

        self.emb_dim = pretrained.decoder.in_features

        self.embedding = pretrained.embedding
        self.pos_encoder = pretrained.pos_encoder
        self.transformer_encoder = pretrained.transformer_encoder

        if self.freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False
            for param in self.pos_encoder.parameters():
                param.requires_grad = False
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

    def forward(self, src):
        word_pos, x, src_key_padding_mask, x_char, src_char_key_padding_mask = src

        if self.train_words:
            inp = x
            in_key_padding_mask = src_key_padding_mask
        else:
            inp = x_char
            in_key_padding_mask = src_char_key_padding_mask

        # process data (no mask in transformer used)
        src1 = self.embedding(inp) * math.sqrt(self.emb_dim)
        src2 = src1.transpose(0, 1)
        src3 = self.pos_encoder(src2)
        out1 = self.transformer_encoder(src3, src_key_padding_mask=in_key_padding_mask)

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
            self.seq_length = clf.trans_max_doc_words
        else:
            # this is actually for characters but for compatibility we use the same variable name
            self.voc_size = clf.char_voc_size
            self.voc_mask_id = clf.char_voc.get_mask_idx()
            self.seq_length = clf.trans_max_doc_chars
        self.perc_masked_token = args['perc_masked_token']
        # number of masked tokens
        self.num_masked_token = int(self.perc_masked_token * self.seq_length)

        self.emb_dim = args["emb_dim"]

        self.embedding = nn.Embedding(self.voc_size, self.emb_dim)
        self.pos_encoder = PositionalEncoding(self.emb_dim, args['dropout_trans'])
        encoder_layers = TransformerEncoderLayer(self.emb_dim, args['num_heads'], args['dim_inner'],
                                                 args['dropout_trans'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, args['num_trans_layers'])
        self.decoder = nn.Linear(self.emb_dim, self.voc_size)
        self.decoder_out = nn.Linear(self.seq_length, self.num_masked_token)

    def forward(self, src, src_key_padding_mask):
        # process data
        src1 = self.embedding(src) * math.sqrt(self.emb_dim)
        src2 = src1.transpose(0, 1)
        src3 = self.pos_encoder(src2)
        src4 = self.transformer_encoder(src3, src_key_padding_mask=src_key_padding_mask)

        out1 = self.decoder(src4)
        out2 = out1.permute(1, 2, 0)
        output = self.decoder_out(out2)

        return output

    def get_pretrained(self, finetuning=False):
        freeze = not finetuning
        return PretrainedTransformerBlock(self, freeze)

    def get_input_and_targets(self, x, src_key_padding_mask):
        """
        inputs are masked docs
        targets are the values of the mask
        """
        batch_size = x.size(0)
        inp = copy.deepcopy(x)
        target = torch.empty(batch_size, self.num_masked_token, dtype=inp.dtype).to(device=inp.device)
        # for each document in the batch
        for i, _ in enumerate(x):
            # get masking indices
            masked_idx = np.random.choice(self.seq_length, self.num_masked_token, replace=False)
            # get tokens of masked indices as targets
            target[i, :] = copy.copy(inp[i, masked_idx])
            # mask indices
            inp[i, masked_idx] = self.voc_mask_id

        # padding mask
        inp_padding_mask = src_key_padding_mask

        # return input, target
        return inp, inp_padding_mask, target
