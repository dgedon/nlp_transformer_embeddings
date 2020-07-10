import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def generate_random_sequence_mask(sz, perc_masking):
    """
    According to attention definition the same mask is used for all sequences in the batch.
    Mask is a [sz x sz] matrix. If the value [i,j] is masked by a value of -inf, then for the
    computation of output j the input i is masked, meaning that no attention is used for this input.

    sz - sequence size
    perc_masking - percentage of all masked samples
    """

    # gives perc_masking percentage to -inf and 1-perc_masking percentage to 0
    mask = np.random.choice([-np.inf, 0], size=(sz, sz), p=[perc_masking, 1 - perc_masking])
    mask = torch.tensor(mask)

    return mask


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
        x, word_pos, src_key_padding_mask = src

        # process data (no mask in transformer used)
        src1 = self.embedding(x) * math.sqrt(self.emb_dim)
        src2 = src1.transpose(0, 1)
        src3 = self.pos_encoder(src2)
        out1 = self.transformer_encoder(src3, src_key_padding_mask=src_key_padding_mask)

        output = (out1, word_pos)
        return output


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
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# new pre-trained model
class MyTransformer(nn.Module):
    """My Transformer:
    inspired by https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, args, voc_size):
        super(MyTransformer, self).__init__()

        self.perc_masked_samples = args['perc_masked_samp']
        self.emb_dim = args["emb_dim"]
        self.voc_size = voc_size

        self.embedding = nn.Embedding(self.voc_size, self.emb_dim)
        self.pos_encoder = PositionalEncoding(self.emb_dim, args['dropout_trans'])
        encoder_layers = TransformerEncoderLayer(self.emb_dim, args['num_heads'], args['dim_inner'],
                                                 args['dropout_trans'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, args['num_trans_layers'])
        self.decoder = nn.Linear(self.emb_dim, self.voc_size)

    def forward(self, src, src_key_padding_mask):
        batch_size, seq_len = src.shape
        # generate mask
        self.mask = generate_random_sequence_mask(seq_len, self.perc_masked_samples).to(next(self.parameters()).device)

        # process data
        src1 = self.embedding(src) * math.sqrt(self.emb_dim)
        src2 = src1.transpose(0, 1)
        src3 = self.pos_encoder(src2)
        src4 = self.transformer_encoder(src3, mask=self.mask, src_key_padding_mask=src_key_padding_mask)

        out1 = self.decoder(src4)
        output = out1.permute(1, 2, 0)

        return output

    def get_pretrained(self, finetuning=False):
        freeze = not finetuning
        return PretrainedTransformerBlock(self, freeze)

    def get_input_and_targets(self, x):
        inp = x
        target = x

        # return input, target
        return inp, target
