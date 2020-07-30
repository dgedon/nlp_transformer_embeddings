from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


# %% Extract Pre-trained block
class PretrainedTransformerBlock(nn.Module):
    """Get reusable part from MyTransformer and return new model. Include Linear block with the given output_size."""

    def __init__(self, pretrained, freeze=True, train_words=False):
        super(PretrainedTransformerBlock, self).__init__()
        self.freeze = freeze
        self.train_words = train_words

        self.emb_dim = pretrained.decoder[0].in_features
        self.decoder = pretrained.decoder

        self.embedding = pretrained.embedding
        self.pos_encoder = pretrained.pos_encoder
        self.transformer_encoder = pretrained.transformer_encoder

        if self.freeze:
            tqdm.write("...transformer requires_grad set to False...")
            for param in self._modules['embedding'].parameters():
                param.requires_grad = False
            for param in self._modules['pos_encoder'].parameters():
                param.requires_grad = False
            for param in self._modules['transformer_encoder'].parameters():
                param.requires_grad = False
        else:
            tqdm.write("...transformer requires_grad set to True...")

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

    def predict_mask_token(self, src):

        # process data (no mask in transformer used)
        src1 = self.embedding(src) * math.sqrt(self.emb_dim)
        src2 = src1.transpose(0, 1)
        src3 = self.pos_encoder(src2)
        out1 = self.transformer_encoder(src3)

        out2 = out1.transpose(0, 1)
        # generate continuous bag of features
        output = self.decoder(out2)

        return output


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

    def __init__(self, args, clf):
        super(MyTransformer, self).__init__()

        self.voc_size = len(clf.voc)
        self.voc_mask_id = clf.voc.get_mask_idx()
        self.seq_length = args['seq_length']

        self.perc_masked_token = args['perc_masked_token']
        # number of masked tokens
        self.num_masked_token = int(self.perc_masked_token * self.seq_length)

        self.emb_dim = args["emb_dim"]

        self.embedding = nn.Embedding(self.voc_size, self.emb_dim)
        self.pos_encoder = PositionalEncoding(self.emb_dim, args['dropout_trans'])
        self.embedding_norm = nn.LayerNorm(self.emb_dim)
        encoder_layers = TransformerEncoderLayer(self.emb_dim, args['num_heads'], args['dim_inner'],
                                                 args['dropout_trans'])
        self.transformer_encoder = TransformerEncoder(encoder_layers, args['num_trans_layers'])

        # LM decoder
        self.dense = nn.Linear(self.emb_dim, self.emb_dim)
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        self.decoder_layer = nn.Linear(self.emb_dim, self.voc_size)
        self.decoder = nn.Sequential(
            self.dense,
            self.layer_norm,
            nn.ReLU(),
            self.decoder_layer
        )

    def forward(self, src):
        # process data
        src1 = self.embedding(src) * math.sqrt(self.emb_dim)
        src2 = self.pos_encoder(src1)
        src3 = src2.transpose(0, 1)
        src4 = self.embedding_norm(src3)
        src5 = self.transformer_encoder(src4)

        # src5 is of shape (batch_size, seq_length, embedding_dim)
        out1 = src5.transpose(0, 1)
        output = self.decoder(out1)

        return output

    def get_pretrained(self, finetuning=False, train_words=False):
        freeze = not finetuning
        return PretrainedTransformerBlock(self, freeze, train_words)

    def get_input_and_targets(self, x):
        """
        inputs are masked docs
        targets are the values of the mask
        """

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
