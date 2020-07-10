from collections import Counter
import torch
from torch.utils.data import Dataset


# %% Read data
def read_data(corpus_file):
    x = []
    y = []
    lemma = []
    word_pos = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            sense_key, lemma_temp, pos_temp, doc = line.strip().split(maxsplit=3)
            x.append(doc)
            y.append(sense_key)
            lemma.append(lemma_temp)
            word_pos.append(int(pos_temp))
    return x, y, word_pos, lemma


# %% Words to int and vice-versa
class Vocabulary:
    """Manages the numerical encoding of the vocabulary."""

    def __init__(self, tokenizer=None, max_voc_size=None):

        self.PAD = '___PAD___'
        self.UNKNOWN = '___UNKNOWN___'
        self.NUMBER = '___NUMBER___'

        # String-to-integer mapping
        self.stoi = None

        # Integer-to-string mapping
        self.itos = None

        # Tokenizer that will be used to split document strings into words.
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = lambda s: s.split()

        # Maximally allowed vocabulary size.
        self.max_voc_size = max_voc_size

    def build(self, docs):
        """Builds the vocabulary, based on a set of documents."""

        # Sort all words by frequency
        word_freqs = Counter(w for doc in docs for w in self.tokenizer(doc))
        word_freqs = sorted(((f, w) for w, f in word_freqs.items()), reverse=True)

        # Build the integer-to-string mapping. The vocabulary starts with the two dummy symbols,
        # and then all words, sorted by frequency. Optionally, limit the vocabulary size.
        if self.max_voc_size:
            self.itos = [self.PAD, self.UNKNOWN] + [w for _, w in word_freqs[:self.max_voc_size - 2]]
        else:
            self.itos = [self.PAD, self.UNKNOWN] + [w for _, w in word_freqs]

        # Build the string-to-integer map by just inverting the aforementioned map.
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, docs):
        """Encodes a set of documents."""
        unkn_index = self.stoi[self.UNKNOWN]
        return [[self.stoi.get(w, unkn_index) for w in self.tokenizer(doc)] for doc in docs]

    def get_unknown_idx(self):
        """Returns the integer index of the special dummy word representing unknown words."""
        return self.stoi[self.UNKNOWN]

    def get_pad_idx(self):
        """Returns the integer index of the special padding dummy word."""
        return self.stoi[self.PAD]

    def __len__(self):
        return len(self.itos)


# %% Document Batching
class DocumentDataset(Dataset):
    def __init__(self, x, y, word_pos):
        self.x = x
        self.y = y
        self.word_pos = word_pos

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.word_pos[idx]

    def __len__(self):
        return len(self.x)


class DocumentBatcher:
    def __init__(self, voc):
        # Find the integer index of the dummy padding word.
        self.pad = voc.get_pad_idx()

    def __call__(self, data):
        # How long is the longest document in this batch?
        max_len = max(len(x) for x, _, _ in data)

        # Build the document tensor. We pad the shorter documents so that all documents have the same length.
        x_padded = torch.as_tensor([x + [self.pad] * (max_len - len(x)) for x, _, _ in data])
        # generate padding mask (0 for non padded, 1 for padded)
        src_key_padding_mask = torch.as_tensor([len(x)*[0] + [1] * (max_len - len(x)) for x, _, _ in data]).bool()

        # Build the label tensor.
        y = torch.as_tensor([y for _, y, _ in data])

        # Build the position tensor.
        word_pos = torch.as_tensor([pos for _, _, pos in data])

        return x_padded, y, word_pos, src_key_padding_mask
