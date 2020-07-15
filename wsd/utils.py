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

    def __init__(self, max_voc_size=None, character=False):

        self.PAD = '___PAD___'
        self.UNKNOWN = '___UNKNOWN___'
        self.MASK = '___MASK___'

        self.character = character
        self.max_voc_size = max_voc_size

        # String-to-integer mapping
        self.stoi = None
        # Integer-to-string mapping
        self.itos = None
        # Tokenizer that will be used to split document strings into words.
        self.tokenizer = lambda s: s.split()

    def build(self, docs):
        """Builds the vocabulary, based on a set of documents."""

        # Sort all words by frequency
        if self.character:
            # actually here its char freqs but meh.
            """res = []
            for doc in docs:
                temp = []
                for w in self.tokenizer(doc):
                    for c in w:
                        temp.append(c)
            res.append(temp)"""
            docs = [[c for w in self.tokenizer(doc) for c in w] for doc in docs]
        else:
            """res = []
            for doc in docs:
                temp = []
                for w in self.tokenizer(doc):
                    temp.append(w)
                res.append(temp)"""
            docs = [[w for w in self.tokenizer(doc)] for doc in docs]

        word_freqs = Counter(w for doc in docs for w in doc)
        word_freqs = sorted(((f, w) for w, f in word_freqs.items()), reverse=True)

        # Build the integer-to-string mapping. The vocabulary starts with the two dummy symbols,
        # and then all words, sorted by frequency. Optionally, limit the vocabulary size.
        if self.max_voc_size:
            self.itos = [self.PAD, self.UNKNOWN, self.MASK] + [w for _, w in word_freqs[:self.max_voc_size - 2]]
        else:
            self.itos = [self.PAD, self.UNKNOWN, self.MASK] + [w for _, w in word_freqs]

        # Build the string-to-integer map by just inverting the aforementioned map.
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, docs):
        """Encodes a set of documents."""
        unkn_index = self.stoi[self.UNKNOWN]
        if self.character:
            """
            res = []
            for doc in docs:
                temp1 = []
                for w in self.tokenizer(doc):
                    temp2 =[]
                    for c in w:
                        temp2.append(self.stoi.get(c, unkn_index))
                    temp1.append(temp2)
                res.append(temp1)"""
            encoded = [[[self.stoi.get(c, unkn_index) for c in w] for w in self.tokenizer(doc)] for doc in docs]
        else:
            """
            res = []
            for doc in docs:
                temp = []
                for w in self.tokenizer(doc):
                    temp.append(self.stoi.get(w, unkn_index))
                res.append(temp)
            """
            encoded = [[self.stoi.get(w, unkn_index) for w in self.tokenizer(doc)] for doc in docs]
        return encoded

    def get_unknown_idx(self):
        """Returns the integer index of the special dummy word representing unknown words."""
        return self.stoi[self.UNKNOWN]

    def get_pad_idx(self):
        """Returns the integer index of the special padding dummy word."""
        return self.stoi[self.PAD]

    def get_mask_idx(self):
        """Returns the integer index of the special masking word."""
        return self.stoi[self.MASK]

    def __len__(self):
        return len(self.itos)


# %% Document Batching
class DocumentDataset(Dataset):
    def __init__(self, x, y, word_pos, x_char):
        self.x = x
        self.y = y
        self.word_pos = word_pos
        self.x_char = x_char

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.word_pos[idx], self.x_char[idx]

    def __len__(self):
        return len(self.x)


class DocumentBatcher:
    def __init__(self, voc):
        # Find the integer index of the dummy padding word.
        self.pad = voc.get_pad_idx()

    def __call__(self, data):
        # data is a list of tuples with words, tags, word_positions, word_chars

        # How long is the longest document in this batch?
        max_doc_len = max(len(x) for x, _, _, _ in data)
        # Build the document-word tensor.
        # We pad the shorter documents so that all documents have the same length.
        x_padded = torch.as_tensor([x + [self.pad] * (max_doc_len - len(x)) for x, _, _, _ in data])

        # generate padding mask (0 for non padded, 1 for padded)
        src_key_padding_mask = torch.as_tensor([len(x)*[0] + [1] * (max_doc_len - len(x)) for x, _, _, _ in data]).bool()

        # Build the label tensor.
        y = torch.as_tensor([y for _, y, _, _ in data])

        # Build the word position tensor.
        word_pos = torch.as_tensor([pos for _, _, pos, _ in data])

        # How long is the longest word in this batch?
        max_word_len = max(len(w) for _, _, _, x_char in data for w in x_char)
        # build the document-char tensor.
        # We pad the shorter documents so that all documents have the same length.
        x_char_padded = [x_char + [[]] * (max_doc_len - len(x_char)) for _, _, _, x_char in data]
        # We pad the shorter words so that all words have the same length.
        x_char_padded = [[w + [self.pad] * (max_word_len - len(w)) for w in x_char] for x_char in x_char_padded]
        x_char_padded = torch.as_tensor(x_char_padded)

        return x_padded, y, word_pos, x_char_padded, src_key_padding_mask
