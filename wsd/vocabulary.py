from collections import Counter
from transformers import RobertaTokenizerFast
from tqdm import tqdm
import sys


class Vocabulary:
    def __init__(self, max_voc_size=None, character=False, stoi = None, itos=None):
        self.character = character
        self.max_voc_size = max_voc_size

        if self.character:
            self.tokenizer = lambda s: s.lower()

            self.PAD = '___PAD___'
            self.UNKNOWN = '___UNKNOWN___'
            self.MASK = '___MASK___'

            # String-to-integer mapping
            if stoi is not None:
                self.stoi = stoi
            else:
                self.stoi = None
            # Integer-to-string mapping
            if itos is not None:
                self.itos = itos
            else:
                self.itos = None

        else:
            self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

            self.pad_id = self.tokenizer.pad_token_id
            self.mask_id = self.tokenizer.mask_token_id

    def build(self, docs):
        """
        Builds the vocabulary, based on a set of documents.
        ONLY FOR CHARACTERS. For words we use the roberta tokenizer with internal vocabulary.
        """
        if self.stoi is None or self.itos is None:

            if self.character:
                docs = [[c for w in doc for c in w] for doc in docs]
            else:
                # docs = [[w for w in self.tokenizer(doc)] for doc in docs]
                tqdm.write("only for characters defined!")
                sys.exit()

            # Sort all token by frequency
            token_freqs = Counter(w for doc in docs for w in doc)
            token_freqs = sorted(((f, w) for w, f in token_freqs.items()), reverse=True)

            # Build the integer-to-string mapping. The vocabulary starts with the two dummy symbols,
            # and then all words, sorted by frequency. Optionally, limit the vocabulary size.
            if self.max_voc_size:
                self.itos = [self.PAD, self.UNKNOWN, self.MASK] + [w for _, w in token_freqs[:self.max_voc_size - 3]]
            else:
                self.itos = [self.PAD, self.UNKNOWN, self.MASK] + [w for _, w in token_freqs]

            # Build the string-to-integer map by just inverting the aforementioned map.
            self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, docs):
        """Encodes a set of documents."""
        if self.character:
            unkn_index = self.stoi[self.UNKNOWN]
            encoded = [[self.stoi.get(w, unkn_index) for w in doc] for doc in docs]
        else:
            encoded = self.tokenizer(docs)['input_ids']
        return encoded

    def get_unknown_idx(self):
        """Returns the integer index of the special dummy word representing unknown words."""
        return self.stoi[self.UNKNOWN]

    def get_pad_idx(self):
        """Returns the integer index of the special padding dummy word."""
        if self.character:
            return self.stoi[self.PAD]
        else:
            return self.pad_id

    def get_mask_idx(self):
        """Returns the integer index of the special masking word."""
        if self.character:
            return self.stoi[self.MASK]
        else:
            return self.mask_id

    def __len__(self):
        if self.character:
            return len(self.itos)
        else:
            return self.tokenizer.vocab_size
