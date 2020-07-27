from collections import Counter
from transformers import AutoTokenizer


# %% Words to int and vice-versa
class Vocabulary:
    """Manages the numerical encoding of the vocabulary."""

    def __init__(self, max_voc_size=None, character=False, bag_of_chars=False, tokenizer_choice=None, stoi = None,
                 itos=None):

        self.PAD = '___PAD___'
        self.UNKNOWN = '___UNKNOWN___'
        self.MASK = '___MASK___'

        self.character = character
        self.bag_of_chars = bag_of_chars
        self.max_voc_size = max_voc_size

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
        # Tokenizer that will be used to split document strings into words.
        if self.character:
            self.tokenizer = lambda s: s.lower()
        else:
            if tokenizer_choice is None or tokenizer_choice == 'simple':
                # use the following tokenizer for simplified tokenization
                self.tokenizer = lambda s: s.lower().split()
            elif tokenizer_choice.lower() == 'distilbert-base-uncased':
                bert_model_name = tokenizer_choice.lower()
                tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
                self.tokenizer = lambda s: tokenizer.tokenize(s)


    def build(self, docs):
        """Builds the vocabulary, based on a set of documents."""
        if self.stoi is None or self.itos is None:

            # Sort all words by frequency
            if self.character:
                # actually here its char freqs and not word freqs but meh.
                if self.bag_of_chars:
                    docs = [[c for w in doc for c in w] for doc in docs]
                else:
                    docs = [[c for w in self.tokenizer(doc) for c in w] for doc in docs]
            else:
                docs = [[w for w in self.tokenizer(doc)] for doc in docs]

            word_freqs = Counter(w for doc in docs for w in doc)
            word_freqs = sorted(((f, w) for w, f in word_freqs.items()), reverse=True)

            # Build the integer-to-string mapping. The vocabulary starts with the two dummy symbols,
            # and then all words, sorted by frequency. Optionally, limit the vocabulary size.
            if self.max_voc_size:
                self.itos = [self.PAD, self.UNKNOWN, self.MASK] + [w for _, w in word_freqs[:self.max_voc_size - 3]]
            else:
                self.itos = [self.PAD, self.UNKNOWN, self.MASK] + [w for _, w in word_freqs]

            # Build the string-to-integer map by just inverting the aforementioned map.
            self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, docs):
        """Encodes a set of documents."""
        unkn_index = self.stoi[self.UNKNOWN]
        if self.character:
            if self.bag_of_chars:
                encoded = [[self.stoi.get(w, unkn_index) for w in doc] for doc in docs]
            else:
                encoded = [[[self.stoi.get(c, unkn_index) for c in w] for w in self.tokenizer(doc)] for doc in docs]
        else:
            encoded = [[self.stoi.get(w, unkn_index) for w in self.tokenizer(doc)] for doc in docs]
        return encoded

    def encode_pretrain(self, doc):
        unkn_index = self.stoi[self.UNKNOWN]
        encoded = [self.stoi.get(w, unkn_index) for w in doc]

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
