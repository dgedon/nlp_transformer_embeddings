from torch.utils.data import Dataset
import torch
from transformers import RobertaTokenizerFast


# %% Read data pretrain
def read_data_dataset_pretrain(file_path, get_characters, max_tokens, seq_length):
    if get_characters:
        tokenizer = lambda s: s.lower()
    else:
        autotokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        tokenizer = lambda s: autotokenizer.tokenize(s)

    with open(file_path, encoding="utf-8") as f:
        lines = []
        n_tokens = 0
        for line in f.read().splitlines():
            if (len(line) > 0 and not line.isspace()) and n_tokens <= max_tokens:
                if get_characters:
                    line_tokenized = tokenizer(line)
                    lines.append(line_tokenized)
                    n_tokens += len(line_tokenized) if len(line_tokenized)<seq_length else seq_length
                else:
                    lines.append(line)
                    n_tokens += len(tokenizer(line))
    return lines

# %% Document Batching pretraining

class DocumentDatasetPretrain(Dataset):
    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)

class DocumentBatcherPretrain:
    def __init__(self, voc, seq_length):  # , max_doc_words, max_docs_chars):
        # Find the integer index of the dummy padding word.
        self.pad = voc.get_pad_idx()
        self.max_doc_len_words = seq_length

    def __call__(self, data):
        # data is a list of tuples with words, tags, word_positions, word_chars

        # How long is the longest document in this batch?
        max_doc_len = self.max_doc_len_words
        # Build the document-word tensor.
        # We pad the shorter documents so that all documents have the same length.
        x_padded = torch.as_tensor(
            [x[:max_doc_len] + [self.pad] * (max_doc_len - len(x[:max_doc_len])) for x in data])

        return x_padded
