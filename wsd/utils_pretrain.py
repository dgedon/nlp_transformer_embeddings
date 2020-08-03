from torch.utils.data import Dataset
import torch
from transformers import RobertaTokenizerFast
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0.
    """

    def lr_lambda(current_step: int):
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# %% Read data pretrain
def read_data_dataset_pretrain_new(file_path, get_characters, max_tokens):
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
                    n_tokens += len(line_tokenized)
                else:
                    lines.append(line)
                    n_tokens += len(tokenizer(line))
    return lines

def read_data_dataset_pretrain(path, get_characters, max_tokens, tokenizer_choice):
    if get_characters:
        tokenizer = lambda s: s.lower()
    else:
        if tokenizer_choice is None or tokenizer_choice == 'simple':
            # use the following tokenizer for simplified tokenization
            tokenizer = lambda s: s.lower().split()
        elif tokenizer_choice.lower() == 'roberta':
            autotokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            tokenizer = lambda s: autotokenizer.tokenize(s)

    doc = []
    with open(path, encoding='utf-8') as source:
        for line in source:
            doc.extend(tokenizer(line))
            n_tokens = len(doc)
            if n_tokens >= max_tokens:
                if get_characters:
                    return doc[:max_tokens]
                else:
                    return doc[:max_tokens]

# %% Document Batching pretraining


class DocumentDatasetPretrain(Dataset):
    def __init__(self, x, seq_length):
        self.x = x
        self.seq_length = seq_length

    def __getitem__(self, idx):
        range_min = idx * self.seq_length
        range_max = (idx + 1) * self.seq_length
        return self.x[range_min:range_max]

    def __len__(self):
        return len(self.x) // self.seq_length


class DocumentBatcherPretrain:
    def __init__(self, seq_length):
        super(DocumentBatcherPretrain, self).__init__()
        self.seq_length = seq_length

    def __call__(self, data):
        # data is a list of length batch_size
        # each list element contains a list of length sequence length

        # reduce batch_size if not a full sequence length can be obtained
        x = torch.stack([torch.tensor(batch) for batch in data if len(batch) == self.seq_length])

        return x


# %%

class DocumentDatasetPretrain_new(Dataset):
    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)

class DocumentBatcherPretrain_new:
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
