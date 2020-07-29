from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer


# %% Read data pretrain
def read_data_dataset_pretrain(path, get_characters, max_tokens, tokenizer_choice):
    if get_characters:
        tokenizer = lambda s: s.lower()
    else:
        if tokenizer_choice is None or tokenizer_choice == 'simple':
            # use the following tokenizer for simplified tokenization
            tokenizer = lambda s: s.lower().split()
        elif tokenizer_choice.lower() == 'distilbert-base-uncased':
            bert_model_name = tokenizer_choice.lower()
            autotokenizer = AutoTokenizer.from_pretrained(bert_model_name)
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
