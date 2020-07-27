import torch
from torch.utils.data import Dataset


# %% Read data finetuning
def read_data_dataset_finetuning(path):
    x = []
    y = []
    lemma = []
    word_pos = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            sense_key, lemma_temp, pos_temp, doc = line.strip().split(maxsplit=3)
            x.append(doc)
            y.append(sense_key)
            lemma.append(lemma_temp)
            word_pos.append(int(pos_temp))
    return x, y, word_pos, lemma


# %% Document Batching training
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
    def __init__(self, voc, bag_of_chars):  # , max_doc_words, max_docs_chars):
        # Find the integer index of the dummy padding word.
        self.pad = voc.get_pad_idx()
        self.bag_of_chars = bag_of_chars
        #self.max_doc_len_words = max_doc_words  # 192
        #self.max_doc_len_chars = max_docs_chars  # 896

    def __call__(self, data):
        # data is a list of tuples with words, tags, word_positions, word_chars

        # How long is the longest document in this batch?
        # limit max doc len to 192 words for constant seq_len in each batch
        max_doc_len = max(len(x) for x, _, _, _ in data)
        #max_doc_len = self.max_doc_len_words
        # Build the document-word tensor.
        # We pad the shorter documents so that all documents have the same length.
        x_padded = torch.as_tensor(
            [x[:max_doc_len] + [self.pad] * (max_doc_len - len(x[:max_doc_len])) for x, _, _, _ in data])

        # generate word padding mask (0 for non padded, 1 for padded)
        src_key_padding_mask = torch.as_tensor(
            [len(x[:max_doc_len]) * [0] + [1] * (max_doc_len - len(x[:max_doc_len])) for x, _, _, _ in data]).bool()

        # Build the label tensor.
        y = torch.as_tensor([y for _, y, _, _ in data])

        # Build the word position tensor.
        word_pos = torch.as_tensor([pos for _, _, pos, _ in data])

        # check if we need a bag of characters or if we have characters for each word
        if self.bag_of_chars:
            # how long (many chars) is the longest document? (max = 1024)
            max_doc_len = max(len(x_char) for _, _, _, x_char in data)
            # limit max doc len to 896 chars for constant seq_len in each batch
            # max_doc_len = self.max_doc_len_chars
            # Build the document-word tensor.
            # We pad the shorter documents so that all documents have the same length.
            x_char_padded = torch.as_tensor(
                [x_char[:max_doc_len] + [self.pad] * (max_doc_len - len(x_char[:max_doc_len])) for _, _, _, x_char in data])

            # generate char padding mask (0 for non padded, 1 for padded)
            src_char_key_padding_mask = torch.as_tensor(
                [len(x_char[:max_doc_len]) * [0] + [1] * (max_doc_len - len(x_char[:max_doc_len])) for _, _, _, x_char in data]).bool()
        else:
            # How long is the longest word in this batch?
            max_word_len = max(len(w) for _, _, _, x_char in data for w in x_char)
            # build the document-char tensor.
            # We pad the shorter documents so that all documents have the same length.
            x_char_padded = [x_char + [[]] * (max_doc_len - len(x_char)) for _, _, _, x_char in data]
            # We pad the shorter words so that all words have the same length.
            x_char_padded = [[w + [self.pad] * (max_word_len - len(w)) for w in x_char] for x_char in x_char_padded]
            x_char_padded = torch.as_tensor(x_char_padded)

            # generate char padding mask (0 for non padded, 1 for padded) [how to do as list comprehension?]
            res = []
            for x, _, _, x_char in data:
                temp = []
                for char in x_char:
                    temp.append(len(char) * [0] + [1] * (max_word_len - len(char)))
                for i in range(max_doc_len - len(x_char)):
                    temp.append([0] + [1] * (max_word_len - 1))
                res.append(torch.as_tensor(temp))

            src_char_key_padding_mask = torch.stack(res).bool()

        return y, word_pos, x_padded, src_key_padding_mask, x_char_padded, src_char_key_padding_mask
