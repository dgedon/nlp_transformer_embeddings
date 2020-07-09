# %% Read data
def read_data(corpus_file):
    X = []
    Y = []
    Lemma = []
    Pos = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            sense_key, lemma, pos, doc = line.strip().split(maxsplit=3)
            X.append(doc)
            Y.append(sense_key)
            Lemma.append(lemma)
            Pos.append(int(pos))
    return X, Y, Pos, Lemma