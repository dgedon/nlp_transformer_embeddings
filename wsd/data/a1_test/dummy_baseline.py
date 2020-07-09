
from collections import defaultdict, Counter

def train(infile):
    stats = defaultdict(Counter)
    with open(infile) as f:
        for l in f:
            sense, lemma, _, = l.split(maxsplit=2)
            stats[lemma][sense] += 1

    return { l: stats_l.most_common(1)[0][0] for l, stats_l in stats.items() }
            
def run(model, infile, outfile):
    f_out = open(outfile, 'w')
    with open(infile) as f_in:
        for l in f_in:
            sense, lemma, _, = l.split(maxsplit=2)
            print(model[lemma], file=f_out)
    f_out.close()

TRAIN_DATA = 'wsd_train.txt'
TEST_DATA = 'wsd_test_blind.txt'
OUTPUT = 'dummy_baseline.txt'

model = train(TRAIN_DATA)
run(model, TEST_DATA, OUTPUT)
