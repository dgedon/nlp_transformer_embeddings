
import sys

from collections import defaultdict, Counter

def acc_str(stats, key):
    corr = stats[key]['corr']
    count = stats[key]['count']
    acc = corr/count

    kc = key + ':'
    out = f'{kc:15} {corr:4} / {count:5} = {acc:.4f}'
    return out

if __name__ == '__main__':

    gf = open(sys.argv[1])
    pf = open(sys.argv[2])

    SKIP = 0
    COL = 0
    
    for _ in range(SKIP):
        pf.readline()
    
    stats = defaultdict(Counter)
    
    for g, p in zip(gf, pf):

        g = g.strip()
        p = p.strip()

        if COL == 0:
            pass
        elif COL == 1:
            ix = p.rindex(',')
            p = p[ix+1:]
        else:
            raise Exception('COL > 1')
            
        gt = g.split('\t')

        gold_sense = gt[0]
        lemma = gt[1]

        #print(lemma, gold_sense, p)
        
        stats['All']['count'] += 1
        stats[lemma]['count'] += 1

        if p == gold_sense:
            stats['All']['corr'] += 1
            stats[lemma]['corr'] += 1            

    print(acc_str(stats, 'All'))
    print('-------------------------------------')
    for key in sorted(stats):
        if key != 'All':
            print(acc_str(stats, key))
    
