import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    option_list = ['simple_char', 'simple_word', 'simple_word_char',
                   'pretrain_char', 'pretrain_word', 'pretrain_word_char']
    legend_list = ['simple chars', 'simple words', 'simple words + char',
                   'transformer chars', 'transformer words', 'transformer words + char']
    colorlist = ['b', 'g', 'r']

    # finetuning or no finetuning history
    history = []
    for option in option_list:
        path = os.path.join(os.getcwd(), 'server', option, 'history.csv')
        history.append(pd.read_csv(path))

    # print validation accuracy
    for attrib, attrib_legend in (['train_acc', 'Training Accuracy'], ['valid_acc', 'Validation Accuracy']):
        plt.figure(dpi=300)
        for i, option in enumerate(option_list):
            # plotting settings
            if option.startswith('simple'):
                linestyle = 'dashed'
                color = colorlist[i%len(colorlist)]
            else:
                linestyle = 'solid'
                color = colorlist[i % len(colorlist)]
            plt.plot(history[i]['epoch'], history[i][attrib], linestyle=linestyle, color=color)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(attrib_legend)
        plt.grid()
        plt.legend(legend_list)
        plt.savefig(os.path.join(os.getcwd(), 'server', attrib))
        plt.show()
