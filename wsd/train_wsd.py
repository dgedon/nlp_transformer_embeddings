# Generic imports
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
# user defined inputs
from wsd.utils import *


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'data')
    data_file_train = 'wsd_train.txt'
    data_file_valid = 'wsd_test.txt'
    data_file_test = 'wsd_test_blind.txt'

    # read the data
    X, Y, Pos, Lemma = read_data(os.path.join(data_dir, data_file_train))
