from __future__ import print_function

import sys
import codecs
import numpy as np

from docopt import docopt

import train

def main():
    args = docopt("""Visualize the training outputs of a model.

    Usage:
        visualize_leanring.py <values.traininfo> 

        <values.traininfo> = gets output after training
    """)

    file = args['<values.traininfo>']
    with open (file) as f:
        content = f.readlines()

    name = content[0].strip()
    time = content[1].strip()

    amounts = [int(val) for val in content[2].strip().split(' ')]
    acc_train = [float(val) for val in content[3].strip().split(' ')]
    acc_dev = [float(val) for val in content[4].strip().split(' ')]
    mean_loss = [float(val) for val in content[5].strip().split(' ')]
    print('amounts', amounts)
    print('acc_train', acc_train)
    print('acc_dev', acc_dev)
    print('mean_loss', mean_loss)



if __name__ == '__main__':
    main()
