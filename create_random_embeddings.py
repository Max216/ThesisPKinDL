import model as m
import mydataloader
import train
import embeddingholder
import config

import torch
import torch.autograd as autograd

from docopt import docopt

import numpy as np

import mydataloader

def main():
    args = docopt("""Create new random embeddings for a dataset

    Usage:
        create_random_embeddings.py create <dataset_path> <name> <dimensions>
    """)

    dataset_path = args['<dataset_path>']
    name = args['<name>']
    dimensions = int(args['<dimensions>'])

    data = mydataloader.load_snli(dataset_path)

    create_embeddings(data, name, dimensions)


def create_embeddings(data, name, dimensions):
    # first get vocab
    vocab = set()
    for p, h, _ in data:
        vocab |= set(p + h)

    print('Found', len(vocab), 'distinct words.')

    vocab = list(vocab)

    # now create random matrix based on vocab: <#words> X <#dimens>
    embedding_matrix = m.cuda_wrap(torch.FloatTensor(len(vocab), dimensions))
    torch.nn.init.xavier_uniform(embedding_matrix)

    # store in files
    vocab_name = name + '.vocab'
    with open(vocab_name, 'w') as vocab_out:
        vocab_out.write('\n'.join(vocab))

    
    np.save(name, embedding_matrix.cpu().numpy())
    print('Done.')





if __name__ == '__main__':
    main()
