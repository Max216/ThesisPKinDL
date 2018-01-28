'''Evaluate a model'''


import sys, os
sys.path.append('./../')

from libs import model_tools, embeddingholder, config, data_handler
from libs import evaluate as ev

import torch
import torch.autograd as autograd

from docopt import docopt

def main():
    args = docopt("""Evaluate on given dataset in terms of accuracy.

    Usage:
        evaluate.py eval <model> <data> [<embeddings>]

        <model> = Path to trained model
        <data>  = Path to data to test model with 
        <embeddings>  = New embedding file to use unknown words from 
    """)

    model_path = args['<model>']
    data_path = args['<data>']
    embeddings_path = args['<embeddings>']


    if args['eval']:
        evaluate(model_path, data_path, embeddings_path)


def evaluate(model_path, data_path, new_embeddings=None, twister=None):
    # Load model

    embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
    embeddings_diff = []
    if new_embeddings != None:
        print ('Merge embeddings')
        embedding_holder_new = embeddingholder.EmbeddingHolder(new_embeddings)
        embeddings_diff = embedding_holder.add_unknowns_from(embedding_holder_new)

    print('Load model ...')
    _,classifier, _2 = model_tools.load(model_path, embedding_holder=embedding_holder)

    # todo look with merging ....
    if len(embeddings_diff) != 0 and embeddings_diff.shape[1] != 0:
        # Merge into model
        classifier.inc_embedding_layer(embeddings_diff)

    print('Load data ...')
    data = data_handler.Datahandler(data_path).get_dataset(embedding_holder)
    print(len(data), 'samples loaded.')
    print('Evaluate ...')
    classifier.eval()

    print('Accuracy:', ev.eval(classifier, data, 32, embedding_holder.padding()))

if __name__ == '__main__':
    main()
