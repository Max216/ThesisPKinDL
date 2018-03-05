'''Evaluate a model'''


import sys, os
sys.path.append('./../')

from libs import model_tools, embeddingholder, config, data_handler
from libs import evaluate as ev
from libs import model as m

import torch
import torch.autograd as autograd

from docopt import docopt

def main():
    args = docopt("""Evaluate on given dataset in terms of accuracy.

    Usage:
        evaluate.py eval <model> <data> [<embeddings>] [--embd1=<embd1>] [--embd2=<embd2>]

        <model> = Path to trained model
        <data>  = Path to data to test model with 
        <embeddings>  = New embedding file to use unknown words from 
    """)

    model_path = args['<model>']
    data_path = args['<data>']
    embeddings_path = args['<embeddings>']
    embd1 = args['--embd1']
    embd2 = args['--embd2']


    if args['eval']:
        embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
        embeddings_diff = []
        if embeddings_path != None:
            print ('Merge embeddings')
            embedding_holder_new = embeddingholder.EmbeddingHolder(new_embeddings)
            embeddings_diff = embedding_holder.add_unknowns_from(embedding_holder_new)
        if embd1:
            embedding_holder.concat(embeddingholder.EmbeddingHolder(embd1))
        if embd2:
            embedding_holder.concat(embeddingholder.EmbeddingHolder(embd2))
        evaluate(model_path, data_path, embedding_holder, embeddings_diff=embeddings_diff)


def evaluate(model_path, data_path, embedding_holder, twister=None, embeddings_diff=False):
    # Load model


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
    classifier = m.cuda_wrap(classifier)

    print('Accuracy:', ev.eval(classifier, data, 32, embedding_holder.padding()))

if __name__ == '__main__':
    main()
