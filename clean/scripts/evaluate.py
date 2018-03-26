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
        evaluate.py ea <model>
        evaluate.py eam <model>
        evaluate.py misclassified_adv <amount>

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

    elif args['misclassified_adv']:
        embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
        _,classifier, _2 = model_tools.load(model_path, embedding_holder=embedding_holder)
        dataholder = data_handler.Datahandler(config.PATH_ADV_DATA, data_format='snli_adversarial')
        categories = dataholder.get_categories()
        amount = int(args['<amount>'])

        for category in categories:
            data = dataholder.get_dataset_for_category_including_sents(embedding_holder, category)
            print('#', category)
            ev.print_misclassified(classifier, data, 32, embeddingholder.padding(), amount=amount)


    elif args['ea']:
        print('Evaluate all')
        embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
        _,classifier, _2 = model_tools.load(model_path, embedding_holder=embedding_holder)
        classifier = m.cuda_wrap(classifier)

        for name, dp in [('train data', config.PATH_TRAIN_DATA), ('dev data', config.PATH_DEV_DATA), ('test data',config.PATH_TEST_DATA)]:
            data = data_handler.Datahandler(dp).get_dataset(embedding_holder)
            classifier.eval()
            print('Accuracy on', name, ':', round(ev.eval(classifier, data, 32, embedding_holder.padding()) * 100, 2))

        # Adversarial
        dataholder = data_handler.Datahandler(config.PATH_ADV_DATA, data_format='snli_adversarial')
        categories = dataholder.get_categories()
        print('New dataset:')
        print('Accuracy on new dataset ->', round(ev.eval(classifier, dataholder.get_dataset(embedding_holder), 1, embedding_holder.padding()) * 100, 2))
        for category in sorted(categories):
            data = dataholder.get_dataset_for_category(embedding_holder, category)
            accuracy = ev.eval(classifier, data, 1, embedding_holder.padding())
            print('Accuracy on', category, '->', round(accuracy * 100, 2))

    elif args['eam']:
        print('Evaluate all merged labels contradiction+neutral')
        embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
        _,classifier, _2 = model_tools.load(model_path, embedding_holder=embedding_holder)
        classifier = m.cuda_wrap(classifier)

        for name, dp in [('train data', config.PATH_TRAIN_DATA), ('dev data', config.PATH_DEV_DATA), ('test data',config.PATH_TEST_DATA)]:
            dh = data_handler.Datahandler(dp)
            data = dh.get_dataset(embedding_holder)
            classifier.eval()
            print('Accuracy on', name, ':', round(ev.eval_merge_contr_neutr(classifier, data, 32, embedding_holder.padding(), dh.tag_to_idx) * 100, 2))

        # Adversarial
        dataholder = data_handler.Datahandler(config.PATH_ADV_DATA, data_format='snli_adversarial')
        categories = dataholder.get_categories()
        print('New dataset:')
        print('Accuracy on new dataset ->', round(ev.eval_merge_contr_neutr(classifier, dataholder.get_dataset(embedding_holder), 1, embedding_holder.padding(), dataholder.tag_to_idx) * 100, 2))
        for category in sorted(categories):
            data = dataholder.get_dataset_for_category(embedding_holder, category)
            accuracy = ev.eval_merge_contr_neutr(classifier, data, 1, embedding_holder.padding(), dataholder.tag_to_idx)
            print('Accuracy on', category, '->', round(accuracy * 100, 2))

        



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
