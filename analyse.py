import sys
import os
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import model
from model import cuda_wrap, EntailmentClassifier

from docopt import docopt
import re

import nltk
from nltk import word_tokenize

import embeddingholder
import config
import mydataloader
import train

import matplotlib.pyplot as plt

FOLDER_ANALYSES = './analyses/'

def left_number(val):
    '''
    Remove the letters from a value of a model name. e.g. 0_001lr -> 0.001
    '''
    return re.split('[a-z]', val)[0]

def lbl_to_float(val):
    '''
    Map a value from a name to a float.
    '''
    return float(val.replace('_', '.'))

def idx_to_w(idxs, word_dict):
    return [word_dict[i] for i in idxs.data.numpy()]


def analyse(model, data, embedding_holder):
    '''
    Analyse what the model learned by checking where max-activation from the sentence
    representation are coming from.
    '''

    loader = DataLoader(data, 
                        drop_last = False,   
                        batch_size=1, 
                        shuffle=False, 
                        #num_workers=0, 
                        collate_fn=train.CollocateBatch(embedding_holder.padding()))


    for i_batch, (batch_p, batch_h, batch_lbl) in enumerate(loader):
        premises = autograd.Variable(cuda_wrap(batch_p))
        hypothesis = autograd.Variable(cuda_wrap(batch_h))
        predictions, indizes = model(premises, hypothesis, output_sent_info=True)

        predictions = predictions.data

        indizes_p = indizes[0].data
        indizes_h = indizes[1].data
        last_index_forward = indizes[0].size()[1] // 2 - 1

        # Can return here, has only one value
        return (indizes_p, indizes_h, last_index_forward, predictions)


def examine_sent(raw, indizes):
    # no need to make it efficient since only very low scale
    indizes = indizes.squeeze().numpy()
    result = []
    for i_word, word in enumerate(raw):
        word_activations = [position for position, val in enumerate(indizes) if val == i_word]
        result.append(word_activations)

    return result

def plot_word_activations(words, activations, last_idx_forward, directory, embedding_holder):

    # add information about the amount of activations per word
    absolute_amounts = [len(a) for a in activations]
    num_all = sum(absolute_amounts)
    relative_amounts = [round(abs_amount / num_all, 2) for abs_amount in absolute_amounts]
    word_labels = embedding_holder.replace_unk(words)
    word_labels = [word_labels[i] + ' \n(' + str(absolute_amounts[i]) + ')(' + str(relative_amounts[i]) + ')' for i in range(len(words))]

    y_axis = [i for i in reversed(range(len(words)))]
    plt.yticks(y_axis, word_labels)
    line_x = last_idx_forward + 0.5
    plt.plot((line_x, line_x), (0, y_axis[0]), 'k-', linewidth=0.2)
    for i in range(len(y_axis)):
        y_value = [y_axis[i] for idx in range(len(activations[i]))]
        plt.scatter(activations[i], y_value, s=1)
    plt.title(' '.join(words))

    path = directory + '_'.join(words) +'analyse.png'
    plt.savefig(path)
    print('Saved plot', path)
    plt.clf()


def main():
    args = docopt("""Analyse the model.

    Usage:
        analyse.py <model> <premise> <hypothesis> 

        <model>         Path to trained model that gets analysed.
        <premise>       Premise as a sentence.
        <hypothesis>    Hypothesis as a sentence
    """)

    model_path = args['<model>']
    premise = word_tokenize(args['<premise>'])
    hypothesis = word_tokenize(args['<hypothesis>'])
    
    model_name = model_path.split('/')[-1]
    splitted = model_name.split('-')

    lr = lbl_to_float(left_number(splitted[0]))
    hidden_dim = int(left_number(splitted[1]))
    lstm_dim = [int(i) for i in left_number(splitted[2]).split('_')]
    batch_size = int(left_number(splitted[3]))
    dropout = lbl_to_float(left_number(splitted[6]))

    if splitted[5] == 'relu':
        nonlinearity = F.relu
    else:
        raise Eception('Unknown activation function.', splitted[5])


    print('Load embeddings ...')
    embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
    print('embeddings loaded.')
    classifier = cuda_wrap(EntailmentClassifier(embedding_holder.embeddings, 
                                            dimen_hidden=hidden_dim, 
                                            dimen_out=3, 
                                            dimen_sent_encoder=lstm_dim,
                                            nonlinearity=nonlinearity, 
                                            dropout=dropout))

    print('Load model ...')
    classifier.load_state_dict(torch.load(model_path))
    print('Loaded.')

    dummy_label = 'neutral' # is not used anyway
    data = mydataloader.SNLIDataset([(premise, hypothesis, dummy_label)], embedding_holder)

    idx_p, idx_h, last_idx_forward, predictions = analyse(classifier, data, embedding_holder)

    # create destination for results
    directory = FOLDER_ANALYSES + model_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Analyze prediction
    predictions = predictions.squeeze().numpy()
    predictions = [str(predictions[i]) + ' [' + mydataloader.index_to_tag[i] + ']' for i in range(len(predictions))]
    print(predictions)

    result_p = examine_sent(premise, idx_p)
    result_h = examine_sent(hypothesis, idx_h)

    plot_word_activations(premise, result_p, last_idx_forward, directory, embedding_holder)
    plot_word_activations(hypothesis, result_h, last_idx_forward, directory, embedding_holder)

if __name__ == '__main__':
    main()


