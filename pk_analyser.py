'''Analyse data and model with some resource'''

import word_resource
import embeddingholder
import mydataloader
import model as m
import config

from docopt import docopt

import torch
import numpy as np

import os

def to_classifier_folder(model_name):
    return './analyses/for_model/' + model_name + '_folder/' 

class PkWordPair:
    '''
    Holding all information of a word pair of an external resource and train data and a model
    '''

    def __init__(self, w1, w2, path, load=False):
        '''
        Create a new instance.

        :param path     Will store results here
        :param load     If true, it will load results here on init.
        '''
        self.w1 = w1
        self.w2 = w2
        self.path = path
        self.predictions = dict()
        self.samples = []
        self.pairs = dict()

        if load:
            print('TODO load')
    

    def add_sample(self, sample_idx, gold, predicted, dims1, reps1, dims2, reps2):
        
        self.samples.append(sample_idx)
        if sample_idx not in self.pairs:
            self.pairs[sample_idx] = []

        self.pairs[sample_idx].append((dims1, reps1, dims2, reps2))

        if gold not in self.predictions:
            self.predictions[gold] = dict()
        if predicted not in self.predictions[gold]:
            self.predictions[gold][predicted] = 1
        else:
            self.predictions[gold][predicted] += 1

def create_pk_analyse_data(classifier_path, data, w_res):
    '''
    Create data files to analyse samples containing word pairs for known relations.
    '''

    classifier, classifier_name = m.load_model(classifier_path)
    classifier = m.cuda_wrap(classifier)
    classifier.eval()

    embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)

    dest_folder = to_classifier_folder(classifier_name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    all_pairs = dict()

    for index, (premise, hypothesis, gold_label) in enumerate(data):
        word_indizes = w_res.get_word_pairs(premise, hypothesis)
        scores, activations, representations = m.predict(classifier, embedding_holder, premise, hypothesis)

        _, predicted_idx = torch.max(scores, dim=1)
        predicted_label = mydataloader.index_to_tag[predicted_idx.data[0]]

        p_act = activations[0].data.cpu().numpy()[0]
        h_act = activations[1].data.cpu().numpy()[0]
        p_rep = representations[0].data.cpu().numpy()[0]
        h_rep = representations[1].data.cpu().numpy()[0]

        for ip, ih in word_indizes:
            key = premise[ip] + '_' + hypothesis[ih]
            if key not in all_pairs:
                all_pairs[key] = PkWordPair(premise[ip], hypothesis[ih], dest_folder + key + '.pkpair')

            selected_dims_p = np.where(p_act == ip)[0]
            selected_dims_h = np.where(h_act == ih)[0]

            selected_reps_p = np.take(p_rep, selected_dims_p)
            selected_reps_h = np.take(h_rep, selected_dims_h)

            pkpair = all_pairs[key]
            pkpair.add_sample(index, gold_label, predicted_label, selected_dims_p, selected_reps_p, selected_dims_h, selected_reps_h)


def main():
    args = docopt("""Analyse.

    Usage:
        pk_analyser.py create <model> <data> <resource> <resource_label>
    """)

    if args['create']:
        model_path = args['<model>']
        data_path = args['<data>']
        res_path = args['<resource>']
        res_label = args['<resource_label>']

        data = mydataloader.load_snli(data_path)
        w_res = word_resource.WordResource(res_path, build_fn='snli', interested_relations=[res_label])

        create_pk_analyse_data(model_path, data, w_res)


if __name__ == '__main__':
    main()