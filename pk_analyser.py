'''Analyse data and model with some resource'''

import word_resource
import embeddingholder
import mydataloader
import model as m
import config
import plotting as pt

from docopt import docopt

import torch
import numpy as np

import os
from collections import defaultdict

def to_classifier_folder(model_name):
    return './analyses/for_model/' + model_name + '_folder/' 

def stringify(arr):
    return ' '.join(str(v) for v in arr)

class PkWordPair:

    FILE_DATA = config.PATH_TRAIN_DATA_CLEAN

    '''
    Holding all information of a word pair of an external resource and train data and a model
    '''

    def __init__(self, path, w1=None, w2=None, load=False):
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
            self.load()
    
    def get_sents(self, gold_label, predicted_label, max_len):
        ids = []
        for id in self.samples:
            for pair in self.pairs[id]:
                if pair[4] == gold_label and pair[5] == predicted_label:
                    ids.append(id)
        ids = ids[:max_len]

        if len(ids) == 0:
            return []

        samples = []
        with open(self.FILE_DATA) as f_in:
            cnt = 0
            for line in f_in:
                if cnt in ids:
                    p, h, lbl = mydataloader.extract_snli(line.strip())
                    samples.append((p, h))
                elif cnt > ids[-1]:
                    break
                cnt += 1
        return samples

    def add_sample(self, sample_idx, gold, predicted, dims1, reps1, dims2, reps2):
        
        if sample_idx not in self.pairs:
            self.pairs[sample_idx] = []

            # Only count prediction once.
            if gold not in self.predictions:
                self.predictions[gold] = defaultdict(int)
            if predicted not in self.predictions[gold]:
                self.predictions[gold][predicted] = 1
            else:
                self.predictions[gold][predicted] += 1

        self.samples.append(sample_idx)
        
        self.pairs[sample_idx].append((dims1, reps1, dims2, reps2, gold, predicted))

    def get_class_counts(self, labels=['entailment', 'neutral', 'contradiction']):
        '''
        Get the amount of samples per gold label together with the assoziated amount of classifications
        '''

        counts = []
        for lbl_gold in labels:
            if lbl_gold not in self.predictions:
                self.predictions[lbl_gold] = defaultdict(int)
            predictions = [self.predictions[lbl_gold][lbl_predicted] for lbl_predicted in labels]

            counts.append((lbl_gold, [sum(predictions)] + predictions))

        print(counts)
        return counts


    def accuracy(self):
        cnt_correct = 0
        cnt_incorrct = 0

        for lbl_gold in self.predictions:
            for lbl_predicted in self.predictions[lbl_gold]:
                if lbl_gold == lbl_predicted:
                    cnt_correct += self.predictions[lbl_gold][lbl_predicted]
                else:
                    cnt_incorrct += self.predictions[lbl_gold][lbl_predicted]

        return cnt_correct / (cnt_correct + cnt_incorrct)

    def precision_recall(self, label):
        tp = 0.00000000001
        tn = 0.00000000001
        fp = 0.00000000001
        fn = 0.00000000001

        for lbl_gold in self.predictions:
            for lbl_predicted in self.predictions[lbl_gold]:

                if label == lbl_gold:
                    if lbl_gold == lbl_predicted:
                        tp += self.predictions[lbl_gold][lbl_predicted]
                    else:
                        fn += self.predictions[lbl_gold][lbl_predicted]
                elif label == lbl_predicted and label != lbl_gold:
                    fp += self.predictions[lbl_gold][lbl_predicted] 
                else:
                    tn += self.predictions[lbl_gold][lbl_predicted] 

        precision =  tp / (tp + fp)
        recall = tp / (tp + fn)

        return precision, recall

    def str_precision_recall(self, label,round_to=2, separator='\n'):
        prec, recall = self.precision_recall(label)
        return separator.join(['Precision: ' + str(round(prec, round_to)), 'Recall: ' + str(round(recall, round_to))])



    def sample_len(self, count_doubles=True):
        cnt_items = self.samples
        if count_doubles == False:
            cnt_items = list(set(cnt_items))
        return len(cnt_items)

    def load(self):
        with open(self.path) as f_in:
            lines = [line.strip() for line in f_in.readlines()]

        self.w1 = lines[0]
        self.w2 = lines[1]
        self.samples = [int(v) for v in lines[6].split(' ')]

        lines_labels = lines[7::5]
        lines_dims1 = lines[8::5]
        lines_reps1 = lines[9::5]
        lines_dims2 = lines[10::5]
        lines_reps2 = lines[11::5]

        for i in range(len(lines_labels)):
            splitted_labels = lines_labels[i].split(' ')
            sample_idx = int(splitted_labels[0])
            lbl_gold = splitted_labels[1]
            lbl_predicted = splitted_labels[2]

            dims1 = [int(v) for v in lines_dims1[i].split(' ')]
            dims2 = [int(v) for v in lines_dims2[i].split(' ')]
            reps1 = [float(v) for v in lines_reps1[i].split(' ')]
            reps2 = [float(v) for v in lines_reps2[i].split(' ')]

            self.add_sample(sample_idx, lbl_gold, lbl_predicted, dims1, reps1, dims2, reps2)


    def store(self):
        lines_general = [
            self.w1, self.w2,
            str(self.accuracy()),
            stringify(self.precision_recall('entailment')),
            stringify(self.precision_recall('contradiction')),
            stringify(self.precision_recall('neutral')),
            stringify(self.samples)
        ]

        with open(self.path, 'w') as f_out:
            f_out.write('\n'.join(lines_general))
            f_out.write('\n')

            for key in self.pairs:
                for (dims1, reps1, dims2, reps2, gold, predicted) in self.pairs[key]:
                    f_out.write(' '.join([str(key), gold, predicted]) + '\n')
                    f_out.write(stringify(dims1) + '\n')
                    f_out.write(stringify(reps1) + '\n') 
                    f_out.write(stringify(dims2) + '\n') 
                    f_out.write(stringify(reps2) + '\n') 

def get_summary_items(summary_file, sort='size', reverse=True, min_amount=-1):
    sort_idx = dict([
        ('w1', 0), ('w2', 1), ('size', 2), ('ind_size', 3), ('acc', 4)
    ])

    if sort not in sort_idx:
        print('Specify one of the following for sorting:', sort_idx.keys())
        return
    else:
        sidx = sort_idx[sort]

    with open(summary_file) as f_in:
        data = [line.strip().split(' ') for line in f_in]

    data = sorted([(d[0], d[1], int(d[2]), int(d[3]), float(d[4]), d[5]) for d in data], key=lambda x: x[sidx], reverse=reverse)
    
    if min_amount > -1:
        data = [d for d in data if d[2] >= min_amount]

    return data



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
    print('Start going through data ...')
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
                all_pairs[key] = PkWordPair(dest_folder + key + '.pkpair', premise[ip], hypothesis[ih])

            selected_dims_p = np.where(p_act == ip)[0]
            selected_dims_h = np.where(h_act == ih)[0]

            selected_reps_p = np.take(p_rep, selected_dims_p)
            selected_reps_h = np.take(h_rep, selected_dims_h)

            pkpair = all_pairs[key]
            pkpair.add_sample(index, gold_label, predicted_label, selected_dims_p, selected_reps_p, selected_dims_h, selected_reps_h)

    print('Done.')
    print('Write out files ...')
    summary_data = []
    for key in all_pairs:
        pkpair = all_pairs[key]
        pkpair.store()

        summary_data.append((pkpair.w1, pkpair.w2, pkpair.sample_len(), pkpair.sample_len(count_doubles=True), pkpair.accuracy(), pkpair.path))
    print('Done.')
    print('Write summary ... ')
    with open(dest_folder + 'summary.txt', 'w') as f_out:
        f_out.write('\n'.join([stringify(data) for data in summary_data]))
    print('Done.')

def create_pk_analyse_data_for_pair(classifier_path, data, w1, w2):
    w_res = word_resource.WordResource((w1, w2, ''), build_fn='single_pair')
    create_pk_analyse_data(classifier_path, data, w_res)

def main():
    args = docopt("""Analyse.

    Usage:
        pk_analyser.py create_single <model> <data> <w1> <w2>
        pk_analyser.py create <model> <data> <resource> <resource_label>
        pk_analyser.py summary <summary_file> <sort_type> <direction> [--ma=<min_amount>]
        pk_analyser.py show <file> <amount>
    """)


    if args['create_single']:
        model_path = args['<model>']
        data_path = args['<data>']
        w1 = args['<w1>']
        w2 = args['<w2>']
        data = mydataloader.load_snli(data_path)
        create_pk_analyse_data_for_pair(model_path, data, w1, w2)
    elif args['create']:
        model_path = args['<model>']
        data_path = args['<data>']
        res_path = args['<resource>']
        res_label = args['<resource_label>']

        print('Load data ...')
        data = mydataloader.load_snli(data_path)
        print('Done.')
        print('Load ressource ...')
        w_res = word_resource.WordResource(res_path, build_fn='snli', interested_relations=[res_label])
        print('Done.')

        create_pk_analyse_data(model_path, data, w_res)
    elif args['summary']:
        summary_file = args['<summary_file>']
        sort_type = args['<sort_type>']
        direction = args['<direction>']
        min_amount = int(args['--ma'] or -1)
        print(min_amount)

        if direction == 'normal':
            reverse = False
        else:
            reverse = True
        data = get_summary_items(summary_file, sort=sort_type, reverse=reverse, min_amount=min_amount)

        for w1, w2, amount, amount2, acc, _ in data:
            print(w1 + '-' + w2 + ': ' + str(amount) + ', ' + str(amount2) + '; Acc: ' + str(acc))

    elif args['show']:
        labels = ['entailment', 'neutral', 'contradiction']
        file = args['<file>']
        amount = int(args['<amount>'])


        pkpair = PkWordPair(file, load=True)
        general_data = pkpair.get_class_counts(labels=labels)

        general_data = [(lbl + '\n' + pkpair.str_precision_recall(lbl) ,  data) for lbl, data in general_data]

        legend_labels = ['gold'] + ['predicted '+lbl for lbl in labels]
        title = file.split('/')[-1].split('.')[0] +' (' + str(pkpair.accuracy()) + ')'

        for lg in labels:
            for lp in labels:
                print('#', lg, lp)
                for p, h in pkpair.get_sents(lg, lp, amount):
                    print('[p]', ' '.join(p))
                    print('[h]', ' '.join(h))
                    print()

        pt.plot_multi_bar_chart(general_data, title, legend_labels=legend_labels)




if __name__ == '__main__':
    main()