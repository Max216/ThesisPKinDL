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

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def replace_word(w_test, w_replace, w_target):
    '''
    :param w_test gets replaced by :param w_target if w_test == w_replace
    '''
    if w_test == w_replace:
        return w_target
    return w_test

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
        self.type = path.split('.')[-1]
        self.sample_types = []
        self.dim = 2048

        if load:
            self.load()
    
    def get_sents(self, gold_label, predicted_label, max_len):
        ids = []
        stypes = []
        swapped_type = self.type == 'spkpair'

        for i, id in enumerate(self.samples):
            for pair in self.pairs[id]:
                if pair[4] == gold_label and pair[5] == predicted_label:
                    ids.append(id)
                    if swapped_type:
                        stypes.append(self.sample_types[i])
        ids = ids[:max_len]

        if len(ids) == 0:
            return []

        samples = []
        cnt_found = 0
        with open(self.FILE_DATA) as f_in:
            cnt = 0
            for line in f_in:
                if cnt in ids:
                    p, h, lbl = mydataloader.extract_snli(line.strip())

                    if swapped_type:
                        current_type = stypes[cnt_found]
                        if current_type == 'premise':
                            sent = p    
                        elif current_type == 'hypothesis':
                            sent = h
                        else:
                            print('Invalid type:', current_type)
                            return 1/0
                        samples.append((sent, [replace_word(w, self.w1, self.w2) for w in sent]))
                    else:
                        samples.append((p, h))

                    cnt_found += 1
                elif cnt > ids[-1]:
                    break
                cnt += 1
        return samples

    def get_common_dims(self, t_percent, t_min):
        '''
        First filters all dimensions s.t. only those remain with a value higher thn t_min. Then check if 
        t_percent of all samples have this.
        '''

        dim_cnt_1 = np.zeros(self.dim, dtype=int)
        dim_cnt_2 = np.zeros(self.dim, dtype=int)

        rep_sum_1 = []
        rep_sum_2 = []
        
        total_len = 0
        for key in self.pairs:
            for (dims1, reps1, dims2, reps2, gold, predicted) in self.pairs[key]:

                filtered_idx1 = [(i,dim) for i, dim in enumerate(dims1) if reps1[i] >= t_min]
                filtered_idx2 = [(i,dim) for i, dim in enumerate(dims2) if reps2[i] >= t_min]
                filtered_dims1 = [dim for i, dim in filtered_idx1]
                filtered_dims2 = [dim for i, dim in filtered_idx2]

                # Add dimension indizes
                add_dims_1 = np.zeros(self.dim, dtype=int)
                np.put(add_dims_1, filtered_dims1, np.ones(len(dims1)))
                
                add_dims_2 = np.zeros(self.dim, dtype=int)
                np.put(add_dims_2, filtered_dims2, np.ones(len(dims2)))

                # Add values
                add_vals_1 = np.zeros(self.dim, dtype=float)
                add_vals_2 = np.zeros(self.dim, dtype=float)
                for i, dim in filtered_idx1:
                    add_vals_1[dim] = reps1[i]
                for i, dim in filtered_idx2:
                    add_vals_2[dim] = reps2[i]

                dim_cnt_1 += add_dims_1
                dim_cnt_2 += add_dims_2
                rep_sum_1.append(add_vals_1)
                rep_sum_2.append(add_vals_2)
                total_len += 1
        
        min_len = total_len * t_percent
        
        relevant_dims1 = [i for i in range(self.dim) if dim_cnt_1[i] >= min_len]
        relevant_dims2 = [i for i in range(self.dim) if dim_cnt_2[i] >= min_len]

        rep_sum_1 = np.asmatrix(rep_sum_1)
        rep_sum_2 = np.asmatrix(rep_sum_2)

        coverage = []
        for dimension in f7(relevant_dims1 + relevant_dims2):
            cov1 = dim_cnt_1[dimension] / total_len
            cov2 = dim_cnt_2[dimension] / total_len
            coverage.append((dimension, [cov1, cov2]))

        def dimify(dim, dim_cnt, rep_sum):
            current_cnt = dim_cnt[dim]
            vals = np.asarray([v for v in np.asarray(rep_sum[:,dim]).flatten() if v > 0.0])
            
            if len(vals) == 0:
                mean = 0
                sd = 0
            else:
                mean = np.mean(vals)
                sd = np.std(vals)

            return (dim, mean, sd, current_cnt)
        print(relevant_dims2)
        print(relevant_dims1)
        print([d for d in relevant_dims2 if d not in relevant_dims1])
        relevant_dims1 = [dimify(d, dim_cnt_1, rep_sum_1) for d in relevant_dims1]
        relevant_dims2 = [dimify(d, dim_cnt_2, rep_sum_2) for d in relevant_dims2]

        return (relevant_dims1, relevant_dims2, coverage)


    def add_sample(self, sample_idx, gold, predicted, dims1, reps1, dims2, reps2, find_at=None):
        if self.type == 'spkpair':
            if find_at == None:
                print('Must specify "premise" or "hypothesis" when adding a sample to ".spkpair".')
                return
            else:
                self.sample_types.append(find_at)

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

        return counts

    def get_prediction_counts(self, labels=['entailment', 'neutral', 'contradiction']):
        counts = defaultdict(int)
        for key in self.predictions:
            for predicted in self.predictions[key]:
                counts[predicted] += self.predictions[key][predicted]
        return [(lbl, counts[lbl]) for lbl in labels]

    def accuracy(self):
        cnt_correct = 0
        cnt_incorrct = 0

        for lbl_gold in self.predictions:
            for lbl_predicted in self.predictions[lbl_gold]:
                if lbl_gold == lbl_predicted:
                    cnt_correct += self.predictions[lbl_gold][lbl_predicted]
                else:
                    cnt_incorrct += self.predictions[lbl_gold][lbl_predicted]

        return cnt_correct / (cnt_correct + cnt_incorrct + 0.00000000000001)

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
        self.samples = []

        start = 7
        _sample_types = None
        if self.type == 'spkpair':
            _sample_types = lines[7].split(' ')
            start = 8

        lines_labels = lines[start::5]
        lines_dims1 = lines[start+1::5]
        lines_reps1 = lines[start+2::5]
        lines_dims2 = lines[start+3::5]
        lines_reps2 = lines[start+4::5]

        for i in range(len(lines_labels)):
            splitted_labels = lines_labels[i].split(' ')
            sample_idx = int(splitted_labels[0])
            lbl_gold = splitted_labels[1]
            lbl_predicted = splitted_labels[2]

            dims1 = [int(v) for v in lines_dims1[i].split(' ')]
            dims2 = [int(v) for v in lines_dims2[i].split(' ')]
            reps1 = [float(v) for v in lines_reps1[i].split(' ')]
            reps2 = [float(v) for v in lines_reps2[i].split(' ')]

            if _sample_types != None:
                self.add_sample(sample_idx, lbl_gold, lbl_predicted, dims1, reps1, dims2, reps2, _sample_types[i])
            else:
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

            if self.type == 'spkpair':
                f_out.write(' '.join(self.sample_types))
                f_out.write('\n')

            indexer = defaultdict(int)
            for key in self.samples:
                (dims1, reps1, dims2, reps2, gold, predicted) = self.pairs[key][indexer[key]]
                indexer[key] += 1
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

def create_pk_analyse_data_for_swapped(model_path, data, w1, w2, assumed_label, twister=None):
    '''
    Create data files to analyse samples by using sentenves with <w1> as premise and the same sentence with 
    <w1> replaced by <w2> as hypthesis

    :param model_path   path to classifier
    :param data         all data
    :param w1           look for sentenceas containing w1
    :param w2           replace w1 with w2
    :param assumed_label    assumed label after replacing those two
    '''
    embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)

    classifier, classifier_name = m.load_model(model_path, embedding_holder)
    classifier = m.cuda_wrap(classifier)
    classifier.eval()

    dest_folder = to_classifier_folder(classifier_name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    print('Go through data ... ')
    if twister == None:
        spkpair = PkWordPair(dest_folder + w1 + '_' + w2 + '.spkpair', w1, w2)
    else:
        spkpair = PkWordPair(dest_folder + w1 + '_' + w2 + '_' + twister.name + '.spkpair', w1, w2)
    for idx, (premise, hypothesis, _ ) in enumerate(data):
        for key, sentence in [('premise', premise), ('hypothesis', hypothesis)]:
            if w1 in sentence:
                w_idx = sentence.index(w1)
                copy = [replace_word(w, w1, w2) for w in sentence]

                # Predict stuff
                scores, activations, representations = m.predict(classifier, embedding_holder, sentence, copy, twister=twister)
                _, predicted_idx = torch.max(scores, dim=1)
                predicted_label = mydataloader.index_to_tag[predicted_idx.data[0]]
                w1_act = activations[0].data.cpu().numpy()[0]
                w2_act = activations[1].data.cpu().numpy()[0]
                w1_rep = representations[0].data.cpu().numpy()[0]
                w2_rep = representations[1].data.cpu().numpy()[0]

                selected_dims_w1 = np.where(w1_act == w_idx)[0]
                selected_dims_w2 = np.where(w2_act == w_idx)[0]

                selected_reps_w1 = np.take(w1_rep, selected_dims_w1)
                selected_reps_w2 = np.take(w2_rep, selected_dims_w2)

                spkpair.add_sample(idx, assumed_label, predicted_label, selected_dims_w1, selected_reps_w1, selected_dims_w2, selected_reps_w2, key)
    print('Store', spkpair.path)
    spkpair.store()



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

def experiment2(model_path, data_path):
    #zero_dims = [402, 837, 1221, 1301, 1826]
    #zero_dims = [50, 72, 246, 341, 402, 731, 837, 1221, 1301, 1763, 1826]
    #zero_dims = [50, 72, 90, 246, 247, 266, 318, 341, 344, 402, 731, 979, 1062, 1221, 1227, 1310, 1667, 1713, 1751, 1763, 1826, 1934, 1990]
    #zero_dims = [95, 114, 266, 777, 877, 1352, 1388, 1505, 1606, 1657, 1665, 1667, 1731]
    zero_dims = [80, 95, 114, 266, 295, 372, 547, 754, 777, 877, 1069, 1301, 1352, 1388, 1565, 1606, 1657, 1665, 1667, 1705, 1713, 1998]

    name = 'zero_' + '_'.join([str(d) for d in zero_dims])
    data = mydataloader.load_snli(data_path)

    def twist_fn(representation, sent_type, _):
        if sent_type == 'hypothesis':
            for d in zero_dims:
                representation[0,d] = float(0.7)

        return representation

    twister = m.ModelTwister(twist_fn, name=name)
    create_pk_analyse_data_for_swapped(model_path, data, 'basketball', 'sport', 'entailment', twister=twister)

def experiment1(model_path, data_path):
    data = mydataloader.load_snli(data_path)

    swap_premise = ['football', 'basketball', 'hockey']
    swap_hyp = swap_premise

    def run (swaps_p, swaps_h, lbl):
        for sp in swaps_p:
            for sh in swaps_h:
                if sp != sh:
                    create_pk_analyse_data_for_swapped(model_path, data, sp, sh, lbl)

    #run(swap_premise, swap_hyp, 'contradiction')

    swap_hyp = ['sport']
    #run(swap_premise, swap_hyp, 'entailment')

    swap_premise = ['inside', 'outside']
    swap_hyp = swap_premise
    #run(swap_premise, swap_hyp, 'contradiction')

    swap_premise = ['river', 'lake', 'sea']
    swap_hyp = swap_premise
    #run(swap_premise, swap_hyp, 'contradiction')

    swap_hyp = ['water']
    #run(swap_premise, swap_hyp, 'entailment')

    swap_premise = ['different', 'same']
    run(swap_premise, swap_premise, 'contradiction')

    swap_premise = ['closed', 'open']
    run(swap_premise, swap_premise, 'contradiction')

    swap_premise = ['short', 'long']
    run(swap_premise, swap_premise, 'contradiction')
    

def main():
    args = docopt("""Analyse.

    Usage:
        pk_analyser.py create_single <model> <data> <w1> <w2> 
        pk_analyser.py replace <model> <data> <w1> <w2> <lbl>
        pk_analyser.py create <model> <data> <resource> <resource_label>
        pk_analyser.py summary <summary_file> <sort_type> <direction> [--ma=<min_amount>]
        pk_analyser.py show <file> <amount>
        pk_analyser.py comp <file> <t_percent> <t_min>
        pk_analyser.py experiment1 <model> <data>
        pk_analyser.py experiment2 <model> <data>
    """)

    if args['replace']:
        model_path = args['<model>']
        data_path = args['<data>']
        w1 = args['<w1>']
        w2 = args['<w2>']
        assumed_label = args['<lbl>']
        data = mydataloader.load_snli(data_path)
        create_pk_analyse_data_for_swapped(model_path, data, w1, w2, assumed_label)
    elif args['experiment1']:
        experiment1(model_path = args['<model>'],data_path = args['<data>'])
    elif args['experiment2']:
        experiment2(model_path = args['<model>'],data_path = args['<data>'])
    elif args['comp']:
        labels = ['entailment', 'neutral', 'contradiction']
        file = args['<file>']
        t_percent = float(args['<t_percent>'])
        t_min = float(args['<t_min>'])
        pkpair = PkWordPair(file, load=True)
        common_dims1, common_dims2, coverage = pkpair.get_common_dims(t_percent, t_min)
        data1 = [(str(dim), mean, std) for dim, mean, std, cnt in common_dims1]
        data2 = [(str(dim), mean, std) for dim, mean, std, cnt in common_dims2]
        
        title = pkpair.w1 + '_' + pkpair.w2 + ' (coverage:' + str(t_percent) + ', min val:' + str(t_min) + ')'
        pt.plot_double_chart_w_std(data1, data2, title, 'dimension', 'mean value', [pkpair.w1, pkpair.w2], block=False)

        coverage = [(str(dim), vals) for dim, vals in coverage]
        title_cov = title + '; coverage'
        pt.plot_multi_bar_chart(coverage, title_cov, [pkpair.w1, pkpair.w2], width=.35, rotate=90)

    elif args['create_single']:
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
        legend_labels = ['predicted '+lbl for lbl in labels]


        if pkpair.type == 'pkpair':
            general_data = pkpair.get_class_counts(labels=labels)
            general_data = [(lbl + '\n' + pkpair.str_precision_recall(lbl) ,  data) for lbl, data in general_data]
            legend_labels = ['gold'] + legend_labels
            title = file.split('/')[-1].split('.')[0] +' (' + str(pkpair.accuracy()) + ')'

            for lg in labels:
                for lp in labels:
                    print('#', lg, lp)
                    for p, h in pkpair.get_sents(lg, lp, amount):
                        print('[p]', ' '.join(p))
                        print('[h]', ' '.join(h))
                        print()

            pt.plot_multi_bar_chart(general_data, title, legend_labels=legend_labels)
        else:
            title = pkpair.w1 + '_' + pkpair.w2
            data = pkpair.get_prediction_counts()

            for lg in labels:
                for lp in labels:
                    print('#', lg, lp)
                    for p, h in pkpair.get_sents(lg, lp, amount):
                        print('[p]', ' '.join(p))
                        print('[h]', ' '.join(h))
                        print()

            pt.plot_single_bar_chart(data, title, 'predicted label', '# samples')




if __name__ == '__main__':
    main()