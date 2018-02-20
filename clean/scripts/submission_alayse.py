'''Evaluate a model'''


import sys, os, collections, json, re
import torch
sys.path.append('./../')

from libs import data_tools

from docopt import docopt

import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy import spatial

import matplotlib.pyplot as plt

COLOR_DECOMPOSABLE = '#FFC312'
COLOR_ESIM = '#ED4C67'
COLOR_RESIDUAL = '#1289A7'


def create_word_mapping():
    keep = ['New Zealand','dining room', 'prison cell','acoustic guitar','North Korean','South Korean','common room','can not','hot chocolate', 'North Korea', 'living room', 'no one', 'Saudi Arabia', 'electric guitar', 'french horn']
    adapt = [
    ('in a bathroom', ['bathroom']),
    ('in a garage', ['garage']), 
    ('in a kitchen', ['kitchen']),
    ('plenty of', ['plenty']),
    ('in a building', ['building']),
    ('far from', ['far']), 
    ('far away from', ['far', 'away']),
    ('at night', ['night']),
    ('close to', ['close']), 
    ('a lot of', ['lot']),
    ('in a hallway', ['hallway']), 
    ('in a room', ['room']), 
    ('during the day', ['during', 'day']),
     ('in front of', ['front'])]

    result_dict = dict()
    for wp in keep:
        result_dict[wp] = wp.split()
        result_dict[wp.lower()] = [w.lower() for w in wp.split()]

    for wp_from, wp_to in adapt:
        result_dict[wp_from] = wp_to
        result_dict[wp_from.lower()] = [w.lower() for w in wp_to]

    return result_dict

word_mapper = create_word_mapping()

def main():
    args = docopt("""For submission

    Usage:
        submission_alayse.py create_counts <data_in> <file_out>
        submission_alayse.py create_counts_lower <data_in> <file_out>
        submission_alayse.py wc <wordcount> <word>
        submission_alayse.py create_esim_anl <esim_results> <dataset> <original_dataset> <wordcount> <out>
        submission_alayse.py create_res_anl <esim_results> <dataset> <original_dataset> <wordcount> <out>
        submission_alayse.py create_decomp_anl <esim_results> <dataset> <original_dataset> <wordcount> <out>
        submission_alayse.py stats <results>
        submission_alayse.py create_cos <results> <embeddings> <path_out> [<lower>]
        submission_alayse.py plot_cos <cosfile>
        submission_alayse.py eval <results>
        submission_alayse.py plot_freq_acc <esim> <residual> <decomp>
        submission_alayse.py labelstats <results>
        submission_alayse.py plot_file <file>
        submission_alayse.py plot_freq_acc_file <file>
        submission_alayse.py find_samples <testset> <file> <group>
        submission_alayse.py validate_esim <esim_results>
        submission_alayse.py add_wc_snli_mnli <word_count> <results_in> <results_out>
    """)


    if args['create_counts']:
        create_counts(args['<data_in>'], args['<file_out>'])
    elif args['add_wc_snli_mnli']:
        add_wc_snli_mnli(args['<word_count>'], args['<results_in>'], args['<results_out>'])
    elif args['validate_esim']:
        validate_esim(args['<esim_results>'])
    elif args['wc']:
        word_count(args['<wordcount>'], args['<word>'])
    elif args['create_counts_lower']:
        create_counts_lower(args['<data_in>'], args['<file_out>'])
    elif args['create_esim_anl']:
        create_esim_analyse_file(args['<esim_results>'], args['<dataset>'], args['<original_dataset>'], args['<wordcount>'], args['<out>'])
    elif args['create_res_anl']:
        create_residual_analyse_file(args['<esim_results>'], args['<dataset>'], args['<original_dataset>'], args['<wordcount>'], args['<out>'])
    elif args['create_decomp_anl']:
        create_decomposition_analyse_file(args['<esim_results>'], args['<dataset>'], args['<original_dataset>'], args['<wordcount>'], args['<out>'])
    elif args['stats']:
        print_stats(args['<results>'])
    elif args['create_cos']:
        create_cosine_similarity(args['<results>'], args['<embeddings>'], args['<path_out>'], args['<lower>'])
    elif args['plot_cos']:
        plot_cos(args['<cosfile>'])
    elif args['eval']:
        evaluate(args['<results>'])
    elif args['plot_freq_acc']:
        plot_freq_acc(args['<esim>'], args['<residual>'], args['<decomp>'])
    elif args['plot_file']:
        plot_file(args['<file>'])
    elif args['plot_freq_acc_file']:
        plot_freq_acc_file(args['<file>'])
    elif args['labelstats']:
        label_stats(args['<results>'])
    elif args['find_samples']:
        find_samples(args['<testset>'],args['<file>'], args['<group>'])


def add_wc_snli_mnli(word_count_path, result_in_path, result_out_path):
    word_count_snli_mnli =torch.load(word_count_path)

    with open(result_in_path) as f_in:
        samples = [json.loads(line.strip()) for line in f_in.readlines()]

    for sample in samples:
        w1 = sample['replaced1'].split(' ')
        w2 = sample['replaced2'].split(' ')

        if len(w1) == 1:
            cnt_w1 = word_count_snli_mnli[w1[0]]
        else:
            cnt_w1 = min([word_count_snli_mnli[w] for w in w1])

        if len(w2) == 1:
            cnt_w2 = word_count_snli_mnli[w2[0]]
        else:
            cnt_w2 = min([word_count_snli_mnli[w] for w in w2])

        sample['count1'] = cnt_w1
        sample['count2'] = cnt_w2

    with open(result_out_path, 'w') as f_out:
        for s in samples:
            f_out.write(json.dumps(s) + '\n')


def validate_esim(esim_file):
    with open(esim_file) as f_in:
        content = [line.strip().split('\t') for line in f_in.readlines()]
    GOLD = 2
    PREDICTED = 3

    incorrect = 0
    correct = 0
    for c in content:
        if c[GOLD] != c[PREDICTED]:
            incorrect += 1
        else:
            correct += 1

    print(correct / len(content), '==', correct / (correct + incorrect))




def load_dataset(path):
    with open(path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]
    return parsed

def load_embeddings(embedding_path):
    with open(embedding_path) as f_in:
        lines = [line.strip() for line in f_in.readlines()]

    lines = [line.split(' ', 1) for line in lines]
    return dict([(line[0], np.asarray([float(v) for v in line[1].split()])) for line in lines])

def label_stats(result_file):
    data = load_dataset(result_file)
    data = [d for d in data if d['gold_label'] == 'contradiction']

    contr_dict = collections.defaultdict(lambda: collections.defaultdict(int))
    count_entailemt = 0
    count_neutral = 0
    count_contradiction = 0
    for d in data:
        lbl = d['predicted_label']
        contr_dict[d['category']][lbl] += 1

        if lbl == 'contradiction':
            count_contradiction += 1
        elif lbl == 'neutral':
            count_neutral += 1
        elif lbl == 'entailment':
            count_entailemt += 1
        else:
            1/0


    for cat in contr_dict:
        print('#', cat)
        cnt = 0
        for pred in contr_dict[cat]:
            cnt += contr_dict[cat][pred]
            print(pred, '->', contr_dict[cat][pred])
        print('total:', cnt)
        print()

    print('general:')
    print('entailment:', count_entailemt)
    print('contradiction:', count_contradiction)
    print('neutral:', count_neutral)
    print('verify:', count_entailemt + count_contradiction + count_neutral, '==', len(data))

def get_embedding(embeddings, words, lower=False):
    if lower == 'lower':
        print('lowering!')
        words = words.lower()
    splitted = words.split()
    if len(splitted) == 1:
        if words == 'cannot':
            return np.mean(np.array([embeddings['can'], embeddings['not']]), axis=0)
        else:
            return embeddings[words]
    else:
        mapped_words = word_mapper[words]
        all_vecs = np.array([embeddings[w] for w in mapped_words])
        return np.mean(all_vecs, axis=0)


def cnt_word_or_phrase(word_dict, w):
    splitted = w.split()
    if len(splitted) == 1:
        return word_dict[w]
    else:
        return min([word_dict[w] for w in splitted])

def cos_sim(a, b):
    #return 1 - spatial.distance.cosine(a, b)
    return dot(a, b)/(norm(a)*norm(b))

def word_count(wordcount_file, word):
    wc = torch.load(wordcount_file)
    print(word, wc[word])

def create_cosine_similarity(result_path, embeddings_path, path_out, lower=False):
    results = load_dataset(result_path)
    embeddings = load_embeddings(embeddings_path)
    results = [r for r in results if r['gold_label'] == 'contradiction']
    print('Preselect', len(results), 'results')
    
    final_values = []
    for sample in results:
        embd1 = get_embedding(embeddings, sample['replaced1'], lower=lower)
        embd2 = get_embedding(embeddings, sample['replaced2'], lower=lower)
        if len(embd2) != 300:
            print('oh no!', len(embd2))
            1/0
        similarity = cos_sim(embd1, embd2)
        final_values.append((sample['replaced1'], sample['replaced2'], sample['gold_label'], sample['predicted_label'], sample['count1'], sample['count2'],  sample['category'], similarity))

    all_similarities = sorted([fv[-1] for fv in final_values])
    print(all_similarities)

    with open(path_out, 'w') as f_out:
        print('Saving', len(final_values), 'values')
        for w1, w2, gold_lbl, predicted_lbl, cnt1, cnt2, category, similarity in final_values:
            f_out.write('\t'.join([w1,w2,gold_lbl,predicted_lbl,str(cnt1), str(cnt2),category,str(similarity)]) + '\n')

def find_samples(testset_path, file, group):
    dataset = load_dataset(file)
    testset = load_dataset(testset_path)

    group_words = set()
    for sample in testset:
        if sample['category'] == group and sample['gold_label'] == 'contradiction':
            group_words.add(sample['replaced1'])
            group_words.add(sample['replaced2'])
    print('group words:', group_words)
    regexps = [(re.compile('\\b' + w + '\\b')) for w in list(group_words)]
    counter = 0
    for sample in dataset:
        if sample['gold_label'] == 'contradiction':
            for i in range(len(group_words)):
                r1 = regexps[i]
                if r1.search(sample['sentence1']):
                    for j in range(len(regexps)):
                        if i != j:
                            r2 = regexps[j]
                            if r2.search(sample['sentence2']):
                                counter += 1
                                print(sample['sentence1'])
                                print(sample['sentence2'])
                                print()
                                break
    print('total:', counter)






def plt_file_acc(samples):
    correct = 0
    for sample in samples:
        gold = sample[2]
        predicted = sample[3]

        if gold == predicted:
            correct += 1

    return correct / len(samples)

def create_bins(samples, bin_size=0.05):
    sorted_content = sorted(samples, key=lambda x: x[-1])

    # verify
    for c in sorted_content:
        if c[2] != 'contradiction':
            print('something is wrong!! no contradiction:', c[2])

    bins = []
    max_bound =  bin_size
    while(len(sorted_content)) > 0:
        #print('find bin with max bound:', max_bound, len(sorted_content))
        for i in range(len(sorted_content)):
            if sorted_content[i][-1] > max_bound:
                bins.append((max_bound - bin_size, max_bound, sorted_content[:i]))
                sorted_content = sorted_content[i:]
                max_bound += bin_size
                break

            # if reached here it is over
            if i == len(sorted_content) - 1:
                bins.append((max_bound - bin_size, max_bound, sorted_content))
                sorted_content = []
    return bins

def acc_predictiondict(pd):
    correct = 0
    incorrect = 0
    for gold in pd:
        for pred in pd[gold]:
            if pred == gold:
                correct += pd[gold][pred]
            else:
                incorrect += pd[gold][pred]

    return correct / (correct + incorrect)

def recall_precision_prediction_dict(prediction_dict, label):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    ZERO = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    for gold_label in prediction_dict:
        for predicted_label in prediction_dict[gold_label]:
            if gold_label == predicted_label:
                if gold_label == label:
                    tp += prediction_dict[gold_label][predicted_label]
                else:
                    tn += prediction_dict[gold_label][predicted_label]
            else:
                if predicted_label == label:
                    fp += prediction_dict[gold_label][predicted_label]
                elif gold_label == label:
                    fn += prediction_dict[gold_label][predicted_label]

    recall = tp / (tp + fn + ZERO)
    precision = tp / (tp + fp+ ZERO)
    return (recall, precision)

def evaluate(result_path):
    data = load_dataset(result_path)

    correct = len([d for d in data if d['gold_label'] == d['predicted_label']])
    print('###', correct / len(data))

    cat_dict = collections.defaultdict(list)
    for d in data:
        cat_dict[d['category']].append(d)

    for cat in cat_dict:
        print('# category:', cat)
        n_entailment = len([d for d in cat_dict[cat] if d['gold_label'] == 'entailment'])
        n_neutral = len([d for d in cat_dict[cat] if d['gold_label'] == 'neutral'])
        n_contradiction = len([d for d in cat_dict[cat] if d['gold_label'] == 'contradiction'])
        print('size:', len(cat_dict[cat]), ', entailment:', n_entailment, ', neutral:', n_neutral, ', contradiction:', n_contradiction)

        prediction_dict = collections.defaultdict(lambda: collections.defaultdict(int))
        for sample in cat_dict[cat]:
            prediction_dict[sample['gold_label']][sample['predicted_label']] += 1
            #print(sample['gold_label'], '--', sample['predicted_label'])

        # accuracy
        print('accuracy:', acc_predictiondict(prediction_dict))

        # precision entailment
        e_rec, e_prec = recall_precision_prediction_dict(prediction_dict, 'entailment')
        print('entailment: prec =', e_prec, ', recall =', e_rec)

        # precision contradiction
        c_rec, c_prec = recall_precision_prediction_dict(prediction_dict, 'contradiction')
        print('contradiction: prec =', c_prec, ', recall =', c_rec)

        # precision neutral
        n_rec, n_prec = recall_precision_prediction_dict(prediction_dict, 'neutral')
        print('neutral: prec =', n_prec, ', recall =', n_rec)

def plot_cos(cos_file, bin_size= 0.1):
    #([w1,w2,gold_lbl,predicted_lbl,str(cnt1), str(cnt2),category,str(similarity)])
    GOLD = 2
    PRED = 3
    CNT1 = 4
    CNT2 = 5
    with open(cos_file) as f_in:
        content = [line.strip().split('\t') for line in f_in.readlines()]

    # verify contradiction
    for c in content:
        if c[GOLD] != 'contradiction':
            print('Not good data!')
            1/0
        c[CNT1] = int(c[CNT1])
        c[CNT2] = int(c[CNT2])
        c[-1] = float(c[-1])

    print('Validated only contradiction.')
    accuracies = []
    bins = create_bins(content, bin_size=bin_size)
    for vstart, vend, samples in bins:
        print(vstart, '-', vend, '->', len(samples))
        correct = 0
        for s in samples:
            if s[GOLD] == s[PRED]:
                correct += 1
        accuracies.append(correct / len(samples))



    y_vals = [round(x * 100,2) for x in accuracies]
    half_bin = bin_size/2
    x_vals = [half_bin + i*bin_size for i in range(len(y_vals))]
    #y_vals = [y for x,y in data]
    #x_indizes = np.arange(len(data))
    width = 1/(len(y_vals) * 1.1)
    color = COLOR_DECOMPOSABLE

    plt.bar(x_vals, y_vals, width, color=color)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Cosine similarity')
    #plt.xticks(x_indizes, x_labels)
    #plt.title(title)

    plt.show()



def plot_file(file):
    bin_size = 0.2
    with open(file) as f_in:
        content = [line.strip().split() for line in f_in.readlines()]

    y_vals = [round(float(c[1]) * 100,2) for c in content]
    half_bin = bin_size/2
    x_labels = [c[0] for c in content]
    #y_vals = [y for x,y in data]
    #x_indizes = np.arange(len(data))
    width = 0.8#1/(len(content))
    color = COLOR_DECOMPOSABLE

    index = np.arange(len(content))

    plt.bar(index, y_vals, width, color=COLOR_DECOMPOSABLE)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Cosine similarity')
    plt.xticks(index , x_labels, rotation=0)
    #plt.xticks(x_indizes, x_labels)
    #plt.title(title)

    plt.show()


def freq_bins(content, bins = [500,1500,5000,25000,50000]):
    contents = [[] for i in range(len(bins))]
    for c in content:
        max_freq = max([c[4], c[5]])
        for i in range(len(bins)):
            if i < len(bins) - 1:
                if max_freq < bins[i]:
                    contents[i].append(c)
                    break
            else:
                # last bin

                contents[-1].append(c)

    test = contents[-1]
    print('verify:', min([max([t[4], t[5]]) for t in test]))

    results = []
    for i in range(len(bins)):
        bin_result = []
        if i < len(bins) - 1:
            print('max-freq', bins[i],'>>', len(contents[i]), plt_file_acc(contents[i]))
            bin_result.append(plt_file_acc(contents[i]))
        else:
            print('min-freq', bins[i],'>>', len(contents[i]), plt_file_acc(contents[i]))
            bin_result.append(plt_file_acc(contents[i]))
        results.append(bin_result)

    return bins, [r[0] for r in results]

    
def asym_freq_bins(content, split1=500, split2=1500):
    content_rare = []
    content_semi_rare = []
    content_semi_often = []
    content_often = []


    for c in content:
        count1 = c[4]
        count2 = c[5]

        if count1 <= split1 and count2 <= split1:
            content_rare.append(c)
        elif count1 <= split2 and count2 <= split1:
            content_semi_rare.append(c)
        elif count1 <= split1 and count2 <= split2:
            content_semi_rare.append(c)
        elif count1 <= split2 or count2 <= split2:
            content_semi_often.append(c)
        else:
            content_often.append(c)

    print('content_rare', len(content_rare), plt_file_acc(content_rare))
    print('content_semi_rare', len(content_semi_rare), plt_file_acc(content_semi_rare))
    print('content_semi_often', len(content_semi_often), plt_file_acc(content_semi_often))
    print('content_often', len(content_often), plt_file_acc(content_often))

def load_txt_result(path):
    #([w1,w2,gold_lbl,predicted_lbl,str(cnt1), str(cnt2),category,str(similarity)])
    with open(path) as f_in:
        content = [line.strip().split('\t') for line in f_in.readlines()]

    for c in content:
        c[4] = int(c[4])
        c[5] = int(c[5])
        c[-1] = float(c[-1])
        if c[2] != 'contradiction':
            print('###', c[2])
            1/0

    return content


def plot_multi_bar_chart(data, legend_labels, colors, width=0.2, rotate=0, ncol=3, xlabel='Accuracy (%)', ylabel='Occurences of most frequent word of word-pair'):
    '''
    Plot a bar chart with several bars per x value

    :param data     data to plot: [(label_x, [v1, v2, ...]), (...)]
    :param x_labels   x_labels,
    :param title   title
    '''
    x_labels = [lbl for lbl, _ in data]
    data = [vals for _, vals in data]

    plot_data = [[] for i in range(len(data[0]))]
    for d in data:
        for i in range(len(d)):
            plot_data[i].append(d[i])

    print('## plotdata', plot_data)


    num_groups = len(x_labels)

    fig, ax = plt.subplots()
    index = np.arange(num_groups)
    bar_width = width

    for i, lbl in enumerate(legend_labels):
        if ncol == 3:
            plt.bar(index + i * bar_width, plot_data[i], bar_width,  label=lbl, color=colors[i])
        else:
            plt.bar(index + i * bar_width, plot_data[i], bar_width,  label=lbl, color=colors[i])

    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    #plt.title(title)
    if ncol == 3:
        plt.xticks(index + (i-1) * bar_width , x_labels, rotation=rotate)
    else:
        plt.xticks(index + i * bar_width/2, x_labels, rotation=rotate)
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=ncol)
    plt.subplots_adjust(top=0.8)


    plt.show()

def plot_freq_acc_file(file_path):
    LABEL = 0
    SNLI_FREQ = 1
    SNLI_ACC = 2
    SNLI_MNLI_FREQ = 3
    SNLI_MNLI_ACC = 4

    with open(file_path) as f_in:
        content = [line.strip().split() for line in f_in.readlines()][1:]

    for i in range(len(content)):
        content[i][SNLI_FREQ] = int(content[i][SNLI_FREQ])
        content[i][SNLI_MNLI_FREQ] = int(content[i][SNLI_MNLI_FREQ])
        content[i][SNLI_ACC] = float(content[i][SNLI_ACC]) * 100
        content[i][SNLI_MNLI_ACC] = float(content[i][SNLI_MNLI_ACC]) * 100

    x_labels = [c[LABEL] for c in content]
    num_groups = len(x_labels)
    ncol = 2


    # plot accuracy per category
    data_acc = [[content[i][SNLI_ACC], content[i][SNLI_MNLI_ACC]] for i in range(len(content))]
    print('acc data', data_acc)

    plot_data_acc = [[] for i in range(len(data_acc[0]))]
    for d in data_acc:
        for i in range(len(d)):
            plot_data_acc[i].append(d[i])

    plt.subplot(2, 1, 1)
    index = np.arange(num_groups)
    #bar_width = 1/3
    rotate = 0
    legend_labels = ['Trained on SNLI', 'Trained on SNLI+MultiNLI']
    colors = ['#d95f0e', '#2c7fb8']

    for i, lbl in enumerate(legend_labels):
        plt.plot(index, plot_data_acc[i], '-o',  label=lbl, color=colors[i])
    plt.xticks(index, x_labels, rotation=rotate)
    plt.ylabel('Accuracy (%)')
    plt.xlabel(' ')

    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=ncol)

    ## Plot the amount of samples covered
    data_freq = [[content[i][SNLI_FREQ], content[i][SNLI_MNLI_FREQ]] for i in range(len(content))]
    print('acc data', data_freq)

    plot_data_freq = [[] for i in range(len(data_freq[0]))]
    for d in data_freq:
        for i in range(len(d)):
            plot_data_freq[i].append(d[i])


    plt.subplot(2, 1, 2)
    for i, lbl in enumerate(legend_labels):
        plt.plot(index, plot_data_freq[i], '-o',  label=lbl, color=colors[i])
    plt.xticks(index, x_labels, rotation=rotate)
    plt.ylabel('Amount of samples')
    plt.xlabel('Amount of contradiction samples in train data containing a word-pair')
    #plt.subplots_adjust(top=-0.5, bottom=-0.6)

    plt.show()


def plot_freq_acc_file_old(file_path):
    LABEL = 0
    SNLI_FREQ = 1
    SNLI_ACC = 2
    SNLI_MNLI_FREQ = 3
    SNLI_MNLI_ACC = 4

    with open(file_path) as f_in:
        content = [line.strip().split() for line in f_in.readlines()][1:]

    for i in range(len(content)):
        content[i][SNLI_FREQ] = int(content[i][SNLI_FREQ])
        content[i][SNLI_MNLI_FREQ] = int(content[i][SNLI_MNLI_FREQ])
        content[i][SNLI_ACC] = float(content[i][SNLI_ACC]) * 100
        content[i][SNLI_MNLI_ACC] = float(content[i][SNLI_MNLI_ACC]) * 100




    x_labels = [c[LABEL] for c in content]
    data = []
    for i in range(len(x_labels)):
        data.append([content[i][val_idx] for val_idx in [SNLI_ACC, SNLI_FREQ, SNLI_MNLI_ACC, SNLI_MNLI_FREQ]])

    data_freq = [d[:] for d in data[:]]
    for i in range(len(data_freq)):
        data_freq[i][0] = 0
        data_freq[i][2] = 0

    data_acc = [d[:] for d in data[:]]
    for i in range(len(data_acc)):
        data_acc[i][1] = 0.0
        data_acc[i][3] = 0.0

    print('acc data', data_acc)
    print('freq_data', data_freq)

    plot_data_freq = [[] for i in range(len(data_freq[0]))]
    for d in data_freq:
        for i in range(len(d)):
            plot_data_freq[i].append(d[i])

    plot_data_acc = [[] for i in range(len(data_acc[0]))]
    for d in data_acc:
        for i in range(len(d)):
            plot_data_acc[i].append(d[i])

    print('## plotdata freq', plot_data_freq)


    num_groups = len(x_labels)

    fig, ax1 = plt.subplots()
    index = np.arange(num_groups)
    bar_width = 1/(num_groups + 1)
    rotate = 0
    legend_labels = ['Accuracy (SNLI)', 'Amount of samples (SNLI)', 'Accuracy (SNLI + MultiNLI)', 'Amount of samples (SNLI + MultiNLI)']
    colors = ['#d95f0e', '#fe9929', '#2c7fb8', '#41b6c4']

    ncol = 2
    for i, lbl in enumerate(legend_labels):
        ax1.bar(index + i * (bar_width + .02), plot_data_acc[i], bar_width,  label=lbl, color=colors[i])

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Amount of contradiction samples in train data with the replaced words')

    ax2 = ax1.twinx()
    for i, lbl in enumerate(legend_labels):
        ax2.bar(index + i * (bar_width+.02), plot_data_freq[i], bar_width,  label=lbl, color=colors[i])
    ax2.set_ylabel('Amount of samples in test data')
    #plt.title(title)
    #if ncol == 3:
    #    plt.xticks(index + (i-1) * bar_width , x_labels, rotation=rotate)
    #else:
    plt.xticks(index + i * (bar_width/2+0.01), x_labels, rotation=rotate)
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=ncol)
    plt.subplots_adjust(top=0.8)

    plt.show()

def plot_freq_acc(esim_file, residual_file, decomposable_file):
    
    esim_content = load_txt_result(esim_file)
    residual_content = load_txt_result(residual_file)
    decomposable_content = load_txt_result(decomposable_file)

    print('# esim')
    bin_sizes, result_esim = freq_bins(esim_content)
    #asym_freq_bins(esim_content)
    print()

    print('# residual')
    bin_sizes, result_res = freq_bins(residual_content)
    #asym_freq_bins(residual_content)
    print()

    print('# decomposable')
    bin_sizes, result_decomp = freq_bins(decomposable_content)
    #asym_freq_bins(decomposable_content)
    print()

    PLOT_ALL = True

    x_labels = [str(bin_sizes[i]) for i in range(len(bin_sizes) - 1)]
    x_labels.append(str(bin_sizes[-2]) + '+')
    if PLOT_ALL:
        data = [(x_labels[i], (result_esim[i] * 100, result_res[i] * 100, result_decomp[i] * 100)) for i in range(len(x_labels))]
        legend_labels = ['ESIM', 'Residual Encoder', 'Decomposable Attention']
        colors = [COLOR_ESIM, COLOR_RESIDUAL, COLOR_DECOMPOSABLE]
        ncol=3

    else:
        data = [(x_labels[i], (result_esim[i] * 100, result_res[i] * 100)) for i in range(len(x_labels))]
        legend_labels = ['ESIM', 'Residual Encoder']
        colors = [COLOR_ESIM, COLOR_RESIDUAL]
        ncol=2


    plot_multi_bar_chart(data, legend_labels, colors=colors, ncol=ncol)

def plot_cos_evalshit(cos_file, bin_size = 0.05):

    with open(cos_file) as f_in:
        content = [line.strip().split('\t') for line in f_in.readlines()]

    for i in range(len(content)):
        content[i][-1] = float(content[i][-1])
        content[i][4] = int(content[i][4])
        content[i][5] = int(content[i][5])

    SPLIT1 = 400
    SPLIT2 = 1500

    content_rare = []
    content_semi_rare = []
    content_semi_often = []
    content_often = []


    for c in content:
        count1 = c[4]
        count2 = c[5]

        if count1 <= SPLIT1 and count2 <= SPLIT1:
            content_rare.append(c)
        elif count1 <= SPLIT2 and count2 <= SPLIT1:
            content_semi_rare.append(c)
        elif count1 <= SPLIT1 and count2 <= SPLIT2:
            content_semi_rare.append(c)
        elif count1 <= SPLIT2 or count2 <= SPLIT2:
            content_semi_often.append(c)
        else:
            content_often.append(c)

    print('content_rare', len(content_rare), plt_file_acc(content_rare))
    print('content_semi_rare', len(content_semi_rare), plt_file_acc(content_semi_rare))
    print('content_semi_often', len(content_semi_often), plt_file_acc(content_semi_often))
    print('content_often', len(content_often), plt_file_acc(content_often))

    print('content_rare')
    create_bins(content_rare, bin_size=0.2)

    print('content_semi_rare')
    create_bins(content_semi_rare, bin_size=0.2)

    print('content_semi_often')
    create_bins(content_semi_often, bin_size=0.2)

    print('content_often')
    create_bins(content_often, bin_size=0.2)

    print('all')
    create_bins(content, bin_size=0.05)

    ##
    print('#### test2')
    max_vals = [500,1500,4500,10000,50000]

    content1 = []
    content2 = []
    content3 = []
    content4 = []
    content5 = []
    content6 = []
    for c in content:
        max_freq = max([c[4], c[5]])
        if max_freq <= 500:
            content1.append(c)
        elif max_freq <= 1500:
            content2.append(c)
        elif max_freq <= 4500:
            content3.append(c)
        elif max_freq <= 10000:
            content4.append(c)
        elif max_freq <= 50000:
            content5.append(c)
        else:
            content6.append(c)

    print('max-freq 500', len(content1), plt_file_acc(content1))
    print('max-freq 1500', len(content2), plt_file_acc(content2))
    print('max-freq 4500', len(content3), plt_file_acc(content3))
    print('max-freq 10000', len(content4), plt_file_acc(content4))
    print('max-freq 50000', len(content5), plt_file_acc(content5))
    print('min-freq 50000+', len(content6), plt_file_acc(content6))



    #x_labels = [x for x,y in data]
    #y_vals = [y for x,y in data]
    #x_indizes = np.arange(len(data))
    #width = 0.35
    #color = [color_palette[i] for i in range(len(data))]

    #plt.bar(x_indizes, y_vals, width, color=color)
    #plt.ylabel(y_axis_name)
    #plt.xlabel(x_axis_name)
    #plt.xticks(x_indizes, x_labels)
    #plt.title(title)

    #plt.show()


def print_stats(result_path):
    results = load_dataset(result_path)
    set_phrases = set()

    cnt_samples = 0
    cnt_samples_contradiction = 0
    for pd in results:
        added  = False
        if len(pd['replaced1'].split()) > 1:
            set_phrases.add(pd['replaced1'])
            added = True
        if len(pd['replaced2'].split()) > 1:
            set_phrases.add(pd['replaced2'])
            added = True

        if added:
            cnt_samples +=1 
            if pd['gold_label'] == 'contradiction':
                cnt_samples_contradiction += 1

    print('phrases:', set_phrases)
    print('Affected samples:', cnt_samples)
    print('Affected contradiction:', cnt_samples_contradiction)



def create_residual_analyse_file(result_file, dataset_file, original_dataset_file, wordcount_file, out_file):
    with open(result_file) as f_in:
        plain_results = [line.strip().split('\t') for line in f_in.readlines()]

    print(collections.Counter([v[-1] for v in plain_results]).most_common())

    dataset = load_dataset(dataset_file)
    original_dataset = load_dataset(original_dataset_file)
    original_dict = dict([(str(pd['id']), pd) for pd in original_dataset])
    dataset_dict = dict([(str(pd['pairID']), pd) for pd in dataset])
    wordcount = torch.load(wordcount_file)

    results = []
    for _id, predicted, category in plain_results:
        _id = str(_id[1:])
        orig_sample = original_dict[_id]
        data_sample = dataset_dict[_id]
        if data_sample['category'] != category:
            print('Someethinhg is wrong!', orig_sample['category'], category, _id)
            1/0
        data_sample['predicted_label'] = predicted
        data_sample['replaced1'] = orig_sample['replaced1']
        data_sample['replaced2'] = orig_sample['replaced2']
        data_sample['count1'] = cnt_word_or_phrase(wordcount, orig_sample['replaced1'])
        data_sample['count2'] = cnt_word_or_phrase(wordcount, orig_sample['replaced2'])

        results.append(data_sample)

    print('Write out:', len(results))
    with open(out_file, 'w') as f_out:
        for pd in results:
            f_out.write(json.dumps(pd) + '\n')


def create_esim_analyse_file(result_file, dataset_file, original_dataset_file, wordcount_file, out_file):
    
    dic = ['entailment','neutral','contradiction']

    dataset = load_dataset(dataset_file)
    original_dataset = load_dataset(original_dataset_file)
    original_dict = dict([(pd['id'], pd) for pd in original_dataset])
    wordcount = torch.load(wordcount_file)

    results = []
    with open(result_file) as f_in:
        plain_results = [line.strip().split('\t') for line in f_in.readlines()]

    plain_results_dict = collections.defaultdict(lambda: dict())
    for pr in plain_results:
        premise = pr[0]
        hyp = pr[1]
        gold = dic[int(pr[2])]
        predicted = dic[int(pr[3])]


        plain_results_dict[premise][hyp] = (predicted, gold)

    out_set = []
    for pd in dataset:
        premise = ' '.join(data_tools._tokenize(pd['sentence1']))
        hypothesis = ' '.join(data_tools._tokenize(pd['sentence2']))

        predicted, gold = plain_results_dict[premise][hypothesis]
        if gold != pd['gold_label']:
            print('Somthing is wrong...')
            print(premise)
            print(hypothesis)
            print('gold:', gold)
            print('predicted:', predicted)
            1/0
        pd['predicted_label'] = predicted

        orgininal_sample = original_dict[pd['pairID']]
        pd['replaced1'] = orgininal_sample['replaced1']
        pd['replaced2'] = orgininal_sample['replaced2']
        pd['count1'] = cnt_word_or_phrase(wordcount, orgininal_sample['replaced1'])
        pd['count2'] = cnt_word_or_phrase(wordcount, orgininal_sample['replaced2'])
        out_set.append(pd)

    print('write out', len(out_set), 'samples')
    with open(out_file, 'w') as f_out:
        for pd in out_set:
            f_out.write(json.dumps(pd) + '\n')



def create_decomposition_analyse_file(result_file, dataset_file, original_dataset_file, wordcount_file, out_file):
    dataset = load_dataset(dataset_file)
    original_dataset = load_dataset(original_dataset_file)
    original_dict = dict([(pd['id'], pd) for pd in original_dataset])
    wordcount = torch.load(wordcount_file)

    results = []
    with open(result_file) as f_in:
        plain_results = [line.strip().split('\t') for line in f_in.readlines()]

    plain_results_dict = collections.defaultdict(lambda: dict())
    for pr in plain_results:
        premise = pr[0]
        hyp = pr[1]
        gold = pr[2]
        predicted = pr[3]


        plain_results_dict[premise][hyp] = (predicted, gold)

    out_set = []
    for pd in dataset:
        premise = pd['sentence1']
        hypothesis = pd['sentence2']

        predicted, gold = plain_results_dict[premise][hypothesis]
        if gold != pd['gold_label']:
            print('Somthing is wrong...')
            print(premise)
            print(hypothesis)
            print('gold:', gold)
            print('predicted:', predicted)
            1/0
        pd['predicted_label'] = predicted

        orgininal_sample = original_dict[pd['pairID']]
        pd['replaced1'] = orgininal_sample['replaced1']
        pd['replaced2'] = orgininal_sample['replaced2']
        pd['count1'] = cnt_word_or_phrase(wordcount, orgininal_sample['replaced1'].lower())
        pd['count2'] = cnt_word_or_phrase(wordcount, orgininal_sample['replaced2'].lower())
        out_set.append(pd)

    print('write out', len(out_set), 'samples')
    with open(out_file, 'w') as f_out:
        for pd in out_set:
            f_out.write(json.dumps(pd) + '\n')

def create_counts(dataset, out):
    word_count = collections.defaultdict(int)

    with open(dataset) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    for pd in parsed:
        if pd['gold_label'] != '-':
            tokenized_premise = data_tools._tokenize(pd['sentence1'])
            tokenized_hyp = data_tools._tokenize(pd['sentence2'])

            for sentence in [tokenized_hyp, tokenized_premise]:
                for word in sentence:
                    word_count[word] += 1

    torch.save(word_count, out)

    # test
    loaded = torch.load(out)
    print(loaded['a'])


def create_counts_lower(dataset_path, out_path):
    word_count = collections.defaultdict(int)
    dataset = load_dataset(dataset_path)

    for pd in dataset:
        if pd['gold_label'] != '-':
            tokenized_premise = data_tools._tokenize(pd['sentence1'])
            tokenized_hyp = data_tools._tokenize(pd['sentence2'])

            for sentence in [tokenized_hyp, tokenized_premise]:
                for word in sentence:
                    word_count[word.lower()] += 1

    torch.save(word_count, out_path)

    # test
    loaded = torch.load(out_path)
    print(loaded['a'])




if __name__ == '__main__':
    main()
