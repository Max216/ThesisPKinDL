'''Evaluate a model'''


import sys, os, collections, json
import torch
sys.path.append('./../')

from libs import data_tools

from docopt import docopt

import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy import spatial

import matplotlib.pyplot as plt



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
    """)


    if args['create_counts']:
        create_counts(args['<data_in>'], args['<file_out>'])
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

def load_dataset(path):
    with open(path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]
    return parsed

def load_embeddings(embedding_path):
    with open(embedding_path) as f_in:
        lines = [line.strip() for line in f_in.readlines()]

    lines = [line.split(' ', 1) for line in lines]
    return dict([(line[0], np.asarray([float(v) for v in line[1].split()])) for line in lines])

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
    return 1 - spatial.distance.cosine(a, b)
    #return dot(a, b)/(norm(a)*norm(b))

def word_count(wordcount_file, word):
    wc = torch.load(wordcount_file)
    print(word, wc[word])

def create_cosine_similarity(result_path, embeddings_path, path_out, lower=False):
    results = load_dataset(result_path)
    embeddings = load_embeddings(embeddings_path)
    results = [r for r in results if r['gold_label'] == 'contradiction']
    
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
        for w1, w2, gold_lbl, predicted_lbl, cnt1, cnt2, category, similarity in final_values:
            f_out.write('\t'.join([w1,w2,gold_lbl,predicted_lbl,str(cnt1), str(cnt2),category,str(similarity)]) + '\n')


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
                bins.append(sorted_content[:i])
                sorted_content = sorted_content[i:]
                max_bound += bin_size
                break

            # if reached here it is over
            if i == len(sorted_content) - 1:
                bins.append(sorted_content)
                sorted_content = []


    print('eval bins:')
    max_bound = bin_size
    for b in bins:
        print('max_bound:', max_bound, '; samples:', len(b))
        if len(b) == 0:
            #print('skip!')
            pass
        else:
            correct = len([sample for sample in b if sample[3] == 'contradiction'])
            print('acc', correct / len(b))
        max_bound += bin_size

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

def plot_cos(cos_file, bin_size = 0.05):

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
