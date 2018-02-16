'''Evaluate a model'''


import sys, os, collections, json
import torch
sys.path.append('./../')

from libs import data_tools

from docopt import docopt

import numpy as np
from numpy import dot
from numpy.linalg import norm


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
        submission_alayse.py plot_cos <results> <embeddings>
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
    elif args['plot_cos']:
        plot_cosine_similarity(args['<results>'], args['<embeddings>'])

def load_dataset(path):
    with open(path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]
    return parsed

def load_embeddings(embedding_path):
    with open(embedding_path) as f_in:
        lines = [line.strip() for line in f_in.readlines()]

    lines = [line.split(' ', 1) for line in lines]
    return dict([line[0], np.asarray([float(v) for v in line[1].split()])])

def get_embedding(embeddings, words):
    splitted = word.split()
    if len(splitted) == 1:
        return embeddings[word]
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
    return dot(a, b)/(norm(a)*norm(b))

def word_count(wordcount_file, word):
    wc = torch.load(wordcount_file)
    print(word, wc[word])

def plot_cosine_similarity(result_path, embeddings_path):
    results = load_dataset(result_path)
    embeddings = load_embeddings(embeddings_path)
    results = [r for r in results if r['gold_label'] == 'contradiction']
    
    final_values = []
    for sample in results:
        embd1 = get_embedding(embeddings, sample['replaced1'])
        embd2 = get_embedding(embeddings, sample['replaced2'])
        if len(embd2) != 300:
            print('oh no!', len(embd2))
            1/0
        similarity = cos_sim(embd1, embd2)
        final_values.append((sample['replaced1'], sample['replaced2'], sample['gold_label'], sample['predicted_label'], sample['category'], similarity))

    all_similarities = sorted([fv[-1] for fv in final_values])
    print(all_similarities)


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
        predicted = [3]


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
