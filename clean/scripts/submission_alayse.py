'''Evaluate a model'''


import sys, os, collections, json
import torch
sys.path.append('./../')

from libs import data_tools

from docopt import docopt

def main():
    args = docopt("""For submission

    Usage:
        submission_alayse.py create_counts <data_in> <file_out>
        submission_alayse.py create_esim_anl <esim_results> <dataset> <original_dataset> <wordcount> <out>
    """)


    if args['create_counts']:
        create_counts(args['<data_in>'], args['<file_out>'])
    elif args['create_esim_anl']:
        create_esim_analyse_file(args['<esim_results>'], args['<dataset>'], args['<original_dataset>'], args['<wordcount>'], args['<out>'])

def load_dataset(path):
    with open(path) as f_in:
        parsed = [json.loads(line.strip() for line in f_in.readlines())]
    return parsed

def create_esim_analyse_file(result_file, dataset_file, original_dataset_file, wordcount_file, out_file):
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
        predicted = pr[2]
        gold = pr[3]

        plain_results_dict[premise][hyp] = (predicted, gold)

    out_set = []
    for pd in dataset:
        premise = data_tools._tokenize(pd['sentence1'])
        hypothesis = data_tools._tokenize(pd['sentence2'])

        predicted, gold = plain_results_dict[premise][hypothesis]
        if gold != pd['gold_label']:
            print('Somthing is wrong...')
            1/0
        pd['predicted_label'] = predicted

        orgininal_sample = original_dict[pd['pairID']]
        pd['replaced1'] = orgininal_sample['replaced1']
        pd['replaced2'] = orgininal_sample['replaced2']
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



if __name__ == '__main__':
    main()
