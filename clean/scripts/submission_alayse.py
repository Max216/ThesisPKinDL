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
        submission_alayse.py create_res_anl <esim_results> <dataset> <original_dataset> <wordcount> <out>
    """)


    if args['create_counts']:
        create_counts(args['<data_in>'], args['<file_out>'])
    elif args['create_esim_anl']:
        create_esim_analyse_file(args['<esim_results>'], args['<dataset>'], args['<original_dataset>'], args['<wordcount>'], args['<out>'])
    elif args['create_res_anl']:
        create_residual_analyse_file(args['<esim_results>'], args['<dataset>'], args['<original_dataset>'], args['<wordcount>'], args['<out>'])

def load_dataset(path):
    with open(path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]
    return parsed

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
        if orig_sample['category'] != category:
            print('Someethinhg is wrong!', orig_sample['category'], category, _id)
            1/0
        data_sample['predicted_label'] = predicted
        data_sample['replaced1'] = orig_sample['replaced1']
        data_sample['replaced2'] = orig_sample['replaced2']
        data_sample['count1'] = wordcount[orig_sample['replaced1']]
        data_sample['count2'] = wordcount[orig_sample['replaced2']]

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
        pd['count1'] = wordcount[orgininal_sample['replaced1']]
        pd['count2'] = wordcount[orgininal_sample['replaced2']]
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
