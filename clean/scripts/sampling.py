import sys, os
from docopt import docopt

sys.path.append('./../')
import json, collections, random

def main():
    args = docopt("""Sample dataset.

    Usage:
        sampling.py samplesame <data> <amount>
        sampling.py find <data> <sent>
        sampling.py findsize <data> <size>
        sampling.py find_label <data> <label> <amount>
    """)

    if args['samplesame']:
        sample_same_premise(args['<data>'], int(args['<amount>']))
    elif args['find']:
        find_samples_with_premise(args['<data>'], args['<sent>'])
    elif args['findsize']:
        find_premise_with_size(args['<data>'], int(args['<size>']))
    elif args['find_label']:
        find_samples_with_label(args['<data>'], args['<label>'], int(args['<amount>']))


def find_samples_with_label(data_path, label, amount):
    with open(data_path) as f_in:
        samples = [json.loads(line.strip()) for line in f_in.readlines()]

    samples = [s for s in samples if s['gold_label'] == label]
    samples = random.sample(samples, amount)

    for s in samples:
        print('[p]', s['sentence1'])
        print('[h]', s['sentence2'])
        print('[lbl]', s['gold_label'])
        print()



def find_premise_with_size(data_path, size):
    with open(data_path) as f_in:
        samples = [json.loads(line.strip()) for line in f_in.readlines()]

    print('results')
    for s in samples:
        if len(s['sentence1'].split(' ')) == size:
            print('[p]', s['sentence1'])
            print('[h]', s['sentence2'])
            print('[lbl]', s['gold_label'])
            print()

def find_samples_with_premise(data_path, sent):
    with open(data_path) as f_in:
        samples = [json.loads(line.strip()) for line in f_in.readlines()]

    print('results')
    for s in samples:
        if s['sentence1'] == sent:
            print('[p]', s['sentence1'])
            print('[h]', s['sentence2'])
            print('[lbl]', s['gold_label'])
            print()

def sample_same_premise(data_path, amount):
    with open(data_path) as f_in:
        samples = [json.loads(line.strip()) for line in f_in.readlines()]

    data = collections.defaultdict(list)
    for sample in samples:
        data[sample['sentence1']].append((sample['sentence2'], sample['gold_label']))

    premises = [p for p in data if len(data[p]) >= 3]
    selected = random.sample(premises, amount)

    for selected_p in selected:
        print('[Premise]', selected_p)
        for hyp, lbl in data[selected_p]:
            print('[Hyp]', hyp, '->', lbl)

        print('###')
        print()




if __name__ == '__main__':
    main()

