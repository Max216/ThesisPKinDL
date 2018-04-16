import sys, os
sys.path.append('./../')
import json, collections, random

def main():
    args = docopt("""Sample dataset.

    Usage:
        sampling.py samplesame <data> <amount>
    """)

    if args['samplesame']:
        sample_same_premise(args['<data>'], int(args['<amount>']))


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

