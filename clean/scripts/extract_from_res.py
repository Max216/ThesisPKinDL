from docopt import docopt

import json, collections

def main():

    args = docopt("""Extract word relations from datasets. 

    Usage:
        extract_from_res.py bless <vocab_file> <data_file> <out_file> 
        extract_from_res.py adv <data_file> <out_file> 
    """)

    vocab_file = args['<vocab_file>']
    data_file = args['<data_file>']
    out_file = args['<out_file>']

    if args['bless']:
        extract_bless(vocab_file, data_file, out_file)
    elif args['adv']:
        extract_adv(data_file, out_file)


def load_vocab(vocab_file):
    with open(vocab_file) as f_in:
        vocab = set([line.strip().lower() for line in f_in.readlines()])

    return vocab

def extract_bless(vocab_file, data_file, out_file):
    vocab = load_vocab(vocab_file)

    irrelevant_relations = set(['random', 'mero', 'event'])

    with open(data_file) as f_in:
        data = [line.strip().split('\t') for line in f_in.readlines()]

    print('Data within BLESS:', len(data))

    # filter relations
    data = [d for d in data if d[2].lower() not in irrelevant_relations]
    print('Data after filtering out', irrelevant_relations, ':', len(data))

    # filter by vocab
    data = [d for d in data if d[0] in vocab and d[1] in vocab]
    print('Data after filtering by vocab:', len(data))

    print('remaining labels:', set([d[2].lower() for d in data]))


def extract_adv(data_file, out_file):
    with open(data_file) as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    wp_dict = collections.defaultdict(list)
    for d in data:
        w1 = d['replaced1']
        w2 = d['replaced2']

        if len(w1.split(' ')) == 1 and len(w2.split(' ')) == 1:
            wp_dict[(w1, w2)].append(d['gold_label'])


    final_data = []
    for (w1, w2) in wp_dict:
        lbl, cnt = collections.Counter(wp_dict[(w1, w2)]).most_common()[0]
        final_data.append([w1, w2, lbl])

    with open(out_file, 'w') as f_out:
        for fd in final_data:
            f_out.write('\t'.join(fd) + '\n')

if __name__ == '__main__':
    main()
