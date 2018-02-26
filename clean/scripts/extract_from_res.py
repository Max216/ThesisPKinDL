from docopt import docopt

def main():

    args = docopt("""Extract word relations from datasets. 

    Usage:
        extract_from_res.py bless <vocab_file> <data_file> <out_file> 
    """)

    vocab_file = args['<vocab_file>']
    data_file = args['<data_file>']
    out_file = args['<out_file>']

    if args['bless']:
        extract_bless(vocab_file, data_file, out_file)


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


if __name__ == '__main__':
    main()
