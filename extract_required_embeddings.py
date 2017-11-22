from docopt import docopt
import nltk
from nltk import word_tokenize

import mydataloader


def extract_vocab(dataset_path):
    '''
    Extract a set of all words occuring in the given dataset.
    '''

    dataset = mydataloader.load_snli(dataset_path)
    p_h_combined = [(p+h) for p,h,lbl in dataset]
    return set([w for p_h in p_h_combined for w in p_h])


def main():
    args = docopt("""Extracts all embeddings that are required for the dataset to reduce the
        size of the embeddings.

    Usage:
        analyse.py <embeddings> <data_train> <data_dev> <name_out>

        <embeddings>         Path to all embeddings.
        <data_train>         Path to train set.
        <data_dev>           Path to dev set.
        <name_out>           Name of the new embedding file.
    """)

    embedding_path = args['<embeddings>']
    data_train_path = args['<data_train>']
    data_dev_path = args['<data_dev>']
    name_out = args['<name_out>']

    # Vocabulary
    voc_train = extract_vocab(data_train_path)
    print('vocab train', len(voc_train))
    voc_dev = extract_vocab(data_dev_path)
    print('vocab dev', len(voc_dev))

    voc = voc_train | voc_dev

    print('Total vocabulary in data:', len(voc))

    # embeddings
    with open(embedding_path) as f:
        used_word_embeddings = [line for line in f if line.split(' ',2)[0] in voc]

    print('word_embeddings:', len(used_word_embeddings))

    # write to file
    with open(name_out, 'w') as f_out:
        for line in used_word_embeddings:
            f_out.write(line)
    

if __name__ == '__main__':
    main()


