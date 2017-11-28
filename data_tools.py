from docopt import docopt
import nltk
from nltk import word_tokenize

import train

import mydataloader


def extract_vocab(dataset_path):
    '''
    Extract a set of all words occuring in the given dataset.
    '''

    dataset = mydataloader.load_snli(dataset_path)
    p_h_combined = [(p+h) for p,h,lbl in dataset]
    return set([w for p_h in p_h_combined for w in p_h])


def req_embeddings(args):
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

def unique_sents(model_path, data_path, amount, name_out):

    # Load model
    print('Load model ...')
    #model, _ = train.load_trained_model(model_path)
    print('Done.')

    # Load data
    print('Load data ...')
    data = mydataloader.load_snli(data_path)
    print('Done.')

    # map premise with all hypothesis
    print('Mapping premise with hypothesis ...')
    hash_to_sent = dict()
    premise_to_hypothesis = dict()
    for p, h, _ in data:
        p_key = hash('_'.join(p))
        h_key = hash('_'.join(h))

        if p_key not in hash_to_sent:
            hash_to_sent[p_key] = p
            premise_to_hypothesis[p_key] = []

        # else ignore

        if h_key not in hash_to_sent:
            if p_key in premise_to_hypothesis:
                hash_to_sent[h_key] = h
                premise_to_hypothesis[p_key].append(h_key)
            # else ignore

    for key in premise_to_hypothesis:
        print('test', key, ' -> ', premise_to_hypothesis[key])

    # only use useful ones (model predict)




    # Check if all premise-hypothesis are correct, only then use them

    # write into new file:
    # w1 w2 w3 . . .
    # repr[0] repr[1] . . . 
    # w_idx . . .
    # . . .

    # verify that no duplicates (cross: premise-hypothesis)

    # print key stats over the sentences

def main():
    args = docopt("""Work with data.

    Usage:
        data_tools.py req_embeddings <embeddings> <data_train> <data_dev> <name_out>
        data_tools.py unique_sents <model_path> <data> <amount> <name_out>

        <embeddings>         Path to all embeddings.
        <data_train>         Path to train set.
        <data_dev>           Path to dev set.
        <name_out>           Name of the new embedding file.
    """)

    if args['req_embeddings']:
        req_embeddings(args)
    elif args['unique_sents']:
        unique_sents(args['<model_path>'], args['<data>'], int(args['<amount>']), args['<name_out>'])
    

if __name__ == '__main__':
    main()


