from docopt import docopt
import nltk
from nltk import word_tokenize
import torch
import torch.autograd as autograd

import train
import embeddingholder
import mydataloader
import model as m
import config

from collections import Counter
import numpy as np


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

def predict(classifier, embedding_holder, p, h):
    p_batch = torch.LongTensor(len(p), 1)
    h_batch = torch.LongTensor(len(h), 1)

    p_batch[:,0] = torch.LongTensor([embedding_holder.word_index(w) for w in p])
    h_batch[:,0] = torch.LongTensor([embedding_holder.word_index(w) for w in h])
    
    return classifier(
        m.cuda_wrap(autograd.Variable(p_batch)), 
        m.cuda_wrap(autograd.Variable(h_batch)), 
        output_sent_info=True)

def is_correct(prediction, gold):
    _, predicted_idx = torch.max(prediction, dim=1)
    return predicted_idx.data[0] == gold

def unique_sents(model_path, data_path, amount, name_out):

    # Load model
    print('Load model ...')
    classifier, _ = m.cuda_wrap(m.load_model(model_path))
    classifier.eval()
    embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
    print('Done.')

    # Load data
    print('Load data ...')
    data = mydataloader.load_snli_with_parse(data_path)
    print('Done.')

    # map premise with all hypothesis
    print('Mapping premise with hypothesis ...')
    hash_to_sent = dict()
    premise_to_hypothesis = dict()
    for p, h, lbl, parse_p, parse_h in data:
        p_key = hash('_'.join(p))
        h_key = hash('_'.join(h))

        if p_key not in hash_to_sent:
            hash_to_sent[p_key] = (p, parse_p)
            premise_to_hypothesis[p_key] = []

        # else ignore

        if h_key not in hash_to_sent:
            if p_key in premise_to_hypothesis:
                hash_to_sent[h_key] = (h, parse_h)
                premise_to_hypothesis[p_key].append((h_key, mydataloader.tag_to_index[lbl]))
            # else ignore

    # only use useful ones (model predict)
    correct_sents = []
    for key in premise_to_hypothesis:
        # must have three hypothesis
        if len(premise_to_hypothesis[key]) == 3:
            p, parse_p = hash_to_sent[key]
            all_h = [(hash_to_sent[k][0], hash_to_sent[k][1], lbl) for k, lbl in premise_to_hypothesis[key]]
            
            all_correct = True          # assume, revert if not true
            p_act = None
            p_repr = None
            h_act = []
            h_repr = []
            for h, parse_h, lbl in all_h:  
                prediction, act, repr = predict(classifier, embedding_holder, p, h)
                if is_correct(prediction, lbl):
                    p_repr = repr[0].data
                    p_act = act[0].data
                    h_repr.append(repr[1].data)
                    h_act.append(act[1].data)
                else:
                    all_correct = False
                    break


            # must all be correct
            if all_correct:
                correct_sents.append((p, parse_p, p_act, p_repr))
                correct_sents.extend([(all_h[i][0], all_h[i][1], h_act[i], h_repr[i]) for i in range(3)])

    print('Found', len(correct_sents), 'possible candidates as correct individual sentences.')
    # find most common length
    sent_lengths = [len(s[0]) for s in correct_sents]
    most_common,num_most_common = Counter(sent_lengths).most_common(1)[0]
    print('Most comment sentence length within:', most_common, '(' + str(num_most_common) + ' times).')
    correct_sents = [s for s in correct_sents if len(s[0]) == most_common]

    # random pick
    rnd_idxs = np.random.choice(len(correct_sents), amount, replace=False)

    # Assume here that there are enough sentences with that length, since so much data available
    subset = [correct_sents[i] for i in rnd_idxs]
    
    # write new file
    with open(name_out, 'w') as f_out:
        # meta info
        f_out.write('# MODEL:' + model_path + ';DATA:' + data_path+'\n')
        f_out.write('# SENTS:' + str(len(subset)) + ';LEN:' + str(most_common) + '\n')
        
        mean, sd, abs_min, abs_max = stats(np.array([r.squeeze().numpy() for s,p,a,r in subset]))
        f_out.write('# STATS: mean, standard_deviation, min, max\n')
        f_out.write(' '.join([str(v) for v in mean]) + '\n')
        f_out.write(' '.join([str(v) for v in sd]) + '\n')
        f_out.write(' '.join([str(v) for v in abs_min]) + '\n')
        f_out.write(' '.join([str(v) for v in abs_max]) + '\n')

        f_out.write('# CONTENT\n')
        for sent, parse, act, repr in subset:
            f_out.write(' '.join(sent) + '\n')
            f_out.write(parse + '\n')
            f_out.write(' '.join([str(v) for v in act.squeeze().numpy()]) + '\n')
            f_out.write(' '.join([str(v) for v in repr.squeeze().numpy()]) + '\n')


def stats(representations):
    mean = np.mean(representations, axis=0)
    sd = np.std(representations, axis=0)
    abs_min = np.amin(representations, axis=0)
    abs_max = np.amax(representations, axis=0)

    return (mean, sd, abs_min, abs_max)


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


