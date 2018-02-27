import json
import spacy
import collections
import torch
nlp = spacy.load('en')

from docopt import docopt

import nltk
from nltk.corpus import wordnet as wn

def main():

    USE_TREE = True

    args = docopt("""Tools to extract a subset of relevant informaation from WordNet. 

    Usage:
        extract_wordnet.py count_hyper <data> <out_counts> <out_words>
        extract_wordnet.py show_hyper_count <data> <amount>
    """)

    if args['count_hyper']:
        count_hypernyms(args['<data>'], args['<out_counts>'], args['<out_words>'])
    elif args['show_hyper_count']:
        show_hypernym_count(args['<data>'], int(args['<amount>']))


def tokenize(sent):
    doc = nlp(sent,  parse=False, tag=False, entity=False)
    return [token.text for token in doc]

def get_token_counts(data_path):
    print('Read data:', data_path)
    with open(data_path) as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    words = []
    for d in data:
        words.extend(tokenize(d['sentence1']))
        words.extend(tokenize(d['sentence2']))

    print('Filter out stopwords')
    words = [w for w in words if w not in spacy.en.language_data.STOP_WORDS]

    word_counter = collections.Counter(words)
    print('Found', len(words), 'tokens and', len(word_counter), 'distinct tokens.')

    return word_counter


def show_hypernym_count(data_path, amount):
    hyper_count = torch.load(data_path)
    counter = collections.Counter(hyper_count)
    print('Most frequent', amount, 'hypernyms:')
    print(counter.most_common()[:amount])


def count_hypernyms(data_path, out_path_counts, out_path_words):
    word_counter = get_token_counts(data_path)

    # Just top print progress
    total_words = len(word_counter)
    checked_words = 0


    # to ensure only to look at nouns
    noun_synsets = set(wn.all_synsets('n'))

    hypernym_counter = collections.defaultdict(int)
    synset_word_counter = collections.defaultdict(dict)

    # Go through all words
    for w in word_counter:
        checked_words += 1
        print('Analyse:', w, '(' + str(checked_words) + '/' + str(total_words) + ')')

        w_synsets = wn.synsets(w, pos=wn.NOUN)

        # SKIP if no synset with POS=Noun was found
        if len(w_synsets) > 0:

            # go through all paths to ROOT
            for synset in w_synsets:
                root_paths = synset.hypernym_paths()

                for root_path in root_paths:

                    # Make sure it is only nouns
                    valid = True
                    for node in root_path:
                        if node not in noun_synsets:
                            valid = False
                            break

                    if valid:
                        # add counts
                        for node in root_path:
                            hypernym_counter[node.name()] += word_counter[w]
                            synset_word_counter[node.name()][w] = word_counter[w]

    torch.save(hypernym_counter, out_path_counts)
    torch.save(synset_word_counter, out_path_words)







if __name__ == '__main__':
    main()

