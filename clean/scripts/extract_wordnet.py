import json
import spacy
import collections
import torch
nlp = spacy.load('en')

import nltk
from nltk.corpus import wordnet as wn

def main():

    USE_TREE = True

    args = docopt("""Tools to extract a subset of relevant informaation from WordNet. 

    Usage:
        extract_wordnet.py count_hyper <data> <out>
    """)

    if args['count_hyper']:
        count_hypernyms(args['<data>'], args['<out>'])


def tokenize(sent):
    doc = nlp(sent,  parse=False, tag=False, entity=False)
    return [token.text for token in doc]

def get_token_counts(data_path):
    print('Read data:', path)
    with open(data_path) as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    words = []
    for d in data:
        words.extend(tokenize(d['sentence1']))
        words.extend(tokenize(d['sentence2']))

    word_counter = collections.Counter(words)
    print('Found', len(words), 'tokens and', len(word_counter), 'distinct tokens.')

    return word_counter





def count_hypernyms(data_path, out_path):
    word_counter = get_token_counts(data_path)

    # Just top print progress
    total_words = len(word_counter)
    checked_words = 0


    # to ensure only to look at nouns
    noun_synsets = set(wn.all_synsets('n'))

    hypernym_counter = collections.defaultdict(int)

    # Go through all words
    for w in word_counter:
        checked_words += 1
        print('Analyse:', w, '(' + str(checked_words) + '/' + str(total_words))

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
                            print('##', node)
                            hypernym_counter[node] += word_counter[w]

    torch.save(hypernym_counter, out_path)







if __name__ == '__main__':
    main()

