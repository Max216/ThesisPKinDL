import spacy
nlp = spacy.load('en')

import sys, json
sys.path.append('./../')
from libs import data_tools

from docopt import docopt

import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))

def main():

    USE_TREE = True

    args = docopt("""wordnet tools

    Usage:
        wn_tools.py wsd <data_path> <out_path>
    """)

    if args['wsd']:
        wsd_snli(args['<data_path>'], args['<out_path>'])

def spacy_lesk(sent, w):
    if w in sw:
        return '-'
    result = lesk(sent, w)
    if result is None:
        result = '-'
    else:
        result = result.name()
    return result

def simple_lesk(sent_nltk, sent_spacy, w):
    pass

def wsd_snli(data_path, out_path):
    with open(data_path) as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    wsd_data = []
    valid_labels = set(['neutral', 'entailment', 'contradiction'])
    for d in data:
        if d['gold_label'] in valid_labels:
            p_tokenized = data_tools._tokenize_spacy(d['sentence1'])
            #p_nltk = data_tools._tokenize_nltk(d['sentence1'])
            h_tokenized = data_tools._tokenize_spacy(d['sentence2'])
            #h_nltk = data_tools._tokenize_nltk(d['sentence2'])

            #synsets_p = [simple_lesk(p_nltk, p_spacy, w for w in p_nltk)]
            #synsets_h = [simple_lesk(h_nltk, h_spacy, w for w in h_nltk)]

            synsets_p = [spacy_lesk(p_tokenized, w) for w in p_tokenized]
            synsets_h = [spacy_lesk(h_tokenized, w) for w in h_tokenized]

            print(p_tokenized)
            print(synsets_p)




if __name__ == '__main__':
    main()