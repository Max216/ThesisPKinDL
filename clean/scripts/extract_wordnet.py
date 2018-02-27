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
        extract_wordnet.py words <word_data> <synset>
        extract_wordnet.py create <count_data> <vocab> <out>
    """)

    if args['count_hyper']:
        count_hypernyms(args['<data>'], args['<out_counts>'], args['<out_words>'])
    elif args['show_hyper_count']:
        show_hypernym_count(args['<data>'], int(args['<amount>']))
    elif args['words']:
        show_words(args['<word_data>'], args['<synset>'])
    elif args['create']:
        create_data(args['<count_data>'], args['<vocab>'], args['<out>'])


def tokenize(sent):
    doc = nlp(sent,  parse=False, tag=False, entity=False)
    return [token.text for token in doc]

def show_words(word_data, synset):
    words = torch.load(word_data)
    print(words[synset])

def create_data(count_path, vocab_path, out_path):

    # use maximum this amount of synsets
    MAX_AMOUNT_SYNSETS = 1

    # look for this distant hypernyms/hyponyms
    SEARCH_DEPTH = 1

    # helper functions
    hyper = lambda s: s.hypernyms()
    hypo = lambda s: s.hyponyms()

    count_data = torch.load(count_path)

    with open(vocab_path) as f_in:
        vocab = [line.strip() for line in f_in.readlines()]

    result = []

    for w in vocab:
        # find synsets
        synsets = wn.synsets(w, pos=wn.NOUN)

        # Map them with direct hypernyms
        synset_pairs = [(syns, list(syns.closure(hyper, depth=SEARCH_DEPTH))) for syns in synsets]

        # score and divide them
        synset_with_hypernym = sorted([(syns, hyper, count_data.get(hyper.name(), 0)) for syns, hypernyms in synset_pairs for hyper in hypernyms], key=lambda x: -x[-1])

        # select best synsets and hypernyms
        selected_synsets = set()
        selected_hypernyms = set()
        selected_relations = []
        for i in range(len(synset_with_hypernym)):
            syns, hypern, score = synset_with_hypernym[i]
            selected_synsets.add(syns)
            selected_hypernyms.add(hypern)
            selected_relations.append((syns, hypern))

            if max(len(selected_synsets), len(selected_hypernyms)) >=MAX_AMOUNT_SYNSETS:
                break


        # Get synonyms
        synonyms = []
        for syns in selected_synsets:
            syns_in_vocab = set(syns.lemma_names()).intersection(vocab)
            for s1 in syns_in_vocab:
                for s2 in syns_in_vocab:
                    if s1 != s2:
                        synonyms.append((s1, s2, 'synonym'))
        result.extend(list(set(synonyms)))

        # Get Antonyms
        antonyms = []
        for syns in selected_synsets:
            lemmas = syns.lemmas()
            lemmas = [lemma for lemma in lemmas if lemma.name() in vocab]

            for lemma  in lemmas:
                antonym_names = [anto.name() for anto in lemma.antonyms()]
                antonym_names = [anto for anto in antonym_names if anto in vocab]
                lemma_names = [lemma.name() for lemma in lemmas]

                for w1 in lemma_names:
                    for w2 in antonym_names:
                        antonyms.append((w1, w2, 'antonym'))

        result.extend(list(set(antonyms)))

        # Get Cohyponyms, hypernyms
        hypernyms = []
        cohyponyms = []
        for syns, hypern in selected_relations:
            
            # first hypernyms
            hypernym_in_vocab = list(set(hypern.lemma_names()).intersection(vocab))
            syns_in_vocab = set(syns.lemma_names()).intersection(vocab)
            for w1 in syns_in_vocab:
                for w2 in hypernym_in_vocab:
                    hypernyms.append((w1, w2, 'hypernym'))

            # Then co-hyponyms
            hyponyms = hypern.closure(hypo, depth=SEARCH_DEPTH)
            hypernyms = [h for h in hypern if h.name() != syns.name()]

            hypernym_names = [n for n in h.lemma_names() for h in hypernyms]
            hypernym_names = list(set(hypernym_names).intersection(vocab))

            for w1 in syns_in_vocab:
                for w2 in hypernym_names:
                    cohyponyms.append((w1, w2, 'cohyponym'))

        result.extend(list(set(hypernyms)))
        result.extend(list(set(cohyponyms)))


    with open(out_path, 'w') as f_out:
        for w1, w2, relation in result:
            f_out.write('\t'.join([w1, w2, relation]) + '\n')




def get_token_counts(data_path):
    print('Read data:', data_path)
    with open(data_path) as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    words = []
    for d in data:
        words.extend(tokenize(d['sentence1']))
        words.extend(tokenize(d['sentence2']))

    print('Filter out stopwords')
    words = [w for w in words if w not in spacy.en.language_data.STOP_WORDS and w.lower() not in spacy.en.language_data.STOP_WORDS]

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

