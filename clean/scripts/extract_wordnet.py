import json
import spacy
import collections
import torch
import random
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
        extract_wordnet.py sample <file> <amount>
        extract_wordnet.py create_using_first <vocab> <outpath>
        extract_wordnet.py finalize_data <data_path> <out_path>
        extract_wordnet.py merge_labels1 <data_path> <out_path>
        extract_wordnet.py merge_labels <data_path> <lbl_same> <lbl_different> <out_path>
        extract_wordnet.py locations <out_path> <vocab>
    """)

    if args['count_hyper']:
        count_hypernyms(args['<data>'], args['<out_counts>'], args['<out_words>'])
    elif args['show_hyper_count']:
        show_hypernym_count(args['<data>'], int(args['<amount>']))
    elif args['words']:
        show_words(args['<word_data>'], args['<synset>'])
    elif args['create']:
        create_data(args['<count_data>'], args['<vocab>'], args['<out>'])
    elif args['sample']:
        sample(args['<file>'], int(args['<amount>']))
    elif args['create_using_first']:
        create_data_using_first_synset(args['<vocab>'], args['<outpath>'])
    elif args['finalize_data']:
        finalize_data(args['<data_path>'], args['<out_path>'])
    elif args['merge_labels1']:
        merge_labels1(args['<data_path>'], args['<out_path>'])
    elif args['merge_labels']:
        merge_labels_general(args['<data_path>'],args['<lbl_same>'], args['<lbl_different>'], args['<out_path>'])
    elif args['locations']:
        create_location_data(args['<out_path>'], args['<vocab>'])

def extract_syns_words(syns, vocab):
    return [lemma for lemma in syns.lemma_names() if len(lemma.split(' ')) == 1 and lemma in vocab]

def create_location_data(out_path, vocab_path):

    with open(vocab_path) as f_in:
        vocab = set([line.strip() for line in f_in.readlines()])

    country_syns = wn.synset('country.n.02')

    # all instance hyponyms are countries
    # TODO also lower case
    countries1 = country_syns.instance_hyponyms()
    countries1 = [(extract_syns_words(s, vocab), s) for s in countries1]
    countries1 = [(lemmas, syn) for lemmas, syn in countries1 if len(lemmas) > 0]
    print('Countries so far:', countries1)

    # other countries: hyponyms with upper case letters

    countries2_hyper = [hypo for hypo in country_syns.hyponyms()]
    countries2_hyper = [h for h in countries2_hyper if h.lemma_names()[0][0].isupper()]
    countries2_hypo = []
    for h in countries2_hyper:
        countries2_hypo.extend(h.instance_hyponyms() + h.hyponyms())

    countries2_hypo = list(set(countries2_hypo))

    countries2 = [(extract_syns_words(syn, vocab), syn) for syn in countries2_hypo]
    countries2 = [(lemmas, syn) for lemmas, syn in countries2 if len(lemmas) > 0]
    print('use countries:', countries2)







def merge_labels_general(data_path, lbl_same, lbl_different, out_path):
    with open(data_path) as f_in:
        data = [line.strip().split('\t') for line in f_in.readlines()]

    data = [(d[0], d[1], d[2]) for d in data]

    data_same = [(w1, w2, 'entailment') for w1, w2, lbl in data if lbl == lbl_same]
    data_different = [(w1, w2, 'contradiction') for w1, w2, lbl in data if lbl == lbl_different]

    with open(out_path, 'w') as f_out:
        for w1, w2, lbl in data_same + data_different:
            f_out.write('\t'.join([w1, w2, lbl]) + '\n')



def merge_labels1(data_path, out_path):
    '''
    combine labels for synonym+hypernym, antonym+cohyponym, ignore hyponym 
    '''
    with open(data_path) as f_in:
        data = [line.strip().split('\t') for line in f_in.readlines()]

    new_data = []
    for d in data:
        if d[2] == 'cohyponym' or d[2] == 'antonym':
            new_data.append((d[0], d[1], 'contradiction'))
        elif d[2] == 'synonym' or d[2] == 'hypernym':
            new_data.append((d[0], d[1], 'entailment'))
        elif d[2] == 'hyponym':
            pass
        else:
            print(d[2])
            1/0

    with open(out_path, 'w') as f_out:
        for w1,w2,lbl in new_data:
            f_out.write('\t'.join([w1, w2, lbl]) + '\n')

def finalize_data(data_path, out_path):
    with open(data_path) as f_in:
        data = [line.strip().split('\t') for line in f_in.readlines()]

    label_dict = collections.defaultdict(lambda: [])
    for d in data:
        label_dict[d[2]].append((d[0], d[1]))

    max_count = max([len(label_dict[key]) for key in label_dict])

    for key in label_dict:
        orig_data = label_dict[key][:]

        missing_samples = max_count - len(orig_data)
        while missing_samples > len(orig_data):
            missing_samples -= len(orig_data)
            label_dict[key].extend(orig_data)

        if missing_samples > 0:
            label_dict[key].extend(random.sample(orig_data, missing_samples))


    for key in label_dict:
        print(key, len(label_dict[key]))

    lines = []
    with open(out_path, 'w') as f_out:
        for key in label_dict:
            for val in label_dict[key]:
                sample = [val[0], val[1], key]
                lines.append('\t'.join(sample) + '\n')

        random.shuffle(lines)
        for line in lines:
            f_out.write(line)


def first_hypernym(syns, vocab=None, min_dist_to_top=4):
    hypernyms = syns.hypernyms() + syns.instance_hypernyms()
    if len(hypernyms) == 0:
        if vocab == None:
            return None, False
        else:
            return None, False, []
    else:
        hyper = hypernyms[0]
        dist_to_root = min([len(p) for p in hyper.hypernym_paths()])
        if dist_to_root < min_dist_to_top:
            if vocab == None:                
                return None, False 
            else:
                return None, False, []

        if vocab != None:
            found_lemmas = []
            for lemma_name in hyper.lemma_names():
                splitted = lemma_name.split(' ')
                if len(splitted) == 1:
                    lemma = lemma_name
                else:
                    lemma = splitted[-1]

                if lemma in vocab:
                    found_lemmas.append(lemma)

            if len(found_lemmas) == 0:
                return None, False, []
            else:
                return hyper, True, found_lemmas
        else:
            return hyper, True

def closest_hypernym(syns, vocab=None, min_dist_to_top=4):
    found = None
    found_bool = False
    while not found_bool:
        hypernyms = syns.hypernyms() + syns.instance_hypernyms()
        if len(hypernyms) == 0:
            return None, False
        else:
            hyper = hypernyms[0]
            dist_to_root = min([len(p) for p in hyper.hypernym_paths()])
            if dist_to_root < min_dist_to_top:
                return None, False

            if vocab != None:
                # make sure that it is in vocab
                for lemma_name in hyper.lemma_names():
                    if lemma_name in vocab:
                        found = hyper
                        found_bool = True
                        break

                if not found_bool:
                    syns = hyper



            else:
                # just use it
                found = hyper
                found_bool = True
    return found, found_bool

def get_hyponyms_excluding_syns(hypernym, syns):
    all_hyponyms = hypernym.hyponyms() + hypernym.instance_hyponyms()
    # exclude
    all_hyponyms = [hypo for hypo in all_hyponyms if hypo.name() != syns.name()]

    return all_hyponyms

def lemma_in_vocab(syns, vocab_set):
    return list(set(syns.lemma_names()) & vocab_set)

def legal_stopwords():
    return set(['whole', 'somehow', 'two', 'enough', 'always', 'never', 'eight', 'alone', 'below', 'beforehand', 'much', 'less', 'nothing', 'same', 'say', 'more', 'above', 'besides', 'against', 'least', 'name',  'most', 'well', 'anyone', 'became', 'noone', 'part','front', 'keep', 'none','see', 'first', 'perhaps', 'anywhere', 'everywhere', 'used', 'several', 'after', 'mostly', 'eleven', 'down', 'done', 'often', 'meanwhile', 'around', 'behind', 'now', 'others', 'until', 'back', 'another', 'third', 'beside', 'become', 'someone', 'hundred', 'using', 'four', 'something', 'very', 'some', 'during', 'under', 'twenty', 'three', 'anything', 'between', 'over', 'amount', 'nine', 'fifty', 'other', 'move', 'cannot', 'next', 'nowhere', 'within', 'give', 'various', 'no', 'go', 'before', 'somewhere', 'serious', 'amongst', 'both', 'last', 'together', 'without', 'once', 'one', 'few', 'himself', 'six', 'all', 'almost', 'take', 'fifteen', 'forty', 'full', 'nobody', 'beyond', 'twelve', 'each', 'show', 'former', 'sometime', 'top', 'becomes', 'across',  'five', 'every', 'out', 'everything', 'call', 'ten', 'bottom', 'many', 'sixty', 'afterwards', 'latterly', 'empty', 'everyone'])



def create_data_using_first_synset(vocab_path, out_path):

    MIN_DIST_TO_TOP_COHYPO = 3
    MIN_DIST_TO_TOP_HYPER = 5

    allowed_stopwords = legal_stopwords()

    with open(vocab_path) as f_in:
        vocab = [line.strip() for line in f_in.readlines()]
        vocab = [w for w in vocab if (w not in spacy.en.language_data.STOP_WORDS and w.lower() not in spacy.en.language_data.STOP_WORDS) or w.lower() in allowed_stopwords]
    vocab_set = set(vocab)
    result = []
    total_len = len(vocab)
    cnt = 0
    for word in vocab:
        cnt += 1
        print(word, '--', cnt, '/', total_len)
        all_syns = wn.synsets(word)

        if len(all_syns) > 0:
            syns = all_syns[0]

            # get hypernyms/hyponyms
            #hyper, found_hyper = closest_hypernym(syns, vocab_set)
            hyper, found_hyper, hyper_lemmas = first_hypernym(syns, vocab_set, min_dist_to_top=MIN_DIST_TO_TOP_HYPER)
            if found_hyper:
                #hyper_lemmas = lemma_in_vocab(hyper, vocab_set)
                for lemma in hyper_lemmas:
                    result.append((word, lemma, 'hypernym'))
                    result.append((lemma, word, 'hyponym'))

            # get synonyms
            syns_lemmas = lemma_in_vocab(syns, vocab_set)
            for w1 in syns_lemmas:
                for w2 in syns_lemmas:
                    result.append((w1, w2, 'synonym'))
            # as this is lemmatized, also include original word
            for w1 in syns_lemmas:
                result.append((word, w1, 'synonym'))
                result.append((w1, word, 'synonym'))

            # get antonyms
            lemmas = syns.lemmas()
            lemmas = [lemma for lemma in lemmas if lemma.name() in vocab]

            all_antonyms = []
            for lemma in lemmas:
                antonym_names = [anto.name() for anto in lemma.antonyms()]
                antonym_names = [anto for anto in antonym_names if anto in vocab]
                all_antonyms.extend(antonym_names)

            lemma_names = list(set(lemma_in_vocab(syns, vocab_set) + [word]))
            all_antonyms = list(set(all_antonyms))

            for lemma_name in lemma_names:
                for anto in all_antonyms:
                    result.append((lemma_name, anto, 'antonym'))
                    result.append((anto, lemma_name, 'antonym'))

            # get cohyponyms
            hyper, found_hyper = first_hypernym(syns, min_dist_to_top=MIN_DIST_TO_TOP_COHYPO)#closest_hypernym(syns)
            if found_hyper:
                hyponyms = get_hyponyms_excluding_syns(hyper, syns)
                hyponym_names = []
                for hypo in hyponyms:
                    hypo_lemmas = [w for w in list(set(hypo.lemma_names()) & vocab_set) if w not in lemma_names]
                    hyponym_names.extend(hypo_lemmas)

                hyponym_names = list(set(hyponym_names))
                for w in lemma_names:
                    for cohypo in hyponym_names:
                        result.append((w, cohypo, 'cohyponym'))
                        result.append((cohypo, w, 'cohyponym'))

    # Clean data
    print('Found:', len(result))
    result = list(set(result))
    print('Remove simple duplicates:', len(result))

    # Clean data
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: set()))
    for w1, w2, relation in result:
        data[w1][w2].add(relation)

    for w1 in data:
        for w2 in data[w1]:
            if len(data[w1][w2]) > 1:
                data[w1][w2] = resolve_label_conflict(data[w1][w2])
            else:
                data[w1][w2] = data[w1][w2].pop()

    cnt_final = 0
    with open(out_path, 'w') as f_out:
        for w1 in data:
            current = data[w1]
            for w2 in current:
                cnt_final += 1
                print('\t'.join([w1, w2, current[w2]]))
                f_out.write('\t'.join([w1, w2, current[w2]]) + '\n')

    print('Wrote out:', cnt_final)

def sample(file_path, amount):
    with open(file_path) as f_in:
        data = [line.strip() for line in f_in.readlines()]

    samples = random.sample(data, amount)
    for s in samples:
        print(s)

def tokenize(sent):
    doc = nlp(sent,  parse=False, tag=False, entity=False)
    return [token.text for token in doc]

def show_words(word_data, synset):
    words = torch.load(word_data)
    print(words[synset])

def resolve_label_conflict(labels):
    if 'synonym' in labels:
        return 'synonym'
    if 'antonym' in labels:
        return 'antonym'
    if 'hypernym' in labels:
        return 'hypernym'
    if 'hyponym' in labels:
        return 'hyponym'
    if 'cohyponym' in labels:
        return 'cohyponym'

    # SHOULD NOT BE HERE
    1/0

def create_data(count_path, vocab_path, out_path):

    # use maximum this amount of synsets
    MAX_AMOUNT_SYNSETS = 1

    # look for this distant hypernyms/hyponyms
    SEARCH_DEPTH = 1



    # Specially treat animals
    #animal_synsets = set(list(list(wn.synsets('animal', wn.NOUN))[0].closure(lambda s: s.hyponyms()))).difference(
    #    set(list(list(wn.synsets('person', wn.NOUN))[0].closure(lambda s: s.hyponyms()))))

    # helper functions
    hyper = lambda s: s.hypernyms()
    hypo = lambda s: s.hyponyms()

    count_data = torch.load(count_path)

    with open(vocab_path) as f_in:
        vocab = [line.strip() for line in f_in.readlines()]
        vocab = [w for w in vocab if w not in spacy.en.language_data.STOP_WORDS and w.lower() not in spacy.en.language_data.STOP_WORDS]

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
            hyponyms = [h for h in hyponyms if h.name() != syns.name()]

            hyponym_names = [name for h in hyponyms for name in h.lemma_names() ]
            hyponym_names = list(set(hyponym_names).intersection(vocab))
            hyponym_names = [name for name in hyponym_names if name not in syns_in_vocab]

            for w1 in syns_in_vocab:
                for w2 in hyponym_names:
                    cohyponyms.append((w1, w2, 'cohyponym'))

        result.extend(list(set(hypernyms)))
        result.extend(list(set(cohyponyms)))


    # Clean data
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: set()))
    for w1, w2, relation in result:
        data[w1][w2].add(relation)

    for w1 in data:
        for w2 in data[w1]:
            if len(data[w1][w2]) > 1:
                data[w1][w2] = resolve_label_conflict(data[w1][w2])
            else:
                data[w1][w2] = data[w1][w2].pop()

    with open(out_path, 'w') as f_out:
        for w1 in data:
            current = data[w1]
            for w2 in current:
                f_out.write('\t'.join([w1, w2, current[w2]]) + '\n')
        #for w1, w2, relation in data:
        #    f_out.write('\t'.join([w1, w2, relation]) + '\n')




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


