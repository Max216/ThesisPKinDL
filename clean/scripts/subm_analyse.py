import sys, json, re
sys.path.append('./../')

from docopt import docopt

import collections
import numpy as np

import nltk

from nltk.corpus import wordnet as wn
from libs import evaluate as ev


def main():
    args = docopt("""Analyse creation and baselines for the submission. 

    Usage:
        subm_analyse.py word_dist <train_data> <newtest> <out>
        subm_analyse.py wn_baseline <newtest>
        subm_analyse.py test

    """)

    if args['word_dist']:
        analyse_word_distribution(args['<train_data>'], args['<newtest>'], args['<out>'])
    elif args['wn_baseline']:
        calc_wn_baseline(args['<newtest>'])
    elif args['test']:
        test()

NOT_IDX = 999999

def is_synonym(synsets1, synsets2):
    result = False
    best_idx = NOT_IDX

    for i, s1 in enumerate(synsets1):
        for j, s2 in enumerate(synsets2):
            if s1.name() == s2.name():
                result = True
                idx = max([i,j])
                if i >= best_idx and j >= best_idx:
                    break
                elif idx < best_idx:
                    best_idx = idx

    return (best_idx, result)

def is_antonym(synsets1, synsets2):

    result = False
    best_idx = NOT_IDX


    for i, s1 in enumerate(synsets1):
        antonyms1 = [anto.name() for lemma in s1.lemmas() for anto in lemma.antonyms() ]
        if len(antonyms1) == 0:
            continue
        else:
            for j, s2 in enumerate(synsets2):
                if len(set(antonyms1) & set(s2.lemma_names())) > 0:
                    result = True
                    idx = max([i,j])
                    if i >= best_idx and j >= best_idx:
                        break
                    elif idx < best_idx:
                        best_idx = idx
    return (best_idx, result)


def all_hypernyms(syn, max_dist):

    def recursive_search(synset, all_hypernyms, current_dist):
        if current_dist > max_dist:
            return

        synset_hypernyms = synset.hypernyms() + synset.instance_hypernyms()
        if len(synset_hypernyms) > 0:
            all_hypernyms += synset_hypernyms
            for syn_hyp in synset_hypernyms:
                recursive_search(syn_hyp, all_hypernyms, current_dist + 1)

    hypernyms = []
    recursive_search(syn, hypernyms, 0)

    return set(hypernyms)

def is_hypernym(synsets1, synsets2, max_dist = 9999999):
    '''
    test if s1 is hypernym of s2
    '''
    result = False
    best_idx = NOT_IDX

    for j, s2 in enumerate(synsets2):
        hypernyms2 = all_hypernyms(s2, max_dist)
        hyper_names = set([h.name() for h in hypernyms2])
        for i, s1 in enumerate(synsets1):
            if s1.name() in hyper_names:
                result = True
                idx = max([i,j])
                if i >= best_idx and j >= best_idx:
                    break
                elif idx < best_idx:
                    best_idx = idx
    return (best_idx, result)

def is_hyponym(synsets1, synsets2, max_dist=9999999):
    '''test if s1 is a hyponym of s2'''
    return is_hypernym(synsets2, synsets1, max_dist=max_dist)

def is_cohyponym(synsets1, synsets2, max_dist=2):
    result = False
    best_idx = NOT_IDX

    for i, s1 in enumerate(synsets1):
        hyper1 = all_hypernyms(s1, max_dist)
        for h1 in hyper1:
            for j, s2 in enumerate(synsets2):
                # ignore if synsets have hypernym/hyponym relationship
                if s1 in set([s for s in s2.closure(lambda x: x.hyponyms() + x.instance_hyponyms())]):
                    continue
                elif s2 in set([s for s in s1.closure(lambda x: x.hyponyms() + x.instance_hyponyms())]):
                    continue
                elif s1.name() == s2.name():
                    continue
                _, ishyper =  is_hypernym([h1], [s2], max_dist=max_dist)
                if ishyper:
                    result = True
                    idx = max([i,j])
                    if i >= best_idx and j >= best_idx:
                        break
                    elif idx < best_idx:
                        best_idx = idx

    return (best_idx, result)



def test():
    w1 = 'pretty'
    w2 = 'beautiful'


    syn_w1  = wn.synsets(w1)
    syn_w2  = wn.synsets(w2)

    print('syn', is_synonym(syn_w1, syn_w2))
    print('anto', is_antonym(syn_w1, syn_w2))
    print('hypo', is_hyponym(syn_w1, syn_w2))
    print('hyper', is_hypernym(syn_w1, syn_w2))
    print('cohypo', is_cohyponym(syn_w1, syn_w2))



def to_single_word(w):
    if len(w) == 1:
        return w[0]

    mapping = dict([
        ('in a garage', 'garage'),
        ('close to', 'close'),
        ('a lot of', 'lot'),
        ('far away from', 'far'),
        ('in a kitchen', 'kitchen'),
        ('living room', 'living_room'),
        ('Saudi Arabia', 'saudi_arabia'),
        ('french horn', 'french_horn'),
        ('North Korea', 'north_korea'),
        ('South Korean', 'south_korea'),
        ('electric guitar', 'electric_guitar'),
        ('in a room', 'room'),
        ('New Zealand', 'new_zealand'),
        ('in a bathroom', 'bathroom'),
        ('plenty of', 'plenty'),
        ('during the day', 'day'),
        ('prison cell', 'prison_cell'),
        ('dining room', 'dining_room'),
        ('in front of', 'in_front'),
        ('in a building', 'building'),
        ('acoustic guitar', 'acoustic_guitar'),
        ('far from', 'far'),
        ('common room', 'common_room'),
        ('hot chocolate', 'hot_chocolate'),
        ('North Korean', 'north_korean'),
        ('at night', 'night'),
        ('in a hallway', 'hallway'),
        ('can not', 'not'),
        ('no one', 'no_one'),
    ])

    return mapping[' '.join(w)]

def predict(w1, w2, lbl):
    synsets1 = wn.synsets(w1)
    synsets2 = wn.synsets(w2)

    if len(synsets1) == 0 or len(synsets2) == 0:
        return ('other', 'other')

    syn_score, syn = is_synonym(synsets1, synsets2)
    anto_score, anto = is_antonym(synsets1, synsets2)
    p_hypo_score, p_hypo = is_hyponym(synsets1, synsets2)
    p_hyper_score, p_hyper = is_hypernym(synsets1, synsets2)
    cohypo_score, cohypo = is_cohyponym(synsets1, synsets2)


    lbl_best = None
    lbl_first = None
    # label for heuristic first sense
    min_score = min([syn_score, anto_score, p_hypo_score, p_hyper_score, cohypo_score])
    if syn and syn_score == min_score:
        lbl_first = 'entailment'
    elif anto and anto_score == min_score:
        lbl_first = 'contradiction'
    elif p_hypo and p_hypo_score == min_score:
        lbl_first = 'entailment'
    elif p_hyper and p_hyper_score == min_score:
        lbl_first = 'neutral'
    elif cohypo and cohypo_score == min_score:
        lbl_first = 'contradiction'
    else:
        lbl_first = 'other'

    # label for heuristic best sense

    # default to first heuristic
    lbl_best = lbl_first

    if lbl == 'entailment':
        if syn or p_hypo:
            lbl_best = 'entailment'
    elif lbl == 'neutral':
        if p_hyper:
            lbl_best == 'neutral'
    elif lbl == 'contradiction':
        if anto or cohypo:
            lbl_best = 'contradiction'

    return (lbl_first, lbl_best)

def percent_of_pred_lbl(pred_dict, lbl):

    cnt_total = 0
    cnt_lbl = 0

    for gold in pred_dict:
        for pred in pred_dict:
            cnt_total += pred_dict[gold][pred]
            if pred == lbl:
                cnt_lbl += pred_dict[gold][pred]

    print('total cnt', lbl, cnt_lbl)
    return cnt_lbl / cnt_total

def print_evaluation(pred_dict):
    categories = sorted([k for k in pred_dict.keys()])

    print('# By category:')
    all_predictions = collections.defaultdict(lambda: collections.defaultdict(int))
    for cat in categories:
        wn_rel_percent = 1 - percent_of_pred_lbl(pred_dict[cat], 'other')
        print(cat, ':', ev.accuracy_prediction_dict(pred_dict[cat]), 'percent with wn relations:', wn_rel_percent)
        for gold in pred_dict[cat]:
            for pred in pred_dict[cat][gold]:
                all_predictions[gold][pred] += pred_dict[cat][gold][pred]

    print('# General')
    print('Accuracy:', ev.accuracy_prediction_dict(all_predictions))
    recall_e, prec_e  = ev.recall_precision_prediction_dict(all_predictions, 'entailment')
    recall_c, prec_c  = ev.recall_precision_prediction_dict(all_predictions, 'contradiction')
    recall_n, prec_n  = ev.recall_precision_prediction_dict(all_predictions, 'neutral')
    print('Entailment: recall:', recall_e, 'prec:', prec_e)
    print('Neutral: recall:', recall_n, 'prec:', prec_n)
    print('Contradiction: recall:', recall_c, 'prec:', prec_c)

def print_misclassified(mis_dict):
    categories = sorted([k for k in mis_dict.keys()])
    for cat in categories:
        print('#', cat)
        for lbl in mis_dict[cat]:
            print(lbl, '-->', mis_dict[cat][lbl])

def calc_wn_baseline(newtest):
    print('Read new test-set ...')
    with open(newtest) as f_in:
        test = [json.loads(line.strip()) for line in f_in.readlines()]
    print('Done.')
    print('Get Replacement words')
    repl_words1 = set([d['replaced1'] for d in test])
    repl_words2 = set([d['replaced2'] for d in test])
    repl_words = list(repl_words1 | repl_words2)

    # find the ones with more than one word
    #multi_word = [w for w in [rw.split(' ') for rw in repl_words] if len(w) > 1]
    #print(multi_word)
    #print('Validate')

    #for w in multi_word:
    #    print(wn.synsets(to_single_word(' '.join(w))))

    print('Evaluate ...')
    correct = 0
    total = 0

    test = [(to_single_word(d['replaced1'].split(' ')), to_single_word(d['replaced2'].split(' ')), d['gold_label'], d['category']) for d in test]

    # by category, label_gold, label_predicted, amount
    predictiondict_first = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
    predictiondict_best = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
    
    misclassified_first = collections.defaultdict(lambda: collections.defaultdict(lambda: set()))
    misclassified_best = collections.defaultdict(lambda: collections.defaultdict(lambda: set()))
    for cnt, (w1, w2, lbl, category) in enumerate(test):
        if cnt % 10 == 0:
            print('samples done:' + str(cnt), end='\r')
        result =  predict(w1, w2, lbl)
        lbl_first, lbl_best = result
        predictiondict_first[category][lbl][lbl_first] += 1
        predictiondict_best[category][lbl][lbl_best] += 1

        if lbl != lbl_first:
            misclassified_first[category][lbl_first].add((w1, w2))
        if lbl != lbl_best:
            misclassified_best[category][lbl_best].add((w1, w2))

    print('')

    print('Done.')
    print('Results for heuristic: first synset:')
    print_evaluation(predictiondict_first)
    print('results for heuristic: best synset:')
    print_evaluation(predictiondict_best)
    print()
    print()
    print('## misclassified first')
    print_misclassified(misclassified_first)
    print()
    print('## misclassified_best')
    print_misclassified(misclassified_best)



def analyse_word_distribution(train_data, newtest, out_file):
    print('Read train data ...')
    with open(train_data) as f_in:
        train = [json.loads(line.strip()) for line in f_in.readlines()]
    print('Done.')

    print('Read new test-set ...')
    with open(newtest) as f_in:
        test = [json.loads(line.strip()) for line in f_in.readlines()]
    print('Done.')

    print('Find replacements ...')
    replacement_counter = collections.Counter([data['replaced2'] for data in test])
    print('Most Common:')
    print(replacement_counter.most_common()[:5])
    print('Least common:', replacement_counter.most_common()[-3:])

    print('Count replaced words in train data ...')
    regexps = [((w, re.compile('\\b' + w + '\\b'))) for w in replacement_counter]
    print('Created', len(regexps), 'regexp')
    correct_labels = set(['entailment', 'contradiction', 'neutral'])
    orig_counts = collections.defaultdict(int)
    for i, data in enumerate(train):
        if data['gold_label'] in correct_labels:
            for w, regex in regexps:
                if regex.search(data['sentence1']):
                    orig_counts[w] += 1
                if regex.search(data['sentence2']):
                    orig_counts[w] += 1

            if i  % 10000 == 0:
                print('Checked:', i+1)

    print('Done')
    only_orig_counts = sorted([orig_counts[w] for w in orig_counts])
    print('maximum counts', only_orig_counts[-3:])
    print('minimum counts:', only_orig_counts[:3])
    print('mean:', np.mean(np.asarray(only_orig_counts)),'validate::', sum(only_orig_counts) / len(only_orig_counts))
    print('first percintile', np.percentile(np.asarray(only_orig_counts), 25))
    print('median:', np.median(np.asarray(only_orig_counts)))
    print('third percintile', np.percentile(np.asarray(only_orig_counts), 75))

    print('Write details out ...')
    with open(out_file, 'w') as f_out:
        counts = sorted([(w, orig_counts[w]) for w in orig_counts], key=lambda x: x[1])
        print('#word\t#orig_count\t#replacements_count')
        for w, c in counts:
            f_out.write(w + '\t' + str(c) + '\t' + str(replacement_counter[w]) + '\n')

    print('Done.')



if __name__ == '__main__':
    main()