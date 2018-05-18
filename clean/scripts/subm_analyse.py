import sys, json, re
sys.path.append('./../')

from docopt import docopt

import collections
import numpy as np

import nltk

from nltk.corpus import wordnet as wn
#from libs import evaluate as ev


def main():
    args = docopt("""Analyse creation and baselines for the submission. 

    Usage:
        subm_analyse.py word_dist <train_data> <newtest> <out>
        subm_analyse.py wn_baseline <newtest>
        subm_analyse.py test
        subm_analyse.py merge <orig_data> <wp_data> <out_data>
        subm_analyse.py find_relevant <wn_data> <dataset> <out>
        subm_analyse.py eval_hypern <data_path> <base_path> <vocab_path>

    """)

    if args['word_dist']:
        analyse_word_distribution(args['<train_data>'], args['<newtest>'], args['<out>'])
    elif args['wn_baseline']:
        calc_wn_baseline(args['<newtest>'])
    elif args['test']:
        test()
    elif args['merge']:
        merge(args['<orig_data>'], args['<wp_data>'], args['<out_data>'])
    elif args['find_relevant']:
        find_relevant(args['<wn_data>'], args['<dataset>'], args['<out>'])
    elif args['eval_hypern']:
        eval_hypern(args['<data_path>'], args['<base_path>'], args['<vocab_path>'])

NOT_IDX = 999999

mapping = dict([
        ('in a garage', 'garage'),
        ('close to', 'close'),
        ('far away from', 'far'),
        ('in a kitchen', 'kitchen'),
        ('in a room', 'room'),
        ('in a bathroom', 'bathroom'),
        ('plenty of', 'plenty'),
        ('during the day', 'day'),
        ('in a building', 'building'),
        ('far from', 'far'),
        ('at night', 'night'),
        ('in a hallway', 'hallway')
    ])


def _map(w):
    if w in mapping:
        return mapping[w]
    return w

def get_hyper_lemmas(w, amount=5):
    synsets = wn.synsets(w)

    count = 0
    done = False
    lemma_list = []
    while count < amount and not done:
        if len(synsets) == 0:
            done = True
        else:
            syns = synsets[0]
            hypernyms = syns.hypernyms() + syns.instance_hypernyms()
            if len(hypernyms) == 0:
                done = True
            else:
                lemmas = [lemma.name() for lemma in hypernyms[0].lemmas() if len(lemma.name().split(' ')) == 1]
                if len(lemmas) > 0:
                    count +=1
                    lemma_list.extend(lemmas)
                synsets = hypernyms

    return set(lemma_list)


def eval_hypern(data_path, base_path, vocab_path):
    print('Read vocab')
    with open(vocab_path) as f_in:
        vocab = set(line.strip() for line in f_in.readlines())

    bins_total = [0,3,6,9]
    bins_percentage = 0.1

    print('Read data')
    with open(data_path) as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    with open(base_path) as f_in:
        data_base = [json.loads(line.strip()) for line in f_in.readlines()]
    base_dict = dict()
    for db in data_base:
        if db['gold_label'] == db['predicted_label']:
            base_dict[db['pairID']] = True
        else:
            base_dict[db['pairID']] = False

    correct_correct = 0
    correct_incorrect = 0
    incorrect_correct = 0
    incorrect_incorrect = 0

    correct = collections.defaultdict(list)
    incorrect = collections.defaultdict(list)

    incorrect_correct_dict = collections.defaultdict(list)
    correct_incorrect_dict = collections.defaultdict(list)

    print('Sort data')
    for d in data:
        w1 = _map(d['replaced1'])
        w2 = _map(d['replaced2'])

        is_correct = d['gold_label'] == d['predicted_label']

        if len(w1.split(' ')) > 1 or len(w2.split(' ')) > 1 or w1 not in vocab or w2 not in vocab:
            pass
        else:
            sample = (w1, w2, d['category'])
            if is_correct:
                correct[d['gold_label']].append(sample)
            else:
                incorrect[d['gold_label']].append(sample)

        if is_correct:
            if base_dict[d['pairID']]:
                correct_correct += 1
                
            else:
                incorrect_correct += 1
                incorrect_correct_dict[d['gold_label']].append(sample)
        else:
            if base_dict[d['pairID']]:
                correct_incorrect += 1
                correct_incorrect_dict[d['gold_label']].append(sample)
            else:
                incorrect_incorrect += 1

    print('correct_correct', correct_correct)
    print('correct_incorrect', correct_incorrect)
    print('incorrect_correct', incorrect_correct)
    print('incorrect_incorrect', incorrect_incorrect)

    print('incorrect_correct', 'ent', len(incorrect_correct_dict['entailment']), 'cont', len(incorrect_correct_dict['contradiction']))
    print('correct_incorrect', 'ent', len(correct_incorrect_dict['entailment']), 'cont', len(correct_incorrect_dict['contradiction']))

    #incorrect_correct_anton = []
    print('Calc')
    for tag, sample_dict in [('correct', correct), ('incorrect', incorrect)]:
        print('#', tag)
        for lbl, samples in [('entailment', sample_dict['entailment']), 
        ('contradiction', sample_dict['contradiction']), 
        ('entail_incorrect_correct', incorrect_correct_dict['entailment']),
        ('entail_correct_incorrect', correct_incorrect_dict['entailment']),
        ('contr_incorrect_correct', incorrect_correct_dict['contradiction']),
        ('contr_correct_incorrect', correct_incorrect_dict['contradiction'])]:
            print('lbl', lbl)
            results = collections.defaultdict(int)
            results_percentage  = list()
            results_percentage_cat = list()
            for w1, w2, cat in samples:
                set1 = get_hyper_lemmas(w1)
                set2 = get_hyper_lemmas(w2)

                len1 = len(list(set1))
                len2 = len(list(set2))
                len_total = len(list(set1 | set2))
                if len1 + len2 == 0:
                    shared = -1
                else:
                    shared = len(list(set1 & set2))
                    results_percentage.append(shared/(len_total))
                    results_percentage_cat.append(cat)
                results[shared] += 1



            #for k in results:
            #    print('shared:', k, 'amount:', results[k], round(results[k] / len(samples), 2))
            print('--bins')
            binz = [0 for i in range(len(bins_total))]
            for k in results:
                added = False
                for j in range(len(bins_total)):
                    if k < bins_total[j]:
                        binz[j-1] += results[k]
                        added = True
                        break

                if not added:
                    binz[-1] += results[k]

            #for j in range(len(binz)):
            #    print('bin:', bins_total[j],':', binz[j], round(binz[j]/ len(samples), 3))


            print('mean:', np.mean(np.asarray(results_percentage)), 'std:',  np.std(np.asarray(results_percentage)), 'median:', np.median(np.asarray(results_percentage)))
            
            min_val = min(results_percentage)
            max_val = max(results_percentage)
            amount_bins = max_val // bins_percentage

            start = 0.0
            bins = []
            while len(bins) < amount_bins:
                bins.append(start)
                start += bins_percentage

            amount_per_bins = [0 for i in range(len(bins))]
            cats_per_bins = [[] for i in range(len(bins))]


            for cnt, v in enumerate(results_percentage):
                added = False
                for j in range(len(bins)):
                    if v < bins[j]:
                        amount_per_bins[j-1] += 1
                        cats_per_bins[j-1].append(results_percentage_cat[cnt])
                        added = True
                        break
                if not added:
                    amount_per_bins[-1] += 1
                    cats_per_bins[-1].append(results_percentage_cat[cnt])
           
            print('percentage bins')
            for i in range(len(bins)):
                print(bins[i], ':', amount_per_bins[i], round(amount_per_bins[i] / len(samples), 3), collections.Counter(cats_per_bins[i]).most_common())
            print()


def find_relevant(data_path, dataset_path, out_path):

    mapping = dict([
        ('in a garage', 'garage'),
        ('close to', 'close'),
        ('far away from', 'far'),
        ('in a kitchen', 'kitchen'),
        ('in a room', 'room'),
        ('in a bathroom', 'bathroom'),
        ('plenty of', 'plenty'),
        ('during the day', 'day'),
        ('in a building', 'building'),
        ('far from', 'far'),
        ('at night', 'night'),
        ('in a hallway', 'hallway')
    ])

    def map(w):
        if w in mapping:
            return mapping[w]
        else:
            return w

    print('Load dataset')

    with open(dataset_path) as f_in:
        dataset = [json.loads(line.strip()) for line in f_in.readlines()]

    categories = collections.defaultdict(list)
    for sample in dataset:
        categories[sample['category']].append(sample)

    print('Load data')
    with open(data_path) as f_in:
        data = [line.strip().split('\t') for line in f_in.readlines()]
        data = [d for d in data if len(d) == 3]
    print('Found:', len(data))

    covered_pairs_in = set()
    covered_pairs_out = set()
    for w1, w2, lbl in data:
        if lbl == 'entailment':
            covered_pairs_in.add(w1 + '_' + w2)
        elif lbl == 'contradiction':
            covered_pairs_out.add(w1 + '_' + w2)
        else:
            1/0

    print('Filter')
    final_data = []
    final_data_out = []
    for cat in categories.keys():
        print('Category:', cat)
        initial_amount = len(categories[cat])
        category_samples = []
        for sample in categories[cat]:
            key = map(sample['replaced1']) + '_' + map(sample['replaced2'])
            if sample['gold_label'] == 'contradiction':
                if key in covered_pairs_out:
                    category_samples.append(sample)
                else:
                    final_data_out.append(sample)
            elif sample['gold_label'] == 'entailment':
                if key in covered_pairs_in:
                    category_samples.append(sample)
                else:
                    final_data_out.append(sample)
            else:
                final_data_out.append(sample)
            

        print('Captured:', len(category_samples),'/', initial_amount)
        final_data.extend(category_samples)


    print('Write out IN')
    with open(out_path + '.in', 'w') as f_out:
        for s in final_data:
            f_out.write(json.dumps(s) + '\n')

    print('Write out OUT')
    with open(out_path + '.out', 'w') as f_out:
        for s in final_data_out:
            f_out.write(json.dumps(s) + '\n')


def merge(path_orig, path_wp, path_out):
    with open(path_orig) as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]
    with open(path_wp) as f_in:
        word_pair_data = [json.loads(line.strip()) for line in f_in.readlines()]
        word_pairs = dict()
        for sample in word_pair_data:
            word_pairs[sample['pairID']] = [sample['replaced1'], sample['replaced2']]

    with open(path_out, 'w') as f_out:
        for sample in data:
            sample['replaced'] = word_pairs[sample['pairID']]
            f_out.write(json.dumps(sample) + '\n')

def is_synonym(synsets1, synsets2):
    result = False
    best_idx = NOT_IDX

    for i, s1 in enumerate(synsets1):
        s1_all = set([s.name()for s in s1.similar_tos()] + [s1.name()])
        for j, s2 in enumerate(synsets2):
            s2_all = set([s.name() for s in s2.similar_tos()] + [s2.name()])
            #if s1.name() == s2.name():
            if len(s1_all & s2_all) > 0:
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


    # extend synsets2


    for i, s1_orig in enumerate(synsets1):
        for s1 in [s1_orig] + [s for s in s1_orig.similar_tos()]:
            antonyms1 = [anto.name() for lemma in s1.lemmas() for anto in lemma.antonyms() ]
            if len(antonyms1) == 0:
                continue
            else:
                for j, s2_orig in enumerate(synsets2):
                    for s2 in [s2_orig] + [s for s in s2_orig.similar_tos()]:
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
    w1 = 'power_drill'
    w2 = 'tool'


    s1 = wn.synsets(w1)[0]
    s2 = wn.synsets(w2)[0]

    print(s1.name(), s1.definition())
    print(s2.name(), s2.definition())

    common = s1.lowest_common_hypernyms(s2)[0]
    print('lowest:', common.name(), common.definition())
    print('dist:', s1.shortest_path_distance(s2))



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
        for pred in pred_dict[gold]:
            cnt_total += pred_dict[gold][pred]
            if pred == lbl:
                cnt_lbl += pred_dict[gold][pred]

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