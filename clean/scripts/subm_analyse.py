import sys, json, re
sys.path.append('./../')

from docopt import docopt

import collections
import numpy as np

def main():
    args = docopt("""Analyse creation and baselines for the submission. 

    Usage:
        subm_analyse.py word_dist <train_data> <newtest> <out>
        subm_analyse.py wn_baseline <newtest>

    """)

    if args['word_dist']:
        analyse_word_distribution(args['<train_data>'], args['<newtest>'], args['<out>'])
    elif args['wn_baseline']:
        calc_wn_baseline(args['<newtest>'])

def calc_wn_baseline(newtest):
    print('Read new test-set ...')
    with open(newtest) as f_in:
        test = [json.loads(line.strip()) for line in f_in.readlines()]
    print('Done.')
    print('Get Replacement words')
    repl_words1 = set([d['replaced1'] for d in test])
    repl_words2 = set([d['replaced1'] for d in test])
    repl_words = list(repl_words1 | repl_words2)

    # find the ones with more than one word
    multi_word = [w for w in [rw.split(' ') for rw in repl_words] if len(w) > 1]
    print(multi_word)


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