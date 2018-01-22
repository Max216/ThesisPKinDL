'''Deal with raw data to manipulate'''

import re
import os
import json

from libs import data_tools, config

class ReplacedDataHolder:
    '''
    Maintains replaced data
    '''

    SUMMARY_NAME = 'SUMMARY.sjson'

    def __init__(self):
        self.pairs = []
        self.counter = dict()
        self.real_sample_counter = dict()

    def update_counts(self, counter, min_val=1):
        '''
        remember the counts per word within the data
        :param counter      dictionary(word, counts)
        :param min_val      onyl remember values >= this value
        '''

        for w in counter:
            if counter[w] >= min_val:
                self.counter[w] = counter[w]


    def add_samples(self, w1, w2, label, amount_real_samples, samples):
        '''
        Adds generted samples to this instance
        :param w1                   relevant word in premise
        :param w2                   relevant word in hypothesis
        :param label                assumed label
        :param amount_real_samples  amount of real samples contining these word in this order
        :param samples              generated samples: [(premise, hypothesis, label, generation_replacement), ...]
        '''

        self._add_real_sample_count(w1, w2, amount_real_samples)
        self.pairs.append((w1, w2, label, samples))

    def get_internal_stats(self):
        '''
        Get info about the amount of samples generated per example
        '''
        return [(w1, w2, label, self._to_data_name(w1, w2), len(samples)) for w1, w2, label, samples in self.pairs]

    def write_summary(self, directory):
        self._ensure_directory(directory)

        # write out summary of data
        out_path = os.path.join(directory, self.SUMMARY_NAME)
        with open(out_path, 'w') as f_out:
            for w1, w2, label, filepath, amount in self.get_internal_stats():
                sample_json = {
                    'word_p': w1,
                    'word_h': w2,
                    'assumed_label': label,
                    'rel_path': filepath,
                    'amount': amount,
                    'sents_with_word_p': self.counter[w1],
                    'sents_with_word_h': self.counter[w2],
                    'real_sample_count': self.real_sample_counter[w1][w2]
                }
                f_out.write(json.dumps(sample_json) + '\n')

    def write_dataset(self, directory):
        self._ensure_directory(directory)
        for w1, w2, _, samples in self.pairs:
            out_path = os.path.join(directory, self._to_data_name(w1, w2))
            with open(out_path, 'w') as f_out:
                for premise, hypothesis, assumed_label, generation_type in samples:
                    json_out = {
                        'sentence1': premise,
                        'sentence2': hypothesis,
                        'gold_label': assumed_label,
                        'generation_replaced': generation_type
                    }
                    f_out.write(json.dumps(json_out) + '\n')

    def merge(self, other):
        '''
        Merge 'other' ReplacedDataHolder inti this instance 
        :param other    ReplacedDataHolder instance
        '''

        self.pairs.extend(other.pairs)
        self.update_counts(other.counter)

        for key_p in other.real_sample_counter:
            for key_h in other.real_sample_counter[key_p]:
                self._add_real_sample_count(key_p, key_h, other.real_sample_counter[key_p][key_h])

    def _to_data_name(self, w1, w2):
        '''
        Create the name of the pair of words
        '''
        return '_'.join(w1.split(' ')) + '__' + '_'.join(w2.split(' ')) + '.jsonl'

    def _ensure_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _add_real_sample_count(self, w1, w2, amount):
        if w1 not in self.real_sample_counter:
            self.real_sample_counter[w1] = dict()
        if w2 not in self.real_sample_counter[w1]:
            self.real_sample_counter[w1][w2] = amount
        else:
            print('Should not happen. Alread have real sample info:', self.real_sample_counter[w1][w2],'. Want to override with', amount)
            1/0


class DataManipulator:
    '''Load, manipulate and store data without'''

    def __init__(self, data=[]):
        '''
        create a new instance with data.
        '''
        self.samples = data    


    def load(self, path=config.PATH_TRAIN_DATA, valid_labels=data_tools.DEFAULT_VALID_LABELS):
        '''
        Loads data at the specified path. Replaces all existing.
        :param path             load data from here
        :param valid_labeels    only load samples with these labels
        '''
        with open(path) as f_in:
            lines = [line.strip() for line in f_in.readlines()]

        self.samples = data_tools._load_snli(lines, valid_labels=valid_labels, tokenize=False)
        
        return self

    def generate_by_replacement(self, replacements, replace='any'):
        '''
        Generate new samples by replacing regexp
        :param replacements         words to replace as a list: [(string-to-replace, replacing-string, label), ...]
        :param replace              Specifies the type of generation, 
                                        if any word occurence can be replaced ('any')
                                        if only sentences containing string1 can be replaced ('w1')
                                        if only sentences containing string2 can be replaced ('w2')
        :param ignore_case          ignores the case by finding matches. Upper cased get only replaced by upper cased and vice versa.
        '''

        ALLOWED_REPLACEMENTS = ['any', 'w1', 'w2']
        direction_output = dict([('any', '<-->'), ('w1', '--->'), ('w2', '<---')])

        if replace not in ALLOWED_REPLACEMENTS:
            print('replace method must be one of the following:',ALLOWED_REPLACEMENTS)
            1/0

        
        # Iterate over all replacement samples
        replacement_holder = ReplacedDataHolder()

        counter = dict()

        for w1, w2, label in replacements:

            counted_w1 = False
            counted_w2 = False
            # skip if not both words in dataset
            if w1 in counter:
                counted_w1 = True
                if counter[w1] == 0:
                    continue
            if w2 in counter:
                counted_w2  = True
                if counter[w2] == 0:
                    continue

            regexp1 = re.compile('\\b' + w1 + '\\b')
            regexp2 = re.compile('\\b' + w2 + '\\b')

            # remember generated samples to avoid duplicates
            sample_hashes = set() 
            new_samples = []

            # remember how many samples with the same label have the two words (same order)
            real_sample_count = 0  
            count_w1 = 0
            count_w2 = 0        

            # iterate over all data
            for sent1, sent2, sample_label in self.samples:

                searched_s1_w1 = False
                searched_s2_w1 = False
                searched_s1_w2 = False
                searched_s2_w2 = False

                s1_contains_w1 = None
                s2_contains_w1 = None
                s1_contains_w2 = None
                s2_contains_w2 = None

                # check if natural sample
                if sample_label ==  label:
                    s1_contains_w1 = regexp1.search(sent1)
                    s2_contains_w2 = regexp2.search(sent2)
                    searched_s1_w1 = True
                    searched_s2_w2 = True

                    if s1_contains_w1 and s2_contains_w2:
                        real_sample_count += 1

                # always check if word not counted yet
                if not counted_w1:
                    if not searched_s1_w1:
                        s1_contains_w1 = regexp1.search(sent1)
                        searched_s1_w1 = True

                    # sent 2 can't be checked yet
                    s2_contains_w1 = regexp1.search(sent2)
                    searched_s2_w1 = True

                    # count
                    if s1_contains_w1:
                        count_w1 += 1
                    if s2_contains_w1:
                        count_w1 += 1

                # same as above for other word
                if not counted_w2:
                    if not searched_s2_w2:
                        s2_contains_w2 = regexp2.search(sent2)
                        searched_s2_w2 = True

                    # sent 1 can't be checked yet
                    s1_contains_w2 = regexp2.search(sent1)
                    searched_s1_w2 = True

                    # count
                    if s1_contains_w2:
                        count_w2 += 1
                    if s2_contains_w2:
                        count_w2 += 1

                # now check for sentences to generate
                both_sents = [
                    (searched_s1_w1, s1_contains_w1, searched_s1_w2, s1_contains_w2, sent1),
                    (searched_s2_w1, s2_contains_w1, searched_s2_w2, s2_contains_w2, sent2)
                ]
                for checked_w1, contains_w1, checked_w2, contains_w2, sent in both_sents:

                    swap_with_w1 = False
                    swap_with_w2 = False

                    if replace == 'any':
                        if not checked_w1:
                            contains_w1 = regexp1.search(sent)
                        if not checked_w2:
                            contains_w2 = regexp2.search(sent)

                        if contains_w1:
                            swap_with_w1 = True
                        if contains_w2:
                            swap_with_w2 = True

                    elif replace == 'w1':
                        if not checked_w1:
                            contains_w1 = regexp1.search(sent)

                        swap_with_w1 = contains_w1

                    else:
                        if not checked_w2:
                            contains_w2 = regexp2.search(sent)

                        swap_with_w2 = contains_w2


                    # Generate samples by swapping
                    if swap_with_w1:
                        replaced_w1 = self._replace_in_sent(sent, regexp1, w2)
                        hashed_sample = hash(sent+replaced_w1)
                        if hashed_sample not in sample_hashes:
                            sample_hashes.add(hashed_sample)
                            new_samples.append((sent, replaced_w1, label, '1'))

                    if swap_with_w2:
                        replaced_w2 = self._replace_in_sent(sent, regexp2, w1)
                        hashed_sample = hash(replaced_w2+sent)
                        if hashed_sample not in sample_hashes:
                            sample_hashes.add(hashed_sample)
                            new_samples.append((replaced_w2, sent, label, '2'))

            
            # Here: Checked all data for samples and generated if possible, counted everything
            
            # Update counts
            if not counted_w1:
                counter[w1] = count_w1

            if not counted_w2:
                counter[w2] = count_w2

            # remove results if not both words are in data
            if counter[w1] < 1 or counter[w2] < 1:
                new_samples = []


            # add data to container
            if len(new_samples) > 0:
                replacement_holder.add_samples(w1, w2, label, real_sample_count, new_samples)
                print('Added', len(new_samples), 'samples for replacing:', w1, direction_output[replace], w2)
            else:
                print('No samples found for:', w1, direction_output[replace], w2)

        
        replacement_holder.update_counts(counter)
        print('Specified words NOT within data:', '; '.join([w for w in counter if counter[w] == 0]))
        return replacement_holder

    def _replace_in_sent(self, sent, regexp, replacing_str):
        '''
        create a new sentence by replacing the specified regexp by the specified string.
        :param sent             string to work on
        :param regexp           regular expression to find substrings to be relaced
        :param replacing_str    replace with this
        '''
        return re.sub(regexp, replacing_str, sent)