'''Deal with raw data to manipulate'''

import re
import os
import json

from libs import data_tools, config

class ReplacedDataHolder:
    '''
    Maintains replaced data
    '''

    SUMMARY_NAME = 'summary.sjson'

    def __init__(self):
        self.pairs = []

    def add_samples(self, w1, w2, label, samples):
        '''
        Adds generted samples to this instance
        :param w1       relevant word in premise
        :param w2       relevant word in hypothesis
        :param label    assumed label
        :param samples  generated samples: [(premise, hypothesis, label, generation_replacement), ...]
        '''

        self.pairs.append((w1, w2, label, samples))

    def get_internal_stats(self):
        '''
        Get info about the amount of samples generated per example
        '''
        return [(w1, w2, label, self._to_data_name(w1, w2), len(samples)) for w1, w2, label, samples in self.pairs]

    def write_summary(self, directory):
        self._ensure_directory(directory)
        out_path = os.path.join(directory, self.SUMMARY_NAME)
        with open(out_path, 'w') as f_out:
            for w1, w2, label, filepath, amount in self.get_internal_stats():
                sample_json = {
                    'word_p': w1,
                    'word_h': w2,
                    'assumed_label': label,
                    'rel_path': filepath,
                    'amount': amount
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

    def _to_data_name(self, w1, w2):
        '''
        Create the name of the pair of words
        '''
        return '_'.join(w1.split(' ')) + '__' + '_'.join(w2.split(' ')) + '.jsonl'

    def _ensure_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


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
        for w1, w2, label in replacements:

            regexp1 = re.compile('\\b' + w1 + '\\b')
            regexp2 = re.compile('\\b' + w2 + '\\b')

            # remember seen sentences in order to avoid unnecessary searches
            seen_sentences = set()

            # remember generated samples to avoid duplicates
            sample_hashes = set() 
            new_samples = []


            # iterate over all data
            for sent1, sent2, _ in self.samples:

                # iterate over all sentences
                for sent in [sent1, sent2]:

                    # remember seen sentences
                    hashed_sent = hash(sent)
                    if hashed_sent not in seen_sentences:
                        seen_sentences.add(hashed_sent)

                        if replace == 'any':
                            contains_w1 = regexp1.search(sent)
                            contains_w2 = regexp2.search(sent)
                        elif replace == 'w1':
                            contains_w1 = regexp1.search(sent)
                            contains_w2 = False
                        else:
                            contains_w1 = False
                            contains_w2 = regexp2.search(sent)

                        if contains_w1:
                            replaced_w1 = self._replace_in_sent(sent, regexp1, w2)
                            hashed_sample = hash(sent+replaced_w1)
                            if hashed_sample not in sample_hashes:
                                sample_hashes.add(hashed_sample)
                                new_samples.append((sent, replaced_w1, label, '1'))
                        if contains_w2:
                            replaced_w2 = self._replace_in_sent(sent, regexp2, w1)
                            hashed_sample = hash(replaced_w2+sent)
                            if hashed_sample not in sample_hashes:
                                sample_hashes.add(hashed_sample)
                                new_samples.append((replaced_w2, sent, label, '2'))
            
            # add data to container
            if len(new_samples) > 0:
                replacement_holder.add_samples(w1, w2, label, new_samples)
                print('Added', len(new_samples), 'samples for replacing:', w1, direction_output[replace], w2)
            else:
                print('No samples found for:', w1, direction_output[replace], w2)

        return replacement_holder

    def _replace_in_sent(self, sent, regexp, replacing_str):
        '''
        create a new sentence by replacing the specified regexp by the specified string.
        :param sent             string to work on
        :param regexp           regular expression to find substrings to be relaced
        :param replacing_str    replace with this
        '''
        return re.sub(regexp, replacing_str, sent)