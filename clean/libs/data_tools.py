'''
Methods to deal with the data
'''

import json

import torch
from torch.utils.data import Dataset

import spacy
nlp = spacy.load('en')

from libs import config

DEFAULT_DATA_FORMAT = 'snli'
DEFAULT_VALID_LABELS = ['neutral', 'contradiction', 'entailment']

# Internal Helper functions
def _is_sublist(list1, list2):
    '''
    Check if list1 is a sublist (with keeping order) of list2
    :return True/False
    '''

    if len(list1) == 0:
        return True

    first_elm = list1[0]
    indizes_first = [i for i, item in enumerate(list2) if first_elm == item]

    if len(indizes_first) == 0:
        return False

    # Check for remaining words
    # single word is easy
    if len(list1) == 1:
        return True

    # multi word
    remaining1 = list1[1:]
    for idx in indizes_first:
        if idx + len(remaining1) >= len(list2):
            continue

        assume_result = True
        for item_to_check in remaining1:
            idx += 1
            if item_to_check != list2[idx]:
                assume_result = False
                break

        if assume_result == True:
            return True

    return False

def _convert_snli_out(samples, out_name):
    if out_name.split('.')[-1] != 'jsonl':
        out_name += '.jsonl'

    return (
        out_name, 
        [json.dumps({ 'sentence1' : ' '.join(p), 'sentence2' : ' '.join(h), 'gold_label' : lbl }) for p, h, lbl in samples]
    )


def _load_snli(lines, valid_labels=DEFAULT_VALID_LABELS):
    '''
    Extract each line into a (string) sample from snli format.
    :return [(premise, hypothesis, label)]
    '''

    def extract_snli_line(line):
        parsed_data = json.loads(line)
        return (_tokenize(parsed_data['sentence1']), _tokenize(parsed_data['sentence2']), parsed_data['gold_label'])
         
    samples = [extract_snli_line(line) for line in lines]
    if valid_labels == None:
        return samples
    else:
        return [(p, h, lbl) for (p, h, lbl) in samples if lbl in valid_labels]


def _tokenize(sent):
    doc = nlp(sent,  parse=False, tag=False, entity=False)
    return [token.text for token in doc]

# Classes
class SentEncoderDataset(Dataset):
    '''
    Dataset format to give to classifier
    '''

    def __init__(self, samples, embedding_holder, tag_to_idx):
        '''
        Create a new dataset for the given samples
        :param samples              parsed samples of the form [(premise, hypothesis, label)] (all strings)
        :paraam embedding_holder    To map from word to number
        :param tag_to_idx         dictionary mapping the string label to a number
        '''
        
        self.converted_samples = [(
            torch.LongTensor([embedding_holder.word_index(w) for w in p]),
            torch.LongTensor([embedding_holder.word_index(w) for w in h]),
            tag_to_idx[lbl]
        ) for (p, h, lbl) in samples]

    def __len__(self):
        return len(self.converted_samples)

    def __getitem__(self, idx):
        return self.converted_samples[idx]

class ExtResPairhandler:
    '''
    Manages pairs of words coming from an external resource
    '''

    def __init__(self, path, data_format=DEFAULT_DATA_FORMAT):
        self.data_format = data_format

        with open(path) as f_in:
            if data_format == 'snli':
                self.knowledge = _load_snli(f_in.readlines())
            else:
                print('Unknown data format:', data_format)
                1/0

    def filter(self, data_handler, min_count=5, same_label=True):
        '''
        Filters the pairs s.t. only those remain if at least <min_count> samples contain
        the first word within the premise and the 2nd word within the hypothesis.

        :param data_handler         Datahandler with the data to check
        :param min_count            At least this many sample are reuired containing the words
        :param same_label           Only count samples that are labeld as the resource pair.
        '''

        def count_in_data(p, h, lbl):
            count = 0
            for p_data, h_data, lbl_data in data_handler.samples:
                if same_label and lbl_data != lbl:
                    continue
                p_in_data = _is_sublist(p, p_data)
                
                if not p_in_data:
                    continue

                if _is_sublist(h, h_data):
                    count += 1

            return count


        self.knowledge = [(p, h, lbl) for (p, h, lbl) in self.knowledge if count_in_data(p, h, lbl) >= min_count]

    def save(self, out_name):
        if self.data_format == 'snli':
            name, lines = _convert_snli_out(self.knowledge, out_name)

        with open(name, 'w') as f_out:
            f_out.write('\n'.join(lines))



class Datahandler:
    '''
    Loads data.
    '''

    def __init__(self, path, data_format=DEFAULT_DATA_FORMAT, valid_labels=DEFAULT_VALID_LABELS):
        '''
        Create a Datahandler for the data at the given path

        :param path         Path of data file
        :param data_format  format of data ('snli')
        '''

        self.valid_labels = valid_labels
        self.tag_to_idx = dict([(label, i) for i, label in enumerate(valid_labels)])
        self.data_format = data_format

        with open(path) as f_in:
            if data_format == 'snli':
                self.samples = _load_snli(f_in.readlines())
            else:
                print('Unknown data format:', data_format)
                1/0

    def get_dataset(self, embedding_holder):
        return SentEncoderDataset(self.samples, embedding_holder, self.tag_to_idx)

    def merge(self, data_handlers):
        '''
        Merge all other data_handlers into this datahandler.
        :param data_handlers       list of Datahandler that get merged into this one
        '''

        for dh in data_handlers:
            self.samples.extend(dh.samples)


# External Helper functions

def get_datahandler_train(path=None):
    if path == None:
        path = config.PATH_TRAIN_DATA
    return Datahandler(path)

def get_datahandler_dev(path=None):
    if path == None:
        path = config.PATH_DEV_DATA
    return Datahandler(path)