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

        with open(path) as f_in:
            if data_format == 'snli':
                self.samples = self._load_snli(f_in.readlines())
            else:
                print('Unknown data format:', data_format)
                1/0

    def get_dataset(self, embedding_holder):
        return SentEncoderDataset(self.samples, embedding_holder, self.tag_to_idx)

    def _load_snli(self, lines):

        def extract_snli_line(line):
            parsed_data = json.loads(line)
            return (self._tokenize(parsed_data['sentence1']), self._tokenize(parsed_data['sentence2']), parsed_data['gold_label'])
             
        samples = [extract_snli_line(line) for line in lines]
        return [(p, h, lbl) for (p, h, lbl) in samples if lbl in self.valid_labels]

    def _tokenize(self, sent):
        doc = nlp(sent,  parse=False, tag=False, entity=False)
        return [token.text for token in doc]

def get_datahandler_train(path=None):
    if path == None:
        path = config.PATH_TRAIN_DATA
    return Datahandler(path)

def get_datahandler_dev(path=None):
    if path == None:
        path = config.PATH_DEV_DATA
    return Datahandler(path)