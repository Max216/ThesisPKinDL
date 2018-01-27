'''
To access data for training/prediction
'''

import collections
import torch

import torch
from torch.utils.data import Dataset

from libs import config, data_tools

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

    def __init__(self, path, data_format=data_tools.DEFAULT_DATA_FORMAT, valid_labels=data_tools.DEFAULT_VALID_LABELS):
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
                self.samples = data_tools._load_snli(f_in.readlines())
            elif data_format == 'snli_nltk':
                self.samples = data_tools._load_snli_nltk(f_in.readlines())
            else:
                print('Unknown data format:', data_format)
                1/0

    def get_dataset(self, embedding_holder):
        return SentEncoderDataset(self.samples, embedding_holder, self.tag_to_idx)

    def get_samples(self, indizes):
        '''
        get the samples having these indizes
        '''
        return [self.samples[i] for i in indizes]

    def merge(self, data_handlers):
        '''
        Merge all other data_handlers into this datahandler.
        :param data_handlers       list of Datahandler that get merged into this one
        '''

        for dh in data_handlers:
            self.samples.extend(dh.samples)

    def vocab(self):
        '''
        Get a list of all vocabularies usied in the dataset.
        :return [word1, ...]
        '''

        combined_premise_hyp = [premise + hyp for premise, hyp, _ in self.samples]
        return set([w for p_h in combined_premise_hyp for w in p_h])


    def create_word_cnt(self, file_out):
        counter = collections.defaultdict(int)
        for p, h, _ in self.samples:
            for w in p:
                counter[w] += 1
            for w in h:
                counter[w] += 1

        with open(file_out, 'w') as f_out:
            lines = [w + ' ' + str(counter[w]) for w in counter]
            f_out.write('\n'.join(lines))

    def get_word_counter(self, file_in):
        with open(file_in) as f_in:
            lines = [line.strip() for line in f_in.readlines()]

        counter = collections.defaultdict(int)
        for line in lines:
            splitted = line.split(' ')
            counter[splitted[0]] = int(splitted[1])

        return counter



# External Helper functions

def get_datahandler_train(path=None):
    if path == None:
        path = config.PATH_TRAIN_DATA
    print('use the following training data:', path)
    return Datahandler(path)

def get_datahandler_dev(path=None):
    if path == None:
        path = config.PATH_DEV_DATA
    return Datahandler(path)

def get_dataset(samples, embedding_holder, tag_to_idx):
    return SentEncoderDataset(self.samples, embedding_holder, self.tag_to_idx)