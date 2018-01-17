'''
Methods to deal with the data
'''

import json
import collections

import torch
from torch.utils.data import Dataset

import spacy
nlp = spacy.load('en')

from libs import config

DEFAULT_DATA_FORMAT = 'snli'
DEFAULT_VALID_LABELS = ['neutral', 'contradiction', 'entailment']

# Internal Helper functions
def _convert_snli_out(samples, out_name):
    if out_name.split('.')[-1] != 'jsonl':
        out_name += '.jsonl'

    return (
        out_name, 
        [json.dumps({ 'sentence1' : ' '.join(p), 'sentence2' : ' '.join(h), 'gold_label' : lbl }) for p, h, lbl in samples]
    )

def _load_txt_01_cn(lines):
    def extract_line(line):
        splitted = line.split()
        if splitted[-1] == "1":
            lbl = 'neutral'
        elif splitted[-1] == "0":
            lbl = 'contradiction'
        else:
            print('unknown label', splitted[-1])
            1/0
        return (splitted[0], splitted[1], lbl)

    return [extract_line(line.strip()) for line in lines]

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
    Manages pairs of words coming from an external resource, Can only deal with single words.
    '''

    def __init__(self, path, data_format=DEFAULT_DATA_FORMAT):
        self.data_format = data_format

        with open(path) as f_in:
            if data_format == 'snli':
                data = _load_snli(f_in.readlines())
                data = [(p[0], h[0], lbl) for p, h, lbl in data]
            elif data_format == 'txt_01_cn':
                data = _load_txt_01_cn(f_in.readlines())
            else:
                print('Unknown data format:', data_format)
                1/0

        self.knowledge = self.create_knowledge_dict(data)
        
    def add_to_knowledge_dict(self, knowledge_dict, sample):
        wp, wh, lbl = sample
        if lbl not in knowledge_dict:
            knowledge_dict[lbl] = dict()
        if wp not in knowledge_dict[lbl]:
            knowledge_dict[lbl][wp] = set([wh])
        else:
            knowledge_dict[lbl][wp].add(wh)

    def create_knowledge_dict(self, data):
        knowledge = dict()
        for wp, wh, lbl in data:
            self.add_to_knowledge_dict(knowledge, (wp, wh, lbl))
        return knowledge

    def add(self, samples):
        '''
        Adds samples to the dictionary.
        :param samples          list of samples to add [(w1, w2, lbl) ...]
        '''
        for sample in samples:
            self.add_to_knowledge_dict(self.knowledge, sample)

    def remove(self, p, h, lbl):
        '''
        Remove this sample from the knowledge.
        '''

        # delete pair
        self.knowledge[lbl][p].remove(h)

        # check if premise still has hypothesis
        if len(self.knowledge[lbl][p]) == 0:
            del self.knowledge[lbl][p]

        # check if label still has samples
        if len(self.knowledge[lbl]) == 0:
            del self.knowledge[lbl]


    def extend_from_own(self, extend_fn):
        '''
        Extend this resource by iterating over all samples and adding new samples generated by the given function.
        :param extend_fn        function(premise, hypothesis, label) returns (premise, hypothesis, label)
        '''

        new_samples = []

        for label in self.knowledge:
            c_knowledge = self.knowledge[label]
            for p in c_knowledge:
                c_p_knowledge = c_knowledge[p]
                new_samples.extend([extend_fn(p, h, label) for h in c_p_knowledge])

        print('Adding', len(new_samples), 'new samples')
        for sample in new_samples:
            self.add_to_knowledge_dict(self.knowledge, sample)


    def filter_vocab(self, vocab):
        '''
        Only keeps samples if both words are contained in the given vocab
        :param vocab    list of vocabulary
        '''
        print('Filter by vocab using', len(vocab), 'vocabularies.')
        data = []
        for label in self.knowledge:
            all_w_p = self.knowledge[label]
            for wp in all_w_p:
                if wp in vocab:
                    all_w_h = all_w_p[wp]
                    data.extend([(wp, wh, label) for wh in all_w_h if wh in vocab])

        print('Previously:', self.__len__(), 'samples.')
        self.knowledge = self.create_knowledge_dict(data)
        print('Now:', self.__len__(), 'samples.')

    def items(self):
        '''
        :return [(premise, hypothesis, label), ...] for all samples stored.
        '''
        samples = []
        for label in self.knowledge:
            all_w_p = self.knowledge[label]
            for wp in all_w_p:
                all_w_h = all_w_p[wp]
                samples.extend([(wp, wh, label) for wh in all_w_h])

        return samples

    def save(self, out_name, data_format=None):
        if data_format == None:
            data_format = self.data_format

        if data_format == 'snli':
            data = [([p], [h], lbl) for lbl in self.knowledge for p in self.knowledge[lbl] for h in self.knowledge[lbl][p]]
            name, lines = _convert_snli_out(data, out_name)

        with open(name, 'w') as f_out:
            f_out.write('\n'.join(lines))


    def __len__(self):
        count = 0
        for label in self.knowledge:
            for wp in self.knowledge[label]:
                count += len(self.knowledge[label][wp])
        return count

    def count(self, label):
        '''Count the amount of samples for the givn label.'''
        if label not in self.knowledge:
            return 0
        count = 0

        for wp in self.knowledge[label]:
            count += len(self.knowledge[label][wp])
        return count

    def get_label_counts(self):
        '''
        Return an array of all counts per label.
        :return     [(label-name, count), ...]
        '''

        return [(label, self.count(label)) for label in self.knowledge]



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