'''
Methods to deal with the data
'''

import random
import json
import collections

import torch
from torch.utils.data import Dataset

import spacy
import nltk
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

def _load_snli_nltk(lines, valid_labels=DEFAULT_VALID_LABELS):
    def extract_snli_line(line):
        parsed_data = json.loads(line)
        return (nltk.word_tokenize(parsed_data['sentence1']), nltk.word_tokenize(parsed_data['sentence2']), parsed_data['gold_label'])
         
    samples = [extract_snli_line(line) for line in lines]
    if valid_labels == None:
        return samples
    else:
        return [(p, h, lbl) for (p, h, lbl) in samples if lbl in valid_labels]


def _tokenize(sent):
    doc = nlp(sent,  parse=False, tag=False, entity=False)
    return [token.text for token in doc]
#def _tokenize(sent):
#    return nltk.word_tokenize(sent)

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

class ExtResPairData:
    '''Contains information to all samples of a train set. Original samples as well as adversarial samples'''

    def __init__(self, w1, w2, label, valid_labels):
        self.w1 = w1
        self.w2 = w2
        self.label = label
        self.valid_labels = valid_labels
        self.samples = dict([(l, []) for l in valid_labels])
        self.adversarial_samples_w1_premise = []
        self.adversarial_samples_w1_hyp = []
        self.adversarial_samples_w2_premise = []
        self.adversarial_samples_w2_hyp = []
        self.adversarial_hashes = set()
        self.cnt_w1 = 0
        self.cnt_w2 = 0

    def _replace_word(self, sent, word_to_replace, replacing_word):
        return [w if w != word_to_replace else replacing_word for w in sent]

    def absorb_if_matches(self, index, sample):
        '''
        Adds this sample and/or creates adversarial samples, if it matches the criteria for this pair. 
        '''
        p, h, slbl = sample

        # make sure to only add once
        added_adversarials = set()

        if self.w1 in p:
            self.cnt_w1 += 1
            # is "natural" sample
            if self.w2 in h:
                self.samples[slbl].append(index)

            # add adversarial samples
            sample_hash = hash('_'.join(p) + '#' + '_'.join(self._replace_word(p, self.w1, self.w2)))
            if sample_hash not in self.adversarial_hashes:
                self.adversarial_samples_w1_premise.append(index)
                self.adversarial_hashes.add(sample_hash)

        if self.w2 in p:
            self.cnt_w2 += 1
            sample_hash = hash('_'.join(self._replace_word(p, self.w2, self.w1)) + '#' + '_'.join(p))
            if sample_hash not in self.adversarial_hashes:
                self.adversarial_samples_w2_premise.append(index)
                self.adversarial_hashes.add(sample_hash)

        if self.w1 in h:
            self.cnt_w1 += 1
            sample_hash = hash('_'.join(h) + '#' + '_'.join(self._replace_word(h, self.w1, self.w2)))
            if sample_hash not in self.adversarial_hashes:
                self.adversarial_samples_w1_hyp.append(index)
                self.adversarial_hashes.add(sample_hash)

        if self.w2 in h:
            self.cnt_w2 += 1
            sample_hash = hash('_'.join(self._replace_word(h, self.w2, self.w1)) + '#' + '_'.join(h))
            if sample_hash not in self.adversarial_hashes:
                self.adversarial_samples_w2_hyp.append(index)
                self.adversarial_hashes.add(sample_hash)

    def get_natural_dataset(self, datahandler, embedding_holder, gold_label):
        '''
        Get the dataset containing all natural samples for the given label
        '''
        sample_set = self.get_natural_samples(datahandler, gold_label, amount=-1)
        return SentEncoderDataset(sample_set, embedding_holder, datahandler.tag_to_idx)

    def get_dataset_sample_idx_w1_replaced(self):
        '''
        return all samples in a dataset that have been created by replacing w1
        :return [(idx_in_dataset, sample_idx)]
        '''
        len_w1_replaced = len(self.adversarial_samples_w1_premise) + len(self.adversarial_samples_w1_hyp)
        len_w2_replaced = len(self.adversarial_samples_w2_premise) + len(self.adversarial_samples_w2_hyp)

        db_indizes = [len_w2_replaced + i for i in range(len_w1_replaced)]
        sample_indizes = self.adversarial_samples_w1_premise + self.adversarial_samples_w1_hyp
        return [(db_indizes[i], sample_indizes[i]) for i in range(len(db_indizes))]

    def get_dataset_sample_idx_w2_replaced(self):
        '''
        return all samples in a dataset that have been created by replacing w2
        :return [(idx_in_dataset, sample_idx)]
        '''
        sample_indizes = self.adversarial_samples_w2_hyp + self.adversarial_samples_w2_premise
        return [(i, sample_indizes[i]) for i in range(len(sample_indizes))]

    def get_dataset_sample_idx_any_replaced(self):
        sample_indizes = self.adversarial_samples_w2_hyp + self.adversarial_samples_w2_premise + self.adversarial_samples_w1_hyp + self.adversarial_samples_w1_premise
        return [(i, sample_indizes[i]) for i in range(len(sample_indizes))]

    def get_adversarial_dataset(self, datahandler, embedding_holder):
        '''
        Get the dataset containing all adversarial samples
        '''
        sample_set = self.get_adversarial_samples(datahandler, amount=-1)
        return SentEncoderDataset(sample_set, embedding_holder, datahandler.tag_to_idx)

    def get_natural_samples(self, datahandler, gold_label, amount=-1):
        '''
        return readable samples as they occur in the data
        :param amount       max amount to return
        :param gold_label   only look for samples having this label
        :param datahandler  must be the same data as on generation!
        '''
        if gold_label not in self.samples:
            return []

        if amount == -1:
            return datahandler.get_samples(self.samples[gold_label])
        return datahandler.get_samples(self.samples[gold_label])[:10]

    def get_adversarial_samples_for(self, sample_indizes, datahandler):
        results = []
        for index in sample_indizes:

            sent_to_use = None
            replace_w1 = None
            p, h, lbl = datahandler.samples[index]

            if index in self.adversarial_samples_w2_hyp:
                replace_w1 = False
                sent_to_use = h

            elif index in self.adversarial_samples_w2_premise:
                replace_w1 = False
                sent_to_use = p

            elif index in self.adversarial_samples_w1_hyp:
                replace_w1 = True
                sent_to_use = h

            elif index in self.adversarial_samples_w1_premise:
                replace_w1 = True
                sent_to_use = p

            else:
                print('Should not happen:', index)
                1/0

            
            if replace_w1:
                w_replaced = self.w1
                w_replacer = self.w2

            else:
                w_replaced = self.w2
                w_replacer = self.w1

            generated_sentence = sent_to_use[:].replace(w_replaced, w_replacer)

            if replace_w1:
                results.append((sent_to_use, generated_sentence))
            else:
                results.append((generated_sentence, sent_to_use))

        return results

    def get_adversarial_samples(self, datahandler, amount=-1):
        '''
        return readable adversarial samples from data.
        :param amount       max amount to return
        :param datahandler  must be the same data as on generation!
        '''
        W2_HYP = 0
        W1_HYP = 1
        W2_PREM = 2
        W1_PREM = 3

        adv_w2_h = [(W2_HYP, idx) for idx in self.adversarial_samples_w2_hyp]
        adv_w2_p = [(W2_PREM, idx) for idx in self.adversarial_samples_w2_premise]
        adv_w1_h = [(W1_HYP, idx) for idx in self.adversarial_samples_w1_hyp]
        adv_w1_p = [(W1_PREM, idx) for idx in self.adversarial_samples_w1_premise]


        all_adv = adv_w2_h + adv_w2_p + adv_w1_h + adv_w1_p
        if len(all_adv) <= amount or amount == -1:
            request_samples = all_adv
        else:
            print('using random sample!')
            random.seed(1)
            request_samples = random.sample(all_adv, amount)

        # now create samples
        indizes = [idx for _, idx in request_samples]
        samples = datahandler.get_samples(indizes)

        results = []
        for i, (p, h, _) in enumerate(samples):
            typ = request_samples[i][0]
            if typ == W2_HYP:
                results.append((self._replace_word(h, self.w2, self.w1), h, self.label))
            elif typ == W1_HYP:
                results.append((h, self._replace_word(h, self.w1, self.w2), self.label))
            elif typ == W1_PREM:
                results.append((p, self._replace_word(p, self.w1, self.w2), self.label))
            elif typ == W2_PREM:
                results.append((self._replace_word(p, self.w2, self.w1), p, self.label))
            else:
                print('Should not happen')
                1/0

        return results







class ExtResPairhandler:
    '''
    Manages pairs of words coming from an external resource, Can only deal with single words.
    '''

    def __init__(self, path=None, data_format=DEFAULT_DATA_FORMAT):
        self.data_format = data_format

        if path != None:
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
        else:
            self.knowledge = self.create_knowledge_dict([])

    def init_with_samples(self, samples):
        '''
        Initialize with samples. This will override all existing knowledge.
        '''
        self.knowledge = self.create_knowledge_dict(samples)
        return self
        
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
        print('before', self.items())
        new_samples = []

        for label in self.knowledge:
            c_knowledge = self.knowledge[label]
            for p in c_knowledge:
                c_p_knowledge = c_knowledge[p]
                new_samples.extend([extend_fn(p, h, label) for h in c_p_knowledge])

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

        self.knowledge = self.create_knowledge_dict(data)

    def filter_data(self, datahandler, keep_order=True, req_label=None):
        '''
        Ony keep samples if there is at least one example in the given data, where one word
        appears in the premise and the other word appears in the hypothsis.
        :param datahandler      data to check in
        :param keep_order       if true, the first word must be in the premise and the 2nd in the hypothesis.
                                if false it can also be the other way around
        :param req_label        if set, only look at samples in data with the specified label
        '''

        if req_label != None:
            data = [(p, h, lbl) for p, h, lbl in datahandler.samples if lbl == req_label]
        else:
            data = datahandler.samples

        knowledge_samples_to_check = self.items()
        valid_samples = []
        for p, h, lbl in data:
            checked = []
            for i, (pk, hk, lblk) in enumerate(knowledge_samples_to_check):
                added = False
                if pk in p and hk in h:
                    added = True
                    valid_samples.append((pk, hk, lblk))
                elif keep_order == False and pk in h and hk in p:
                    added = True
                    valid_samples.append((pk, hk, lblk))

                if added:
                    checked.append(i)

            if len(checked) > 0:
                knowledge_samples_to_check = [s for i, s in enumerate(knowledge_samples_to_check) if i not in checked]

            if len(knowledge_samples_to_check) == 0:
                break

        self.knowledge = self.create_knowledge_dict(valid_samples)


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

    def find_samples(self, datahandler):
        '''
        Return a list of each knowledge sample with data indizes for all samples
        '''

        all_knowledge_items = [ExtResPairData(w1, w2, label, valid_labels=datahandler.valid_labels) for w1, w2, label in self.items()]

        for i, sample in enumerate(datahandler.samples):
            for knowledge_item in all_knowledge_items:
                knowledge_item.absorb_if_matches(i, sample)

        return all_knowledge_items




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
            elif data_format == 'snli_nltk':
                self.samples = _load_snli_nltk(f_in.readlines())
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