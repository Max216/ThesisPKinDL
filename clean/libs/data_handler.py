'''
To access data for training/prediction
'''

import collections
import torch

import torch
from torch.utils.data import Dataset

from libs import config, data_tools, embeddingholder


class SentEncoderDatasetIncludingSents(Dataset):
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
            tag_to_idx[lbl],
            len_p,
            len_h,
            p,
            h
        ) for (p, h, lbl, len_p, len_h) in samples]

    def __len__(self):
        return len(self.converted_samples)

    def __getitem__(self, idx):
        return self.converted_samples[idx]

class SentEncoderDatasetIncludingSentsAndReplacements(Dataset):
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
            tag_to_idx[lbl],
            len_p,
            len_h,
            p,
            h,
            rep1,
            rep2
        ) for (p, h, lbl, len_p, len_h, rep1, rep2) in samples]

    def __len__(self):
        return len(self.converted_samples)

    def __getitem__(self, idx):
        return self.converted_samples[idx]

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
            tag_to_idx[lbl],
            len_p,
            len_h
        ) for (p, h, lbl, len_p, len_h) in samples]

    def __len__(self):
        return len(self.converted_samples)

    def __getitem__(self, idx):
        return self.converted_samples[idx]


class SentEncoderIdDataset(Dataset):
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
            tag_to_idx[lbl],
            len_p,
            len_h,
            p_id,
            h_id
        ) for (p, h, lbl, len_p, len_h, p_id, h_id) in samples]

    def __len__(self):
        return len(self.converted_samples)

    def __getitem__(self, idx):
        return self.converted_samples[idx]


class Datahandler:
    '''
    Loads data.
    '''

    def __init__(self, path, data_format=data_tools.DEFAULT_DATA_FORMAT, valid_labels=data_tools.DEFAULT_VALID_LABELS, include_start_end_token=True, lower=None, sort=True):
        '''
        Create a Datahandler for the data at the given path

        :param path         Path of data file
        :param data_format  format of data ('snli')
        '''

        self.valid_labels = valid_labels
        self.tag_to_idx = dict([(label, i) for i, label in enumerate(valid_labels)])
        self.data_format = data_format
        self.has_categories = False

        with open(path) as f_in:
            if data_format == 'snli':
                self.samples = data_tools._load_snli(f_in.readlines())
            elif data_format == 'snli_nltk':
                self.samples = data_tools._load_snli_nltk(f_in.readlines())
            elif data_format == 'snli_adversarial':
                self.samples = data_tools._load_snli_adversarial(f_in.readlines())
                self.has_categories = True
            elif data_format == 'snli_adversarial_incl_replacements':
                self.has_categories = True
                self.samples = data_tools._load_snli_adversarials_including_replacement(f_in.readlines())
            else:
                print('Unknown data format:', data_format)
                1/0

        
        if include_start_end_token:
            print('Including start/stop symbols')
            samples = []
            for s in self.samples:

                if lower == 'lower':
                    use_p = [w.lower() for w in s[0]]
                    use_h = [w.lower() for w in s[1]]
                    #print('lowerd h', use_h)
                else:
                    use_p = s[0]
                    use_h = s[1]

                p = [embeddingholder.START_SENT] + use_p + [embeddingholder.END_SENT]
                h = [embeddingholder.START_SENT] + use_h + [embeddingholder.END_SENT]
                p_len = s[3] + 2
                h_len = s[4] + 2

                if data_format == 'snli_adversarial':
                    samples.append((p,h,s[2], p_len, h_len, s[5]))
                elif data_format == 'snli_adversarial_incl_replacements':
                    samples.append((p,h,s[2], p_len, h_len, s[5], s[6], s[7]))
                else:# data_format != 'snli_adversarial':
                    samples.append((p,h,s[2], p_len, h_len))
                #else:
                #    samples.append((p,h,s[2], p_len, h_len, s[5]))

            self.samples = samples


        # sort by premise length
        if sort:
            print('sorting')
            self.samples = sorted(self.samples, key=lambda x: x[3])
        else:
            print('non sorting')

    def get_sentences(self):
        used_keys = set()
        unique_sents = []
        for s1, s2, lbl, s1_len, s2_len in self.samples:
            for s in [s1, s2]:
                key = ''.join(s)
                if key not in used_keys:
                    used_keys.add(key)
                    unique_sents.append(s)
        return unique_sents

    def get_dataset_for_category(self, embedding_holder, category):
        curent_samples = [(p, h, lbl, p_len, h_len) for p, h, lbl, p_len, h_len, cat in self.samples if cat == category]
        return SentEncoderDataset(curent_samples, embedding_holder, self.tag_to_idx)

    def get_dataset_for_category_including_sents(self, embedding_holder, category):
        curent_samples = [(p, h, lbl, p_len, h_len) for p, h, lbl, p_len, h_len, cat in self.samples if cat == category]
        return SentEncoderDatasetIncludingSents(curent_samples, embedding_holder, self.tag_to_idx)

    def get_dataset_for_category_including_sents_and_replacement(self, embedding_holder, category):
        curent_samples = [(p, h, lbl, p_len, h_len, rep1, rep2) for p, h, lbl, p_len, h_len, rep1, rep2, cat in self.samples if cat == category]
        return SentEncoderDatasetIncludingSentsAndReplacements(curent_samples, embedding_holder, self.tag_to_idx)

    def get_samples_for_category(self, category):
        return [(p, h, lbl, p_len, h_len) for p, h, lbl, p_len, h_len, cat in self.samples if cat == category]

    def get_categories(self):
        if not self.has_categories:
        #if len(self.samples[0]) != 6:
            print('No categories')
            1/0
    
        return list(set([s[-1] for s in self.samples]))

    def get_dataset(self, embedding_holder):
        '''
        Get a dataset including all samples
        '''
        if len(self.samples[0]) != 5:
            current_samples = [(p, h, lbl, p_len, h_len) for p, h, lbl, p_len, h_len, cat in self.samples]
        else:
            current_samples = self.samples
        return SentEncoderDataset(current_samples, embedding_holder, self.tag_to_idx)

    def get_dataset_including_sents(self, embedding_holder):
        if len(self.samples[0]) != 5:
            current_samples = [(p, h, lbl, p_len, h_len) for p, h, lbl, p_len, h_len, cat in self.samples]
        else:
            current_samples = self.samples
        return SentEncoderDatasetIncludingSents(current_samples, embedding_holder, self.tag_to_idx)

    def get_dataset_id(self, embedding_holder, start_id=0):
        if len(self.samples[0]) != 5:
            current_samples = [(p, h, lbl, p_len, h_len) for p, h, lbl, p_len, h_len, cat in self.samples]
        else:
            current_samples = self.samples

        print('total samples:', len(current_samples))
        id_samples = []
        seen_sents = dict()

        next_id = start_id
        for p, h, lbl, p_len, h_len in current_samples:
            p_key = '__'.join(p)
            h_key = '__'.join(h)

            if p_key not in seen_sents:
                seen_sents[p_key] = next_id
                next_id += 1
            if h_key not in seen_sents:
                seen_sents[h_key] = next_id
                next_id += 1

            #print('After next sample:', next_id)
            id_samples.append((p, h, lbl, p_len, h_len, seen_sents[p_key], seen_sents[h_key]))

        return SentEncoderIdDataset(id_samples, embedding_holder, self.tag_to_idx), next_id


    def get_dataset_splits(self, embedding_holder, split_size=16000):
        splits = []
        start_idx = 0
        while start_idx < len(self.samples):
            print('remain samples to split:', len(self.samples[start_idx:]))
            if len(self.samples[start_idx:]) < split_size:
                print('>> use all')
                splits.append(SentEncoderDataset(self.samples[start_idx:], embedding_holder, self.tag_to_idx))
            else:
                print('use subset')
                splits.append(SentEncoderDataset(self.samples[start_idx:start_idx + split_size], embedding_holder, self.tag_to_idx))

            start_idx += split_size

        return splits

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

        combined_premise_hyp = [premise + hyp for premise, hyp, _, _2,_3 in self.samples]
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

def get_datahandler_train(path=None, lower=None):
    if path == None:
        path = config.PATH_TRAIN_DATA
    print('use the following training data:', path)
    return Datahandler(path, lower=lower)

def get_datahandler_dev(path=None, lower=None, sort=True):
    if path == None:
        path = config.PATH_DEV_DATA
    return Datahandler(path, lower=lower, sort=sort)

def get_dataset(samples, embedding_holder, tag_to_idx):
    return SentEncoderDataset(self.samples, embedding_holder, self.tag_to_idx, lower)