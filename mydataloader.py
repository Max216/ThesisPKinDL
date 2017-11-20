import nltk
from nltk import word_tokenize
import json
import torch
from torch.utils.data import Dataset

tag_to_index = dict()
tag_to_index['neutral'] = 0
tag_to_index['contradiction'] = 1
tag_to_index['entailment'] = 2

def extract_snli(raw_instance):
    """
    @param raw_instance - from the official SNLI dataset the .jsonl file format
    
    Extracts the required information of a raw single instance of the SNLI dataset.
    This will result in a triplet like of the following form:
    
    Applied preprocessing:
    - lowecase
    - split()
    
    ([sent1], [sent2], label)
    """
    
    parsed_data = json.loads(raw_instance)
    return (word_tokenize(parsed_data['sentence1'].lower()), word_tokenize(parsed_data['sentence2'].lower()), parsed_data['gold_label'])

def load_snli(path, valid_labels=['neutral','contradiction','entailment']):
    """
    @param path - Path to the *.jsonl file of the SNLI dataset
    @param valid_labels -  only samples with one of these labels will be considered.

    Load instances of the SNLI dataset into an array.
    """
    with open(path) as file:
        all_lines = [extract_snli(line) for line in open(path)]

    return [(p, h, lbl) for (p, h, lbl) in all_lines if lbl in valid_labels]


class SNLIDataset(Dataset):
    '''
    Load the SNLI dataset
    @param samples - loaded raw samples (premise, hypothesis, label)
    '''
    def __init__(self, samples, embedding_holder):
        self.converted_samples = [(
                torch.LongTensor([embedding_holder.word_index(w) for w in s1]),
                torch.LongTensor([embedding_holder.word_index(w) for w in s2]),
                tag_to_index[lbl]
            ) for (s1, s2, lbl) in samples]
        
    def __len__(self):
        return len(self.converted_samples)
    
    def __getitem__(self, idx):
        return self.converted_samples[idx]

def get_dataset_chunks(filepath, embedding_holder, chunk_size=10000, mark_as=''):
    '''
    Divides the data into several chunks with the premise having approximately the same size of all examples
    to reduce padding.

    @param filepath - path to data to be divided
    @param chunk_size - each resulting chunk will have this size (or less, if not enough data left)
    @param mark_as - just for output message to specify thee data loaded
    '''

    # sort by length
    raw_samples = load_snli(filepath) 
    print('Loaded', len(raw_samples),'samples with valid labels.', mark_as)
    raw_samples = sorted(raw_samples, key=lambda sample: len(sample[0])) # 0 is premise

    return [SNLIDataset(raw_samples[i:i + chunk_size], embedding_holder) for i in range(0, len(raw_samples), chunk_size)]