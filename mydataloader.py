import nltk
from nltk import word_tokenize
import json
import torch
from torch.utils.data import Dataset

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

def load_snli(path, size=-1):
    """
    @param path - Path to the *.jsonl file of the SNLI dataset
    @param size - Amount of examples that are loaded. If not specified or size=-1 all instances are loaded.
    
    Load instances of the SNLI dataset into an array.
    """

    result = []
    cnt=0
    file = open(path, "r")
    for line in file:
        s1,s2,lbl = extract_snli(line)
        if lbl in ['neutral','contradiction','entailment']:
            result.append((s1,s2,lbl))
            cnt += 1
        if size != -1 and cnt >= size:
            break
        
    return result
    
def tag_to_index(tag):
    """
    Maps a label as a string into an integer number.
    """
 
    if tag == 'neutral':
        return 0
    elif tag == 'contradiction':
        return 1
    elif tag == 'entailment':
        return 2
    else:
        print(tag)
        raise Exception('invalid tag', tag)
        



class SNLIDataset(Dataset):
    '''
    Load the SNLI dataset
    @param file - file-path to samples
    @param max_size (optional) - limit to a maximum size
    '''
    def __init__(self, file, embedding_holder, max_size=-1):
        self.converted_samples = []
        
        raw_samples = load_snli(file, max_size)
        # to tensors
        for i in range(len(raw_samples)):
            s1, s2, lbl = raw_samples[i]
            self.converted_samples.append((
                torch.LongTensor([embedding_holder.word_index(w) for w in s1]),
                torch.LongTensor([embedding_holder.word_index(w) for w in s2]),
                tag_to_index(lbl)
            ))
        
    def __len__(self):
        return len(self.converted_samples)
    
    def __getitem__(self, idx):
        return self.converted_samples[idx]