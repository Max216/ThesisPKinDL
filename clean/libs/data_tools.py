'''
Methods to deal with the data
'''

import json

import spacy
import nltk

from libs import config

if not config.ONLY_TEST:
    nlp = spacy.load('en')

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

def _load_snli(lines, valid_labels=DEFAULT_VALID_LABELS, tokenize=True, include_lengths=True):
    '''
    Extract each line into a (string) sample from snli format.
    :param valid_labels         Only load samples with labels specified here
    :param tokenize             Only tokenize if specified (default=yes)
    :return [(premise, hypothesis, label)]
    '''

    def extract_no_tokenize(line):
        parsed_data = json.loads(line)
        return (parsed_data['sentence1'], parsed_data['sentence2'], parsed_data['gold_label'])

    def extract_snli_line(line):
        parsed_data = json.loads(line)
        return (_tokenize(parsed_data['sentence1']), _tokenize(parsed_data['sentence2']), parsed_data['gold_label'])

    if tokenize:
        extract_fn = extract_snli_line
    else:
        extract_fn = extract_no_tokenize
         
    samples = [extract_fn(line) for line in lines]

    # ensure valid labels
    if valid_labels != None:
        samples = [(p, h, lbl) for (p, h, lbl) in samples if lbl in valid_labels]

    # include sentence length
    if include_lengths:
        samples = [(p, h, lbl, len(p), len(h)) for p, h, lbl in samples]

    return samples

def _load_snli_adversarial(lines, valid_labels=DEFAULT_VALID_LABELS):
    parsed_data = [json.loads(line) for line in lines]
    data = [(_tokenize(pd['sentence1']), _tokenize(pd['sentence2']), pd['gold_label'], pd['category']) for pd in parsed_data]
    return [(p, h, lbl, len(p), len(h), cat) for p,h,lbl,cat in data]

def _load_snli_adversarials_including_replacement(lines, valid_labels=DEFAULT_VALID_LABELS):
    parsed_data = [json.loads(line) for line in lines]
    data = [(_tokenize(pd['sentence1']), _tokenize(pd['sentence2']), pd['gold_label'], pd['category'], pd['replaced1'], pd['replaced2']) for pd in parsed_data]
    return [(p, h, lbl, len(p), len(h), rep1, rep2, cat) for p,h,lbl,cat,rep1, rep2 in data]

def _load_snli_nltk(lines, valid_labels=DEFAULT_VALID_LABELS, include_lengths=True):
    def extract_snli_line(line):
        parsed_data = json.loads(line)
        return (nltk.word_tokenize(parsed_data['sentence1']), nltk.word_tokenize(parsed_data['sentence2']), parsed_data['gold_label'])
         
    samples = [extract_snli_line(line) for line in lines]
    if valid_labels != None:
        samples = [(p, h, lbl) for (p, h, lbl) in samples if lbl in valid_labels]

    if include_lengths:
        samples = [(p, h, lbl, len(p), len(h)) for p, h, lbl in samples]

    return samples



def _tokenize_spacy(sent):
    doc = nlp(sent,  parse=False, tag=False, entity=False)
    return [token.text for token in doc]
def _tokenize_nltk(sent):
    return nltk.word_tokenize(sent)

_tokenize = None
if config.ONLY_TEST:
    _tokenize = _tokenize_nltk
else:
    _tokenize = _tokenize_spacy