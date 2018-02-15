'''Evaluate a model'''


import sys, os, collections, json
import torch
sys.path.append('./../')

from libs import data_tools

from docopt import docopt

def main():
    args = docopt("""For submission

    Usage:
        submission_alayse.py create_counts <data_in> <file_out>
    """)


    if args['create_counts']:
        create_counts(args['<data_in>'], args['<file_out>'])



def create_counts(dataset, out):
    word_count = collections.defaultdict(int)

    with open(dataset) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    for pd in parsed:
        tokenized_premise = data_tools._tokenize(pd['sentence1'])
        tokenized_hyp = data_tools._tokenize(pd['sentence2'])

        for sentence in [tokenized_hyp, tokenized_premise]:
            for word in sentence:
                word_count[word] += 1

    torch.save(word_count, out)

    # test
    loaded = torch.load(out)
    print(loaded['a'])



if __name__ == '__main__':
    main()
