'''Manipulate snli data'''
import sys, json
sys.path.append('./../') 

from docopt import docopt

def split_dataset(path_data, path_out_neutral_entailment, path_put_contradiction):
    '''
    Splits the dataset into samples containing neutral/entailment and contadiction
    :param path_data                    Path to dataset to split
    :param path_out_neutral_entailment  save neutral/entailing samples here
    :param path_out_contradiction       save contradiction samples here
    '''

    with open(path_data) as f_in:
        with open(path_out_neutral_entailment, 'w') as out_ne:
            with open(path_out_contradiction, 'w') as out_contr:
                for line in f_in:
                    parsed = json.loads(line.strip())
                    label = parsed['gold_label']
                    if label == 'entailment' or label == 'neutral':
                        out_ne.write(line)
                    elif label == 'contradiction':
                        out_contr.write(line)
    print('done.')

def reverse(path_in, path_out):
    '''
    Reverse the order of premise/hypothesis (might not be correct label anymore)
    :param path_in      Path to dataset to reverse
    :param path_out     Resulting dataset path
    '''
    with open(path_in) as f_in:
        with open(path_out, 'w') as f_out:
            for line in f_in:
                parsed = json.loads(line.strip())
                tmp = parsed['sentence1']
                parsed['sentence1'] = parsed['sentence2']
                parsed['sentence2'] = tmp
                f_out.write(json.dumps(parsed) + '\n')

def main():
    args = docopt("""Manipulate SNLI jsonl data

    Usage:
        manipulate_data.py split <in_path> <out_path_neutral_entailment> <out_path_contradiction>
        manipulate_data.py reverse <in_path> <out_path>

    """)

    if args['split']:
        split_dataset(args['<in_path>'], args['<out_path_neutral_entailment>'], args['<out_path_contradiction>'])
    elif args['reverse']:
        reverse(args['<in_path>'], args['<out_path>'])
if __name__ == '__main__':
    main()