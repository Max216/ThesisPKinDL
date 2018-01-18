import sys
sys.path.append('./../')

from docopt import docopt

from libs import model_tools
from libs import compatability


def main():
    args = docopt("""Evaluate how good a model performs with adversarial examples created by swapping words.

    Usage:
        eval_replacements.py file <model> <swap_word_filel>

    """)

    path_model = args['<model>']
    swap_words = args['<swap_word_filel>']
    #datahandler_train = data_tools.get_datahandler_train(path_train)

    # load model
    if path_model.split('.')[-1] == 'model':
        # use compatability and older model
        print('Use compatability...')
        compaibility = 0
    else:
        # use normal model
        model_tools.load(path_model)


    if args['file']:
        print('yay')
        

if __name__ == '__main__':
    main()