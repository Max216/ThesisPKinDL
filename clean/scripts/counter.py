import sys
sys.path.append('./../')

from docopt import docopt
 
from libs import data_tools

def create_word_cnt(datahandler, out_name):
    datahandler.create_word_cnt(out_name)

def main():
    args = docopt("""Preprocess data.

    Usage:
        counter.py create_word_count <data> <out_name>
    """)

    path_data  = args['<data>']
    out_name  = args['<out_name>']

    data_handler = data_tools.get_datahandler_train(path_data)

    if args['create_word_count']:
        create_word_cnt(data_handler, out_name)

if __name__ == '__main__':
    main()