import sys
sys.path.append('./../')

from docopt import docopt
 
from libs import data_tools
from libs import model as m

def filter(data_handler, res_handler, out_name):
    res_handler.filter(data_handler, min_count=5)
    res_handler.save(out_name)


def main():
    args = docopt("""Preprocess external resource files.

    Usage:
        ext_res_tools.py filter <data> <ext_res> <out_name>

    """)

    path_data  = args['<data>']
    path_res  = args['<ext_res>']
    out_name  = args['<out_name>']

    data_handler = data_tools.get_datahandler_train(path_data)
    ext_handler = data_tools.ExtResPairhandler(path_res)

    if args['filter']:
        filter(data_handler, ext_handler, out_name)

if __name__ == '__main__':
    main()