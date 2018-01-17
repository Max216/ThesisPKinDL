'''
To deal with embeddings
'''

import sys
sys.path.append('./../')

from docopt import docopt
 
from libs import data_tools

def cfd(embedding_path, data1_path, data2_path, name_out):
    datahandler1 = data_tools.Datahandler(data1_path)
    datahandler2 = data_tools.Datahandler(data2_path)
    datahandler1.merge([datahandler2])
    vocab = datahandler1.vocab()

    print('Total vocab in files:', len(vocab))
    with open(embedding_path) as f_in:
        used_embeddings = [line for line in f_in if line.split(' ', maxsplit=2)[0] in vocab]

    print('Word embeddings for vocab:', len(used_embeddings))
    with open(name_out, 'w') as f_out:
        for line in used_embeddings:
            f_out.write(line)

def main():
    args = docopt("""Deal with embeddings. 
        cfd  = create for data: Create embedding files for a given dataset.

    Usage:
        embedding_tools.py cfd <embeddings> <data_train> <data_dev> <name_out>

    """)

    embeddings  = args['<embeddings>']
    data_train  = args['<data_train>']
    data_dev  = args['<data_dev>']
    name_out  = args['<name_out>']

    if args['cfd']:
        cfd(embeddings, data_train, data_dev, name_out)


if __name__ == '__main__':
    main()