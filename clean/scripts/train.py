import sys
sys.path.append('./../')

from docopt import docopt
 
from libs import model_tools, data_tools, train
from libs import model as m

def main():
    args = docopt("""Train a neural network.

    Usage:
        train.py new [--tdata=<train_data>] [--ddata=<dev_data>] [--encoding=<encoding_dim>] [--hidden=<hidden_dim>] [--embeddings=<embedding_path>] [--sentfn=<sent_fn>]

    """)

    path_train  = args['--tdata']
    path_dev = args['--ddata']
    encoding_dim = args['--encoding']
    hidden_dim = args['--hidden']
    embedding_path = args['--embeddings']
    sent_fn = args['--sentfn'] or 'normal'
    m_settings = m.ModelSettings([('sent-rep', sent_fn)])

    datahandler_train = data_tools.get_datahandler_train(path_train)
    datahandler_dev =  data_tools.get_datahandler_dev(path_dev)

    if embedding_path != None:
        embedding_holder = eh.EmbeddingHolder(embedding_path)
    else:
        embedding_holder = None

    if args['new']:
        print('Create model ... ')
        if encoding_dim != None:
            encoding_dim = [int(encoding_dim), int(encoding_dim), int(encoding_dim)]
            print(encoding_dim)
        model_name, classifier, embedding_holder = model_tools.create_model(encoding_dim, embedding_holder, hidden_dim, opts=m_settings)
        train_set = datahandler_train.get_dataset(embedding_holder)
        dev_set = datahandler_dev.get_dataset(embedding_holder)
        train.train_model(model_name, classifier, embedding_holder.padding(), train_set, dev_set)

if __name__ == '__main__':
    main()