import sys
sys.path.append('./../')

from docopt import docopt
 
from libs import model_tools, data_tools, train, data_handler
from libs import model as m
from libs import embeddingholder as eh
from libs import multitask, config

import torch
import numpy as np
import random

def main():
    args = docopt("""Train a neural network.

    Usage:
        train.py new [--tdata=<train_data>] [--ddata=<dev_data>] [--encoding=<encoding_dim>] [--hidden=<hidden_dim>] [--embeddings=<embedding_path>] [--sentfn=<sent_fn>] [--appendix=<appendix>] [--embd1=<embd1>] [--embd2=<embd2>]
        train.py new_mt_sent [--tdata=<train_data>] [--ddata=<dev_data>] [--encoding=<encoding_dim>] [--hidden=<hidden_dim>] [--embeddings=<embedding_path>] [--sentfn=<sent_fn>] [--appendix=<appendix>] [--embd1=<embd1>] [--embd2=<embd2>] [--mt1=<mt1>]
        train.py multitask <multitask_type> <multitask_data> [--embeddings=<embedding_path>] [--load=<model_path>] [--tdata=<train_data>]
    """)

    torch.manual_seed(12)
    np.random.seed(12)
    random.seed(12)

    path_train  = args['--tdata']
    path_dev = args['--ddata']
    encoding_dim = args['--encoding']
    hidden_dim = args['--hidden']
    embedding_path = args['--embeddings']
    embd1 = args['--embd1']
    embd2 = args['--embd2']
    sent_fn = args['--sentfn'] or 'normal'
    appendix = args['--appendix'] or ''
    m_settings = m.ModelSettings([('sent-rep', sent_fn)])

    datahandler_train = data_handler.get_datahandler_train(path_train)
    datahandler_dev =  data_handler.get_datahandler_dev(path_dev)

    if embedding_path != None:
        embedding_holder = eh.EmbeddingHolder(embedding_path)
    else:
        embedding_holder = eh.create_embeddingholder()

    if embd1 != None:
        embd1_holder = eh.EmbeddingHolder(embd1, include_oov_padding = False, include_start_end=False)
        embedding_holder.concat(embd1_holder)
        # merge

    if embd2 != None:
        embd2_holder = eh.EmbeddingHolder(embd2, include_oov_padding = False, include_start_end=False)
        embedding_holder.concat(embd2_holder)

    if args['new']:
        print('Create model ... ')
        if encoding_dim != None:
            encoding_dim = [int(encoding_dim), int(encoding_dim), int(encoding_dim)]
        model_name, classifier, embedding_holder = model_tools.create_model(encoding_dim, embedding_holder, hidden_dim, opts=m_settings, hint=appendix)
        print('Store result as', model_name)
        train_set = [datahandler_train.get_dataset(embedding_holder)]
        dev_set = datahandler_dev.get_dataset(embedding_holder)
        train.train_model(model_name, classifier, embedding_holder.padding(), train_set, dev_set)

    elif args['new_mt_sent']:
        print('MultiTask Sentence training')
        print('Create model ... ')
        if encoding_dim != None:
            encoding_dim = [int(encoding_dim), int(encoding_dim), int(encoding_dim)]
        model_name, classifier, embedding_holder = model_tools.create_model(encoding_dim, embedding_holder, hidden_dim, opts=m_settings, hint=appendix)
        print('Store result as', model_name)
        train_set = [datahandler_train.get_dataset(embedding_holder)]
        dev_set = datahandler_dev.get_dataset(embedding_holder)

        if path_train == None:
            path_train = config.PATH_TRAIN_DATA
        mt_target = multitask.SentenceInOutTarget(args['--mt1'], embedding_holder, path_train).get_target_dataset()
        multitask_learner = multitask.MTNetwork(classifier, 600 * 2 + embedding_holder.dim(), 2)
        train.train_model_multitask_sent(model_name, classifier, embedding_holder.padding(), train_set, dev_set,multitask_learner, mt_target)
    
    elif args['multitask']:
        print('Multitask learning')


        load_path = args['--load']

        if load_path == None:
            # None becuse of default settings
            model_name, classifier, embedding_holder = model_tools.create_model(None, embedding_holder, None, opts=m_settings, hint=appendix)
        else:
            model_name, classifier, embedding_holder = model_tools.load(load_path, embedding_holder)
            model_name += '.loaded'
        
        model_name += '.' + args['<multitask_type>']

        print('Store result as', model_name)
        train_set = [datahandler_train.get_dataset(embedding_holder)]
        dev_set = datahandler_dev.get_dataset(embedding_holder)
        if path_train == None:
            path_train = config.PATH_TRAIN_DATA

        multitask.train_simult(model_name, classifier, embedding_holder, train_set, dev_set, path_train, args['<multitask_type>'], args['<multitask_data>'])

    elif args['new_mt_sent_simult']:
        print('MultiTask Sentence training')
        print('Create model ... ')
        if encoding_dim != None:
            encoding_dim = [int(encoding_dim), int(encoding_dim), int(encoding_dim)]
        model_name, classifier, embedding_holder = model_tools.create_model(encoding_dim, embedding_holder, hidden_dim, opts=m_settings, hint=appendix)
        print('Store result as', model_name)
        train_set = [datahandler_train.get_dataset(embedding_holder)]
        dev_set = datahandler_dev.get_dataset(embedding_holder)


        if path_train == None:
            path_train = config.PATH_TRAIN_DATA
        mt_target = multitask.SentenceInOutTarget(args['--mt1'], embedding_holder, path_train).get_target_dataset()
        multitask_learner = multitask.MTNetwork(classifier, 600 * 2 + embedding_holder.dim(), 2)
        train.train_model_multitask_sent_simult(model_name, classifier, embedding_holder.padding(), train_set, dev_set,multitask_learner, mt_target)

if __name__ == '__main__':
    main()