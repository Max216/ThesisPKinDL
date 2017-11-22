"""
Usage:
    train_single_model.py new <dim_hidden> <dim_s1> <dim_s2> <dim_s3> [--iterations=<iterations>] [--validate_after=<validate_after>] [--batch=<batch>] [--chunk=<chunk>] [--lr=<lr>] [--path_train=<path_train>] [--path_dev=<path_dev>] [--appendix=<appendix>] 
    train_single_model.py continue <file>

Options:
    <dim_hidden>    Hidden dimension of MLP.
    <dim_s1>        Hidden dimension in first sentence encoding LSTM.
    <dim_s2>        Hidden dimension in second sentence encoding LSTM.
    <dim_s3>        Hidden dimension in third sentence encoding LSTM.
    <file>          Already trained model. (not renamed)

    --iterations=<iterations>           How many iterations to train.
    --validate_after=<validate_after>   After how many examples evaluate on the entire data (train & dev).
    --batch=<batch>                     Minibatch size.
    --chunk=<chunk>                     Chunk size (data is splitted into chunks with roughly even size to minimize padding)
    --lr=<lr>                           Learning rate.
    --path_train=<path_train>           Path to train set.
    --path_dev=<path_dev>               Path to development set.
    --appendix=<appendix>               Appendix to add to the name.
"""
from docopt import docopt

import embeddingholder
import mydataloader
import config
from config import *
import train

def main():

    args = docopt(__doc__)

    DEFAULT_ITERATIONS = 2
    DEFAULT_LR = 0.0002
    DEFAULT_VAL_AFTER = 2000
    DEFAULT_BATCH = 32
    DEFAULT_CHUNK = 32*400
    
    if args['continue']:
        print('continue learning a model')
        name = args['<file>'].split('/')[-1]
        print(name)
        # Parse name
        # Do when needed

    elif args['new']:
        print('new model')
        dim_hidden = int(args['<dim_hidden>'])
        dim_s1 = int(args['<dim_s1>'])
        dim_s2 = int(args['<dim_s2>'])
        dim_s3 = int(args['<dim_s3>'])

        iterations = int(args.get('--iterations') or DEFAULT_ITERATIONS)
        validate_after = int(args.get('--validate_after') or DEFAULT_VAL_AFTER)
        batch = int(args.get('--batch') or DEFAULT_BATCH)
        chunk = int(args.get('--chunk') or DEFAULT_CHUNK)
        lr = float(args.get('--lr') or DEFAULT_LR)
        path_train = args.get('--path_train') or PATH_TRAIN_DATA
        path_dev = args.get('--path_dev') or PATH_DEV_DATA
        appendix = args.get('--appendix') or ''


        print('Start training a new model:')
        

        embedding_holder = embeddingholder.EmbeddingHolder(PATH_WORD_EMBEDDINGS)
        snli_train = mydataloader.get_dataset_chunks(path_train, embedding_holder, chunk_size=chunk, mark_as='[train]')
        snli_dev = mydataloader.get_dataset_chunks(path_dev, embedding_holder, chunk_size=chunk, mark_as='[dev]')

        train.search_best_model(snli_train, snli_dev, embedding_holder, [lr], [dim_hidden], [[dim_s1, dim_s2, dim_s3]], [batch], epochs=iterations, validate_after=validate_after, appendix=appendix)

if __name__ == '__main__':
    main()
