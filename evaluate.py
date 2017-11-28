import model as m
import mydataloader
import train
import embeddingholder
import config

from docopt import docopt

def main():
    args = docopt("""Evaluate on given dataset in terms of accuracy.

    Usage:
        evaluate.py <model> <data> [<embeddings>]

        <model> = Path to trained model
        <data>  = Path to data to test model with 
        <embeddings>  = New embedding file to use unknown words from 
    """)

    model_path = args['<model>']
    data_path = args['<data>']
    embeddings_path = args['<embeddings>']

    evaluate(model_path, data_path, embeddings_path)



def evaluate(model_path, data_path, new_embeddings=None):
    # Load model

    embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
    embeddings_diff = []
    if new_embeddings != None:
        print ('Merge embeddings')
        embedding_holder_new = embeddingholder.EmbeddingHolder(new_embeddings)
        embeddings_diff = embedding_holder.add_unknowns_from(embedding_holder_new)

    print('Load model ...')
    classifier, _ = m.load_model(model_path, embedding_holder=embedding_holder)

    if embeddings_diff.shape[1] != 0:
        # Merge into model
        classifier.inc_embedding_layer(embeddings_diff)

    print('Load data ...')
    data = mydataloader.simple_load(data_path)
    print(len(data), 'samples loaded.')
    print('Evaluate ...')
    classifier.eval()
    print('Accuracy:', train.evaluate(classifier, [data], size=32, padding_token=embedding_holder.padding()))

if __name__ == '__main__':
    main()
