import model as m
import mydataloader
import train
import embeddingholder
import config

import torch
import torch.autograd as autograd

from docopt import docopt

def main():
    args = docopt("""Evaluate on given dataset in terms of accuracy.

    Usage:
        evaluate.py eval <model> <data> [<embeddings>]
        evaluate.py test <model> <premise> <hypothesis>

        <model> = Path to trained model
        <data>  = Path to data to test model with 
        <embeddings>  = New embedding file to use unknown words from 
    """)

    model_path = args['<model>']
    data_path = args['<data>']
    embeddings_path = args['<embeddings>']


    if args['eval']:
        evaluate(model_path, data_path, embeddings_path)
    else:
        test(model_path, args['<premise>'], args['<hypothesis>'])

def test(model_path, p, h):
    embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
    vec_p, vec_h, _ = mydataloader.load_test_pair(p, h, embedding_holder)
    classifier, _ = m.load_model(model_path, embedding_holder=embedding_holder)
    var_p = autograd.Variable(m.cuda_wrap(vec_p.view(-1, 1)))
    var_h = autograd.Variable(m.cuda_wrap(vec_h).view(-1, 1))
    out = classifier(var_p, var_h)
    _, predicted_idx = torch.max(out, dim=1)
    print('Predict:', mydataloader.index_to_tag[predicted_idx.data[0]])

def evaluate(model_path, data_path, new_embeddings=None, twister=None):
    # Load model

    embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
    embeddings_diff = []
    if new_embeddings != None:
        print ('Merge embeddings')
        embedding_holder_new = embeddingholder.EmbeddingHolder(new_embeddings)
        embeddings_diff = embedding_holder.add_unknowns_from(embedding_holder_new)

    print('Load model ...')
    classifier, _ = m.load_model(model_path, embedding_holder=embedding_holder)

    # todo look with merging ....
    if len(embeddings_diff) != 0 and embeddings_diff.shape[1] != 0:
        # Merge into model
        classifier.inc_embedding_layer(embeddings_diff)

    print('Load data ...')
    data = mydataloader.simple_load(data_path)
    print(len(data), 'samples loaded.')
    print('Evaluate ...')
    classifier.eval()
    classifier = m.cuda_wrap(classifier)
    print('Accuracy:', train.evaluate(classifier, [data], size=32, padding_token=embedding_holder.padding(), twister=twister))

if __name__ == '__main__':
    main()
