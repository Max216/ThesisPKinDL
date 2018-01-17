'''
Methods to create/load/store a model.
'''
from time import gmtime, strftime
import os

import torch

from libs import config
from libs import embeddingholder as eh
from libs import model as m

DEFAULT_SENT_ENCODING_DIMS = [600, 600, 600]
DEFAULT_HIDDEN_DIM = 800
DEFAULT_NUM_CLASSES = 3

def create_model_name(classifier, version=1, hint='', opts=m.ModelSettings()):
    '''
    Create a model name that makes it possible to obtain the same model when loading. the name looks
    as follows: 
    [<dimension-details>_<...>_<hint>].[<option_details>_<...>].<timestamp>.model<version>
    '''

    sent_encoder = classifier.sent_encoder
    dim_details = '_'.join([
        str(classifier.dimen_hidden),
        str(sent_encoder.dimen1), str(sent_encoder.dimen2), str(sent_encoder.dimen_out),
        hint
    ])

    opts_details = '_'.join([setting + '=' + opts.get_val(setting) for setting in opts.settings])

    time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    model_version = 'model' + str(version)

    return '.'.join([dim_details, opts_details, time, model_version])

def create_model(sent_encoding_dims=None, embedding_holder=None, mlp_dim=None, num_classes=None, opts=m.ModelSettings()):
    '''
    Create a model from parameters

    :param sent_encoding_dims           [dim1, dim2, dim3] for the encoding BiLSTMs
    :param embedding_holder             embeddingholder class for the used embeddings
    :param mlp_dim                      dimension for the hidden layer in the MLP
    :param opts                         Specify options using :class ModelSettings

    :return (name, model, embedding_holder)
    '''

    # Create sentence encoder
    if embedding_holder == None:
        embedding_holder = eh.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)

    embedding_dim = embedding_holder.dim()

    if sent_encoding_dims != None and len(sent_encoding_dims == 3):
        print('ayya')
        sent_lstm_dim_1 = sent_encoding_dims[0]
        sent_lstm_dim_2 = sent_encoding_dims[1]
        sent_lstm_dim_3 = sent_encoding_dims[2]
    else:
        sent_lstm_dim_1 = DEFAULT_SENT_ENCODING_DIMS[0]
        sent_lstm_dim_2 = DEFAULT_SENT_ENCODING_DIMS[1]
        sent_lstm_dim_3 = DEFAULT_SENT_ENCODING_DIMS[2]

    sent_encoder = m.SentenceEncoder(
        embedding_dim=embedding_dim, 
        dimen1=sent_lstm_dim_1,
        dimen2=sent_lstm_dim_2,
        dimen_out=sent_lstm_dim_3,
        options=opts
    )

    # Create model
    hidden_dim = mlp_dim or DEFAULT_HIDDEN_DIM
    mlp_out_dim = num_classes or DEFAULT_NUM_CLASSES

    classifier = m.EntailmentClassifier(
        pretrained_embeddings=embedding_holder.embedding_matrix(),
        padding_idx=embedding_holder.padding(),
        dimen_hidden=hidden_dim,
        dimen_out=mlp_out_dim,
        sent_encoder=sent_encoder
    )

    name = create_model_name(classifier, opts=opts)

    return (name, m.cuda_wrap(classifier), embedding_holder)

def store(name, classifier_state, result_type):
    '''
    Store the current model

    :param name                 name of file
    :param classifier_state     weights of classifier
    :param result_type                 final ('final') or temporary ('temp') result
    '''
    current_dir = os.path.dirname(os.path.realpath(__file__))
    path = current_dir + '/../results/models/' + result_type + '/'
    torch.save(classifier_state, path + name)

