'''
Methods to create/load/store a model.
'''
from time import gmtime, strftime
import os

import torch
import torch.cuda as cu

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

    sent_encoder_type, dim = sent_encoder.type()
    if sent_encoder_type == 'mlp-sent-encoder':
        last_dim = str(sent_encoder.dimen3)
    else:
        last_dim = str(sent_encoder.dimen_out)


    dim_details = '_'.join([
        str(classifier.dimen_hidden),
        str(sent_encoder.dimen1), str(sent_encoder.dimen2), last_dim,
        hint
    ])

    opts.add_val(sent_encoder_type, str(dim))

    opts_details = '_'.join([setting + '=' + opts.get_val(setting) for setting in opts.all_keys()])

    time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    model_version = 'model' + str(version)

    return '.'.join([dim_details, opts_details, time, model_version])

def params_from_model_name(name):
    categories = name.split('.')

    result = dict()

    # parse category1
    cat1 = categories[0].split('_')
    result['dim_hidden'] = int(cat1[0])
    result['dim_encoder_1'] = int(cat1[1])
    result['dim_encoder_2'] = int(cat1[2])
    result['dim_encoder_3'] = int(cat1[3])

    # parse category2
    cat2 = categories[1].split('_')
    opts = [(splitted[0], splitted[1]) for splitted in [s.split('=') for s in cat2]]
    result['opts'] = m.ModelSettings(opts)

    return result

def create_model(sent_encoding_dims=None, embedding_holder=None, mlp_dim=None, num_classes=None, opts=m.ModelSettings(), hint=None, mlpsent=None):
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

    if sent_encoding_dims != None and len(sent_encoding_dims) == 3:
        sent_lstm_dim_1 = sent_encoding_dims[0]
        sent_lstm_dim_2 = sent_encoding_dims[1]
        sent_lstm_dim_3 = sent_encoding_dims[2]
    else:
        sent_lstm_dim_1 = DEFAULT_SENT_ENCODING_DIMS[0]
        sent_lstm_dim_2 = DEFAULT_SENT_ENCODING_DIMS[1]
        sent_lstm_dim_3 = DEFAULT_SENT_ENCODING_DIMS[2]

    print('Create model:')
    print('sent encoder:', sent_lstm_dim_1, sent_lstm_dim_2, sent_lstm_dim_3)

    if  mlpsent == None:
        sent_encoder = m.SentenceEncoder(
            embedding_dim=embedding_dim, 
            dimen1=sent_lstm_dim_1,
            dimen2=sent_lstm_dim_2,
            dimen_out=sent_lstm_dim_3,
            options=opts
        )   
    else:
        print('Create MLP sent encoder:', mlpsent)
        sent_encoder = m.SentenceEncoderMLP(
            embedding_dim=embedding_dim, 
            dimen1=sent_lstm_dim_1,
            dimen2=sent_lstm_dim_2,
            dimen3=sent_lstm_dim_3,
            dimen_out=mlpsent,
            options=opts
        ) 

    # Create model
    hidden_dim = mlp_dim or DEFAULT_HIDDEN_DIM
    print('MLP:', hidden_dim)
    mlp_out_dim = num_classes or DEFAULT_NUM_CLASSES

    classifier = m.EntailmentClassifier(
        pretrained_embeddings=embedding_holder.embedding_matrix(),
        padding_idx=embedding_holder.padding(),
        dimen_hidden=hidden_dim,
        dimen_out=mlp_out_dim,
        sent_encoder=sent_encoder
    )

    hint = hint or ''
    name = create_model_name(classifier, opts=opts, hint=hint)

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

def load(path, embedding_holder=None):
    model_name = path.split('/')[-1]
    params = params_from_model_name(model_name)
    if embedding_holder == None:
        embedding_holder = eh.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)

    opts = params['opts']
    print(opts)

    if 'mlp-sent-encoder' in opts.all_keys():
        sent_encoder = m.SentenceEncoderMLP(
            embedding_dim=embedding_holder.dim(), 
            dimen1=params['dim_encoder_1'],
            dimen2=params['dim_encoder_2'],
            dimen3=params['dim_encoder_3'],
            dimen_out=int(opts['mlp_sent_encoder']),
            options=params['opts']
        )
    else:
        sent_encoder = m.SentenceEncoder(
            embedding_dim=embedding_holder.dim(), 
            dimen1=params['dim_encoder_1'],
            dimen2=params['dim_encoder_2'],
            dimen_out=params['dim_encoder_3'],
            options=params['opts']
        )

    classifier = m.cuda_wrap(m.EntailmentClassifier(
        pretrained_embeddings=embedding_holder.embedding_matrix(),
        padding_idx=embedding_holder.padding(),
        dimen_hidden=params['dim_hidden'],
        dimen_out=DEFAULT_NUM_CLASSES,
        sent_encoder=sent_encoder
    ))

    if cu.is_available():
        classifier.load_state_dict(torch.load(path))
    else:
        classifier.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    classifier.eval()

    return (model_name, classifier, embedding_holder)

