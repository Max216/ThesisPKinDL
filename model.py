
# coding: utf-8

# For running on cluster
import os; 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cu

import embeddingholder
import config

import re


# for cuda
def make_with_cuda(t):
    return t.cuda()

def make_without_cuda(t):
    return t

# always applies cuda() on model/tensor when available
cuda_wrap = None

if cu.is_available():
    print('Running with cuda enabled.')
    cuda_wrap = make_with_cuda
else:
    print('Running without cuda.')
    cuda_wrap = make_without_cuda

class SentenceEncoder(nn.Module):
    """
    Encodes a sentence. This is later used to compare different sentences.
    """
    
    def __init__(self, embedding_dim, dimen1, dimen2, dimen_out):
        """
        Encode a sentence of variable length into a fixed length representation.
        
        @param embedding_dim - size of pretrained embeddings
        @param dimen_out - size of the resulting vector
        """
        super(SentenceEncoder, self).__init__()
        
        self.directions = 2  # bidirectional
        
        self.dimen_out = dimen_out
        self.dimen1 = dimen1
        self.dimen2 = dimen2
        self.layers = 1      # number of lstm layers 
                 
        self.input_dim_2 = embedding_dim + dimen1 * self.directions
        self.input_dim_3 = self.input_dim_2 + dimen2 * self.directions
        
        
        # Encode via LSTM
        bidirectional = self.directions == 2
        self.lstm1 = nn.LSTM(embedding_dim, dimen1, bidirectional=bidirectional)
        self.lstm2 = nn.LSTM(self.input_dim_2, dimen2, bidirectional=bidirectional)
        self.lstm3 = nn.LSTM(self.input_dim_3, dimen_out, bidirectional=bidirectional)
        
    def init_hidden(self, batch_size):
        # (num_layers*directions, minibatch_size, hidden_dim)
        self.hidden_state1 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen1)))
        self.cell_state1 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen1)))
        self.hidden_state2 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen2)))
        self.cell_state2 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen2)))
        self.hidden_state3 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen_out)))
        self.cell_state3 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen_out)))
    
    def forward(self, sents):
        # init for current batch size
        self.init_hidden(sents.size()[1])
        
        output1, (h_n1, c_n1) = self.lstm1(sents, (self.hidden_state1, self.cell_state1))
        
        # shortcuts of hidden state to word embeddings
        input_lstm2 = torch.cat((sents, output1), dim=2)
        output2, (h_n2, c_n2) = self.lstm2(input_lstm2, (self.hidden_state2, self.cell_state2))
        input_lstm3 = torch.cat((input_lstm2, output2), dim=2)
        output3, (h_n3, c_n3) = self.lstm3(input_lstm3, (self.hidden_state3, self.cell_state3))
        return output3

class EntailmentClassifier(nn.Module):
    """
    Classifier using the SentEncoder to encode sentences with three LSTM, 
    followed by a FeedForward network of three layers.
    """
    
    def __init__(self, pretrained_embeddings, padding_idx, dimen_hidden, dimen_out, dimen_sent_encoder = [64,128,256], nonlinearity=F.relu, dropout=0.1, sent_repr='all'):
        """
        @param pretrained_embeddings - Already trained embeddings to initialize the embeddings layer
        @param dimen_sent_repr - Size of the learned representation of a single sentence
        @param dimen_hidden1 - Amount of output units of the first hidden layer of the FF
        @param dimen_hidden2 - Amount of output units of the second hidden layer of the FF
        @param dimen_out - Amount of output units of the output layer of the FF
        @param nonlinearity - Nonlinearity function that is applied to the ouputs of the hidden layers.
        @param dropout - Dropout rate applied within the FF 
        @param sent_repr - "all" means the concatenation [p,p,p-h,p*h] is classified
                           "relative" means that only [p-h,p*h] is classified
        """ 
        
        super(EntailmentClassifier, self).__init__()
        
        if len(dimen_sent_encoder) != 3:
            raise Exception('must have three values for dimen_sent_encoder.')

        self.nonlinearity = nonlinearity
        self.sent_repr = sent_repr
                
        self.embeddings = nn.Embedding(pretrained_embeddings.shape[0], pretrained_embeddings.shape[1], padding_idx=padding_idx)
        #self.embeddings = nn.Embedding(pretrained_embeddings.shape[0], pretrained_embeddings.shape[1])
        # Use pretrained values
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
                                               
        # since it is bidirectional, use half size of wanted dimensions
        self.sent_encoder = SentenceEncoder(pretrained_embeddings.shape[1], dimen_sent_encoder[0], dimen_sent_encoder[1], dimen_sent_encoder[2])          
                
        # 3 layer Feedforward 
        dimen_sent_repr = dimen_sent_encoder[2] * 2 # multiplication because bidirectional
        self.dimen_sent_repr = dimen_sent_repr

        if sent_repr == "all":
            features = 4
        elif sent_repr == "relative":
            features = 2
        else:
            raise Exception('Invalid keyword for sent_repr. Must be "all" or "relative".')

        self.hidden1 = nn.Linear(dimen_sent_repr * features, dimen_hidden) # multiplication because of feature concatenation
        self.hidden2 = nn.Linear(dimen_hidden, dimen_hidden)
        self.hidden3 = nn.Linear(dimen_hidden, dimen_out)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, sent1, sent2, output_sent_info=False, twister=None):
        batch_size = sent1.size()[1]
        
        # Map to embeddings
        embedded1 = self.embeddings(sent1)
        embedded2 = self.embeddings(sent2)
        
        # Get sentence representation
        sent1_representation = self.sent_encoder(embedded1)
        sent2_representation = self.sent_encoder(embedded2)
        


        # Max pooling
        sent1_representation, idxs1 = torch.max(sent1_representation, dim=0)
        sent2_representation, idxs2 = torch.max(sent2_representation, dim=0)

        sent1_representation = sent1_representation.view(batch_size, -1)
        sent2_representation = sent2_representation.view(batch_size, -1)
        idxs1 = idxs1.view(batch_size, -1)
        idxs2 = idxs2.view(batch_size, -1)


        if twister != None:
            sent1_representation = twister.twist_representation(sent1_representation, 'premise', activations=idxs1, sent=sent1)
            sent2_representation = twister.twist_representation(sent2_representation, 'hypothesis', activations=idxs2, sent=sent1)
        # Create feature tensor
        if self.sent_repr == "all":
            feedforward_input = torch.cat((
                sent1_representation,
                sent2_representation,
                torch.abs(sent1_representation - sent2_representation),
                sent1_representation * sent2_representation
            ),1)
        else:
            feedforward_input = torch.cat((
                torch.abs(sent1_representation - sent2_representation),
                sent1_representation * sent2_representation
            ),1)
        
        # Run through feed forward network
        out = self.nonlinearity(self.hidden1(feedforward_input))
        out = self.dropout1(out)
        out = self.nonlinearity(self.hidden2(out))
        out = self.dropout2(out)
        out = self.hidden3(out)
        tag_scores = F.softmax(out)

        if output_sent_info:
            return tag_scores, [idxs1, idxs2], [sent1_representation, sent2_representation]

        return tag_scores

    def inc_embedding_layer(self, wv):
        '''
        Increase the embedding layer by the matrix wv.
        '''
        #self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        wv_new = cuda_wrap(torch.from_numpy(wv).float())
        wv_combined = torch.cat([self.embeddings.weight.data, wv_new])

        # adjust embedding layer size
        self.embeddings = nn.Embedding(wv_combined.size()[0], wv_combined.size()[1])
        self.embeddings.weight.data.copy_(wv_combined)
        

def left_number(val):
    '''
    Remove the letters from a value of a model name. e.g. 0_001lr -> 0.001
    '''
    return re.split('[a-z]', val)[0]

def lbl_to_float(val):
    '''
    Map a value from a name to a float.
    '''
    return float(val.replace('_', '.'))

def predict(classifier, embedding_holder, p, h, twister=None):
    p_batch = torch.LongTensor(len(p), 1)
    h_batch = torch.LongTensor(len(h), 1)

    p_batch[:,0] = torch.LongTensor([embedding_holder.word_index(w) for w in p])
    h_batch[:,0] = torch.LongTensor([embedding_holder.word_index(w) for w in h])
    
    return classifier(
        cuda_wrap(autograd.Variable(p_batch)), 
        cuda_wrap(autograd.Variable(h_batch)), 
        output_sent_info=True,
        twister=twister)

def load_model(model_path, embedding_holder = None):
    '''
    Load a model. Parameters are retrieved from the given model name.

    @param stored_model     Path to the stored model. Name must be untouched.
    @param embedding_holder Optional, if not specified the one inf config.py is used. Must have correct dimensions
                            with the trained model.

    @return (model, model_name)
    '''

    if embedding_holder == None:
        embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)

    # Model params:
    model_name = model_path.split('/')[-1]
    splitted = model_name.split('-')
    lr = lbl_to_float(left_number(splitted[0]))
    hidden_dim = int(left_number(splitted[1]))
    lstm_dim = [int(i) for i in left_number(splitted[2]).split('_')]
    batch_size = int(left_number(splitted[3]))
    dropout = lbl_to_float(left_number(splitted[6]))

    if splitted[5] == 'relu':
        nonlinearity = F.relu
    else:
        print('Unknown:', splitted[5])
        raise Exception('Unknown activation function.', splitted[5])

    # Create new model with correct dimensions
    model = cuda_wrap(EntailmentClassifier(embedding_holder.embeddings, 
                                            embedding_holder.padding(),
                                            dimen_hidden=hidden_dim, 
                                            dimen_out=3, 
                                            dimen_sent_encoder=lstm_dim,
                                            nonlinearity=nonlinearity, 
                                            dropout=dropout))

    # Model state:
    if cu.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    model.eval()
    return model, model_name

class ModelTwister:
    '''
    Manipulate dimensions of the sentence representation using this class.
    '''

    def __init__(self, twist, tools = None, name = None):
        '''
        :param twist - function(representation, ['premise'|'hypothesis'], tools) to twist the representation
        :param tools - additional information to use in the twist function
        '''
        self.twist = twist
        self.tools = tools
        self.name = name

    def twist_representation(self, representation, sent_type, activations, sent):
        return self.twist(representation, sent_type, self.tools, activations=activations, sent=sent)