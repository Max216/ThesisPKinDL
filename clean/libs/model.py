'''
Includes The model classes.
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cu

from collections import defaultdict

# For running on cluster
import os; 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

class ModelSettings:
    '''
    Some settings can be specified here
    '''

    def __init__(self, opts=None):
        '''
        Create a new object with the settings specified.

        :param opts = [(<option-name>, <option-value>)]         Specify the options:        
                                                                name: 'sent-rep', value: 'normal' | 'relu'
        '''
        self.settings = ['sent-rep']
        self.opts_dict = defaultdict(lambda: 'normal')
        if opts != None:
            for key, val in opts:
                 self.opts_dict[key] = val

    def get_val(self, setting):
        return self.opts_dict[setting]

    def add_val(self, k, v):
        self.opts_dict[k] = v

    def get_fn(self, key):
        '''
        Get the function assoziated with the given option
        '''

        if key == 'sent-rep':
            return self._get_sent_rep_fn(self.opts_dict[key])
        else:
            print('Unkown option key:', key)
            1/0

    def _get_sent_rep_fn(self, val):

        # Define possibilities
        def identity(v):
            return v

        def relu_sent(v):
            return F.relu(v)

        if val == 'normal':
            return identity
        elif val == 'relu':
            return relu_sent
        else:
            print('Invalid option for "sent-rep":', val)
            1/0


class SentenceEncoderMLP(nn.Module):
    """
    Encodes a sentence. This is later used to compare different sentences. This implementation uses an MLP on top of the last LSTM representaion
    """
    
    def __init__(self, embedding_dim, dimen1, dimen2, dimen3, dimen_out, options=ModelSettings()):
        """
        Encode a sentence of variable length into a fixed length representation.
        
        @param embedding_dim - size of pretrained embeddings
        @param dimen_out - size of the resulting vector
        """
        super(SentenceEncoderMLP, self).__init__()

        print('Create mlpsent encoder')
        
        self.directions = 2  # bidirectional
        self.dimen_out = dimen_out
        self.dimen1 = dimen1
        self.dimen2 = dimen2
        self.dimen3 = dimen3

        print('dimen1', dimen1)
        print('dimen2', dimen2)
        print('dimen3', dimen3)
        print('dimen out', dimen_out)

        self.layers = 1      # number of lstm layers        
        self.input_dim_2 = embedding_dim + dimen1 * self.directions
        self.input_dim_3 = self.input_dim_2
        self.embedding_dim = embedding_dim
        
        # Encode via LSTM
        bidirectional = self.directions == 2
        self.lstm1 = nn.LSTM(embedding_dim, dimen1, bidirectional=bidirectional)
        self.lstm2 = nn.LSTM(self.input_dim_2, dimen2, bidirectional=bidirectional)
        self.lstm3 = nn.LSTM(self.input_dim_3, dimen3, bidirectional=bidirectional)

        # options
        self.sent_rep_fn = options.get_fn('sent-rep')

        self.base_tensor_type = cuda_wrap(torch.FloatTensor([0.0]))

        self.linear = nn.Linear(self.dimen3 * self.directions, self.dimen_out)
        
    def init_hidden(self, batch_size):

        # (num_layers*directions, minibatch_size, hidden_dim)
        self.hidden_state1 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen1).zero_())
        self.cell_state1 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen1).zero_())
        self.hidden_state2 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen2).zero_())
        self.cell_state2 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen2).zero_())
        self.hidden_state3 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen3).zero_())
        self.cell_state3 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen3).zero_())

        #self.hidden_state1 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen1)))
        #self.cell_state1 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen1)))
        #self.hidden_state2 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen2)))
        #self.cell_state2 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen2)))
        #self.hidden_state3 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen_out)))
        #self.cell_state3 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen_out)))
    
    def forward(self, sents):
        # init for current batch size
        self.init_hidden(sents.size()[1])
        
        output1, (h_n1, c_n1) = self.lstm1(sents, (self.hidden_state1, self.cell_state1))
        
        # residual of hidden state to word embeddings. here concat
        input_lstm2 = torch.cat((sents, output1), dim=2)
        output2, (h_n2, c_n2) = self.lstm2(input_lstm2, (self.hidden_state2, self.cell_state2))

        # here add
        # use initial embeddings, add lstm outputs
        additional = output1 + output2
        input_lstm3 = torch.cat((sents, additional), dim=2)
        output3, (h_n3, c_n3) = self.lstm3(input_lstm3, (self.hidden_state3, self.cell_state3))
        lstm_out = self.sent_rep_fn(output3)

        return F.relu(self.linear(lstm_out))

    def sent_dim(self):
        '''
        :return the dimension of the resulting sentence represenations
        '''
        return self.dimen_out# * self.directions

    def type(self):
        return ('mlp_sent_encoder', self.dimen_out)


class SentenceEncoder(nn.Module):
    """
    Encodes a sentence. This is later used to compare different sentences.
    """
    
    def __init__(self, embedding_dim, dimen1, dimen2, dimen_out, options=ModelSettings()):
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
        self.input_dim_3 = self.input_dim_2
        self.embedding_dim = embedding_dim
        
        # Encode via LSTM
        bidirectional = self.directions == 2
        self.lstm1 = nn.LSTM(embedding_dim, dimen1, bidirectional=bidirectional)
        self.lstm2 = nn.LSTM(self.input_dim_2, dimen2, bidirectional=bidirectional)
        self.lstm3 = nn.LSTM(self.input_dim_3, dimen_out, bidirectional=bidirectional)

        # options
        self.sent_rep_fn = options.get_fn('sent-rep')

        self.base_tensor_type = cuda_wrap(torch.FloatTensor([0.0]))
        
    def init_hidden(self, batch_size):

        # (num_layers*directions, minibatch_size, hidden_dim)
        self.hidden_state1 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen1).zero_())
        self.cell_state1 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen1).zero_())
        self.hidden_state2 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen2).zero_())
        self.cell_state2 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen2).zero_())
        self.hidden_state3 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen_out).zero_())
        self.cell_state3 = autograd.Variable(self.base_tensor_type.new(self.layers * self.directions, batch_size, self.dimen_out).zero_())

        #self.hidden_state1 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen1)))
        #self.cell_state1 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen1)))
        #self.hidden_state2 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen2)))
        #self.cell_state2 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen2)))
        #self.hidden_state3 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen_out)))
        #self.cell_state3 = autograd.Variable(cuda_wrap(torch.zeros(self.layers * self.directions, batch_size, self.dimen_out)))
    
    def forward(self, sents):
        # init for current batch size
        self.init_hidden(sents.size()[1])
        
        output1, (h_n1, c_n1) = self.lstm1(sents, (self.hidden_state1, self.cell_state1))
        
        # residual of hidden state to word embeddings. here concat
        input_lstm2 = torch.cat((sents, output1), dim=2)
        output2, (h_n2, c_n2) = self.lstm2(input_lstm2, (self.hidden_state2, self.cell_state2))

        # here add
        # use initial embeddings, add lstm outputs
        additional = output1 + output2
        input_lstm3 = torch.cat((sents, additional), dim=2)
        output3, (h_n3, c_n3) = self.lstm3(input_lstm3, (self.hidden_state3, self.cell_state3))
        return self.sent_rep_fn(output3)

    def sent_dim(self):
        '''
        :return the dimension of the resulting sentence represenations
        '''
        return self.dimen_out * self.directions

    def type(self):
        return ('lstm_only', 0)

class EntailmentClassifier(nn.Module):
    """
    Classifier using the SentEncoder to encode sentences with three LSTM, 
    followed by a FeedForward network of three layers.
    """
    
    def __init__(self, pretrained_embeddings, padding_idx, dimen_hidden, dimen_out, sent_encoder, nonlinearity=F.relu, dropout=0.1, multitask='none'):
        """
        @param pretrained_embeddings - Already trained embeddings to initialize the embeddings layer
        @param dimen_sent_repr - Size of the learned representation of a single sentence
        @param dimen_hidden1 - Amount of output units of the first hidden layer of the FF
        @param dimen_hidden2 - Amount of output units of the second hidden layer of the FF
        @param dimen_out - Amount of output units of the output layer of the FF
        @param nonlinearity - Nonlinearity function that is applied to the ouputs of the hidden layers.
        @param dropout - Dropout rate applied within the FF 
        """ 
        
        super(EntailmentClassifier, self).__init__()

        self.multitask = multitask
        self.dimen_hidden = dimen_hidden
        self.nonlinearity = nonlinearity                
        self.embeddings = nn.Embedding(pretrained_embeddings.shape[0], pretrained_embeddings.shape[1], padding_idx=padding_idx)
        
        # Use pretrained values
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
                                               
        # since it is bidirectional, use half size of wanted dimensions
        self.sent_encoder = sent_encoder         
                
        # 3 layer Feedforward 
        dimen_sent_repr = self.sent_encoder.sent_dim()
        self.dimen_sent_repr = dimen_sent_repr

        features = 4
        self.hidden1 = nn.Linear(dimen_sent_repr * features, dimen_hidden) # multiplication because of feature concatenation
        self.hidden2 = nn.Linear(dimen_hidden, dimen_out)
        
        self.dropout1 = nn.Dropout(p=dropout)

    #def forward(self, sent1, sent2, output_sent_info=False, twister=None):
    def forward(self, sent1, sent2, output_sent_info=False):
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


        #if twister != None:
        #    sent1_representation = twister.twist_representation(sent1_representation, 'premise', activations=idxs1, sent=sent1)
        #    sent2_representation = twister.twist_representation(sent2_representation, 'hypothesis', activations=idxs2, sent=sent1)
        
        # Create feature tensor
        feedforward_input = torch.cat((
            sent1_representation,
            sent2_representation,
            torch.abs(sent1_representation - sent2_representation),
            sent1_representation * sent2_representation
        ),1)

        # Run through feed forward network
        out = self.nonlinearity(self.hidden1(feedforward_input))
        out = self.dropout1(out)
        tag_scores = F.softmax(self.hidden2(out))

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

    def forward_sent(self, sent1):
        batch_size = sent1.size()[1]
        print('forward sent btch size', batch_size)
        
        # Map to embeddings
        embedded1 = self.embeddings(sent1)
        
        # Get sentence representation
        sent1_representation = self.sent_encoder(embedded1)
        
        # Max pooling
        sent1_representation, idxs1 = torch.max(sent1_representation, dim=0)

        return sent1_representation.view(batch_size, -1)

    def lookup_word(self, word):
        return self.embeddings(word)


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