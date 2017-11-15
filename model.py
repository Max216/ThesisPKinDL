
# coding: utf-8

# In[1]:

import json
import nltk, re, pprint
from nltk import word_tokenize
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from random import shuffle
import time
import torch.cuda as cu
from torch.utils.data import Dataset, DataLoader
import config
from config import *

torch.manual_seed(1)

# For running on cluster
import os; 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class EmbeddingHolder:
    
    """
    Load pretrained GloVe embeddings and makes them accessable.
    Extra symbols are added for OOV and Padding.
    """
    
    OOV = '@@OOV@@'
    PADDING = '@@PADDING@@'
    
    def __init__(self, path):
        cnt = 0
        words = dict()
        vectors = []
        file = open(path, "r")
        for line in file:
            splitted_line = line.split()
            words[splitted_line[0]] = cnt
            vectors.append(np.asarray(splitted_line[1:], dtype='float'))
            cnt += 1
            
            # TODO rm
            #if cnt == 100:
            #    break
                
        self.dimen = len(vectors[0])
        print(len(vectors), 'word embeddings loaded.')
        
        # Add OOV and PADDING
        words[self.OOV] = cnt
        words[self.PADDING] = cnt+1
        oov_vector = np.random.rand(self.dimen)
        vectors.append(oov_vector)
        vectors.append(np.zeros(self.dimen))
        
        self.words = words
        self.embeddings = np.matrix(vectors)
    
    def embedding_matrix(self):
        """
        Get the embedding matrix of the form:
        #vocab X #dimen i.e. every row represents one word
        """
        return self.embeddings
    
    def dim(self):
        """
        Get the dimension of the embeddings
        """
        return self.dimen
    
    def word_index(self, word):
        """
        Get the index of the given word within the embedding matrix.
        """
        if word in self.words:
            return self.words[word]
        else:
            return self.words[self.OOV]
        
    def padding(self):
        """
        Get the index of the Padding symbol.
        """
        return self.word_index(embedding_holder.PADDING)
        
embedding_holder = EmbeddingHolder(PATH_WORD_EMBEDDINGS)


# In[3]:

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


def extract_snli(raw_instance):
    """
    @param raw_instance - from the official SNLI dataset the .jsonl file format
    
    Extracts the required information of a raw single instance of the SNLI dataset.
    This will result in a triplet like of the following form:
    
    Applied preprocessing:
    - lowecase
    - split()
    
    ([sent1], [sent2], label)
    """
    
    parsed_data = json.loads(raw_instance)
    return (word_tokenize(parsed_data['sentence1'].lower()), word_tokenize(parsed_data['sentence2'].lower()), parsed_data['gold_label'])

def load_snli(path, size=-1):
    """
    @param path - Path to the *.jsonl file of the SNLI dataset
    @param size - Amount of examples that are loaded. If not specified or size=-1 all instances are loaded.
    
    Load instances of the SNLI dataset into an array.
    """

    result = []
    cnt=0
    file = open(path, "r")
    for line in file:
        s1,s2,lbl = extract_snli(line)
        if lbl in ['neutral','contradiction','entailment']:
            result.append((s1,s2,lbl))
            cnt += 1
        if size != -1 and cnt >= size:
            break
        
    return result
    
def tag_to_index(tag):
    """
    Maps a label as a string into an integer number.
    """
 
    if tag == 'neutral':
        return 0
    elif tag == 'contradiction':
        return 1
    elif tag == 'entailment':
        return 2
    else:
        print(tag)
        raise Exception('invalid tag', tag)
        



class SNLIDataset(Dataset):
    '''
    Load the SNLI dataset
    @param file - file-path to samples
    @param max_size (optional) - limit to a maximum size
    '''
    def __init__(self, file, embedding_holder, max_size=-1):
        self.converted_samples = []
        
        raw_samples = load_snli(file, max_size)
        # to tensors
        for i in range(len(raw_samples)):
            s1, s2, lbl = raw_samples[i]
            self.converted_samples.append((
                torch.LongTensor([embedding_holder.word_index(w) for w in s1]),
                torch.LongTensor([embedding_holder.word_index(w) for w in s2]),
                tag_to_index(lbl)
            ))
        
    def __len__(self):
        return len(self.converted_samples)
    
    def __getitem__(self, idx):
        return self.converted_samples[idx]
        
        
class CollocateBatch(object):
    '''
    Applies padding to shorter sentences within a minibatch.
    '''
    
    def __init__(self, padding_token):
        self.padding_token = padding_token
        
    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(self.padding_token)])
        
    def __call__(self, batch):
        # TODO maybe make faster by having fixed length padding?
        sizes = torch.LongTensor([[len(premise), len(hypothesis)] for premise, hypothesis, _ in batch])
        (max_length_premise, max_length_hypothesis), idxs = torch.max(sizes, dim=0)
        
        # add padding to shorter sentences than longest within minibatch
        batch_size = len(batch)
        p = torch.LongTensor(max_length_premise, batch_size)
        h = torch.LongTensor(max_length_hypothesis, batch_size)
        l = torch.LongTensor(batch_size)
        
        cnt = 0
        for premise, hypothesis, lbl in batch:
            l[cnt] = lbl
            p[:,cnt] = self.pad(premise, max_length_premise)
            h[:,cnt] = self.pad(hypothesis, max_length_hypothesis)
            cnt +=1
        
        return (p, h, l)


def evaluate(model, data, size, padding_token):
    """
    Evaluate the given model with the given data in terms of accuracy.
    """
    
    loader = DataLoader(data, 
                        drop_last = False,    # drops last batch if it is incomplete
                        batch_size=size, 
                        shuffle=False, 
                        #num_workers=0, 
                        collate_fn=CollocateBatch(padding_token))
    correct = 0
    
    for i_batch, (batch_p, batch_h, batch_lbl) in enumerate(loader):
        predictions = classifier(autograd.Variable(cuda_wrap(batch_p)),
                                 autograd.Variable(cuda_wrap(batch_h))
                                ).data
                                 
        _, predicted_idx = torch.max(predictions, dim=1)
        correct += torch.sum(torch.eq(batch_lbl, predicted_idx))
    
    # Accuracy
    return correct / len(data)




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
    
    def __init__(self, pretrained_embeddings, dimen_hidden, dimen_out, dimen_sent_encoder = [64,128,256], nonlinearity=F.relu, dropout=0.1):
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
        
        if len(dimen_sent_encoder) != 3:
            raise Exception('must have three values for dimen_sent_encoder.')

        self.nonlinearity = nonlinearity
                
        self.embeddings = nn.Embedding(pretrained_embeddings.shape[0], pretrained_embeddings.shape[1])
        # Use pretrained values
        self.embeddings.weight.data.copy_(cuda_wrap(torch.from_numpy(pretrained_embeddings)))
                                               
        # since it is bidirectional, use half size of wanted dimensions
        self.sent_encoder = SentenceEncoder(pretrained_embeddings.shape[1], dimen_sent_encoder[0], dimen_sent_encoder[1], dimen_sent_encoder[2])          
                
        # 3 layer Feedforward 
        dimen_sent_repr = dimen_sent_encoder[2] * 2 # multiplication because bidirectional
        self.hidden1 = nn.Linear(dimen_sent_repr * 4, dimen_hidden) # multiplication because of feature concatenation
        self.hidden2 = nn.Linear(dimen_hidden, dimen_hidden)
        self.hidden3 = nn.Linear(dimen_hidden, dimen_out)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, sent1, sent2):
        batch_size = sent1.size()[1]
        
        # Map to embeddings
        embedded1 = self.embeddings(sent1)
        embedded2 = self.embeddings(sent2)
        
        # Get sentence representation
        sent1_representation = self.sent_encoder(embedded1)
        sent2_representation = self.sent_encoder(embedded2)
        
        # Max pooling
        sent1_representation, _ = torch.max(sent1_representation, dim=0)
        sent2_representation, _ = torch.max(sent2_representation, dim=0)
        
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
        out = self.nonlinearity(self.hidden2(out))
        out = self.dropout2(out)
        out = self.hidden3(out)
        tag_scores = F.softmax(out)
        return tag_scores


def print_params(model, label):
    print(label)
    print('params:')
    for p in model.parameters():
        print(p)

def train_model(model, train_set, dev_set, padding_token, loss_fn, lr, epochs, batch_size, validate_after=5):
    
    loader_train = DataLoader(train_set, 
                        drop_last = True,    # drops last batch if it is incomplete
                        batch_size=batch_size, 
                        shuffle=True, 
                        #num_workers=0, 
                        collate_fn=CollocateBatch(padding_token))
    
    # remember for weight decay
    start_lr = lr
    
    # switch between train/eval mode due to dropout
    model.train()
    start = time.time()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print('Train epoch', epoch + 1)
        
        total_loss = 0
        for i_batch, (batch_p, batch_h, batch_lbl) in enumerate(loader_train):
            model.zero_grad()
            optimizer.zero_grad()
            
            var_p = autograd.Variable(cuda_wrap(batch_p))
            var_h = autograd.Variable(cuda_wrap(batch_h))
            var_label = autograd.Variable(cuda_wrap(batch_lbl))
            
            prediction = model(var_p, var_h)
            loss = loss_fn(prediction, var_label)
            
            total_loss += loss.data
            
            loss.backward()
            optimizer.step()
            
        # apply half decay learn rate (copied from original paper)
        decay = epoch // 2
        lr = start_lr / (2 ** decay)  
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            
        # Validate when specified and after the last epoch
        if (epoch % validate_after == 0) or (epoch == epochs - 1):
            model.eval()
            print('Accuracy on train data:', evaluate(model, train_set, batch_size, padding_token))
            print('Accuracy on dev data:', evaluate(model, dev_set, batch_size, padding_token))
            model.train()
        
        print('Loss:', total_loss)
        print('Running time:', time.time() - start, 'seconds.')
    

# How much data to load
SIZE_TRAIN = 30
SIZE_DEV = 5

lr = 0.00005
        
snli_train = SNLIDataset(PATH_TRAIN_DATA, embedding_holder, max_size=SIZE_TRAIN)
snli_dev = SNLIDataset(PATH_DEV_DATA, embedding_holder, max_size=SIZE_DEV)


classifier = cuda_wrap(EntailmentClassifier(embedding_holder.embeddings, 
                                            dimen_hidden=800, 
                                            dimen_out=3, 
                                            nonlinearity=F.relu, 
                                            dropout=0.1))


train_model(classifier, snli_train, snli_dev, 
            embedding_holder.padding(),
            F.cross_entropy, lr, epochs=30, batch_size=5)




