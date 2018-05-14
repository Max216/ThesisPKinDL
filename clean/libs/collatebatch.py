'''
Use that to automatically apply padding in the dataset
'''

import torch
import numpy as np

from libs import model as m

class CollateBatch(object):
    '''
    Applies padding to shorter sentences within a minibatch.
    '''
    
    def __init__(self, padding_token):
        self.padding_token = padding_token
        
    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(self.padding_token)])
        
    def __call__(self, batch):

        premise, hypothesis, label, len_p, len_h = [list(a) for a in zip(*batch)]

        max_len_premise = int(np.max(len_p))
        max_len_hypothesis = int(np.max(len_h))

        p = torch.cat([self.pad(m.cuda_wrap(p_sent), max_len_premise).view(-1,1) for p_sent in premise], dim=1)
        h = torch.cat([self.pad(m.cuda_wrap(h_sent), max_len_hypothesis).view(-1,1) for h_sent in hypothesis], dim=1)
        l = m.cuda_wrap(torch.LongTensor(label))

        print('return', l)
        return p,h,l

class CollateBatchIncludingSents(object):
    '''
    Applies padding to shorter sentences within a minibatch.
    '''
    
    def __init__(self, padding_token):
        self.padding_token = padding_token
        
    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(self.padding_token)])
        
    def __call__(self, batch):

        premise, hypothesis, label, len_p, len_h, p_sent, h_sent = [list(a) for a in zip(*batch)]

        max_len_premise = int(np.max(len_p))
        max_len_hypothesis = int(np.max(len_h))

        p = torch.cat([self.pad(m.cuda_wrap(p_sent), max_len_premise).view(-1,1) for p_sent in premise], dim=1)
        h = torch.cat([self.pad(m.cuda_wrap(h_sent), max_len_hypothesis).view(-1,1) for h_sent in hypothesis], dim=1)
        l = m.cuda_wrap(torch.LongTensor(label))

        return p,h,l, p_sent, h_sent



class CollateBatchIncludingSentsIncludingReplacements(object):
    '''
    Applies padding to shorter sentences within a minibatch.
    '''
    
    def __init__(self, padding_token):
        self.padding_token = padding_token
        
    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(self.padding_token)])
        
    def __call__(self, batch):

        premise, hypothesis, label, len_p, len_h, p_sent, h_sent, rep1, rep2 = [list(a) for a in zip(*batch)]

        max_len_premise = int(np.max(len_p))
        max_len_hypothesis = int(np.max(len_h))

        p = torch.cat([self.pad(m.cuda_wrap(p_sent), max_len_premise).view(-1,1) for p_sent in premise], dim=1)
        h = torch.cat([self.pad(m.cuda_wrap(h_sent), max_len_hypothesis).view(-1,1) for h_sent in hypothesis], dim=1)
        l = m.cuda_wrap(torch.LongTensor(label))

        return p,h,l, p_sent, h_sent, rep1, rep2

class CollateBatchId(object):
    '''
    Applies padding to shorter sentences within a minibatch.
    '''
    
    def __init__(self, padding_token):
        self.padding_token = padding_token
        
    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(self.padding_token)])
        
    def __call__(self, batch):

        premise, hypothesis, label, len_p, len_h, p_id, h_id = [list(a) for a in zip(*batch)]

        max_len_premise = int(np.max(len_p))
        max_len_hypothesis = int(np.max(len_h))

        p = torch.cat([self.pad(m.cuda_wrap(p_sent), max_len_premise).view(-1,1) for p_sent in premise], dim=1)
        h = torch.cat([self.pad(m.cuda_wrap(h_sent), max_len_hypothesis).view(-1,1) for h_sent in hypothesis], dim=1)
        l = m.cuda_wrap(torch.LongTensor(label))

        return p,h,l, p_id, h_id


class CollateBatchSentWord(object):
    '''
    Applies padding to shorter sentences within a minibatch.
    '''

    def __init__(self, padding_token):
        self.padding_token = padding_token
        
    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(self.padding_token)])
                
    def __call__(self, batch):

        sent, word, label, sent_len = [list(a) for a in zip(*batch)]

        max_sent_len = int(np.max(sent_len))


        sents = torch.cat([self.pad(m.cuda_wrap(s), max_sent_len).view(-1,1) for s in sent], dim=1)
        #sents = torch.cat([self.pad(m.cuda_wrap(sent), max_sent_len).view(-1,1) for s in sent], dim=1)
        w = m.cuda_wrap(torch.LongTensor(word))
        l = m.cuda_wrap(torch.LongTensor(label))

        return sents,w,l



        #sizes = torch.LongTensor([[len(premise), len(hypothesis)] for premise, hypothesis, _, _2, _3 in batch])
        #(max_length_premise, max_length_hypothesis), idxs = torch.max(sizes, dim=0)
        #print('nbatch', batch)
        #sizes = batch[:,-2:]
        #print('sizes', sizes)
        #1/0
        #maxlen, idxs = torch.max(sizes, dim=0)
        #maxlen = maxlen.view(-1)
        #max_length_premise = maxlen[0]
        #max_length_hypothesis = maxlen[1]
        
        # add padding to shorter sentences than longest within minibatch
        #batch_size = len(batch)
        #p = torch.LongTensor(max_length_premise, batch_size)
        #h = torch.LongTensor(max_length_hypothesis, batch_size)
        #l = torch.LongTensor(batch_size)
        
        #cnt = 0
        #for premise, hypothesis, lbl,_,_2 in batch:
        #    l[cnt] = lbl
        #    p[:,cnt] = self.pad(premise, max_length_premise)
        #    h[:,cnt] = self.pad(hypothesis, max_length_hypothesis)
        #    cnt +=1

        #print('final p', p)
        #1/0
        
        #return (p, h, l)