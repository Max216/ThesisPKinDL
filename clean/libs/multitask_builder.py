import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time

import collections

from libs import model as m

DEFAULT_LR = 0.0002

def _zero_grad_nothing(dummy):
    pass
def _zero_grad_obj(obj):
    obj.zero_grad()

class CollateBatchMultiTask(object):
    '''
    Applies padding to shorter sentences within a minibatch.
    '''
    
    def __init__(self):
        pass
                
    def __call__(self, batch):

        data, lbl = [list(a) for a in zip(*batch)]

        data = torch.cat(data, 0)
        lbl = m.cuda_wrap(torch.LongTensor(lbl))


        return data, lbl

class SentMTDataset(Dataset):
    '''
    Dataset format to give to classifier
    '''

    def __init__(self, samples):
        '''
        Create a new dataset for the given samples
        :param samples              parsed samples of the form [(premise, hypothesis, label)] (all strings)
        :paraam embedding_holder    To map from word to number
        :param tag_to_idx         dictionary mapping the string label to a number
        '''
        
        self.converted_samples = samples

    def __len__(self):
        return len(self.converted_samples)

    def __getitem__(self, idx):
        return self.converted_samples[idx]

class MTNetworkSingleLayer(nn.Module):
    """
    Map the embedding to a smaller represerntation
    """

    def __init__(self, classifier, input_dim, output_dim):
        """
        Initialize a new network to create representations based on WordNet information.
        :param pretrained_embeddings    pretrained embedding matrix
        :param hidden_layer_dimension   amoount of hidden nodes
        :param representations          size of the resulting representation
        """
        super(MTNetworkSingleLayer, self).__init__()
        self.classifier = classifier
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, samples):
        #sentence_representation = self.classifier.forward_sent(sent)
        #batch_size = sentence_representation.size()[0]
        #word_representation = self.classifier.lookup_word(target_word).view(batch_size, -1)

        #feed_forward_input = torch.cat((sentence_representation, word_representation), 1)
        
        return F.softmax(self.layer(samples))

    def lookup_word(self, w_idx):
        word = self.classifier.lookup_word(w_idx)
        return word

class MTNetworkTwoLayer(nn.Module):
    """
    Map the embedding to a smaller represerntation
    """

    def __init__(self, classifier, input_dim, hidden_dim, output_dim):
        """
        Initialize a new network to create representations based on WordNet information.
        :param pretrained_embeddings    pretrained embedding matrix
        :param hidden_layer_dimension   amoount of hidden nodes
        :param representations          size of the resulting representation
        """
        super(MTNetworkTwoLayer, self).__init__()
        self.classifier = classifier
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, samples):
        #sentence_representation = self.classifier.forward_sent(sent)
        #batch_size = sentence_representation.size()[0]
        #word_representation = self.classifier.lookup_word(target_word).view(batch_size, -1)

        #feed_forward_input = torch.cat((sentence_representation, word_representation), 1)
        
        out1 = F.relu(self.layer1(samples))
        return F.softmax(self.layer2(out1))

    def lookup_word(self, w_idx):
        word = self.classifier.lookup_word(w_idx)
        return word

class MultitaskBuilder:
    """
    Create all things required for the multitask training
    """

    def __init__(self, params, lr, multitask_targets, classifier, embedding_holder):
        self._multitask_network = params['multitask_network']
        self._optimizer = params['optimizer'](classifier, self._multitask_network, lr)
        self._loss_fn = params['loss_fn']
        self._loss_fn_multitask = params['loss_fn_multitask']
        self._stop_idx = embedding_holder.stop_idx()
        self._classifier = classifier
        self._word_dict = embedding_holder.reverse()
        
        target_words, target_labels = multitask_targets
        print('init:', len(target_words), len(target_labels))
        self._target_words = multitask_targets
        self._target_labels = target_labels

        # helper functions
        if self._multitask_network == None:
            self._zero_grad = _zero_grad_nothing
        else:
            self._zero_grad = _zero_grad_obj

        

        #self._res_word_vec_in_sent = m.cuda_wrap(torch.LongTensor([0 for i in]))

    def zero_grad_multitask(self):
        """ Reset gradients for multitask network and optimizer """
        self._optimizer.zero_grad()
        self._zero_grad(self._multitask_network)


    def optimizer_step(self):
        """ Train step based on optimizer function """
        self._optimizer.step()

    def new_evaluation(self):
        """ Reset evaluation values to start a new evaluation """
        self._correct_multitask_samples = 0
        self._total_count_multitask_samples = 0

    def train(self):
        """ set multitask network in train mode """
        if self._multitask_network != None:
            self._multitask_network.train()

    def eval(self):
        """ set multitask network in evaluation mode """
        if self._multitask_network != None:
            self._multitask_network.eval()

    def add_evaluation(self, premise_info, hypothesis_info):
        """ evaluate the samples and remember the results """
        eval_data, counts = self.get_all_multitask_samples(premise_info, hypothesis_info)
        for samples, lbls in eval_data:
            pred = self._multitask_network(samples)
            _, predicted_idx = torch.max(pred.data, dim=1)
            self._correct_multitask_samples += torch.sum(torch.eq(predicted_idx, lbls))
        self._total_count_multitask_samples += counts

    def print_evaluation(self):
        """ print evaluation """
        print(self._correct_multitask_samples / self._total_count_multitask_samples)
        self._correct_multitask_samples = 0
        self._total_count_multitask_samples = 0

    def loss(self, snli_loss, premise_info, hypothesis_info, premise_ids, hyp_ids):
        """ Calculate the loss for thee gradient """
        #print('now create loss')
        #print(list(self._classifier.sent_encoder.lstm1.parameters())[0])
        multitask_loss = self._loss_fn_multitask(premise_info, hypothesis_info, premise_ids, hyp_ids, self)
        #print('multitask loss', multitask_loss)
        return self._loss_fn(snli_loss, multitask_loss)

    def adjust_lr(self, new_lr):
        """ Adjust the learnrate """
        for pg in self._optimizer.param_groups:
            pg['lr'] = new_lr

    def get_all_multitask_samples(self, premise_info, hypothesis_info, premise_ids, hyp_ids):
        #start = time.time()
        """ Create a dataset based on wordnet and the given sentences """
        premise_var, premise_repr = premise_info
        hyp_var, hyp_repr = hypothesis_info

        #print('premise repr',premise_repr.size())
        samples = []
        for i in range(len(premise_ids)):
            print('##')
            _id = premise_ids[i]
            print('_id',_id)
            print('len target words:', len(self._target_words))
            print('target_words[i].size()', self._target_words[_id].size())
            embds = self._multitask_network.lookup_word(autograd.Variable(m.cuda_wrap(self._target_words[_id])))
            print('embds.size()', embds.size())
            single_repr = premise_repr[i,:]
            print('single_repr.size()', single_repr.size())

            duplicated_repr = torch.cat([single_repr for i in range(embds.size()[1])], 1)
            print('duplicated_repr.size()', duplicated_repr.size())

            concatenated = torch.cat((duplicated_repr, embds), 0)
            print('concatenated size()', concatenated.size())

        return DataLoader(SentMTDataset(samples), drop_last=False, batch_size=512, shuffle=False, collate_fn=CollateBatchMultiTask()), len(samples)
        # samples = []

        # def add(sent_repr, w_idx, lbl):
        #     #print('lookup word', w)
        #     embd = self._multitask_network.lookup_word(autograd.Variable(m.cuda_wrap(w), requires_grad=False)).view(-1)
        #     #print('embd word',embd.size())
        #     #print('repr dim', sent_repr.size())
        #     #print('concatenated', torch.cat((sent_repr, embd), 0).size())
        #     samples.append((torch.cat((sent_repr, embd), 0).view(1,-1), lbl))

        # for i in range(premise_var.size()[1]):
        #     current_sent_indizes = premise_var.data[:,i]
        #     word_set = set()
        #     for j in range(current_sent_indizes.size()[0]):
        #         w_idx = current_sent_indizes[j]
        #         if w_idx == self._stop_idx:
        #             break
        #         else:
        #             word_set.add(w_idx)

        #     entailing_words = set()
        #     contradicting_words = set()
        #     #print('# premise start')
        #     for w_idx in list(word_set):
        #         entailing_words.update(self._in_sent_samples[w_idx])
        #         contradicting_words.update(self._not_in_sent_samples[w_idx])
        #         #print('word:', self._word_dict[w_idx], '-> (e)', [self._word_dict[www[0]] for www in self._in_sent_samples[w_idx]])
        #         #print('word:', self._word_dict[w_idx], '-> (c)', [self._word_dict[www[0]] for www in self._not_in_sent_samples[w_idx]])
        #     #print('# premise end')
            
        #     contradicting_words = list(contradicting_words - entailing_words)
        #     entailing_words = list(entailing_words) 

        #     for w in contradicting_words:
        #         #print('premise size',premise_var.size())
        #         #print('premise_var[i,:]',premise_var[:,i])
                
        #         add(premise_repr[i,:], w, 0)
        #     for w in entailing_words:
        #         #print('premise size',premise_var.size())
        #         #print('premise_var[i,:]',premise_var[:,i])
        #         add(premise_repr[i,:], w, 1)

        # for i in range(hyp_var.size()[1]):
        #     current_sent_indizes = hyp_var.data[:,i]
        #     word_set = set()
        #     for j in range(current_sent_indizes.size()[0]):
        #         w_idx = current_sent_indizes[j]
        #         if w_idx == self._stop_idx:
        #             break
        #         else:
        #             word_set.add(w_idx)

        #     entailing_words = set()
        #     contradicting_words = set()
        #     for w_idx in list(word_set):
        #         entailing_words.update(self._in_sent_samples[w_idx])
        #         contradicting_words.update(self._not_in_sent_samples[w_idx])
            
        #     contradicting_words = list(contradicting_words - entailing_words)
        #     entailing_words = list(entailing_words) 

        #     for w in contradicting_words:
        #         add(hyp_repr[i,:], w, 0)
        #     for w in entailing_words:
        #         #print(hyp_repr[i,:])
        #         add(hyp_repr[i,:], w, 1)

        #print('samples')
        #print(samples)
        #print('# samples:', len(samples))

        #print('time:', time.time() - start)
        



    def predict(self, sent_reprs):
        """
        predict the samples
        """
        return self._multitask_network(sent_reprs)

#
# Dummys
#
def nothing(dummy1=None, dummy2=None, dummy3=None):
    return None

#
# Optimizer creators
#
def get_optimizer_snli_only(classifier, multitask_network, lr):
    return optim.Adam(classifier.parameters(), lr=lr)

def get_optimizer_multitask_only(classifier, multitask_network, lr):
    return optim.Adam(multitask_network.parameters(), lr=lr) 

#
# Combining Loss functions
#
def loss_snli_only(snli_loss, multitask_loss):
    return snli_loss

def loss_multitask_only(snli_loss, multitask_loss):
    print('multitask loss', multitask_loss)
    return multitask_loss


#
# Loss function for MultiTask
#
def loss_multitask_reweighted(premise_info, hypothesis_info, premise_ids, hyp_ids, builder):
    """Average the loss over the batches of all samples created from these sentence pairs"""

    #premise_var, premise_repr = premise_info
    #hyp_var, hyp_repr = hypothesis_info
    samples, sample_count = builder.get_all_multitask_samples(premise_info, hypothesis_info, premise_ids, hyp_ids)

    loss = autograd.Variable(m.cuda_wrap(torch.FloatTensor([0])))
    sample_factor = 1/sample_count
    for batch_samples, batch_lbl in samples:
        
        #print('batch sample')
        #print(batch_samples)
        #print(batch_samples.size())

        batch_size = batch_samples.size()[0]
        batch_factor = sample_factor * batch_size

        #words_var = autograd.Variable(batch_words, requires_grad=False)
        lbl_var = autograd.Variable(m.cuda_wrap(batch_lbl))

        predictions = builder.predict(batch_samples)
        #print('predicted', predictions.size())
        batch_loss = F.cross_entropy(predictions, lbl_var)
        multiplicator_batch_factor = autograd.Variable(batch_loss.data.clone().fill_(batch_factor))
        loss += batch_loss * multiplicator_batch_factor

    return loss


#
# Multitask network factories
#
def get_multitask_nw(classifier, layers=1):
    dim_sent = 1200
    dim_word = 300

    dim_input = dim_word + dim_sent

    if layers == 1:
        mt_network = MTNetworkSingleLayer(classifier, dim_input, 2)
    else:
        mt_network = MTNetworkTwoLayer(classifier, dim_input, 600, 2)

    return m.cuda_wrap(mt_network)

#
# Factory
#
def get_builder(classifier, mt_type, mt_target, lr, embedding_holder):
    params = dict()
    if mt_type == 'test_snli':
        # ignore multitask, verify that SNLI training works
        params['multitask_network'] = None
        params['optimizer'] = get_optimizer_snli_only
        params['loss_fn_multitask'] = nothing
        params['loss_fn'] = loss_snli_only
        #params['target'] = mt_target.get_targets()

        return MultitaskBuilder(params, lr, mt_target, classifier, embedding_holder)

    elif mt_type == 'test_mt':
        # ignore snli, verify that Multitask works
        params['multitask_network'] = get_multitask_nw(classifier, layers=2)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_multitask_only
        #params['target'] = mt_target.get_targets()

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)