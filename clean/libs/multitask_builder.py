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

def _zero_grad_nothing(dummy):
    pass
def _zero_grad_obj(obj):
    obj.zero_grad()


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False

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

    def layers(self):
        return [self.layer]

class MTNetworkSingleLayerDropout(nn.Module):
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
        super(MTNetworkSingleLayerDropout, self).__init__()
        self.classifier = classifier
        self.layer = nn.Linear(input_dim, output_dim)
        self.dropout1 = nn.Dropout(p=0.1)

    def forward(self, samples):
        #sentence_representation = self.classifier.forward_sent(sent)
        #batch_size = sentence_representation.size()[0]
        #word_representation = self.classifier.lookup_word(target_word).view(batch_size, -1)

        #feed_forward_input = torch.cat((sentence_representation, word_representation), 1)
        
        return F.softmax(self.layer(self.dropout1(samples)))

    def lookup_word(self, w_idx):
        word = self.classifier.lookup_word(w_idx)
        return word

    def layers(self):
        return [self.layer]

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

    def layers(self):
        return [self.layer1, self.layer2]

class MTNetworkTwoLayerDoubleDropout(nn.Module):
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
        super(MTNetworkTwoLayerDoubleDropout, self).__init__()
        self.classifier = classifier
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, samples):
        #sentence_representation = self.classifier.forward_sent(sent)
        #batch_size = sentence_representation.size()[0]
        #word_representation = self.classifier.lookup_word(target_word).view(batch_size, -1)

        #feed_forward_input = torch.cat((sentence_representation, word_representation), 1)
        
        out1 = F.relu(self.layer1(self.dropout1(samples)))
        return F.softmax(self.layer2(self.dropout2(out1)))

    def lookup_word(self, w_idx):
        word = self.classifier.lookup_word(w_idx)
        return word

    def layers(self):
        return [self.layer1, self.layer2]


class MTNetworkTwoLayerSingleDropout(nn.Module):
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
        super(MTNetworkTwoLayerSingleDropout, self).__init__()
        self.classifier = classifier
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(p=0.1)

    def forward(self, samples):
        #sentence_representation = self.classifier.forward_sent(sent)
        #batch_size = sentence_representation.size()[0]
        #word_representation = self.classifier.lookup_word(target_word).view(batch_size, -1)

        #feed_forward_input = torch.cat((sentence_representation, word_representation), 1)
        
        out1 = F.relu(self.layer1(samples))
        return F.softmax(self.layer2(self.dropout1(out1)))

    def lookup_word(self, w_idx):
        word = self.classifier.lookup_word(w_idx)
        return word

    def layers(self):
        return [self.layer1, self.layer2]

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
        self._regularization = 1
        self._regularization_update = params['regularization_update']
        self._mask_sentence = False
        if 'mask_sent' in params:
            self._mask_sentence = True
            target_words, target_labels, target_has_content, source_positions = multitask_targets
            self._source_word_positions = source_positions
        else:
            target_words, target_labels, target_has_content = multitask_targets

        if 'after_epoch' in params:
            self._after_epoch = params['after_epoch']
        else:
            self._after_epoch = None
        
        #target_words, target_labels, target_has_content = multitask_targets
        self._target_words = target_words
        self._target_labels = target_labels
        self._has_content = target_has_content

        # helper functions
        if self._multitask_network == None:
            self._zero_grad = _zero_grad_nothing
        else:
            self._zero_grad = _zero_grad_obj

        print('mask sent:', self._mask_sentence)

        #self._res_word_vec_in_sent = m.cuda_wrap(torch.LongTensor([0 for i in]))

    def zero_grad_multitask(self):
        """ Reset gradients for multitask network and optimizer """
        self._optimizer.zero_grad()
        self._zero_grad(self._multitask_network)
        self._zero_grad(self._classifier)

    def next_epoch(self, epoch):
        reg_general, reg_snli, reg_mt = self._regularization_update(epoch, self._regularization)
        self._regularization = reg_general
        self.reg_snli = reg_snli
        self.reg_mt = reg_mt

        print('Regularization for multitask:', self.reg_mt)
        print('Regularization for SNLI:', self.reg_snli)

        if self._after_epoch is not None:
            print('After epoch!')
            self._after_epoch(self, epoch)

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

    def add_evaluation(self, premise_info, hypothesis_info, premise_ids, hyp_ids, activations=None):
        """ evaluate the samples and remember the results """
        eval_data, counts = self.get_all_multitask_samples(premise_info, hypothesis_info, premise_ids, hyp_ids, activations=activations)
        for samples, lbls in eval_data:
            pred = self._multitask_network(samples)
            _, predicted_idx = torch.max(pred.data, dim=1)
            self._correct_multitask_samples += torch.sum(torch.eq(predicted_idx, m.cuda_wrap(lbls)))
        self._total_count_multitask_samples += counts

    def print_evaluation(self):
        """ print evaluation """
        print('Multi-task:',self._correct_multitask_samples / self._total_count_multitask_samples)
        self._correct_multitask_samples = 0
        self._total_count_multitask_samples = 0

    def loss(self, snli_loss, premise_info, hypothesis_info, premise_ids, hyp_ids, activations=None):
        """ Calculate the loss for thee gradient """
        #print('now create loss')
        #print(list(self._classifier.sent_encoder.lstm1.parameters())[0])
        multitask_loss = self._loss_fn_multitask(premise_info, hypothesis_info, premise_ids, hyp_ids, self, activations=activations)
        #print('multitask loss', multitask_loss)
        return self._loss_fn(snli_loss, multitask_loss, self)

    def adjust_lr(self, new_lr):
        """ Adjust the learnrate """
        for pg in self._optimizer.param_groups:
            pg['lr'] = new_lr

    def get_all_multitask_samples(self, premise_info, hypothesis_info, premise_ids, hyp_ids, activations=None):
        #start = time.time()
        """ Create a dataset based on wordnet and the given sentences """
        premise_var, premise_repr = premise_info
        hyp_var, hyp_repr = hypothesis_info
        if not self._mask_sentence:

            #print('premise repr',premise_repr.size())
            samples = []
            count = 0
            for i in range(len(premise_ids)):
                #print('##')
                _id = premise_ids[i]
                #print('_id',_id)

                if self._has_content[_id]:
                    #print('len target words:', len(self._target_words))
                    #print('target_words[i].size()', self._target_words[_id].size())
                    embds = self._multitask_network.lookup_word(autograd.Variable(m.cuda_wrap(self._target_words[_id])))
                    embds = embds.view(embds.size()[0], -1)
                    #print('embds.size()', embds.size())
                    single_repr = premise_repr[i,:].view(1,-1)
                    #print('single_repr.size()', single_repr.size())

                    duplicated_repr = torch.cat([single_repr for i in range(embds.size()[0])], 0)
                    #print(duplicated_repr)
                    #print('duplicated_repr.size()', duplicated_repr.size())

                    concatenated_input = torch.cat((duplicated_repr, embds), 1)
                    #print('concatenated_input size()', concatenated_input.size())
                    labels = self._target_labels[_id]
                    #print('labels.size()', labels.size())

                    samples.append((concatenated_input, labels))
                    count += labels.size()[0]
                else:
                    #print('Skipping one')
                    pass

            for i in range(len(hyp_ids)):
                #print('##')
                _id = hyp_ids[i]
                #print('_id',_id)

                if self._has_content[_id]:
                    #print('len target words:', len(self._target_words))
                    #print('target_words[i].size()', self._target_words[_id].size())
                    embds = self._multitask_network.lookup_word(autograd.Variable(m.cuda_wrap(self._target_words[_id])))
                    embds = embds.view(embds.size()[0], -1)
                    #print('embds.size()', embds.size())
                    single_repr = hyp_repr[i,:].view(1,-1)
                    #print('single_repr.size()', single_repr.size())

                    duplicated_repr = torch.cat([single_repr for i in range(embds.size()[0])], 0)
                    #print(duplicated_repr)
                    #print('duplicated_repr.size()', duplicated_repr.size())

                    concatenated_input = torch.cat((duplicated_repr, embds), 1)
                    #print('concatenated_input size()', concatenated_input.size())
                    labels = self._target_labels[_id]
                    #print('labels.size()', labels.size())

                    samples.append((concatenated_input, labels))
                    count += labels.size()[0]
                else:
                    #print('Skipping one')
                    pass

            return samples, count
        else:
            #print('premise_var', premise_var.size())
            samples = []
            count = 0
            for i in range(len(premise_ids)):
                _id = premise_ids[i]

                if self._has_content[_id]:
                    #print('source word positions for masking', self._source_word_positions[_id])
                    #print('target words, each of them align', self._target_words[_id])
                    #print('target labels to predict', self._target_labels[_id])

                    sentence_samples = []

                    for iidx in range(len(self._source_word_positions[_id])):
                        source_positions = self._source_word_positions[_id][iidx]
                        labels = self._target_labels[_id]
                        target_words = self._multitask_network.lookup_word(autograd.Variable(m.cuda_wrap(self._target_words[_id][iidx])))
                        target_words = target_words.view(target_words.size()[0], -1)
                        single_repr = premise_repr[i,:].view(1,-1)
                        single_act = activations[0][i,:].view(1,-1)
                        binary_masks = torch.cat([(single_act==sp).float() for sp in source_positions], dim=0)
                        final_mask, _ = torch.max(binary_masks, dim=0)

                        # mask single rep
                        masked_repr = single_repr * final_mask
                        duplicated_repr = torch.cat([masked_repr for i in range(target_words.size()[0])], 0)
                        
                        concatenated = torch.cat([duplicated_repr, target_words], dim=1)
                        sentence_samples.append(concatenated)


                    concatenated_sentence_samples = torch.cat(sentence_samples, dim=0)
                    count += concatenated_sentence_samples.size()[0]
                    samples.append((concatenated_sentence_samples, labels))

                else:
                    pass

            for i in range(len(hyp_ids)):
                #print('##')
                _id = hyp_ids[i]
                #print('_id',_id)

                if self._has_content[_id]:
                    #print('len target words:', len(self._target_words))
                    #print('target_words[i].size()', self._target_words[_id].size())
                    sentence_samples = []

                    for iidx in range(len(self._source_word_positions[_id])):
                        source_positions = self._source_word_positions[_id][iidx]
                        labels = self._target_labels[_id]
                        target_words = self._multitask_network.lookup_word(autograd.Variable(m.cuda_wrap(self._target_words[_id][iidx])))
                        target_words = target_words.view(target_words.size()[0], -1)
                        single_repr = hyp_repr[i,:].view(1,-1)
                        single_act = activations[1][i,:].view(1,-1)
                        binary_masks = torch.cat([(single_act==sp).float() for sp in source_positions], dim=0)
                        final_mask, _ = torch.max(binary_masks, dim=0)

                        # mask single rep
                        masked_repr = single_repr * final_mask
                        duplicated_repr = torch.cat([masked_repr for i in range(target_words.size()[0])], 0)
                        
                        concatenated = torch.cat([duplicated_repr, target_words], dim=1)
                        sentence_samples.append(concatenated)


                    concatenated_sentence_samples = torch.cat(sentence_samples, dim=0)
                    count += concatenated_sentence_samples.size()[0]
                    samples.append((concatenated_sentence_samples, labels))
                else:
                    #print('Skipping one')
                    pass
            return samples, count

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
    return optim.Adam(list(set(list(multitask_network.parameters()) + list(classifier.parameters()))), lr=lr) 

#
# Combining Loss functions
#
def loss_snli_only(snli_loss, multitask_loss, builder, activations=None):
    return snli_loss

def loss_multitask_only(snli_loss, multitask_loss, builder, activations=None):
    #print('multitask loss', multitask_loss.data[0])
    return multitask_loss

def loss_equal_both(snli_loss, multitask_loss, builder, activations=None):
    return (snli_loss + multitask_loss) / 2

def loss_on_regularization(snli_loss, multitask_loss, builder, activations=None):
    return builder.reg_snli * snli_loss + builder.reg_mt * multitask_loss


#
# Loss function for MultiTask
#

def loss_multitask_reweighted(premise_info, hypothesis_info, premise_ids, hyp_ids, builder, activations):
    """Average the loss over the batches of all samples created from these sentence pairs"""

    #premise_var, premise_repr = premise_info
    #hyp_var, hyp_repr = hypothesis_info
    samples, sample_count = builder.get_all_multitask_samples(premise_info, hypothesis_info, premise_ids, hyp_ids, activations=activations)

    loss = []#autograd.Variable(m.cuda_wrap(torch.FloatTensor([0])))
    #batch_sizes = 0
    for batch_samples, batch_lbl in samples:
        
        #print('batch sample')
        #print(batch_samples)
        #print(batch_samples.size())

        batch_size = batch_samples.size()[0]
        batch_factor = batch_size / sample_count

        #words_var = autograd.Variable(batch_words, requires_grad=False)
        lbl_var = autograd.Variable(m.cuda_wrap(batch_lbl))

        predictions = builder.predict(batch_samples)
        #print('predictions:', predictions)
        #print('labels', lbl_var)
        #print('#####')
        #print('predicted', predictions.size())
        batch_loss = F.cross_entropy(predictions, lbl_var)
        #print('Batch loss:', batch_loss)
        loss.append(batch_loss)
        #return batch_loss
        loss.append(batch_loss * batch_factor) #* multiplicator_batch_factor
        #batch_sizes += 1
        #batch_loss.backward()
        #builder._optimizer.step()

    return torch.sum(torch.cat(loss, 0)) #/ batch_sizes


#
# Regularization functions
#
def dummy_regularization(epoch, regularization):
    return (1,2,3)

def mt_both_finetune_200_it10(epoch, regularization):
    vals = [0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0,0,0,0]

    factor_multitask = vals[epoch]
    return (1, 1 - factor_multitask, factor_multitask)

def first_mt_then_all_then_snli_it10(epoch, regularization):
    # general, snli, mt
    if epoch == 0:
        print('First tune MultiTask')
        regularization = 0
    elif epoch == 9:
        print('Lastly tune SNLI')
        regularization = 1
    else:
        regularization = 0.5

    factor_multitask = 1 - regularization
    return (regularization, 1 - factor_multitask, factor_multitask)

def constant_25_percent(epoch, regularization):
    return (1.0, 0.75, 0.25) 

def decrease_strong_mt_it10(epoch, regularization):
    vals = [0.75, 0.75, 0.5, 0.5, 0.125, 0.125, 0,0,0,0]
    factor_multitask = vals[epoch]
    return (1 - factor_multitask, 1 - factor_multitask, factor_multitask)

def train_tailvtail_shape_it10(epoch, regularization):
    # first 3 iterations for good sentence representation, last 3 epochs for fine tuning SNLI
    if epoch < 3 or epoch >= 7:
        return (1.0,1.0,0.0)

    # remaining iterations: 3,4,5,6
    factor_multitask = 0.0
    if epoch == 3:
        factor_multitask = 0.5
    elif epoch == 4:
        factor_multitask = 1.0
    elif epoch == 5:
        factor_multitask = 0.75
    else:
        factor_multitask = 0.5

    return (1.0, 1 - factor_multitask, factor_multitask)

#
# Multitask network factories
#
def get_multitask_nw(classifier, layers=1, mlp=600):
    dim_sent = classifier.sent_encoder.sent_dim()
    dim_word = 300
    dim_nw = mlp

    dim_input = dim_word + dim_sent

    if layers == 1:
        mt_network = MTNetworkSingleLayer(classifier, dim_input, 2)
    else:
        mt_network = MTNetworkTwoLayer(classifier, dim_input, dim_nw, 2)

    return m.cuda_wrap(mt_network)

def get_multitask_nw_dropout(classifier, mlp=600):
    dim_sent = classifier.sent_encoder.sent_dim()
    dim_word = 300
    dim_nw = mlp

    return m.cuda_wrap(MTNetworkTwoLayerDoubleDropout(classifier, dim_word + dim_sent, dim_nw, 2))

def get_multitask_nw_dropout1(classifier, mlp=600):
    dim_sent = classifier.sent_encoder.sent_dim()
    dim_word = 300
    dim_nw = mlp

    return m.cuda_wrap(MTNetworkTwoLayerSingleDropout(classifier, dim_word + dim_sent, dim_nw, 2))

def get_multitask_nw_dropout_1layer(classifier):
    dim_sent = classifier.sent_encoder.sent_dim()
    dim_word = 300
    return m.cuda_wrap(MTNetworkSingleLayerDropout(classifier, dim_word + dim_sent, 2))

def freeze_after_first_epoch(epoch, classifier):
    if epoch == 1:
        for layer in classifier.layers():
            freeze_layer(layer)

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
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target, classifier, embedding_holder)

    elif mt_type == 'test_mt':
        # ignore snli, verify that Multitask works
        params['multitask_network'] = get_multitask_nw(classifier, layers=2)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_multitask_only
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'equal_snli_mt':
        print('equal_snli_mt')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, layers=2)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    
    elif mt_type == 'equal_snli_mt_2layer_10':
        print('equal_snli_mt_2layer_10')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, layers=2, mlp=10)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'equal_snli_mt_2layer_50':
        print('equal_snli_mt_2layer_50')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, layers=2, mlp=50)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'equal_snli_mt_2layer_100':
        print('equal_snli_mt_2layer_100')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, layers=2, mlp=100)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'equal_snli_mt_1layer':
        print('equal_snli_mt_1layer')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, layers=1)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    #elif mt_type == 'mt_both_snli':
    #    print('mt_both_snli')
    # #   # weight both results the same, all the time
    #    params['multitask_network'] = get_multitask_nw(classifier, layers=2)
    #    params['optimizer'] = get_optimizer_multitask_only
    #    params['loss_fn_multitask'] = loss_multitask_reweighted
    #    params['loss_fn'] = loss_equal_both
    #    params['regularization_update'] = dummy_regularization

    #    return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_snli_ddropout_600':
        print('mt_both_snli_ddropout_600')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout(classifier, mlp=600)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_snli_ddropout_300':
        print('mt_both_snli_ddropout_300')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_snli_dropout_300':
        print('mt_both_snli_dropout_300')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout1(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_25_snli_ddropout_300':
        print('mt_both_snli_ddropout_300')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = constant_25_percent

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_finetune_200':
        print('mt_both_finetune_200')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, layers=2, mlp=200)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = mt_both_finetune_200_it10

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_finetune_300_dd':
        print('mt_both_finetune_300_dd')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = mt_both_finetune_200_it10

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_strong_decrease':
        print('mt_strong_decrease')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, layers=2)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = decrease_strong_mt_it10

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_strong_decrease_300_d':
        print('mt_strong_decrease_300_d')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout1(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = decrease_strong_mt_it10

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'tail_v_tail10':
        print('tail_v_tail10')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, layers=2)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = train_tailvtail_shape_it10

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_mlpsent_800_d':
        print('mt_both_mlpsent_800_d')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout_1layer(classifier)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_mlpsent_600_d':
        print('mt_both_mlpsent_600_d')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout_1layer(classifier)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_mlpsent_1200_d':
        print('mt_both_mlpsent_1200_d')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout_1layer(classifier)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_mlpsent_600_d_25':
        print('mt_both_mlpsent_600_d_25')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout_1layer(classifier)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = constant_25_percent

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_both_mlpsent_400_d':
        print('mt_both_mlpsent_400_d')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout_1layer(classifier)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(), classifier, embedding_holder)

    elif mt_type == 'mt_freeze1_300':
        print('mt_freeze1_300')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization
        params['after_epoch'] = freeze_after_first_epoch

        return MultitaskBuilder(params, lr, mt_target.get_targets(make_even_dist=True), classifier, embedding_holder)

    elif mt_type == 'mt_freeze1_300_25':
        print('mt_freeze1_300_25')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = constant_25_percent
        params['after_epoch'] = freeze_after_first_epoch

        return MultitaskBuilder(params, lr, mt_target.get_targets(make_even_dist=True), classifier, embedding_holder)

    elif mt_type == 'mt_freeze1_600_25':
        print('mt_freeze1_600')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw(classifier, mlp=600)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = constant_25_percent
        params['after_epoch'] = freeze_after_first_epoch

        return MultitaskBuilder(params, lr, mt_target.get_targets(make_even_dist=True), classifier, embedding_holder)


    elif mt_type == 'mt_25_snli_ddropout_300_noteven':
        print('mt_25_snli_ddropout_300_noteven')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = constant_25_percent

        return MultitaskBuilder(params, lr, mt_target.get_targets(make_even_dist=False), classifier, embedding_holder)


    elif mt_type == 'mt_both_snli_ddropout_600_noteven':
        print('mt_both_snli_ddropout_600_noteven')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout(classifier, mlp=600)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(make_even_dist=False), classifier, embedding_holder)

    elif mt_type == 'mt_both_snli_ddropout_300_noteven':
        print('mt_both_snli_ddropout_300_noteven')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_equal_both
        params['regularization_update'] = dummy_regularization

        return MultitaskBuilder(params, lr, mt_target.get_targets(make_even_dist=False), classifier, embedding_holder)
        
    elif mt_type == 'mt_25_snli_ddropout_300_masking':
        print('mt_25_snli_ddropout_300_masking')
        # weight both results the same, all the time
        params['multitask_network'] = get_multitask_nw_dropout(classifier, mlp=300)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_on_regularization
        params['regularization_update'] = constant_25_percent
        params['mask_sent'] = True

        return MultitaskBuilder(params, lr, mt_target.get_targets_with_positions(), classifier, embedding_holder)




        #return DataLoader(SentMTDataset(samples), drop_last=False, batch_size=512, shuffle=False, collate_fn=CollateBatchMultiTask()), len(samples)
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