import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn as nn




from libs import model as m

DEFAULT_LR = 0.0002

def _zero_grad_nothing(dummy):
    pass
def _zero_grad_obj(obj):
    obj.zero_grad()

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

    def forward(self, sent, target_word):
        sentence_representation = self.classifier.forward_sent(sent)
        batch_size = sentence_representation.size()[0]
        word_representation = self.classifier.lookup_word(target_word).view(batch_size, -1)

        feed_forward_input = torch.cat((sentence_representation, word_representation), 1)
        
        return F.softmax(self.layer(feed_forward_input))

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

    def forward(self, sent, target_word):
        sentence_representation = self.classifier.forward_sent(sent)
        batch_size = sentence_representation.size()[0]
        word_representation = self.classifier.lookup_word(target_word).view(batch_size, -1)

        feed_forward_input = torch.cat((sentence_representation, word_representation), 1)
        
        out1 = F.relu(self.layer1(feed_forward_input))
        return F.softmax(self.layer2(out1))

class MultitaskBuilder:
    """
    Create all things required for the multitask training
    """

    def __init__(self, params, lr, multitask_data, classifier):
        self._multitask_network = params['multitask_network']
        self._optimizer = params['optimizer'](classifier, self._multitask_network, lr)
        self._loss_fn = params['loss_fn']
        self._loss_fn_multitask = params['loss_fn_multitask']

        # helper functions
        if self._multitask_network == None:
            self._zero_grad = _zero_grad_nothing
        else:
            self._zero_grad = _zero_grad_obj

        # create data
        # TODO

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
        pass

    def print_evaluation(self):
        """ print evaluation """
        pass

    def loss(self, snli_loss, premise_info, hypothesis_info):
        """ Calculate the loss for thee gradient """
        multitask_loss = self._loss_fn_multitask(premise_info, hypothesis_info)
        return self._loss_fn(snli_loss, multitask_loss)

    def adjust_lr(self, new_lr):
        """ Adjust the learnrate """
        for pg in self._optimizer.param_groups:
            pg['lr'] = new_lr

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
    return multitask_loss


#
# Loss function for MultiTask
#
def loss_multitask_reweighted(premise_info, hypothesis_info, builder):
    """Average the loss over the batches of all samples created from these sentence pairs"""

    #premise_var, premise_repr = premise_info
    #hyp_var, hyp_repr = hypothesis_info
    samples, sample_count = builder.get_all_multitask_samples(premise_info, hypothesis_info)

    loss = autograd.Variable(torch.FloatTensor([0]))
    for batch_sents, batch_words, batch_lbl in samples:
        words_var = autograd.Variable(batch_words, requires_grad=False)
        predictions = builder.predict(batch_sents, words_var)
        loss = 0


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
        mt_network = MTNetworkSingleLayer(classifier, dim_input, 600, 2)

    return m.cuda_wrap(mt_network)

#
# Factory
#
def get_builder(classifier, mt_type, mt_data, lr):
    params = dict()
    if mt_type == 'test_snli':
        # ignore multitask, verify that SNLI training works
        params['multitask_network'] = None
        params['optimizer'] = get_optimizer_snli_only
        params['loss_fn_multitask'] = nothing
        params['loss_fn'] = loss_snli_only

        return MultitaskBuilder(params, lr, None, classifier)

    elif mt_type == 'test_mt':
        # ignore snli, verify that Multitask works
        params['multitask_network'] = get_multitask_nw(layers=1)
        params['optimizer'] = get_optimizer_multitask_only
        params['loss_fn_multitask'] = loss_multitask_reweighted
        params['loss_fn'] = loss_multitask_only

        return MultitaskBuilder(params, lr, mt_data, classifier)