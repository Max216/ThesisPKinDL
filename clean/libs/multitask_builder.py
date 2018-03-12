import torch
import torch.optim as optim

DEFAULT_LR = 0.0002

def _zero_grad_nothing(dummy):
    pass
def _zero_grad_obj(obj):
    obj.zero_grad()

class MultitaskBuilder:
    """
    Create all things required for the multitask training
    """

    def __init__(self, params, lr, multitask_data, classifier):
        self._multitask_network = params['multitask_network']()
        self._optimizer = params['optimizer'](classifier, self.multitask_network, lr)
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

    def add_evaluation(self):
        """ evaluate the samples and remember the results """
        pass

    def print_evaluation(self, premise_info, hypothesis_info):
        """ print evaluation """
        pass

    def loss(self, snli_loss, premise_info, hypothesis_info):
        """ Calculate the loss for thee gradient """
        multitask_loss = self.loss_fn_multitask(premise_info, hypothesis_info)
        return self._loss_fn(snli_loss, multitask_loss)

    def adjust_lr(self, new_lr):
        """ Adjust the learnrate """
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

def nothing(dummy1=None, dummy2=None):
    return None

def get_optimizer_snli_only(classifier, multitask_network, lr):
    return optim.Adam(classifier.parameters(), lr=lr)

def loss_snli_only(snli_loss, multitask_loss):
    return snli_loss

def get_builder(classifier, mt_type, mt_data, lr):
    params = dict()
    if mt_type == 'test_snli':
        # ignore multitask, verify that SNLI training works
        params['multitask_network'] = nothing
        params['optimizer'] = get_optimizer_snli_only
        params['loss_fn_multitask'] = nothing
        params['loss_fn'] = loss_snli_only

        return MultitaskBuilder(params, lr, mt_data, classifier)

    elif mt_type == 'test_mt':
        # ignore snli, verify that Multitask works
        pass