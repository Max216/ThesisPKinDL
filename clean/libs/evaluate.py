'''
Methods for evaluation
'''

import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd


from libs import collatebatch
from libs import model as m


def eval(classifier, data, batch_size, padding_token, twister=None):
    '''
    Evaluate a model

    :param classifier           To be evaluated
    :param data                 To evaluate on
    :param batch_size           minibatch size
    :param padding_token        to pad sentences
    :param twister              if apply sentence representation twists
    '''
    classifier.eval()
    data_loader = DataLoader(data, drop_last=False, batch_size=batch_size, shuffle=False, collate_fn=collatebatch.CollateBatch(padding_token))

    correct = 0
    total = len(data)

    for premise_batch, hyp_batch, lbl_batch in data_loader:
        prediction = classifier(
            autograd.Variable(m.cuda_wrap(premise_batch)),
            autograd.Variable(m.cuda_wrap(hyp_batch)),
            twister=twister
        ).data

        # count corrects
        _, predicted_idx = torch.max(prediction, dim=1)
        correct += torch.sum(torch.eq(m.cuda_wrap(lbl_batch), predicted_idx))

    return correct / total