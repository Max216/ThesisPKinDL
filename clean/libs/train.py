'''
Functions for training a model
'''
import time
import sys
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.autograd as autograd
import torch.nn.functional as F

from libs import model as m
from libs import model_tools, evaluate, collatebatch


DEFAULT_ITERATIONS = 10
DEFAULT_LR = 0.0002
DEFAULT_VALIDATE_AFTER = 2000
DEFAULT_BATCH_SIZE = 32


def train_model(name, classifier, padding_token, train_set, dev_set, iterations=DEFAULT_ITERATIONS, lr=DEFAULT_LR, validate_after=DEFAULT_VALIDATE_AFTER, batch_size=DEFAULT_BATCH_SIZE):
    '''
    Train a model and always store the best current result.

    :param classifier       Classifier object to be trained
    :param padding_token    Used to apply padding to sentences
    :param train_set        Data for training (Dataset)
    :param dev_set          Data for evaluation (Dataset)
    :param iterations       How many iterations are trained
    :param lr               learn rate
    :param validate_after   after seeing this many samples the model is evaluated
    :param batch_size       minibatch size
    '''

    classifier.train()
    torch.manual_seed(6)

    # remember when to validate:
    until_validation = 0
    
    # remember stats
    samples_seen = 0
    best_dev_acc = -1
    best_train_acc = -1
    best_model = None

    # actual training
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    train_loader = DataLoader(train_set, drop_last=True, batch_size=batch_size, shuffle=True, collate_fn=collatebatch.CollateBatch(padding_token))

    start_time = time.time()
    for epoch in range(iterations):
        print('Train epoch', epoch + 1)

        total_loss = 0
        number_batches = 0

        for premise_batch, hyp_batch, lbl_batch in train_loader:
            
            # update stats
            number_batches += 1
            until_validation -= batch_size
            samples_seen += batch_size

            # reset gradients
            classifier.zero_grad()
            optimizer.zero_grad()

            # predict
            premise_var = autograd.Variable(m.cuda_wrap(premise_batch))
            hyp_var = autograd.Variable(m.cuda_wrap(hyp_batch))
            lbl_var = autograd.Variable(m.cuda_wrap(lbl_batch))

            prediction = classifier(premise_var, hyp_var)
            loss = F.cross_entropy(prediction, lbl_var)
            total_loss += loss.data

            # update model
            loss.backward()
            optimizer.step()

            # Check if validate
            if until_validation <= 0:
                until_validation = validate_after

                # validate
                acc_train = evaluate.eval(classifier, train_set, batch_size, padding_token)
                acc_dev = evaluate.eval(classifier, dev_set, batch_size, padding_token)
                mean_loss = total_loss[0] / number_batches

                print('After seeing', samples_seen, 'samples:')
                print('Accuracy on train data:', acc_train)
                print('Accuracy on dev data:', acc_dev)   
                print('Mean loss:', mean_loss)
                sys.stdout.flush()

                classifier.train()

                if acc_dev > best_dev_acc:
                    best_model = copy.deepcopy(classifier.state_dict())
                    best_dev_acc = acc_dev
                    best_train_acc = acc_train

                    print('Saving current best model!')
                    sys.stdout.flush()
                    model_tools.store(name, best_model, 'temp')

