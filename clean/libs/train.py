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
from libs import model_tools, evaluate, collatebatch, multitask


DEFAULT_ITERATIONS = 6
DEFAULT_LR = 0.0002
DEFAULT_VALIDATE_AFTER = [16000,2000]
DEFAULT_BATCH_SIZE = 32


def train_model_multitask_sent_simult(name, classifier, padding_token, train_set_splits, dev_set, multitask_learner, multitask_target, iterations=DEFAULT_ITERATIONS, lr=DEFAULT_LR, validate_after_vals=DEFAULT_VALIDATE_AFTER, batch_size=DEFAULT_BATCH_SIZE, validate_train=False):
    classifier.train()
    multitask_learner.train()
    multitask_learner = m.cuda_wrap(multitask_learner)
    #torch.manual_seed(6)

    # remember when to validate:
    until_validation = 0
    
    # remember stats
    samples_seen = 0
    best_dev_acc = -1
    best_train_acc = -1
    best_model = None

    #multitask_learner = multitask.MTNetwork(classifier)

    # actual training
    print('params normal:', len(list(classifier.parameters())))
    print('params mt:', len(list(classifier.parameters())))
    print('params mt+normal:', len(list(set(list(classifier.parameters()) + list(multitask_learner.parameters())))))
    1/0
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    


    train_loader = [DataLoader(train_set, drop_last=True, batch_size=batch_size, shuffle=True, collate_fn=collatebatch.CollateBatch(padding_token)) for train_set in train_set_splits]
    mt_loader_train = DataLoader(multitask_target, drop_last=False, batch_size=batch_size, shuffle=True, collate_fn=collatebatch.CollateBatchSentWord(padding_token))
    mt_loader_eval = DataLoader(multitask_target, drop_last=False, batch_size=batch_size, shuffle=False, collate_fn=collatebatch.CollateBatchSentWord(padding_token))

    start_time = time.time()
    start_lr = lr
    for epoch in range(iterations):
        
        print(validate_after_vals)
        if len(validate_after_vals) > epoch:
            print('use current')
            validate_after = validate_after_vals[epoch]
        else:
            print('use last val')
            validate_after = validate_after_vals[-1]

        print('Train epoch', epoch + 1)
        print('Validate after:', validate_after)

        total_loss = 0
        number_batches = 0


        for train_split in train_loader:
            for premise_batch, hyp_batch, lbl_batch in train_split:
                
                # update stats
                number_batches += 1
                until_validation -= batch_size
                samples_seen += batch_size

                # reset gradients
                classifier.zero_grad()
                optimizer.zero_grad()

                # predict
                premise_var = autograd.Variable(premise_batch)
                hyp_var = autograd.Variable(hyp_batch)
                lbl_var = autograd.Variable(lbl_batch)

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
                    if validate_train:
                        acc_train = evaluate.eval_splits(classifier, train_set_splits, batch_size, padding_token)
                    else:
                        acc_train = 'not calculated'
                    acc_dev = evaluate.eval(classifier, dev_set, batch_size, padding_token)
                    mean_loss = total_loss[0] / number_batches

                    print('After seeing', samples_seen, 'samples:')
                    print('Accuracy on train data:', acc_train)
                    print('Accuracy on dev data:', acc_dev)   
                    print('Mean loss:', mean_loss)
                    running_time = time.time() - start_time
                    print('Running time:', running_time, 'seconds.')

                    sys.stdout.flush()

                    classifier.train()

                    if acc_dev > best_dev_acc:
                        best_model = copy.deepcopy(classifier.state_dict())
                        best_dev_acc = acc_dev
                        best_train_acc = acc_train

                        print('Saving current best model!')
                        sys.stdout.flush()
                        model_tools.store(name, best_model, 'temp')
        
        if epoch in mt_epochs:
            # init optimizer
            decay = epoch // 2
            lr = start_lr / (2 ** decay) 
            optimizer_mt = optim.Adam(multitask_learner.parameters(), lr=lr)
 
            # MULTITASK LEARN 
            for i in range(MT_ITER):
                for sent_batch, word_batch, lbl_batch in mt_loader_train:
                    # reset gradients
                    multitask_learner.zero_grad()
                    optimizer_mt.zero_grad()

                    # predict
                    sent_var = autograd.Variable(m.cuda_wrap(sent_batch))
                    word_var = autograd.Variable(m.cuda_wrap(word_batch.view(1,-1)))
                    lbl_var = autograd.Variable(m.cuda_wrap(lbl_batch))

                    print('sent:', sent_var.data.size())
                    print('words:', word_var.data.size())
                    print('labels:', lbl_var.data.size())
                    prediction = multitask_learner(sent_var, word_var)
                    loss = F.cross_entropy(prediction, lbl_var)

                    # update model
                    loss.backward()
                    optimizer_mt.step()

                    # Check if validate

                # validate
                multitask_learner.eval()
                acc_mt = evaluate.eval_simple_2(multitask_learner, mt_loader_eval)
                multitask_learner.train()

                print('After seeing', samples_seen, 'samples:')
                print('Accuracy on multitask:', acc_mt)
                running_time = time.time() - start_time

                sys.stdout.flush()

            # reinit optimizer normal
            optimizer = optim.Adam(classifier.parameters(), lr=start_lr)


        # Half decay lr
        decay = epoch // 2
        lr = start_lr / (2 ** decay)  
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            
        running_time = time.time() - start_time
        print('Running time:', running_time, 'seconds.')

def train_model_multitask_sent(name, classifier, padding_token, train_set_splits, dev_set, multitask_learner, multitask_target, iterations=DEFAULT_ITERATIONS, lr=DEFAULT_LR, validate_after_vals=DEFAULT_VALIDATE_AFTER, batch_size=DEFAULT_BATCH_SIZE, validate_train=False):
    classifier.train()
    multitask_learner.train()
    multitask_learner = m.cuda_wrap(multitask_learner)
    #torch.manual_seed(6)

    # remember when to validate:
    until_validation = 0
    
    # remember stats
    samples_seen = 0
    best_dev_acc = -1
    best_train_acc = -1
    best_model = None

    mt_epochs = set([2,4,6])
    MT_ITER = 2
    #multitask_learner = multitask.MTNetwork(classifier)

    # actual training
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    


    train_loader = [DataLoader(train_set, drop_last=True, batch_size=batch_size, shuffle=True, collate_fn=collatebatch.CollateBatch(padding_token)) for train_set in train_set_splits]
    mt_loader_train = DataLoader(multitask_target, drop_last=False, batch_size=batch_size, shuffle=True, collate_fn=collatebatch.CollateBatchSentWord(padding_token))
    mt_loader_eval = DataLoader(multitask_target, drop_last=False, batch_size=batch_size, shuffle=False, collate_fn=collatebatch.CollateBatchSentWord(padding_token))

    start_time = time.time()
    start_lr = lr
    for epoch in range(iterations):
        
        print(validate_after_vals)
        if len(validate_after_vals) > epoch:
            print('use current')
            validate_after = validate_after_vals[epoch]
        else:
            print('use last val')
            validate_after = validate_after_vals[-1]

        print('Train epoch', epoch + 1)
        print('Validate after:', validate_after)

        total_loss = 0
        number_batches = 0


        for train_split in train_loader:
            for premise_batch, hyp_batch, lbl_batch in train_split:
                
                # update stats
                number_batches += 1
                until_validation -= batch_size
                samples_seen += batch_size

                # reset gradients
                classifier.zero_grad()
                optimizer.zero_grad()

                # predict
                premise_var = autograd.Variable(premise_batch)
                hyp_var = autograd.Variable(hyp_batch)
                lbl_var = autograd.Variable(lbl_batch)

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
                    if validate_train:
                        acc_train = evaluate.eval_splits(classifier, train_set_splits, batch_size, padding_token)
                    else:
                        acc_train = 'not calculated'
                    acc_dev = evaluate.eval(classifier, dev_set, batch_size, padding_token)
                    mean_loss = total_loss[0] / number_batches

                    print('After seeing', samples_seen, 'samples:')
                    print('Accuracy on train data:', acc_train)
                    print('Accuracy on dev data:', acc_dev)   
                    print('Mean loss:', mean_loss)
                    running_time = time.time() - start_time
                    print('Running time:', running_time, 'seconds.')

                    sys.stdout.flush()

                    classifier.train()

                    if acc_dev > best_dev_acc:
                        best_model = copy.deepcopy(classifier.state_dict())
                        best_dev_acc = acc_dev
                        best_train_acc = acc_train

                        print('Saving current best model!')
                        sys.stdout.flush()
                        model_tools.store(name, best_model, 'temp')
        
        if epoch in mt_epochs:
            # init optimizer
            decay = epoch // 2
            lr = start_lr / (2 ** decay) 
            optimizer_mt = optim.Adam(multitask_learner.parameters(), lr=lr)
 
            # MULTITASK LEARN 
            for i in range(MT_ITER):
                for sent_batch, word_batch, lbl_batch in mt_loader_train:
                    # reset gradients
                    multitask_learner.zero_grad()
                    optimizer_mt.zero_grad()

                    # predict
                    sent_var = autograd.Variable(m.cuda_wrap(sent_batch))
                    word_var = autograd.Variable(m.cuda_wrap(word_batch.view(1,-1)))
                    lbl_var = autograd.Variable(m.cuda_wrap(lbl_batch))

                    print('sent:', sent_var.data.size())
                    print('words:', word_var.data.size())
                    print('labels:', lbl_var.data.size())
                    prediction = multitask_learner(sent_var, word_var)
                    loss = F.cross_entropy(prediction, lbl_var)

                    # update model
                    loss.backward()
                    optimizer_mt.step()

                    # Check if validate

                # validate
                multitask_learner.eval()
                acc_mt = evaluate.eval_simple_2(multitask_learner, mt_loader_eval)
                multitask_learner.train()

                print('After seeing', samples_seen, 'samples:')
                print('Accuracy on multitask:', acc_mt)
                running_time = time.time() - start_time

                sys.stdout.flush()

            # reinit optimizer normal
            optimizer = optim.Adam(classifier.parameters(), lr=start_lr)


        # Half decay lr
        decay = epoch // 2
        lr = start_lr / (2 ** decay)  
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            
        running_time = time.time() - start_time
        print('Running time:', running_time, 'seconds.')
def train_model(name, classifier, padding_token, train_set_splits, dev_set, iterations=DEFAULT_ITERATIONS, lr=DEFAULT_LR, validate_after_vals=DEFAULT_VALIDATE_AFTER, batch_size=DEFAULT_BATCH_SIZE, validate_train=False):
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
    #torch.manual_seed(6)

    # remember when to validate:
    until_validation = 0
    
    # remember stats
    samples_seen = 0
    best_dev_acc = -1
    best_train_acc = -1
    best_model = None

    # actual training
    optimizer = optim.Adam(classifier.parameters(), lr=lr)


    train_loader = [DataLoader(train_set, drop_last=True, batch_size=batch_size, shuffle=True, collate_fn=collatebatch.CollateBatch(padding_token)) for train_set in train_set_splits]

    start_time = time.time()
    start_lr = lr
    for epoch in range(iterations):
        print(validate_after_vals)
        if len(validate_after_vals) > epoch:
            print('use current')
            validate_after = validate_after_vals[epoch]
        else:
            print('use last val')
            validate_after = validate_after_vals[-1]

        print('Train epoch', epoch + 1)
        print('Validate after:', validate_after)

        total_loss = 0
        number_batches = 0

        for train_split in train_loader:
            for premise_batch, hyp_batch, lbl_batch in train_split:
                
                # update stats
                number_batches += 1
                until_validation -= batch_size
                samples_seen += batch_size

                # reset gradients
                classifier.zero_grad()
                optimizer.zero_grad()

                # predict
                premise_var = autograd.Variable(premise_batch)
                hyp_var = autograd.Variable(hyp_batch)
                lbl_var = autograd.Variable(lbl_batch)

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
                    if validate_train:
                        acc_train = evaluate.eval_splits(classifier, train_set_splits, batch_size, padding_token)
                    else:
                        acc_train = 'not calculated'
                    acc_dev = evaluate.eval(classifier, dev_set, batch_size, padding_token)
                    mean_loss = total_loss[0] / number_batches

                    print('After seeing', samples_seen, 'samples:')
                    print('Accuracy on train data:', acc_train)
                    print('Accuracy on dev data:', acc_dev)   
                    print('Mean loss:', mean_loss)
                    running_time = time.time() - start_time
                    print('Running time:', running_time, 'seconds.')

                    sys.stdout.flush()

                    classifier.train()

                    if acc_dev > best_dev_acc:
                        best_model = copy.deepcopy(classifier.state_dict())
                        best_dev_acc = acc_dev
                        best_train_acc = acc_train

                        print('Saving current best model!')
                        sys.stdout.flush()
                        model_tools.store(name, best_model, 'temp')

        # Half decay lr
        decay = epoch // 2
        lr = start_lr / (2 ** decay)  
        for pg in optimizer.param_groups:
            pg['lr'] = lr
            
        running_time = time.time() - start_time
        print('Running time:', running_time, 'seconds.')

