# For running on cluster
import os; 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import time
import copy
import matplotlib.pyplot as plt

import model
from model import cuda_wrap, EntailmentClassifier
import embeddingholder
import config
from config import *
import mydataloader

class CollocateBatch(object):
    '''
    Applies padding to shorter sentences within a minibatch.
    '''
    
    def __init__(self, padding_token):
        self.padding_token = padding_token
        
    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(self.padding_token)])
        
    def __call__(self, batch):
        sizes = torch.LongTensor([[len(premise), len(hypothesis)] for premise, hypothesis, _ in batch])
        #(max_length_premise, max_length_hypothesis), idxs = torch.max(sizes, dim=0)
        maxlen, idxs = torch.max(sizes, dim=0)
        maxlen = maxlen.view(-1)
        max_length_premise = maxlen[0]
        max_length_hypothesis = maxlen[1]
        
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

    @param model - model to be evaluated
    @param data - list of data chunks
    @param size - minibatch size
    @param padding_token - padding token embedding
    """

    loader = [DataLoader(chunk, 
                        drop_last = False,    # drops last batch if it is incomplete
                        batch_size=size, 
                        shuffle=False, 
                        #num_workers=0, 
                        collate_fn=CollocateBatch(padding_token)) for chunk in data]
    correct = 0
    total = sum([len(chunk) for chunk in data])
    for chunk in loader:
        for i_batch, (batch_p, batch_h, batch_lbl) in enumerate(chunk):
            predictions = model(autograd.Variable(cuda_wrap(batch_p)),
                                     autograd.Variable(cuda_wrap(batch_h))
                                    ).data
                                     
            _, predicted_idx = torch.max(predictions, dim=1)
            correct += torch.sum(torch.eq(batch_lbl, predicted_idx))
    
    # Accuracy
    return correct / total


def train_model(model, train_set, dev_set, padding_token, loss_fn, lr, epochs, batch_size, validate_after=50):
    '''
    Train the given model.

    @param model - model to train
    @param train_sets - list of chuncks of the train set. sentences of each chunk should be roughly evenly sized
                        in order to reduce padding
    @param dev_set - like above
    @param padding_token - index representing the padding embedding
    @param loss_fn - loss function
    @param lr - initial learning rate
    @param epochs - number of iterations over the whole data
    @param batch_size - minibatch size
    @param validate_after - after seeing this amount of samples the accuracy and error is calculated over the full 
                            train and dev data.
    '''

    # remember best params
    best_model = None
    best_dev_acc = 0
    best_train_acc = 0
    best_data_amount = 0

    loader_train = [DataLoader(chunk_train, 
                        drop_last = True,    # drops last batch if it is incomplete
                        batch_size=batch_size, 
                        shuffle=True, 
                        #num_workers=0, 
                        collate_fn=CollocateBatch(padding_token)) for chunk_train in train_set]
    
    # remember for weight decay
    start_lr = lr
    
    # switch between train/eval mode due to dropout
    model.train()
    start = time.time()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # if this is <= 0, validate model
    until_validation = 0
    amount_trained = 0
    
    all_amount_trained = []
    all_acc_train = []
    all_acc_dev = []
    all_err = []

    train_acc = -1
    dev_acc = -1
    for epoch in range(epochs):
        print('Train epoch', epoch + 1)
        
        total_loss = 0
        number_batches = 0
        # go through all chunks of train set
        for chunk in loader_train:
            # go through all minibatches of chunk
            for i_batch, (batch_p, batch_h, batch_lbl) in enumerate(chunk):
                number_batches += 1
                model.zero_grad()
                optimizer.zero_grad()
                
                var_p = autograd.Variable(cuda_wrap(batch_p))
                var_h = autograd.Variable(cuda_wrap(batch_h))
                var_label = autograd.Variable(cuda_wrap(batch_lbl))
                
                prediction = model(var_p, var_h)

                # calculates mean loss over whole batch
                loss = loss_fn(prediction, var_label)
                
                total_loss += loss.data
                
                loss.backward()
                optimizer.step()

                until_validation -= batch_size
                amount_trained += batch_size

                if until_validation <= 0:
                    #print('Current performance after seeing', amount_trained, 'samples:')
                    until_validation = validate_after # reset
                    model.eval()
                    train_acc = evaluate(model, train_set, batch_size, padding_token)
                    dev_acc = evaluate(model, dev_set, batch_size, padding_token)

                    all_acc_train.append(train_acc)
                    all_acc_dev.append(dev_acc)
                    all_amount_trained.append(amount_trained)
                    print('after seeing', amount_trained, 'samples:')
                    print('Accuracy on train data:', train_acc)
                    print('Accuracy on dev data:', dev_acc)
                    # mean loss per sample
                    all_err.append(total_loss[0] / number_batches)


                    model.train()

                    # remember best dev model
                    if dev_acc > best_dev_acc:
                        best_model = copy.deepcopy(model.state_dict())
                        best_dev_acc = dev_acc
                        best_train_acc = train_acc
                        best_data_amount = amount_trained
                        #print('Stored these model settings as best in this configuration.')
                
            # apply half decay learn rate (copied from original paper)
            decay = epoch // 2
            lr = start_lr / (2 ** decay)  
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        print('mean loss in epoch', epoch+1, ':', total_loss[0] / number_batches)
        print('Running time:', time.time() - start, 'seconds.')

    # Done training, return best settings
    return (best_model, best_data_amount, best_dev_acc, best_train_acc, all_acc_train, all_acc_dev, all_err, all_amount_trained)
    

def to_name(lr, dim_hidden, dim_sent_encoder, batch_size, data_size_train, data_size_dev):
    return str(lr).replace('.','_') + 'lr-' + \
        str(dim_hidden) + 'hidden-' + \
        str(dim_sent_encoder[0]) + '_' + str(dim_sent_encoder[1]) + '_' + str(dim_sent_encoder[2]) + 'lstm-' + \
        str(batch_size) + 'batch-' + \
        str(data_size_train) + '_' + str(data_size_dev) + \
        '-relu-0_1dropout' # fixed for now

def plot_learning(name, amount_data, acc_train, acc_dev, mean_loss):
    plt.plot(amount_data, acc_dev,label='dev set (accuracy)')
    plt.plot(amount_data, acc_train, label='train set (accuracy)')
    plt.plot(amount_data, mean_loss, label='mean loss on train')
    plt.xlabel('# samples')
    plt.ylabel('acccuracy/loss')
    plt.legend()
    plt.title(name)
    plt.savefig('./plots/' + name +'.png')
    plt.clf()


def search_best_model(train_data, dev_data, embedding_holder, lrs, dimens_hidden, dimens_sent_encoder, batch_sizes=[5], nonlinearities=[F.relu], dropouts=[0.1], epochs=50, plot=True, validate_after=50):
    torch.manual_seed(6)

    results = []
    best_model = None
    best_dev_acc = 0
    best_name = None

    for lr in lrs:
        for dim_hidden in dimens_hidden:
            for dim_sent_encoder in dimens_sent_encoder:
                for batch_size in batch_sizes:
                    for nonlinearity in nonlinearities:
                        for dropout in dropouts:

                            settings = (lr, dim_hidden, dim_sent_encoder, batch_size, nonlinearity, dropout)
                            print('#######################')
                            print('Now running:', settings)
                            # create model
                            classifier = cuda_wrap(EntailmentClassifier(embedding_holder.embeddings, 
                                            dimen_hidden=dim_hidden, 
                                            dimen_out=3, 
                                            dimen_sent_encoder=dim_sent_encoder,
                                            nonlinearity=nonlinearity, 
                                            dropout=dropout))

                            # train model
                            trained_model, data_used, dev_acc, train_acc, all_acc_train, all_acc_dev, all_mean_loss, amount_trained = train_model(classifier, 
                                train_set=train_data, 
                                dev_set=dev_data, 
                                padding_token=embedding_holder.padding(),
                                loss_fn=F.cross_entropy, 
                                lr=lr, epochs=epochs, 
                                batch_size=batch_size, 
                                validate_after=validate_after)

                            # remember results
                            result = (settings, data_used, dev_acc, train_acc)
                            results.append(result)

                            if(plot):
                                # plot learning curve
                                name = to_name(lr, dim_hidden, dim_sent_encoder, batch_size, len(train_data), len(dev_data))
                                plot_learning(name, amount_trained, all_acc_train, all_acc_dev, all_mean_loss)
                            
                            # remember best model
                            if dev_acc > best_dev_acc:
                                best_dev_acc = dev_acc
                                best_model = trained_model
                                best_name = to_name(lr, dim_hidden, dim_sent_encoder, batch_size, len(train_data), len(dev_data))
                                print('Stored as best model so far', result)

    print('Results:')
    print(results)
    print('Saving best model into', best_name + '.model')
    torch.save(best_model, 'models/' + best_name + '.model')
    print('Done.')
