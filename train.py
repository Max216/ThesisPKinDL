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

import model
from model import cuda_wrap, EntailmentClassifier
import embeddingholder
import config
from config import *
import mydataloader
from mydataloader import SNLIDataset

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
        predictions = model(autograd.Variable(cuda_wrap(batch_p)),
                                 autograd.Variable(cuda_wrap(batch_h))
                                ).data
                                 
        _, predicted_idx = torch.max(predictions, dim=1)
        correct += torch.sum(torch.eq(batch_lbl, predicted_idx))
    
    # Accuracy
    return correct / len(data)


def train_model(model, train_set, dev_set, padding_token, loss_fn, lr, epochs, batch_size, validate_after=5):
    
    # remember best params
    best_model = None
    best_dev_acc = 0
    best_train_acc = 0
    best_epoch = 0

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
        #decay = epoch // 2
        #lr = start_lr / (2 ** decay)  
        #for pg in optimizer.param_groups:
        #    pg['lr'] = lr
            
        # Validate when specified and after the last epoch
        if (epoch % validate_after == 0) or (epoch == epochs - 1):
            model.eval()
            
            train_acc = evaluate(model, train_set, batch_size, padding_token)
            dev_acc = evaluate(model, dev_set, batch_size, padding_token)
            print('Accuracy on train data:', train_acc)
            print('Accuracy on dev data:', dev_acc)
            model.train()

            # remember best dev model
            if dev_acc > best_dev_acc:
                best_model = copy.deepcopy(model.state_dict())
                best_dev_acc = dev_acc
                best_train_acc = train_acc
                best_epoch = epoch + 1
                print('Stored these model settings as best in this configuration.')
        
        print('Loss:', total_loss[0])
        print('Running time:', time.time() - start, 'seconds.')

    # Done training, return best settings
    return (best_model, best_epoch, best_dev_acc, best_train_acc)
    

def search_best_model(train_data, dev_data, embedding_holder, lrs, dimens_hidden, dimens_sent_encoder, batch_sizes=[5], nonlinearities=[F.relu], dropouts=[0.1], epochs=50):
    torch.manual_seed(6)

    results = []
    best_model = None
    best_dev_acc = 0

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
                            trained_model, epochs, dev_acc, train_acc = train_model(classifier, 
                                train_set=train_data, 
                                dev_set=dev_data, 
                                padding_token=embedding_holder.padding(),
                                loss_fn=F.cross_entropy, 
                                lr=lr, epochs=epochs, 
                                batch_size=batch_size, 
                                validate_after=5)

                            # remember results
                            result = (settings, epochs, dev_acc, train_acc)
                            results.append(result)
                            
                            # remember best model
                            if dev_acc > best_dev_acc:
                                best_dev_acc = dev_acc
                                best_model = trained_model
                                print('Stored as best model so far', result)

    print('Results:')
    print(results)
    print('Saving best model into', FILENAME_BEST_MODEL)
    torch.save(best_model, FILENAME_BEST_MODEL)
    print('Done.')
