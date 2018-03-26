'''
Methods for evaluation
'''

import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd


from libs import collatebatch, data_tools
from libs import model as m

ZERO = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

def predict_untokenized(classifier, embedding_holder, p_sentence, h_sentence, twister=None, index_to_tag=data_tools.DEFAULT_VALID_LABELS):
    p = data_tools._tokenize(p_sentence)
    h = data_tools._tokenize(h_sentence)

    p_batch = torch.LongTensor(len(p), 1)
    h_batch = torch.LongTensor(len(h), 1)

    p_batch[:,0] = torch.LongTensor([embedding_holder.word_index(w) for w in p])
    h_batch[:,0] = torch.LongTensor([embedding_holder.word_index(w) for w in h])
    
    scores = classifier(
        m.cuda_wrap(autograd.Variable(p_batch)), 
        m.cuda_wrap(autograd.Variable(h_batch)), 
        output_sent_info=False,
        twister=twister)

    _, predicted_idx = torch.max(scores, dim=1)
    return index_to_tag[predicted_idx.data[0]]

def predict_tokenized(classifier, embedding_holder, p, h, index_to_tag=data_tools.DEFAULT_VALID_LABELS):
    p_batch = torch.LongTensor(len(p), 1)
    h_batch = torch.LongTensor(len(h), 1)

    p_batch[:,0] = torch.LongTensor([embedding_holder.word_index(w) for w in p])
    h_batch[:,0] = torch.LongTensor([embedding_holder.word_index(w) for w in h])
    
    scores = classifier(
        m.cuda_wrap(autograd.Variable(p_batch)), 
        m.cuda_wrap(autograd.Variable(h_batch)), 
        output_sent_info=False)

    _, predicted_idx = torch.max(scores, dim=1)
    return index_to_tag[predicted_idx.data[0]]

def eval_splits(classifier, data_splits, batch_size, padding_token, twister=None):
    classifier.eval()
    data_loaders = [DataLoader(data, drop_last=False, batch_size=batch_size, shuffle=False, collate_fn=collatebatch.CollateBatch(padding_token)) for data in data_splits]

    correct = 0
    total = sum([len(data) for data in data_loaders])

    for loader in data_loaders:
        for premise_batch, hyp_batch, lbl_batch in loader:
            prediction = classifier(
                autograd.Variable(premise_batch),
                autograd.Variable(hyp_batch)#,
                #twister=twister
            ).data

            # count corrects
            _, predicted_idx = torch.max(prediction, dim=1)
            correct += torch.sum(torch.eq(lbl_batch, predicted_idx))

    return correct / total

def eval_simple_2(classifier, data_loader):
    correct = 0
    total = 0

    for v1, v2, lbl_batch in data_loader:
        prediction = classifier(
            autograd.Variable(v1),
            autograd.Variable(v2)#,
            #twister=twister
        ).data

        # count corrects
        _, predicted_idx = torch.max(prediction, dim=1)
        correct += torch.sum(torch.eq(lbl_batch, predicted_idx))
        total += predicted_idx.size()[0]

    return correct / total

def eval_merge_contr_neutr(classifier, data, batch_size, padding_token, tag_to_idx):

    idx_neutral = tag_to_idx['neutral']
    idx_contradiction = tag_to_idx['contradiction']

    diff_to_contr = idx_contradiction - idx_neutral

    classifier.eval()
    data_loader = DataLoader(data, drop_last=False, batch_size=batch_size, shuffle=False, collate_fn=collatebatch.CollateBatch(padding_token))

    correct = 0
    total = len(data)

    for premise_batch, hyp_batch, lbl_batch in data_loader:
        prediction = classifier(
            autograd.Variable(premise_batch),
            autograd.Variable(hyp_batch)#,
            #twister=twister
        ).data

        # count corrects
        #print(tag_to_idx)
        _, predicted_idx = torch.max(prediction, dim=1)
        merge_neutr_contr_vec_predicted = (predicted_idx == idx_neutral).long() * diff_to_contr
        merge_neutr_contr_vec_gold = (lbl_batch == idx_neutral).long() * diff_to_contr
        #print('predicted_idx before reassign', predicted_idx)
        #print('lbl before reassign', lbl_batch)
        predicted_idx = predicted_idx + merge_neutr_contr_vec_predicted
        lbl_batch = lbl_batch + merge_neutr_contr_vec_gold
        #print('predicted_idx after reassign', predicted_idx)
        #print('predicted_idx after reassign', lbl_batch)
        correct += torch.sum(torch.eq(lbl_batch, predicted_idx))

    return correct / total


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
            autograd.Variable(premise_batch),
            autograd.Variable(hyp_batch)#,
            #twister=twister
        ).data

        # count corrects
        _, predicted_idx = torch.max(prediction, dim=1)
        correct += torch.sum(torch.eq(lbl_batch, predicted_idx))

    return correct / total

def predict_outcomes(classifier, dataset, batch_size, padding_token, twister=None, idx_to_lbl=data_tools.DEFAULT_VALID_LABELS):
    classifier.eval()
    data_loader = DataLoader(dataset, drop_last=False, batch_size=batch_size,shuffle=False, collate_fn=collatebatch.CollateBatch(padding_token))

    predictions = []
    for premise_batch, hyp_batch, lbl_batch in data_loader:
        prediction = classifier(
            autograd.Variable(m.cuda_wrap(premise_batch)),
            autograd.Variable(m.cuda_wrap(hyp_batch))#,
            #twister=twister
        ).data

        # count corrects
        _, predicted_idx = torch.max(prediction, dim=1)
        predictions.extend([idx_to_lbl[i] for i in predicted_idx])
    
    return predictions

def print_misclassified(classifier, dataset, batch_size, padding_token, idx_to_lbl=data_tools.DEFAULT_VALID_LABELS, amount=20):
    classifier.eval()
    data_loader = DataLoader(dataset, drop_last=False, batch_size=batch_size,shuffle=False, collate_fn=collatebatch.CollateBatchIncludingSents(padding_token))

    predictions = []

    correct_samples = []
    incorrect_samples = []

    for premise_batch, hyp_batch, lbl_batch, p_sents, h_sents in data_loader:
        prediction = classifier(
            autograd.Variable(m.cuda_wrap(premise_batch)),
            autograd.Variable(m.cuda_wrap(hyp_batch))#,
        ).data

        # count corrects
        _, predicted_idx = torch.max(prediction, dim=1)
        print('predicted_idx', predicted_idx)
        print('lbl_battch', lbl_batch)
        predicted = predicted_idx.
        for i in range()
        predictions.extend([idx_to_lbl[i] for i in predicted_idx])

def create_prediction_dict(classifier, data, padding_token, idx_to_lbl, identifiers=None, twister=None):
    '''
    Create a dictionary with the prediction like dict[gold][predicted] = #amount. Batch size is one.
    :param classifier       classifier to be evaluated
    :param data             dataset for evaluation
    :param padding_token    to pad input
    :param identifiers      if set, an additional dictionary is returned with the identifiers per label
    :param twister          if apply sentence representation twists
    '''

    classifier.eval()
    data_loader = DataLoader(data, drop_last=False, batch_size=1, shuffle=False, collate_fn=collatebatch.CollateBatch(padding_token))

    prediction_dict = dict([(lbl, dict([(lbl2, 0) for lbl2 in idx_to_lbl])) for lbl in idx_to_lbl])
    identifier_dict = dict([(lbl, dict([(lbl2, []) for lbl2 in idx_to_lbl])) for lbl in idx_to_lbl])

    index = 0
    for premise_batch, hyp_batch, lbl_batch in data_loader:
        prediction = classifier(
            autograd.Variable(m.cuda_wrap(premise_batch)),
            autograd.Variable(m.cuda_wrap(hyp_batch))#,
            #twister=twister
        ).data

        # count corrects
        _, predicted_idx = torch.max(prediction, dim=1)

        gold_lbl = idx_to_lbl[lbl_batch.cpu()[0]]
        predicted_lbl = idx_to_lbl[predicted_idx.cpu()[0]]
        
        prediction_dict[gold_lbl][predicted_lbl] += 1
        if identifiers != None:
            identifier_dict[gold_lbl][predicted_lbl].append(identifiers[index])

        index += 1


    if identifiers == None:
        return prediction_dict
    else:
        return (prediction_dict, identifier_dict) 

def accuracy_prediction_dict(prediction_dict):
    correct = 0
    total = 0
    for gold_label in prediction_dict:
        for predicted_label in prediction_dict[gold_label]:
            if gold_label == predicted_label:
                correct += prediction_dict[gold_label][predicted_label]
            total += prediction_dict[gold_label][predicted_label]

    return correct / (total + ZERO)

def recall_precision_prediction_dict(prediction_dict, label):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for gold_label in prediction_dict:
        for predicted_label in prediction_dict[gold_label]:
            if gold_label == predicted_label:
                if gold_label == label:
                    tp += prediction_dict[gold_label][predicted_label]
                else:
                    tn += prediction_dict[gold_label][predicted_label]
            else:
                if predicted_label == label:
                    fp += prediction_dict[gold_label][predicted_label]
                elif gold_label == label:
                    fn += prediction_dict[gold_label][predicted_label]
            #if predicted_label == label:
            #    if gold_label == predicted_label:
            #        tp += prediction_dict[gold_label][predicted_label]
            #    else:
            #        fp += prediction_dict[gold_label][predicted_label]
            #else:
            #    if gold_label == predicted_label:
            #        fn += prediction_dict[gold_label][predicted_label]
            #    else:
            #        tn += prediction_dict[gold_label][predicted_label]

    #print('tp', tp, 'fp', fp, 'tn', tn, 'fn', fn)
    recall = tp / (tp + fn + ZERO)
    precision = tp / (tp + fp+ ZERO)
    return (recall, precision)


