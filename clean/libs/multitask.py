import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cu
from collections import defaultdict
from docopt import docopt
from torch.utils.data import DataLoader, Dataset

from libs import data_handler, multitask_builder, collatebatch, model_tools

import collections, time, sys, copy, random


# class TargetCreator:
#     """
#     To on the fly create target samples for the multi-task
#     """

#     def __init__(self, data_path, embedding_lookup):
#         self.lookup = embedding_lookup

#         with open(dataset_path) as f_in:
#             data = [line.strip().split('\t') for line in f_in.readlines()]

#         contradictions = collections.defaultdict(lambda: set())
#         entailments = collections.defaultdict(lambda: set())

#         for d in data:
#             if d[2] == 'contradiction':
#                 contradictions[d[0]].add(d[1])
#             else:
#                 entailments[d[0]].add(d[1])


#     def get_samples(self, sent_id, sent_repr):
#         pass




# class SentMTDataset(Dataset):
#     '''
#     Dataset format to give to classifier
#     '''

#     def __init__(self, samples, embedding_holder, tag_to_idx):
#         '''
#         Create a new dataset for the given samples
#         :param samples              parsed samples of the form [(premise, hypothesis, label)] (all strings)
#         :paraam embedding_holder    To map from word to number
#         :param tag_to_idx         dictionary mapping the string label to a number
#         '''
        
#         self.converted_samples = [(
#             torch.LongTensor([embedding_holder.word_index(w_sent) for w_sent in sent]),
#             embedding_holder.word_index(w),
#             tag_to_idx[lbl],
#             len(sent)
#         ) for sent, w, lbl in samples]

#     def __len__(self):
#         return len(self.converted_samples)

#     def __getitem__(self, idx):
#         return self.converted_samples[idx]

# class SentenceInOutTarget:
#     """
#     Creates targets considering only "contradiction" and "entailment" to determine if a word is within a
#     sentence or not.
#     """

#     def __init__(self, data_path, embedding_holder, dataset_path):
#         """
#         data_path for target.tsv
#         dataset for snli.jsonl
#         """

#         self.embedding_holder = embedding_holder
#         self.labels = dict([('entailment', 0), ('contradiction', 1)])

#         with open(data_path) as f_in:
#             data = [line.strip().split('\t') for line in f_in.readlines()]

#         sentence_dataset_handler = data_handler.Datahandler(dataset_path)
#         sentences = sentence_dataset_handler.get_sentences()

#         knowledge_dict_ent = collections.defaultdict(lambda : set())
#         knowledge_dict_contr = collections.defaultdict(lambda : set())
#         for d in data:
#             if d[2] == 'entailment':
#                 knowledge_dict_ent[d[0]].add(d[1])
#             elif d[2] == 'contradiction':
#                 knowledge_dict_contr[d[0]].add(d[1])
#             else:
#                 1/0

#         samples = []
#         for sent in sentences:
#             sent_set = set(sent)
#             entailing_words = set()
#             for w in list(sent_set):
#                 if w in knowledge_dict_ent:
#                     c_knowledge = list(knowledge_dict_ent[w])
#                     entailing_words.update(c_knowledge)
#                     samples.extend([(sent, w2, 'entailment') for w2 in c_knowledge])

#             for w in list(sent_set):
#                 if w in knowledge_dict_contr:
#                     c_knowledge = list(knowledge_dict_contr[w])
#                     samples.extend([(sent, w2, 'contradiction') for w2 in c_knowledge if w2 not in entailing_words])

#         self.samples = samples


#     def get_target_dataset(self):
#         return SentMTDataset(self.samples, self.embedding_holder, self.labels)


# class MTNetwork(nn.Module):
#     """
#     Map the embedding to a smaller represerntation
#     """

#     def __init__(self, classifier, input_dim, output_dim):
#         """
#         Initialize a new network to create representations based on WordNet information.
#         :param pretrained_embeddings    pretrained embedding matrix
#         :param hidden_layer_dimension   amoount of hidden nodes
#         :param representations          size of the resulting representation
#         """
#         super(MTNetwork, self).__init__()
#         self.classifier = classifier
#         self.layer = nn.Linear(input_dim, output_dim)

#     def forward(self, sent, target_word):
#         sentence_representation = self.classifier.forward_sent(sent)
#         batch_size = sentence_representation.size()[0]
#         word_representation = self.classifier.lookup_word(target_word).view(batch_size, -1)

#         print('#sr',sentence_representation.size())
#         print('##wr', word_representation.size())
#         feed_forward_input = torch.cat((sentence_representation, word_representation), 1)
#         print('ff input:', feed_forward_input.size())
#         return F.softmax(self.layer(feed_forward_input))


def make_even(entailing_words, contradicting_words):

    def fill_up(words, max_len):
        # first repeat fulll list
        w_copy = words[:]
        w_len = len(w_copy)
        while max_len - len(words) >= w_len:
            words.extend(w_copy)

        words.extend(random.sample(words, max_len - len(words)))

        return words

    if len(entailing_words) == 0 or len(contradicting_words) == 0:
        return (entailing_words, contradicting_words)
    else:
        if len(entailing_words) > len(contradicting_words):
            contradicting_words = fill_up(contradicting_words, len(entailing_words))
        else:
            entailing_words = fill_up(entailing_words, len(contradicting_words))

        return (entailing_words, contradicting_words)

class MultiTaskTarget:

    def __init__(self, datasets, resource_data_path, embedding_holder):
        #random.seed(7)
        print('init MultiTaskTarget')
        self.tag_to_idx = dict([('entailment', 1), ('contradiction', 0)])
        self.datasets = datasets
        self.resource_data_path = resource_data_path
        self.embedding_holder = embedding_holder

    
    def get_targets_with_positions(self,make_even_dist=True):
        with open(self.resource_data_path) as f_in:
            data = [line.strip().split('\t') for line in f_in.readlines()]

        not_in_sent_samples = collections.defaultdict(list) 
        in_sent_samples = collections.defaultdict(list)

        # go through resource and get word pairs into dictionaries
        for d in data:
            w1 = self.embedding_holder.word_index(d[0])
            w2 = self.embedding_holder.word_index(d[1])
            if d[2] == 'contradiction':
                not_in_sent_samples[w1].append(w2)
            elif d[2] == 'entailment':
                in_sent_samples[w1].append(w2)

        # create empty result arrays
        indizes = [(p_id, h_id) for ds in self.datasets for p,h,l,pl,hl,p_id,h_id in ds]
        max_id = max([_id1 for _id1, _id2 in indizes] + [_id2 for _id1, _id2 in indizes])
        print('maxid', max_id)

        # Iterate through each sentence in each dataset
        count = 0
        targets = [[] for i in range(max_id + 1)]
        all_sents = [[] for i in range(max_id + 1)]
        for dataset in self.datasets:
            for p,h,lbl,p_len,h_len,p_id,h_id in dataset:
                for sent, sent_id in [(p, p_id), (h, h_id)]:
                    if len(targets[sent_id]) == 0:
                        #print('in it')
                        entailing_words = list()
                        contradicting_words = list()
                        for w_idx in list(set(sent)):
                            if len(in_sent_samples[w_idx]) > 0:
                                entailing_words.append((w_idx, in_sent_samples[w_idx]))
                            if len(not_in_sent_samples[w_idx]) > 0:
                                contradicting_words.append((w_idx, not_in_sent_samples[w_idx]))

                        # clean not needed here
                        #contradicting_words = list(contradicting_words - entailing_words)
                        #entailing_words = list(entailing_words) 

                        #if make_even_dist:
                        #    #print('Make even dist')
                        #    entailing_words, contradicting_words = make_even(entailing_words, contradicting_words)
                        #else:
                        #    #print('Not make even dist')
                        #    pass
                        samples = [(source_w, target_ws, 0) for source_w, target_ws in contradicting_words] + [(source_w, target_ws, 1) for source_w, target_ws in entailing_words]
                        targets[sent_id] = samples
                        all_sents[sent_id] = sent

        target_words = [[0] for i in range(len(targets))]
        target_labels = [[0] for i in range(len(targets))]
        source_words = [[0] for i in range(len(targets))]
        target_has_content = [True for i in range(len(targets))]

        for i in range(len(targets)):
            if len(targets[i]) == 0:
                target_words[i] = False
                target_labels[i] = False
                source_words[i] = False
                target_has_content[i] = False
            else:
                #source_words, w_indizes,  labels = [torch.LongTensor(list(a)) for a in zip(*targets[i])]
                #print([a for a in zip(*targets[i])])
                source_w, target_ws, lbl = zip(*targets[i])
                target_ws = list(target_ws)
                target_words[i] = [torch.LongTensor(ws) for ws in target_ws]
                target_labels[i] = [torch.LongTensor([list(lbl)[j]] * len(target_ws[j])).view(-1) for j in range(len(target_ws))]
                target_labels[i] = torch.cat(target_labels[i], dim=0)
                source_words[i] = list(source_w)
                # adapt source words to positions in sentence
                current_sent = all_sents[i]
                source_words[i] = [[i for i in range(current_sent.size()[0]) if current_sent[i] == sw] for sw in source_words[i]]
                #print('pos adding')

            #print('added', target_words[i].size())

        self._target_words = target_words
        self._target_labels = target_labels
        self._target_has_content = target_has_content
        return self._target_words, self._target_labels, self._target_has_content, source_words


    def get_targets(self, make_even_dist=True):

        # create data
        with open(self.resource_data_path) as f_in:
            data = [line.strip().split('\t') for line in f_in.readlines()]
        
        not_in_sent_samples = collections.defaultdict(list) 
        in_sent_samples = collections.defaultdict(list)

        for d in data:
            w1 = self.embedding_holder.word_index(d[0])
            w2 = self.embedding_holder.word_index(d[1])
            if d[2] == 'contradiction':
                not_in_sent_samples[w1].append(w2)
                #print(d[0],d[1], 'c')
            elif d[2] == 'entailment':
                in_sent_samples[w1].append(w2)
                #print(d[0],d[1], 'e')

        indizes = [(p_id, h_id) for ds in self.datasets for p,h,l,pl,hl,p_id,h_id in ds]
        max_id = max([_id1 for _id1, _id2 in indizes] + [_id2 for _id1, _id2 in indizes])
        print('maxid', max_id)
        count = 0
        targets = [[] for i in range(max_id + 1)]
        for dataset in self.datasets:
            for p,h,lbl,p_len,h_len,p_id,h_id in dataset:
                for sent, sent_id in [(p, p_id), (h, h_id)]:
                    if len(targets[sent_id]) == 0:
                        #print('in it')
                        entailing_words = set()
                        contradicting_words = set()
                        for w_idx in sent:
                            #w_idx = embedding_holder.word_index(w)
                            #if w_idx in entailing_words:print('yay')
                            #print('###', in_sent_samples[w_idx])
                            entailing_words.update(in_sent_samples[w_idx])
                            contradicting_words.update(not_in_sent_samples[w_idx])
                            #print('###',not_in_sent_samples[w_idx])

                        contradicting_words = list(contradicting_words - entailing_words)
                        entailing_words = list(entailing_words) 

                        if make_even_dist:
                            #print('Make even dist')
                            entailing_words, contradicting_words = make_even(entailing_words, contradicting_words)
                        else:
                            #print('Not make even dist')
                            pass
                        samples = [(w, 0) for w in contradicting_words] + [(w,1) for w in entailing_words]
                        targets[sent_id] = samples

        #self.targets = targets
        #print(targets)
        #print('len targets:', len(targets))
        target_words = [[] for i in range(len(targets))]
        target_labels = [[] for i in range(len(targets))]
        target_has_content = [True for i in range(len(targets))]

        for i in range(len(targets)):
            if len(targets[i]) == 0:
                target_words[i] = False#torch.LongTensor([])
                target_labels[i] = False#torch.LongTensor(-1)#torch.LongTensor([])
                target_has_content[i] = False
                #print('False adding')
            else:
                #print('targets[i]', targets[i])
                #target_words, target_labels = [list(a) for a in zip(*targets[i])]
                #print('target words', target_words)
                #print('target_labels', target_labels)
                w_indizes,  labels = [torch.LongTensor(list(a)) for a in zip(*targets[i])]
                target_words[i] = w_indizes.view(-1,1)
                target_labels[i] = labels.view(-1)
                #print('pos adding')

            #print('added', target_words[i].size())

        self._target_words = target_words
        self._target_labels = target_labels
        self._target_has_content = target_has_content
        return self._target_words, self._target_labels, self._target_has_content





DEFAULT_ITERATIONS = 10
DEFAULT_LR = 0.0002
DEFAULT_VALIDATE_AFTER = [16000,16000, 8000,8000,2000]
#DEFAULT_VALIDATE_AFTER = [1000, 1000]
DEFAULT_BATCH_SIZE = 32
VALIDATE_AFTER_MT = 64000
def train_simult(model_name, classifier, embedding_holder, train_set, dev_set, train_path, multitask_type, multitask_data, make_even_dist=True):
    
    start_lr = DEFAULT_LR
    iterations = DEFAULT_ITERATIONS
    validate_after_vals = DEFAULT_VALIDATE_AFTER
    batch_size = DEFAULT_BATCH_SIZE


    classifier.train()
    
    mt_target = MultiTaskTarget([train_set, dev_set], multitask_data, embedding_holder)
    builder = multitask_builder.get_builder(classifier, multitask_type, mt_target, start_lr, embedding_holder)
    builder.train()

    train_loader = DataLoader(train_set, drop_last=True, batch_size=batch_size, shuffle=True, collate_fn=collatebatch.CollateBatchId(embedding_holder.padding()))
    dev_loader = DataLoader(dev_set, drop_last=False, batch_size=batch_size, shuffle=False, collate_fn=collatebatch.CollateBatchId(embedding_holder.padding()))

    best_dev_acc_snli = 0
    until_validation = 0
    until_validation_mt = 0
    samples_seen = 0
    start_time = time.time()



    for epoch in range(iterations):
        builder.next_epoch(epoch)

        # Output validation infos
        print(validate_after_vals)
        if len(validate_after_vals) > epoch:
            print('use current')
            validate_after = validate_after_vals[epoch]
        else:
            print('use last val')
            validate_after = validate_after_vals[-1]

        print('Train epoch', epoch + 1)
        print('Validate after:', validate_after)


        for premise_batch, hypothesis_batch, lbl_batch, premise_ids, hyp_ids in train_loader:
            #print('##')
            classifier.zero_grad()
            builder.zero_grad_multitask()

            samples_seen += DEFAULT_BATCH_SIZE
            until_validation -= DEFAULT_BATCH_SIZE
            until_validation_mt -= DEFAULT_BATCH_SIZE

            premise_var = autograd.Variable(premise_batch)
            hyp_var = autograd.Variable(hypothesis_batch)
            lbl_var = autograd.Variable(lbl_batch)

            # Main task prediction and loss
            prediction, activation_indizes, sentence_representations = classifier(premise_var, hyp_var, output_sent_info=True)
            loss = F.cross_entropy(prediction, lbl_var)
            #print('loss size', loss.size(), loss)

            premise_info = (premise_var, sentence_representations[0])
            hypothesis_info = (hyp_var, sentence_representations[1])

            backward_loss = builder.loss(loss, premise_info, hypothesis_info, premise_ids, hyp_ids, activation_indizes)
            backward_loss.backward()
            #torch.autograd.backward(backward_loss)#.backward()

            builder.optimizer_step()


            # Check if validate
            if until_validation <= 0:
                until_validation = validate_after

                if until_validation_mt <= 0:
                    until_validation_mt = VALIDATE_AFTER_MT
                    validate_mt = True
                    builder.new_evaluation()
                else:
                    validate_mt = False

                classifier.eval()
                builder.eval()

                correct_snli = 0
                total_snli = len(dev_set) 

                for premise_batch, hyp_batch, lbl_batch, premise_ids, hyp_ids in dev_loader:
                    premise_var = autograd.Variable(premise_batch)
                    hyp_var = autograd.Variable(hyp_batch)
                    
                    prediction, activation_indizes, sentence_representations = classifier(premise_var, hyp_var, output_sent_info=True)
                    _, predicted_idx = torch.max(prediction.data, dim=1)
                    correct_snli += torch.sum(torch.eq(lbl_batch, predicted_idx))

                    premise_info = (premise_var, sentence_representations[0])
                    hypothesis_info = (hyp_var, sentence_representations[1])

                    if validate_mt:
                        builder.add_evaluation(premise_info, hypothesis_info, premise_ids, hyp_ids, activations=activation_indizes)


                print('Running time:', time.time() - start_time, 'seconds')
                dev_acc = correct_snli / total_snli
                print('After', samples_seen, 'samples: dev accuracy (SNLI):', dev_acc)
                
                if validate_mt:
                    builder.print_evaluation()

                if dev_acc > best_dev_acc_snli:
                    best_model = copy.deepcopy(classifier.state_dict())
                    best_dev_acc_snli = dev_acc
                    model_tools.store(model_name, best_model, 'multitask')
                    print('Saved as best model!')

                sys.stdout.flush()
                classifier.train()
                builder.train()


        # Half weight decay every 2 epochs
        decay = epoch // 2
        lr = start_lr / (2 ** decay)  
        builder.adjust_lr(lr)
        
    print('Done.')
    sys.stdout.flush()



