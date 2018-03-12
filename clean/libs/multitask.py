import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cu
from collections import defaultdict
from docopt import docopt
from torch.utils.data import DataLoader, Dataset

from libs import data_handler, multitask_builder, collatebatch

import collections, time, sys


class TargetCreator:
    """
    To on the fly create target samples for the multi-task
    """

    def __init__(self, data_path, embedding_lookup):
        self.lookup = embedding_lookup

        with open(dataset_path) as f_in:
            data = [line.strip().split('\t') for line in f_in.readlines()]

        contradictions = collections.defaultdict(lambda: set())
        entailments = collections.defaultdict(lambda: set())

        for d in data:
            if d[2] == 'contradiction':
                contradictions[d[0]].add(d[1])
            else:
                entailments[d[0]].add(d[1])


    def get_samples(self, sent_id, sent_repr):
        pass




class SentMTDataset(Dataset):
    '''
    Dataset format to give to classifier
    '''

    def __init__(self, samples, embedding_holder, tag_to_idx):
        '''
        Create a new dataset for the given samples
        :param samples              parsed samples of the form [(premise, hypothesis, label)] (all strings)
        :paraam embedding_holder    To map from word to number
        :param tag_to_idx         dictionary mapping the string label to a number
        '''
        
        self.converted_samples = [(
            torch.LongTensor([embedding_holder.word_index(w_sent) for w_sent in sent]),
            embedding_holder.word_index(w),
            tag_to_idx[lbl],
            len(sent)
        ) for sent, w, lbl in samples]

    def __len__(self):
        return len(self.converted_samples)

    def __getitem__(self, idx):
        return self.converted_samples[idx]

class SentenceInOutTarget:
    """
    Creates targets considering only "contradiction" and "entailment" to determine if a word is within a
    sentence or not.
    """

    def __init__(self, data_path, embedding_holder, dataset_path):
        """
        data_path for target.tsv
        dataset for snli.jsonl
        """

        self.embedding_holder = embedding_holder
        self.labels = dict([('entailment', 0), ('contradiction', 1)])

        with open(data_path) as f_in:
            data = [line.strip().split('\t') for line in f_in.readlines()]

        sentence_dataset_handler = data_handler.Datahandler(dataset_path)
        sentences = sentence_dataset_handler.get_sentences()

        knowledge_dict_ent = collections.defaultdict(lambda : set())
        knowledge_dict_contr = collections.defaultdict(lambda : set())
        for d in data:
            if d[2] == 'entailment':
                knowledge_dict_ent[d[0]].add(d[1])
            elif d[2] == 'contradiction':
                knowledge_dict_contr[d[0]].add(d[1])
            else:
                1/0

        samples = []
        for sent in sentences:
            sent_set = set(sent)
            entailing_words = set()
            for w in list(sent_set):
                if w in knowledge_dict_ent:
                    c_knowledge = list(knowledge_dict_ent[w])
                    entailing_words.update(c_knowledge)
                    samples.extend([(sent, w2, 'entailment') for w2 in c_knowledge])

            for w in list(sent_set):
                if w in knowledge_dict_contr:
                    c_knowledge = list(knowledge_dict_contr[w])
                    samples.extend([(sent, w2, 'contradiction') for w2 in c_knowledge if w2 not in entailing_words])

        self.samples = samples


    def get_target_dataset(self):
        return SentMTDataset(self.samples, self.embedding_holder, self.labels)


class MTNetwork(nn.Module):
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
        super(MTNetwork, self).__init__()
        self.classifier = classifier
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, sent, target_word):
        sentence_representation = self.classifier.forward_sent(sent)
        batch_size = sentence_representation.size()[0]
        word_representation = self.classifier.lookup_word(target_word).view(batch_size, -1)

        print('#sr',sentence_representation.size())
        print('##wr', word_representation.size())
        feed_forward_input = torch.cat((sentence_representation, word_representation), 1)
        print('ff input:', feed_forward_input.size())
        return F.softmax(self.layer(feed_forward_input))


DEFAULT_ITERATIONS = 10
DEFAULT_LR = 0.0002
#DEFAULT_VALIDATE_AFTER = [16000,2000]
DEFAULT_VALIDATE_AFTER = [1000,200]
DEFAULT_BATCH_SIZE = 32
def train_simult(model_name, classifier, embedding_holder, train_set, dev_set, train_path, multitask_type, multitask_data):
    
    start_time = time.time()
    start_lr = DEFAULT_LR
    iterations = DEFAULT_ITERATIONS
    validate_after_vals = DEFAULT_VALIDATE_AFTER
    samples_seen = 0
    batch_size = DEFAULT_BATCH_SIZE

    builder = multitask_builder.get_builder(classifier, multitask_type, multitask_data, start_lr)

    classifier.train()
    builder.train()
    train_loader = DataLoader(train_set, drop_last=True, batch_size=batch_size, shuffle=True, collate_fn=collatebatch.CollateBatch(embedding_holder.padding()))
    dev_loader = DataLoader(dev_set, drop_last=False, batch_size=batch_size, shuffle=False, collate_fn=collatebatch.CollateBatch(embedding_holder.padding()))

    best_dev_acc_snli = 0

    for epoch in range(iterations):

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


        for premise_batch, hypothesis_batch, lbl_batch in train_loader:
            #print('##')
            classifier.zero_grad()
            builder.zero_grad_multitask()

            samples_seen += DEFAULT_BATCH_SIZE
            until_validation -= DEFAULT_BATCH_SIZE

            premise_var = autograd.Variable(premise_batch)
            hyp_var = autograd.Variable(hyp_batch)
            lbl_var = autograd.Variable(lbl_batch)

            # Main task prediction and loss
            prediction, activation_indizes, sentence_representations = classifier(premise_var, hyp_var, output_sent_info=True)
            loss = F.cross_entropy(prediction, lbl_var)

            premise_info = (premise_var, sentence_representations[0])
            hypothesis_info = (hyp_var, sentence_representation[1])

            backward_loss = builder.loss(loss, premise_info, hypothesis_info)
            backward_loss.backward()

            builder.optimizer_step()


            # Check if validate
            if until_validation <= 0:
                until_validation = validate_after

                classifier.eval()
                builder.eval()

                correct_snli = 0
                total_snli = len(dev_set) 

                builder.new_evaluation()

                for premise_batch, hyp_batch, lbl_batch in dev_set:
                    premise_var = autograd.Variable(premise_batch)
                    hyp_var = autograd.Variable(hyp_batch)
                    
                    prediction, activation_indizes, sentence_representations = classifier(premise_var, hyp_var, output_sent_info=True)
                    _, predicted_idx = torch.max(prediction.data, dim=1)
                    correct_snli += torch.sum(torch.eq(lbl_batch, predicted_idx))

                    premise_info = (premise_var, sentence_representations[0])
                    hypothesis_info = (hyp_var, sentence_representation[1])
                    builder.add_evaluation(premise_info, hypothesis_info)


                print('Running time:', time.time() - start_time, 'seconds')
                dev_acc = correct_snli / total_snli
                print('After', samples_seen, 'samples: dev accuracy (SNLI):', dev_acc)
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



