import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cu
from collections import defaultdict
from docopt import docopt
from torch.utils.data import DataLoader, Dataset

from libs import data_handler


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
            torch.LongTensor([embedding_holder.word_index(w)]),
            tag_to_idx[lbl]
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

        self.labels = dict([('entailment', 0), ('contradiction', 1)])

        with open(data_path) as f_in:
            data = [line.strip().split('\t') for line in f_in.readlines()]

        sentence_dataset_handler = data_handler.DataHandler(dataset_path)
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
                    c_knowledge = list(knowledge_dic_ent[w])
                    entailing_words.add(w2)
                    samples.extend([(sent, w2, 'entailment') for w2 in c_knowledge])

            for w in list(sent_set):
                if w in knowledge_dict_contr:
                    c_knowledge = list(knowledge_dict_contr[w])
                    samples.extend([(sent, w2, 'contradiction') for w2 in c_knowledge if w2 not in entailing_words])

        self.samples = samples


    def get_target_dataset(self, embedding_holder):
        return SentMTDataset(self.samples, embedding_holder, self.labels)


class MTNetwork(nn.Module):
    """
    Map the embedding to a smaller represerntation
    """

    def __init__(self, classifier input_dim, output_dim):
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
        feed_forward_input = torch.cat((sentence_representation, target_words), 1)
        return F.softmax(self.layer(feed_forward_input))