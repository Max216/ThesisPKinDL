import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cu
from collections import defaultdict
from docopt import docopt
from torch.utils.data import DataLoader, Dataset

import time
import sys
sys.path.append('./../') 

from libs import embeddingholder as eh

# For running on cluster
import os; 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# for cuda
def make_with_cuda(t):
    return t.cuda()

def make_without_cuda(t):
    return t

# always applies cuda() on model/tensor when available
cuda_wrap = None

if cu.is_available():
    print('Running with cuda enabled.')
    cuda_wrap = make_with_cuda
else:
    print('Running without cuda.')
    cuda_wrap = make_without_cuda


class EmbeddingEncoder(nn.Module):
    """
    Map the embedding to a smaller represerntation
    """

    def __init__(self, pretrained_embeddings, hidden_layer_dimension, representation_dimension):
        """
        Initialize a new network to create representations based on WordNet information.
        :param pretrained_embeddings    pretrained embedding matrix
        :param hidden_layer_dimension   amoount of hidden nodes
        :param representations          size of the resulting representation
        """
        super(EmbeddingEncoder, self).__init__()

        self.nonlinearity = F.relu
        self.representation_dimension = representation_dimension

        num_embeddings = pretrained_embeddings.shape[0]
        embedding_dim = pretrained_embeddings.shape[1]
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.weight.data.copy_(cuda_wrap(torch.from_numpy(pretrained_embeddings)))

        self.hidden_layer = nn.Linear(embedding_dim, hidden_layer_dimension)
        self.out_layer = nn.Linear(hidden_layer_dimension, representation_dimension)

    def forward(self, words):
        #batch_size = words.size()[1]
        embeddings = self.embedding_layer(words)

        out1 = self.nonlinearity(self.hidden_layer(embeddings))
        return F.tanh(self.out_layer(out1))

    def get_dimension(self):
        """
        :return the dimension of the resulting representation.
        """
        return self.representation_dimension


class WordDataset(Dataset):
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
            torch.LongTensor([embedding_holder.word_index(w1)]),
            torch.LongTensor([embedding_holder.word_index(w2)]),
            tag_to_idx[lbl]
        ) for w1, w2, lbl in samples]


    def __len__(self):
        return len(self.converted_samples)

    def __getitem__(self, idx):
        return self.converted_samples[idx]

class EmbeddingMatcher(nn.Module):
    """
    Learn to predict relations between words based on WordNet.
    """

    def __init__(self, embedding_encoder, hidden_layer_dimension, out_dimension):
        """
        todo
        """
        super(EmbeddingMatcher, self).__init__()
        self.nonlinearity = F.relu
        self.embedding_encoder = embedding_encoder
        self.hidden_layer = nn.Linear(embedding_encoder.get_dimension() * 2, hidden_layer_dimension)
        self.out_layer  =nn.Linear(hidden_layer_dimension, out_dimension)

    def forward(self, words1, words2):
        batch_size = words1.size()[0]
        representations1 = self.embedding_encoder(words1).view(batch_size, -1)
        representations2 = self.embedding_encoder(words2).view(batch_size, -1)

        feed_forward_input = torch.cat((representations1, representations2), 1)

        out1 = self.nonlinearity(self.hidden_layer(feed_forward_input))
        return F.softmax(self.out_layer(out1))

class EmbeddingMatcherSimple(nn.Module):
    """
    Learn to predict relations between words based on WordNet.
    """

    def __init__(self, embedding_encoder, out_dimension):
        """
        todo
        """
        super(EmbeddingMatcherSimple, self).__init__()
        self.nonlinearity = F.relu
        self.embedding_encoder = embedding_encoder
        print('embedding-out dimension:', embedding_encoder.get_dimension())
        self.out_layer = nn.Linear(embedding_encoder.get_dimension() * 2, out_dimension)

    def forward(self, words1, words2):
        #print('words shape:', words1.size(), words2.size())
        batch_size = words1.size()[0]
        #print('words1:', words1.size())
        representations1 = self.embedding_encoder(words1).view(batch_size, -1)
        representations2 = self.embedding_encoder(words2).view(batch_size, -1)

        #print('representation sizes:', representations1.size(), representations2.size())
        feed_forward_input = torch.cat((representations1, representations2), 1)

        #print('FF input:', feed_forward_input.size())
        return F.softmax(self.out_layer(feed_forward_input))

class CosSimMatcher(nn.Module):
    """
    Learn to predict relations between words based on WordNet.
    """

    def __init__(self, embedding_encoder):
        """
        todo
        """
        super(CosSimMatcher, self).__init__()
        self.embedding_encoder = embedding_encoder

    def forward(self, words1, words2):
        #print('words shape:', words1.size(), words2.size())
        batch_size = words1.size()[0]
        #print('words1:', words1.size())
        representations1 = self.embedding_encoder(words1).view(batch_size, -1)
        representations2 = self.embedding_encoder(words2).view(batch_size, -1)


        return F.cosine_similarity(representations1, representations2)



def train_cos(data_path, encoder_hidden_dim, encoder_out_dim, out_path, embedding_path):
    lr = 8e-4
    iterations = 600
    validate_after = 1024
    batch_size = 20

    with open(data_path) as f_in:
        data = [line.strip().split('\t') for line in f_in.readlines()]

    data = [(d[0], d[1], d[2]) for d in data]
    labels = sorted(list(set([lbl for w1, w2, lbl in data])))
    tag_to_idx = dict([(labels[i], i) for i in range(len(labels))])
    print(tag_to_idx)

    if embedding_path == None:
        embedding_holder = eh.create_embeddingholder()
    else:
        embedding_holder = eh.EmbeddingHolder(embedding_path)

    # Train
    encoder = cuda_wrap(EmbeddingEncoder(embedding_holder.embedding_matrix(), encoder_hidden_dim, encoder_out_dim))
    matcher = cuda_wrap(CosSimMatcher(encoder))

    dataset = WordDataset(data, embedding_holder, tag_to_idx)
    data_loader = DataLoader(dataset, drop_last=False, batch_size=batch_size, shuffle=True)
    eval_data_loader = DataLoader(dataset, drop_last=False, batch_size=batch_size, shuffle=False)

    start_time = time.time()
    reverse_embeddings = embedding_holder.reverse()
    optimizer = optim.Adam(matcher.parameters(), lr=lr)
    
    until_validation=0
    samples_seen = 0
    matcher.train()

    # verify that entailment is one!
    if tag_to_idx['entailment'] != 1 or len(tag_to_idx) != 2:
        print('entailment must be one, only two labels, or fix that!')
        1/0


    def calc_loss(prediction, lbl):
        multiplicator_entailment = var_lbl.data.clone().fill_(-1) * var_lbl.data 
        multiplicator_contradiction = var_lbl.data.clone().fill_(1) - var_lbl.data
        multiplicator = autograd.Variable(multiplicator_entailment + multiplicator_contradiction, requires_grad=False)
        print('pred',prediction.size())
        print('mult', multiplicator.size())
        loss = prediction * multiplicator.float()
        return loss.sum()

    for i in range(iterations):
        print('Train iteration:', i+1)
        for w1, w2, lbl in data_loader:
            #print(reverse_embeddings[w1[0][0]], reverse_embeddings[w2[0][0]], labels[lbl[0]])
            # reset gradients
            matcher.zero_grad()
            optimizer.zero_grad()

            #print('samples in batch:', lbl.size()[0])
            samples_seen += lbl.size()[0]
            until_validation -= lbl.size()[0]

            # predict
            var_w1 = autograd.Variable(cuda_wrap(w1))
            var_w2 = autograd.Variable(cuda_wrap(w2))
            var_lbl = autograd.Variable(cuda_wrap(lbl))

            prediction = matcher(var_w1, var_w2)

            

            loss = calc_loss(prediction, var_lbl)

            #F.cross_entropy(prediction, var_lbl)
            #total_loss += loss.data

            # update model
            loss.backward()
            optimizer.step()

            if until_validation <= 0:
                until_validation = validate_after

                # evaluate
                matcher.eval()
                correct = 0

                total_loss = 0

                for w1, w2, lbl in eval_data_loader:
                    prediction = matcher(
                        autograd.Variable(cuda_wrap(w1)),
                        autograd.Variable(cuda_wrap(w2))
                    )

                    print('#', prediction.size())
                    print('##', lbl.size())
                    total_loss += calc_loss(prediction, autograd.Variable(cuda_wrap(lbl))).data[0]

                    #_, predicted_idx = torch.max(prediction, dim=1)
                    #correct += torch.sum(torch.eq(cuda_wrap(lbl), predicted_idx))

                #total = len(data)
                #print('Accuracy after samples:', samples_seen, '->', correct/total)
                print('Loss:', total_loss)
                matcher.train()




    # Write out embeddings
    print('Write out to file')
    with open(out_path, 'w') as f_out:
        vocab = list(set([w1 for w1, w2, lbl in data] + [w2 for w1, w2, lbl in data]))
        matcher.eval()
        for w in vocab:
            w_index = autograd.Variable(cuda_wrap(torch.LongTensor([embedding_holder.word_index(w)]).view(1,-1)))
            #print(w_index.size())
            embedding = matcher.embedding_encoder(w_index).data[0].cpu().numpy().tolist()[0]
            f_out.write(w + ' ' + ' '.join([str(v) for v in embedding]) + '\n')

def train(data_path, encoder_hidden_dim, encoder_out_dim, matcher_hidden_dim, out_path, embedding_path):
    lr = 8e-4
    iterations = 600
    validate_after = 1024
    batch_size = 256

    with open(data_path) as f_in:
        data = [line.strip().split('\t') for line in f_in.readlines()]

    data = [(d[0], d[1], d[2]) for d in data]
    labels = sorted(list(set([lbl for w1, w2, lbl in data])))
    tag_to_idx = dict([(labels[i], i) for i in range(len(labels))])
    print(tag_to_idx)

    if embedding_path == None:
        embedding_holder = eh.create_embeddingholder()
    else:
        embedding_holder = eh.EmbeddingHolder(embedding_path)

    # Train
    encoder = cuda_wrap(EmbeddingEncoder(embedding_holder.embedding_matrix(), encoder_hidden_dim, encoder_out_dim))
    if matcher_hidden_dim == 0:
        matcher = cuda_wrap(EmbeddingMatcherSimple(encoder, len(labels)))
    else:
        matcher = cuda_wrap(EmbeddingMatcher(encoder, matcher_hidden_dim, len(labels)))

    dataset = WordDataset(data, embedding_holder, tag_to_idx)
    data_loader = DataLoader(dataset, drop_last=False, batch_size=batch_size, shuffle=True)
    eval_data_loader = DataLoader(dataset, drop_last=False, batch_size=batch_size, shuffle=False)

    start_time = time.time()
    reverse_embeddings = embedding_holder.reverse()
    optimizer = optim.Adam(matcher.parameters(), lr=lr)
    
    until_validation=0
    samples_seen = 0
    matcher.train()
    for i in range(iterations):
        print('Train iteration:', i+1)
        for w1, w2, lbl in data_loader:
            #print(reverse_embeddings[w1[0][0]], reverse_embeddings[w2[0][0]], labels[lbl[0]])
            # reset gradients
            matcher.zero_grad()
            optimizer.zero_grad()

            #print('samples in batch:', lbl.size()[0])
            samples_seen += lbl.size()[0]
            until_validation -= lbl.size()[0]

            # predict
            var_w1 = autograd.Variable(cuda_wrap(w1))
            var_w2 = autograd.Variable(cuda_wrap(w2))
            var_lbl = autograd.Variable(cuda_wrap(lbl))

            prediction = matcher(var_w1, var_w2)
            loss = F.cross_entropy(prediction, var_lbl)
            #total_loss += loss.data

            # update model
            loss.backward()
            optimizer.step()

            if until_validation <= 0:
                until_validation = validate_after

                # evaluate
                matcher.eval()
                correct = 0

                for w1, w2, lbl in eval_data_loader:
                    prediction = matcher(
                        autograd.Variable(cuda_wrap(w1)),
                        autograd.Variable(cuda_wrap(w2))
                    ).data

                    _, predicted_idx = torch.max(prediction, dim=1)
                    correct += torch.sum(torch.eq(cuda_wrap(lbl), predicted_idx))

                total = len(data)
                print('Accuracy after samples:', samples_seen, '->', correct/total)

                matcher.train()




    # Write out embeddings
    print('Write out to file')
    with open(out_path, 'w') as f_out:
        vocab = list(set([w1 for w1, w2, lbl in data] + [w2 for w1, w2, lbl in data]))
        matcher.eval()
        for w in vocab:
            w_index = autograd.Variable(cuda_wrap(torch.LongTensor([embedding_holder.word_index(w)]).view(1,-1)))
            #print(w_index.size())
            embedding = matcher.embedding_encoder(w_index).data[0].cpu().numpy().tolist()[0]
            f_out.write(w + ' ' + ' '.join([str(v) for v in embedding]) + '\n')
    

def main():

    USE_TREE = True

    args = docopt("""Create wordnet embeddings. 

    Usage:
        wn_representation_model.py train <train_data> <hidden_encoder> <representation_dim> <hidden_matcher> <save_path> [--embedding=<embedding>]
        wn_representation_model.py train_cos <train_data> <hidden_encoder> <representation_dim> <save_path> [--embedding=<embedding>]

    """)

    if args['train']:
        train(args['<train_data>'], int(args['<hidden_encoder>']), int(args['<representation_dim>']),  int(args['<hidden_matcher>']),  args['<save_path>'], args['--embedding'])
    elif args['train_cos']:
        train_cos(args['<train_data>'], int(args['<hidden_encoder>']), int(args['<representation_dim>']),  args['<save_path>'], args['--embedding'])


if __name__ == '__main__':
    main()
