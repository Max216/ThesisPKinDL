import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cu
from collections import defaultdict
from docopt import docopt
from torch.utils.data import DataLoader


from libs import embeddingholder as eh

# For running on cluster
import os; 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
        self.embedding_layer.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.hidden_layer = nn.Linear(embedding_dim, hidden_layer_dimension)
        self.out_layer = nn.Linear(hidden_layer_dimension, representation_dimension)

    def forward(self, words):
        batch_size = sent1.size()[1]
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
        self.nonlinearity = F.relu
        self.embedding_encoder = embedding_encoder
        self.hidden_layer = nn.Linear(embedding_encoder.get_dimension(), hidden_layer_dimension * 2)
        self.out_layer  =nn.Linear(hidden_layer_dimension, out_dimension)

    def forward(self, words1, words2):
        batch_size = sent1.size()[1]
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
        self.nonlinearity = F.relu
        self.embedding_encoder = embedding_encoder
        self.out_layer = nn.Linear(embedding_encoder.get_dimension() * 2, out_dimension)

    def forward(self, words1, words2):
        batch_size = sent1.size()[1]
        representations1 = self.embedding_encoder(words1).view(batch_size, -1)
        representations2 = self.embedding_encoder(words2).view(batch_size, -1)

        feed_forward_input = torch.cat((representations1, representations2), 1)

        return F.softmax(self.out_layer(feed_forward_input))


def train(data_path, encoder_hidden_dim, encoder_out_dim, matcher_hidden_dim, out_path):
    lr = 4e-4
    iterations = 10
    validate_after = 100000
    batch_size = 128

    with open(data_path) as f_in:
        data = [line.strip() for line in f_in.readlines()]

    data = [(d[0], d[1], d[2]) for d in data]
    labels = sorted(list(set([lbl for w1, w2, lbl in data])))
    tag_to_idx = dict([(labels[i], i) for i in range(len(labels))])
    print(tag_to_idx)
    embedding_holder = eh.create_embeddingholder()

    # Train
    encoder = EmbeddingEncoder(embedding_holder.embedding_matrix(), encoder_hidden_dim, encoder_out_dim)
    if matcher_hidden_dim == 0:
        matcher = EmbeddingMatcherSimple(encoder, len(labels))
    else:
        matcher = EmbeddingMatcher(encoder, matcher_hidden_dim, len(labels))

    dataset = WordDataset(data, embedding_holder, tag_to_idx)
    data_loader = DataLoader(dataset, drop_last=False, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    for i in range(iterations):
        print('Train iteration:', i+1)
        for w1, w2, lbl in data_loader:
            print(w1, w2, lbl)


    # Write out embeddings
    

def main():

    USE_TREE = True

    args = docopt("""Create wordnet embeddings. 

    Usage:
        wn_representation_model.py train <train_data> <hidden_encoder> <representation_dim> <hidden_matcher> <save_path>

    """)

    if args['train']:
        train(args['<train_data>'], int(args['<hidden_encoder>']), int(args['<representation_dim>']),  int(args['<hidden_matcher>']),  args['<save_path>'])


if __name__ == '__main__':
    main()
