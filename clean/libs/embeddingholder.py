'''
Store and manage word embeddings
'''

import numpy as np
from libs import config

START_SENT = '<<start_sent>>'
END_SENT = '<<end_sent>>'

class EmbeddingHolder:
    
    """
    Load pretrained GloVe embeddings and makes them accessable.
    Extra symbols are added for OOV and Padding.
    """
    
    OOV = '@@oov@@'
    PADDING = '@@padding@@'


    
    def __init__(self, path, include_oov_padding = True, include_start_end=True):


        # red previously stored binary word embeddings and vocab
        wv = np.load(path + '.npy')
        vocab_file = open(path + '.vocab', 'r')
        vocab = [w.rstrip('\n') for w in vocab_file]
        words = dict([(vocab[i], i) for i in range(len(vocab))])
        print('loaded embd', wv.shape)
        amount = wv.shape[0]
        self.dimen = wv.shape[1]
        
        # Add OOV and PADDING
        next_idx = amount
        if include_oov_padding:
            words[self.OOV] = next_idx
            self.oov_index = next_idx
            next_idx += 1

            words[self.PADDING] = next_idx
            next_idx += 1
            unk = np.random.random_sample((wv.shape[1],))
            padding = np.zeros(self.dimen)
            wv = np.vstack((wv, unk, padding))

        if include_start_end:
            words[START_SENT] = next_idx
            next_idx += 1
            words[END_SENT] = next_idx
            next_idx += 1

            start = np.random.random_sample((wv.shape[1]))
            end = np.random.random_sample((wv.shape[1]))

            wv = np.vstack((wv, start, end))
        
        self.words = words
        self.embeddings = wv


    def stop_idx(self):
        return self.word_index(END_SENT)

    def concat(self, other):
        '''
        Concatenate each embedding with an additional vector. All unknowns will be set to zero.
        '''

        copied = np.zeros((self.embeddings.shape[0], other.embeddings.shape[1]))

        count_used_words = 0
        for w in self.words:
            if w in other.words:
                copied[self.words[w]] = other.embeddings[other.words[w]]
                count_used_words += 1


        self.embeddings = np.hstack((self.embeddings, copied))
        self.dimen = self.embeddings.shape[1]
        print('new matrix:', copied)
        print('Used:', count_used_words)
        print('new embedding shape:', self.embeddings.shape)


    
    def embedding_matrix(self):
        """
        Get the embedding matrix of the form:
        #vocab X #dimen i.e. every row represents one word
        """
        return self.embeddings
    
    def dim(self):
        """
        Get the dimension of the embeddings
        """
        return self.dimen
    
    def word_index(self, word):
        """
        Get the index of the given word within the embedding matrix.
        """
        return self.words.get(word, self.oov_index)
        
    def padding(self):
        """
        Get the index of the Padding symbol.
        """
        return self.word_index(self.PADDING)

    def reverse(self):
        """
        Get the reversed dictionary to lookup words from indizes
        """
        return dict((v,k) for k,v in self.words.items())

    def replace_unk(self, words):
        '''
        replaces tokens with "UNK" if they are not known for embeddings.
        '''
        return [w if w in self.words else w + '<UNK>' for w in words]

    def add_unknowns_from(self, other):
        '''
        Add all word embeddings from another embeddingholder that are not known to this instance. Already
        known words are untouched.
        E.g. to increase the embeddings with new vocabulary from the test set.

        @param other    embedding_holder containing new words
        '''

        # find new words
        words_this = list(self.words.keys())
        words_other = list(other.words.keys())
        new_words = np.setdiff1d(words_other, words_this)

        # matrix of new embeddings
        wv = np.asmatrix([other.embedding_matrix()[other.word_index(new_words[i])] for i in range(len(new_words))])

        # add words to vocab
        last_idx = len(self.words) 
        for w in new_words:
            self.words[w] = last_idx
            last_idx += 1

        print('Added', len(new_words), 'vocabs.')


        return wv

def create_embeddingholder(path=None, lower=None):
    if path == None and lower == 'lower':
        path = config.PATH_WORD_EMBEDDINGS_LOWER
    elif path == None:
        path = config.PATH_WORD_EMBEDDINGS

    return EmbeddingHolder(path)