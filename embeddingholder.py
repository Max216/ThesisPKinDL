import numpy as np
import gzip
import codecs

class EmbeddingHolder:
    
    """
    Load pretrained GloVe embeddings and makes them accessable.
    Extra symbols are added for OOV and Padding.
    """
    
    OOV = '@@OOV@@'
    PADDING = '@@PADDING@@'
    
    def __init__(self, path):


        # red previously stored binary word embeddings and vocab
        wv = np.load(path + '.npy')
        vocab_file = open(path + '.vocab', 'r')
        vocab = [w.rstrip('\n') for w in vocab_file]
        words = dict([(vocab[i], i) for i in range(len(vocab))])
        
        amount = wv.shape[0]
        self.dimen = wv.shape[1]
        
        # Add OOV and PADDING
        words[self.OOV] = amount
        words[self.PADDING] = amount+1
        unk = np.random.random_sample((wv.shape[1],))
        padding = np.zeros(self.dimen)
        wv = np.vstack((wv, unk, padding))

        
        self.words = words
        self.embeddings = wv
    
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
        if word in self.words:
            return self.words[word]
        else:
            return self.words[self.OOV]
        
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
