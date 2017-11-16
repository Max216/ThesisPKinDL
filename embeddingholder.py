import numpy as np

class EmbeddingHolder:
    
    """
    Load pretrained GloVe embeddings and makes them accessable.
    Extra symbols are added for OOV and Padding.
    """
    
    OOV = '@@OOV@@'
    PADDING = '@@PADDING@@'
    
    def __init__(self, path):
        cnt = 0
        words = dict()
        vectors = []
        file = open(path, "r")
        for line in file:
            splitted_line = line.split()
            words[splitted_line[0]] = cnt
            vectors.append(np.asarray(splitted_line[1:], dtype='float'))
            cnt += 1
            
            # TODO rm
            #if cnt == 100:
            #    break
                
        self.dimen = len(vectors[0])
        print(len(vectors), 'word embeddings loaded.')
        
        # Add OOV and PADDING
        words[self.OOV] = cnt
        words[self.PADDING] = cnt+1
        oov_vector = np.random.rand(self.dimen)
        vectors.append(oov_vector)
        vectors.append(np.zeros(self.dimen))
        
        self.words = words
        self.embeddings = np.matrix(vectors)
    
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
