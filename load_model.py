import torch
import torch.nn.functional as F
import analyse

import config
import model
import train
import embeddingholder
import mydataloader
from model import *


embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)

# hyperparams of model
dim_hidden = 800
dim_sent_encoder=[64,128,256]
nonlinearity = F.relu
dropout=0.1

classifier = cuda_wrap(EntailmentClassifier(embedding_holder.embeddings, 
                                            dimen_hidden=dim_hidden, 
                                            dimen_out=3, 
                                            dimen_sent_encoder=dim_sent_encoder,
                                            nonlinearity=nonlinearity, 
                                            dropout=dropout))

print('Load best model', config.FILENAME_BEST_MODEL)
classifier.load_state_dict(torch.load(config.FILENAME_BEST_MODEL))
print('loaded')

snli_train = mydataloader.SNLIDataset(config.PATH_TRAIN_DATA, embedding_holder)
#result = train.evaluate(classifier, snli_dev, 32, embedding_holder.padding())

analyse.analyse(classifier, snli_train, 3, embedding_holder)