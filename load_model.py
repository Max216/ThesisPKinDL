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

name='models/5e-05lr-800hidden-64_128_256lstm-5batch-6_3-relu-0_1dropout.model'

print('Load best model', name)

classifier.load_state_dict(torch.load(name))
print('loaded')

snli_train = mydataloader.SNLIDataset(config.PATH_TRAIN_DATA, embedding_holder)
#result = train.evaluate(classifier, snli_dev, 32, embedding_holder.padding())

analyse.analyse(classifier, snli_train, 3, embedding_holder)