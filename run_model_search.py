import embeddingholder
import train
import mydataloader
import config
from config import *

embedding_holder = embeddingholder.EmbeddingHolder(PATH_WORD_EMBEDDINGS)
# How much data to load
SIZE_TRAIN = 30
SIZE_DEV = 5
        
snli_train = mydataloader.SNLIDataset(PATH_TRAIN_DATA, embedding_holder, max_size=SIZE_TRAIN)
snli_dev = mydataloader.SNLIDataset(PATH_DEV_DATA, embedding_holder, max_size=SIZE_DEV)


#model, epochs, dev_acc, train_acc = train_model(classifier, snli_train, snli_dev, 
#            embedding_holder.padding(),
#            F.cross_entropy, lr, epochs=50, batch_size=5, validate_after=5)

#print('best after ', epochs, ' acc:', dev_acc, train_acc)

lrs = [0.00005]
dimens_hidden=[800]
dimens_sent_encoder = [[64,128,256]]
batch_sizes=[5]

train.search_best_model(snli_train, snli_dev, embedding_holder, lrs, dimens_hidden, dimens_sent_encoder, batch_sizes, epochs=50)